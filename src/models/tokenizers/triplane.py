import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange, repeat

from src.constants import FieldName, Names, OutputsType
from src.utils.layers import RotaryPositionalEmbedding
from src.utils.typing import Optional

from .abstract_tokenizer import AbstractTokenizer


class TriplaneLearnablePositionalEmbedding(AbstractTokenizer):
    @dataclass
    class Config(AbstractTokenizer.Config):
        plane_size: int = 32
        num_channels: int = 1024

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.embeddings = nn.Parameter(
            torch.randn(
                (3, self.cfg.num_channels, self.cfg.plane_size, self.cfg.plane_size),
                dtype=torch.float32,
            )
            * 1
            / math.sqrt(self.cfg.num_channels)
        )

    def forward(self, outputs: OutputsType) -> OutputsType:
        batch_size = outputs[Names.BATCH_SIZE]
        return {
            self.cfg.tokenize_key: rearrange(
                repeat(self.embeddings, "Np Ct Hp Wp -> B Np Ct Hp Wp", B=batch_size),
                "B Np Ct Hp Wp -> B 1 Ct (Np Hp Wp)",
            )
        }

    def detokenize(self, outputs: OutputsType) -> OutputsType:
        tokens = outputs[self.cfg.detokenize_key]
        Ct, Nt = tokens.shape[-2:]
        assert Nt == self.cfg.plane_size**2 * 3
        assert Ct == self.cfg.num_channels
        return {
            self.cfg.detokenize_key: rearrange(
                tokens,
                "B 1 Ct (Np Hp Wp) -> B Np Ct Hp Wp",
                Np=3,
                Hp=self.cfg.plane_size,
                Wp=self.cfg.plane_size,
            )
        }

    def produced_keys(self):
        return super().produced_keys().union({self.cfg.tokenize_key})

    def consumed_keys(self):
        return super().consumed_keys().union({self.cfg.detokenize_key})


class SimpleTriplaneTokenizer(AbstractTokenizer):
    @dataclass
    class Config(AbstractTokenizer.Config):
        triplane_key: FieldName = Names.TRIPLANE.raw
        output_dimension: Optional[int] = None
        input_dimension: Optional[int] = None

    def configure(self) -> None:
        super().configure()
        self.cfg.is_output_tokenizer = False
        if self.cfg.output_dimension is not None:
            assert self.cfg.input_dimension is not None
            self.output_projection = nn.Linear(
                self.cfg.input_dimension, self.cfg.output_dimension
            )

    def forward(self, outputs: OutputsType) -> OutputsType:
        x = outputs[self.cfg.triplane_key]
        x = rearrange(x, "b np c h w -> b (np h w) c")
        if self.cfg.output_dimension is not None:
            x = self.output_projection(x)
        return {self.cfg.tokenize_key: x}

    def produced_keys(self):
        return super().produced_keys().union({self.cfg.tokenize_key})

    def consumed_keys(self):
        return super().consumed_keys().union({self.cfg.triplane_key})


class TriplaneTokenizer(AbstractTokenizer):
    """A tokenizer that processes triplane representations using a Vision Transformer architecture.

    This tokenizer takes a triplane representation (3 orthogonal planes) and processes it using
    patch embedding followed by transformer layers with rotary positional embeddings (RoPE).
    """

    @dataclass
    class Config(AbstractTokenizer.Config):
        """Configuration for TriplaneTokenizer.

        Args:
            n_layers (int): Number of transformer layers. Defaults to 5.
            hidden_features (int): Dimension of the transformer hidden states. Defaults to 512.
            activation (str): Activation function to use. Defaults to "relu".
            plane_size (int): Size of each plane (height/width). Defaults to 32.
            num_channels (int): Number of input channels. Defaults to 1024.
            patch_size (int): Size of patches for tokenization. Defaults to 4.
            num_heads (int): Number of attention heads in transformer. Defaults to 8.
            mlp_ratio (float): Ratio for MLP hidden dimension. Defaults to 4.0.
            dropout (float): Dropout rate. Defaults to 0.0.
        """

        n_layers: int = 5
        hidden_features: int = 512
        activation: str = "relu"

        triplane_key: FieldName = Names.TRIPLANE.raw
        plane_size: int = 32
        num_channels: int = 1024

        # Adding ViT-specific configs
        patch_size: int = 4
        num_heads: int = 8
        mlp_ratio: float = 4.0
        dropout: float = 0.0

    cfg: Config

    def configure(self) -> None:
        """Configures the tokenizer by initializing its components.

        Sets up the patch embedding layer, rotary positional embeddings,
        and transformer encoder layers.
        """
        super().configure()

        # Patch embedding layer
        self.patch_embed = nn.Conv2d(
            in_channels=self.cfg.num_channels,
            out_channels=self.cfg.hidden_features,
            kernel_size=self.cfg.patch_size,
            stride=self.cfg.patch_size,
        )

        # RoPE embeddings
        num_patches = (self.cfg.plane_size // self.cfg.patch_size) * (
            self.cfg.plane_size // self.cfg.patch_size * 3
        )

        self.rope = RotaryPositionalEmbedding(self.cfg.hidden_features, num_patches)

        # Use standard TransformerEncoderLayer
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.cfg.hidden_features,
                    nhead=self.cfg.num_heads,
                    dim_feedforward=int(self.cfg.hidden_features * self.cfg.mlp_ratio),
                    dropout=self.cfg.dropout,
                    activation=self.cfg.activation,
                    batch_first=True,
                )
                for _ in range(self.cfg.n_layers)
            ]
        )

        self.cfg.is_output_tokenizer = False

    def forward(self, outputs: OutputsType) -> OutputsType:
        """Processes the triplane representation through the tokenizer.

        Args:
            outputs (OutputsType): Dictionary containing input tensor with key specified
                by cfg.tokenize_key. Expected shape: [B, 3, C, H, W] where B is batch size,
                3 is number of planes, C is channels, H and W are spatial dimensions.

        Returns:
            OutputsType: Dictionary containing processed tokens with the same key.
                Output shape: [B, N, F] where N is number of tokens and F is hidden_features.
        """
        x = outputs[self.cfg.triplane_key]
        B, Np, C, H, W = x.shape

        # Rearrange planes into spatial dimension
        x = rearrange(x, "b np c h w -> b c h (w np)")  # Combine planes with width

        # Extract patches and embed
        embedded = self.patch_embed(x)  # [B, hidden_features, h', w'*Np]

        # Reshape to sequence
        # h = H // self.cfg.patch_size
        # w = W // self.cfg.patch_size * 3  # Width patches per plane
        embedded = rearrange(embedded, "b c h w -> b (h w) c")

        # Apply RoPE
        embedded = self.rope(embedded, self.freqs_cis)

        # Apply transformer layers
        for layer in self.transformer_layers:
            embedded = layer(embedded)

        return {self.cfg.tokenize_key: embedded}

    def produced_keys(self):
        return super().produced_keys().union({self.cfg.tokenize_key})

    def consumed_keys(self):
        return super().consumed_keys().union({self.cfg.triplane_key})

    def detokenize(self, outputs: OutputsType) -> OutputsType:
        """Does not apply here."""
        raise NotImplementedError(
            "Detokenization not implemented for TriplaneTokenizer"
        )
