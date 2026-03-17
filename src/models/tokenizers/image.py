from dataclasses import dataclass

import open_clip
import torch
import torch.nn as nn
from einops import rearrange

from src.constants import FieldName, Names, OutputsType
from src.models.tokenizers.base.dino import ViTModel
from src.models.tokenizers.base.dinov2 import Dinov2Model
from src.models.transformers.attention import Modulation
from src.utils.layers import RotaryPositionalEmbedding
from src.utils.ops import get_activation_module
from src.utils.typing import Optional

from .abstract_tokenizer import AbstractTokenizer


class DINOV2SingleImageTokenizer(AbstractTokenizer):
    @dataclass
    class Config(AbstractTokenizer.Config):
        pretrained_model_name_or_path: str = "facebook/dinov2-base"
        width: int = 224
        height: int = 224
        modulation_key: Optional[FieldName] = None
        modulation_zero_init: bool = False
        modulation_single_layer: bool = False
        modulation_cond_dim: int = 16
        append_conditioning: bool = False
        freeze_backbone_params: bool = True
        enable_memory_efficient_attention: bool = False
        enable_gradient_checkpointing: bool = False
        use_patch_embeddings: bool = False
        patch_embeddings_aggr_method: str = "concat"
        extra_input_key: Optional[FieldName] = None
        extra_input_dim: int = 0
        image_key: FieldName = Names.IMAGE.cond

    cfg: Config

    def configure(self) -> None:
        super().configure()
        if "dinov2" in self.cfg.pretrained_model_name_or_path:
            MODEL = Dinov2Model
        else:
            MODEL = ViTModel

        if self.cfg.freeze_backbone_params:
            # freeze dino backbone parameters
            self.register_non_module(
                "model",
                MODEL.from_pretrained(self.cfg.pretrained_model_name_or_path).to(
                    self.device
                ),
            )

            model = self.non_module("model")
            for p in model.parameters():
                p.requires_grad_(False)
            model.eval()
        else:
            self.model = MODEL.from_pretrained(
                self.cfg.pretrained_model_name_or_path
            ).to(self.device)
            model = self.model

        if self.cfg.extra_input_key:
            assert self.cfg.extra_input_dim, "Extra input key set, but its dimensionality is unknown (set `extra_input_dim`)"
            model.expand_input_channels(self.cfg.extra_input_dim)

        model.set_use_memory_efficient_attention_xformers(
            self.cfg.enable_memory_efficient_attention
        )
        model.set_gradient_checkpointing(self.cfg.enable_gradient_checkpointing)

        # add modulation
        if self.cfg.modulation_key is not None:
            modulations = []
            if self.cfg.modulation_cond_dim != model.config.hidden_size:
                self.modulation_projection = nn.Linear(
                    self.cfg.modulation_cond_dim, model.config.hidden_size
                )
            else:
                self.modulation_projection = nn.Identity()
            for layer in model.encoder.layer:
                norm1_modulation = Modulation(
                    model.config.hidden_size,
                    model.config.hidden_size,  # Use hidden_size instead of modulation_cond_dim
                    zero_init=self.cfg.modulation_zero_init,
                    single_layer=self.cfg.modulation_single_layer,
                )
                norm2_modulation = Modulation(
                    model.config.hidden_size,
                    model.config.hidden_size,  # Use hidden_size instead of modulation_cond_dim
                    zero_init=self.cfg.modulation_zero_init,
                    single_layer=self.cfg.modulation_single_layer,
                )
                layer.register_ada_norm_modulation(norm1_modulation, norm2_modulation)
                modulations += [norm1_modulation, norm2_modulation]
            self.modulations = nn.ModuleList(modulations)

        self.register_buffer(
            "image_mean",
            torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )

    def consumed_keys(self):
        consumed = super().consumed_keys() | {self.cfg.image_key}
        for key in [self.cfg.modulation_key, self.cfg.extra_input_key]:
            if key is not None:
                consumed.add(key)
        return consumed

    def produced_keys(self):
        return super().produced_keys() | {self.cfg.tokenize_key}

    def forward(self, outputs: OutputsType) -> OutputsType:
        modulation_cond = None
        if self.cfg.modulation_key is not None:
            modulation_cond = outputs[self.cfg.modulation_key]
            # Project modulation condition if needed
            modulation_cond = self.modulation_projection(modulation_cond)
        extra_input = (
            outputs[self.cfg.extra_input_key] if self.cfg.extra_input_key else None
        )

        model: Dinov2Model
        if self.cfg.freeze_backbone_params:
            model = self.non_module("model")
        else:
            model = self.model

        images = outputs[self.cfg.image_key]

        packed = False
        if images.ndim == 4:
            packed = True
            images = images.unsqueeze(1)
            if modulation_cond is not None:
                assert modulation_cond.ndim == 2
                modulation_cond = modulation_cond.unsqueeze(1)
            if extra_input is not None:
                assert extra_input.ndim == 4
                extra_input = extra_input.unsqueeze(1)

        images = rearrange(images, "B N H W C -> B N C H W")

        batch_size, n_input_views = images.shape[:2]
        images = (images - self.image_mean) / self.image_std
        if self.cfg.extra_input_key:
            extra_input = rearrange(extra_input, "B N H W C -> B N C H W")
            images = torch.cat([images, extra_input], dim=-3)
        out = model(
            rearrange(images, "B N C H W -> (B N) C H W"),
            modulation_cond=(
                rearrange(modulation_cond, "B N Cc -> (B N) Cc")
                if modulation_cond is not None
                else None
            ),
        )
        local_features = out.last_hidden_state
        if self.cfg.use_patch_embeddings:
            patch_embeddings = out.patch_embeddings
            if self.cfg.patch_embeddings_aggr_method == "concat":
                local_features = torch.cat([local_features, patch_embeddings], dim=1)
            elif self.cfg.patch_embeddings_aggr_method == "add":
                local_features = local_features + patch_embeddings
            else:
                raise NotImplementedError
        local_features = rearrange(
            local_features, "(B N) Nt Ct -> B N Nt Ct", B=batch_size
        )
        if packed:
            local_features = local_features.squeeze(1)

        if self.cfg.append_conditioning and modulation_cond is not None:
            # The modulation is expected to be the same dimensionality as the output tokens.
            local_features = torch.cat(
                [local_features, modulation_cond.unsqueeze(-2)], dim=-2
            )

        return {self.cfg.tokenize_key: local_features}


class CLIPImageTokenizer(AbstractTokenizer):
    @dataclass
    class Config(AbstractTokenizer.Config):
        model_name: str = "ViT-L/14"
        pretrained: str = "laion2b_s32b_b82k"
        freeze_backbone_params: bool = True
        image_key: FieldName = Names.IMAGE.cond
        output_dim: Optional[int] = None

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # Initialize OpenCLIP model
        model, _, _ = open_clip.create_model_and_transforms(
            self.cfg.model_name,
            pretrained=self.cfg.pretrained,
        )
        # Remove text encoder as we only need vision
        self.model = model.visual

        self.model = self.model.to(self.device)

        if self.cfg.freeze_backbone_params:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)

        # Check if the model has a input_resolution attribute
        if not hasattr(self.model, "input_resolution"):
            self.img_size = 224
        else:
            self.img_size = self.model.input_resolution
            # Check if img_size is subscribable and pick the first element
            if hasattr(self.img_size, "__getitem__"):
                self.img_size = self.img_size[0]

        # Register normalization constants
        self.register_buffer(
            "image_mean",
            torch.tensor(open_clip.OPENAI_DATASET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(open_clip.OPENAI_DATASET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

        if self.cfg.output_dim is not None:
            self.output_proj = nn.Linear(self.model.output_dim, self.cfg.output_dim)
        else:
            self.output_proj = nn.Identity()

    def forward(self, outputs: OutputsType) -> OutputsType:
        images = outputs[self.cfg.image_key]
        batch_size = images.shape[0]
        images = nn.functional.interpolate(
            rearrange(images, "B N H W C -> (B N) C H W"),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        images = (images - self.image_mean) / self.image_std
        features = self.model(images)

        features = rearrange(features, "(B N) C -> B N 1 C", B=batch_size)
        features = self.output_proj(features)
        return {self.cfg.tokenize_key: features}

    def consumed_keys(self):
        return super().consumed_keys() | {self.cfg.image_key}

    def produced_keys(self):
        return super().produced_keys() | {self.cfg.tokenize_key}


class ModulatedTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_module(activation)

    def forward(self, x, norm_cond=None, norm1_mod=None, norm2_mod=None):
        # First normalization and attention block
        x_norm = self.norm1(x)
        if norm1_mod is not None and norm_cond is not None:
            x_norm = norm1_mod(x_norm, norm_cond)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_output)

        # Second normalization and FFN block
        x_norm = self.norm2(x)
        if norm2_mod is not None and norm_cond is not None:
            x_norm = norm2_mod(x_norm, norm_cond)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        x = x + self.dropout2(ff_output)

        return x


class VisionTransformerImageTokenizer(AbstractTokenizer):
    @dataclass
    class Config(AbstractTokenizer.Config):
        image_key: FieldName = Names.IMAGE.cond
        n_input_channels: int = 3

        n_blocks: int = 12
        hidden_features: int = 512
        activation: str = "relu"

        width: int = 224
        height: int = 224
        patch_size: int = 4
        num_heads: int = 8
        mlp_ratio: float = 4.0
        dropout: float = 0.0
        modulation_key: Optional[FieldName] = None
        modulation_zero_init: bool = False
        modulation_single_layer: bool = False
        modulation_cond_dim: int = 16
        append_conditioning: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.patch_embed = nn.Conv2d(
            self.cfg.n_input_channels,
            self.cfg.hidden_features,
            self.cfg.patch_size,
            self.cfg.patch_size,
        )

        # RoPE embeddings
        num_patches = (self.cfg.width // self.cfg.patch_size) * (
            self.cfg.height // self.cfg.patch_size
        )

        self.rope = RotaryPositionalEmbedding(self.cfg.hidden_features, num_patches)

        transformer_layers = []
        modulations = []
        for _ in range(self.cfg.n_blocks):
            if self.cfg.modulation_key is not None:
                norm1_modulation = Modulation(
                    self.cfg.hidden_features,
                    self.cfg.modulation_cond_dim,
                    zero_init=self.cfg.modulation_zero_init,
                    single_layer=self.cfg.modulation_single_layer,
                )
                norm2_modulation = Modulation(
                    self.cfg.hidden_features,
                    self.cfg.modulation_cond_dim,
                    zero_init=self.cfg.modulation_zero_init,
                    single_layer=self.cfg.modulation_single_layer,
                )
                modulations.extend([norm1_modulation, norm2_modulation])

            transformer_layers.append(
                ModulatedTransformerEncoderLayer(
                    d_model=self.cfg.hidden_features,
                    nhead=self.cfg.num_heads,
                    dim_feedforward=int(self.cfg.hidden_features * self.cfg.mlp_ratio),
                    dropout=self.cfg.dropout,
                    activation=self.cfg.activation,
                )
            )

        self.transformer_layers = nn.ModuleList(transformer_layers)
        if self.cfg.modulation_key is not None:
            self.modulations = nn.ModuleList(modulations)

    def forward(self, outputs: OutputsType) -> OutputsType:
        modulation_cond = None
        if self.cfg.modulation_key is not None:
            modulation_cond = outputs[self.cfg.modulation_key]

        images = outputs[self.cfg.image_key]
        batch_size = images.shape[0]

        packed = False
        if images.ndim == 4:
            packed = True
            images = images.unsqueeze(1)
            if modulation_cond is not None:
                assert modulation_cond.ndim == 2
                modulation_cond = modulation_cond.unsqueeze(1)

        mod_cond = rearrange(modulation_cond, "B N Cc -> (B N) Cc")
        images = rearrange(images, "B N H W C -> (B N) C H W")

        x = self.patch_embed(images)
        x = rearrange(x, "b f h w -> b (h w) f")
        x = self.rope(x)

        for i, layer in enumerate(self.transformer_layers):
            if self.cfg.modulation_key is not None:
                mod_idx = i * 2
                x = layer(
                    x,
                    norm_cond=mod_cond,
                    norm1_mod=self.modulations[mod_idx],
                    norm2_mod=self.modulations[mod_idx + 1],
                )
            else:
                x = layer(x)

        x = rearrange(x, "(b n) t c -> b n t c", b=batch_size)
        if packed:
            x = x.squeeze(1)

        if self.cfg.append_conditioning and modulation_cond is not None:
            x = torch.cat([x, modulation_cond.unsqueeze(-2)], dim=-2)

        return {self.cfg.tokenize_key: x}

    def consumed_keys(self):
        consumed = super().consumed_keys() | {self.cfg.image_key}
        for key in [self.cfg.modulation_key]:
            if key is not None:
                consumed.add(key)
        return consumed

    def produced_keys(self):
        return super().produced_keys() | {self.cfg.tokenize_key}
