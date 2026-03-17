import math
from dataclasses import dataclass

import torch
from einops import repeat

from src.constants import Names, OutputsType
from src.utils.layers import RotaryPositionalEmbedding

from .abstract_tokenizer import AbstractTokenizer


class VectorProjectionTokenizer(AbstractTokenizer):
    """Linearly projects data to a set of tokens.

    The resulting tokens are each a full linear combination of all input channels."""

    @dataclass
    class Config(AbstractTokenizer.Config):
        in_dim: int = 1
        token_dim: int = 128
        nr_tokens: int = 64

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.linear_in = torch.nn.Linear(
            self.cfg.in_dim, self.cfg.token_dim * self.cfg.nr_tokens
        )
        self.linear_out = torch.nn.Linear(
            self.cfg.token_dim * self.cfg.nr_tokens, self.cfg.in_dim
        )

    def forward(self, outputs: OutputsType) -> OutputsType:
        data = outputs[self.cfg.tokenize_key]
        tokens_proj = self.linear_in(data).view(
            *data.shape[:-1], self.cfg.nr_tokens, self.cfg.token_dim
        )

        return {self.cfg.tokenize_key: tokens_proj}

    def detokenize(self, outputs: OutputsType) -> OutputsType:
        tokens = outputs[self.cfg.detokenize_key]
        tokens = tokens.view(*tokens.shape[:-2], -1)
        data_proj = self.linear_out(tokens)
        return {self.cfg.detokenize_key: data_proj}


class OrderedChannelTokenizer(AbstractTokenizer):
    """Linearly projects data to a set of tokens.

    The resulting tokens are each a variable linear lifting of one of the input channels.
    What this does is scale each of the input channels on their own D-dimensional line,
    and then calculate their distance to an arbitrary plane to gather their final value.
    (This motivates the offsets, as otherwise there is very little to distinguish the various tokens from one another if their channel values are close to zero).
    """

    @dataclass
    class Config(AbstractTokenizer.Config):
        in_dim: int = 1
        token_dim: int = 128
        transpose: bool = False

    cfg: Config

    inplace_module = True

    def configure(self) -> None:
        super().configure()
        # Rather than matrix multiplication, we'll do broadcasted hadamard multiplication.
        if self.cfg.is_input_tokenizer:
            self.matrix_in = torch.nn.Parameter(
                torch.randn(self.cfg.in_dim, self.cfg.token_dim)
            )
            self.offset_in = torch.nn.Parameter(
                torch.randn(self.cfg.in_dim, self.cfg.token_dim)
            )
            torch.nn.init.xavier_uniform_(self.matrix_in)
            torch.nn.init.xavier_uniform_(self.offset_in)
        # Same here: per-channel dot product for the final output.
        if self.cfg.is_output_tokenizer:
            self.matrix_out = torch.nn.Parameter(
                torch.randn(self.cfg.in_dim, self.cfg.token_dim)
            )
            self.offset_out = torch.nn.Parameter(
                torch.randn(self.cfg.in_dim, self.cfg.token_dim)
            )
            torch.nn.init.xavier_uniform_(self.matrix_out)
            torch.nn.init.xavier_uniform_(self.offset_out)

    def forward(self, outputs: OutputsType) -> OutputsType:
        data = outputs[self.cfg.tokenize_key]  #  ..., in_dim, 1
        full_shape = (*[1 for _ in data.shape[:-2]], *self.matrix_in.shape)
        tokens_lift = data * self.matrix_in.view(*full_shape) + self.offset_in.view(
            *full_shape
        )
        return {self.cfg.tokenize_key: tokens_lift}  # ..., in_dim, token_dim

    def detokenize(self, outputs: OutputsType) -> OutputsType:
        # There might be more tokens than we are configured for. If so, the extra ones are just computation registers and can be ignored.
        tokens = outputs[self.cfg.detokenize_key]
        if self.cfg.transpose:
            tokens = tokens.transpose(-1, -2)
        tokens = tokens[..., : self.cfg.in_dim, :]  # ..., in_dim, token_dim
        data_proj = (
            (
                tokens
                - self.offset_out.view(
                    *[1 for _ in tokens.shape[:-2]], *self.offset_out.shape
                )
            )
            * self.matrix_out.view(
                *[1 for _ in tokens.shape[:-2]], *self.matrix_out.shape
            )
        ).sum(dim=-1, keepdim=True)
        return {self.cfg.detokenize_key: data_proj}


class LearnableTokenBank(AbstractTokenizer):
    @dataclass
    class Config(AbstractTokenizer.Config):
        token_count: int = 512
        token_dim: int = 256
        transpose: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.embeddings = torch.nn.Parameter(
            torch.randn(
                (self.cfg.token_count, self.cfg.token_dim),
                dtype=torch.float32,
            )
            * 1
            / math.sqrt(self.cfg.token_dim)
        )

    def forward(self, outputs: OutputsType) -> OutputsType:
        batch_size = outputs[Names.BATCH_SIZE]
        return {
            self.cfg.tokenize_key: repeat(
                self.embeddings,
                "T D -> B D T" if self.cfg.transpose else "T D -> B T D",
                B=batch_size,
            ).contiguous(),
        }

    def produced_keys(self):
        return super().produced_keys().union({self.cfg.tokenize_key})


class RoPeLatentBank(AbstractTokenizer):
    @dataclass
    class Config(AbstractTokenizer.Config):
        token_count: int = 512
        token_dim: int = 256
        transpose: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.embeddings = torch.nn.Parameter(
            torch.randn(
                (self.cfg.token_count, self.cfg.token_dim),
                dtype=torch.float32,
            )
            * 1
            / math.sqrt(self.cfg.token_dim)
        )

        self.rope = RotaryPositionalEmbedding(self.cfg.token_dim, self.cfg.token_count)

    def forward(self, outputs: OutputsType) -> OutputsType:
        batch_size = outputs[Names.BATCH_SIZE]
        return {
            self.cfg.tokenize_key: repeat(
                self.rope(self.embeddings.unsqueeze(0))[0],
                "T D -> B D T" if self.cfg.transpose else "T D -> B T D",
                B=batch_size,
            ).contiguous(),
        }

    def produced_keys(self):
        return super().produced_keys().union({self.cfg.tokenize_key})
