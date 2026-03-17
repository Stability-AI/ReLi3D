import math
from dataclasses import dataclass

import torch
from diffusers.models.attention_processor import Attention
from einops import rearrange

from src.constants import FieldName, Names, OutputsType

from .abstract_tokenizer import AbstractTokenizer


class EnvQueryTokenizer(AbstractTokenizer):
    @dataclass
    class Config(AbstractTokenizer.Config):
        image_key: FieldName = Names.IMAGE.cond
        fg_mask_key: FieldName = Names.OPACITY.cond
        direction_key: FieldName = Names.REFLECTED_VIEW_DIRECTION.cond
        output_key: FieldName = Names.TOKEN.add_suffix("envquery")
        type_embedding_dim: int = 4
        num_heads: int = 16
        head_dim: int = 64
        bank_size: int = 1024
        token_dim: int = 1024  # Set to match DINOv2's token dimensionality.
        is_masked: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.type_embedding = torch.nn.Parameter(
            torch.randn(2, self.cfg.type_embedding_dim)
            / math.sqrt(self.cfg.type_embedding_dim)
        )
        self.learned_token_bank = torch.nn.Parameter(
            torch.randn(self.cfg.bank_size, self.cfg.token_dim)
            / math.sqrt(self.cfg.token_dim)
        )
        self.triplets_to_tokens = torch.nn.Sequential(
            torch.nn.Conv2d(
                6 + self.cfg.type_embedding_dim, self.cfg.token_dim // 16, 3
            ),
            torch.nn.SiLU(),
            torch.nn.Conv2d(
                self.cfg.token_dim // 16,
                self.cfg.token_dim // 8,
                3,
                stride=2,
                padding=1,
            ),
            torch.nn.SiLU(),
            torch.nn.Conv2d(
                self.cfg.token_dim // 8, self.cfg.token_dim // 4, 3, stride=2, padding=1
            ),
            torch.nn.SiLU(),
            torch.nn.Conv2d(
                self.cfg.token_dim // 4, self.cfg.token_dim // 4, 3, stride=2, padding=1
            ),
            torch.nn.SiLU(),
            torch.nn.Conv2d(
                self.cfg.token_dim // 4, self.cfg.token_dim, 3, stride=2, padding=1
            ),
        )
        self.attention = Attention(
            query_dim=self.cfg.token_dim,
            cross_attention_dim=self.cfg.token_dim,
            heads=self.cfg.num_heads,
            dim_head=self.cfg.head_dim,
        )

    def forward(self, outputs: OutputsType) -> OutputsType:
        image = outputs[self.cfg.image_key]
        direction = outputs[self.cfg.direction_key]
        fg_mask = outputs[self.cfg.fg_mask_key] > 0.05
        type_embeddings = self.type_embedding[fg_mask.squeeze(-1).long()]

        # Process (image, type_embeddings, direction) observation triplets to yield fancy tokens.
        # Main issue is that it'll create *way* too many tokens, so we'll use a token register
        # of fixed size to attend to these image-extracted ones and hopefully get some interesting stuff out.
        input_feature_images = torch.cat([image, type_embeddings, direction], dim=-1)
        if self.cfg.is_masked:
            input_feature_images = input_feature_images * fg_mask
        observation_tokens = rearrange(
            self.triplets_to_tokens(
                rearrange(input_feature_images, "... H W C -> (...) C H W")
            ),
            "B C H W -> B (H W) C",
        )
        observation_tokens = observation_tokens.view(
            *image.shape[:-3], *observation_tokens.shape[-2:]
        )

        # Attend to the learned token bank with the raw observations.
        batched_obs_tokens = observation_tokens.view(-1, *observation_tokens.shape[-2:])
        extracted_tokens = self.attention(
            hidden_states=self.learned_token_bank[None].expand(
                batched_obs_tokens.shape[0], -1, -1
            ),
            encoder_hidden_states=batched_obs_tokens,
        ).view(*observation_tokens.shape[:-2], self.cfg.bank_size, self.cfg.token_dim)

        return {self.cfg.tokenize_key: extracted_tokens}
