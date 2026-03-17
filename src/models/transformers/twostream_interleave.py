from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
import xformers
import xformers.ops
from torch import nn

from src.constants import FieldName, Names
from src.utils.base import BaseModule
from src.utils.typing import Optional


class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states, scale: float = 1.0):
        args = ()
        hidden_states, gate = self.proj(hidden_states, *args).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        kv_dim=None,
        num_heads=16,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        memory_efficient_attention: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        kv_dim = dim if not kv_dim else kv_dim
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.memory_efficient_attention = memory_efficient_attention

    def forward(self, x_q, x_kv):
        B, N_q, C = x_q.shape
        B, N_kv, _ = x_kv.shape
        # [B, N_q, C] -> [B, N_q, H, C/H]
        q = self.wq(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads)
        # [B, N_kv, C] -> [B, N_kv, H, C/H]
        k = self.wk(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads)
        v = self.wv(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads)

        # memory-efficient attention
        # TODO: do we have to move the head dim to the batch dim?
        if self.memory_efficient_attention:
            # torch.Size([1, 3089, 16, 64]) torch.Size([1, 27648, 16, 64]) torch.Size([1, 27648, 16, 64])
            x = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, p=self.attn_drop, op=None, scale=self.scale
            )
        else:
            x = torch.nn.functional.scaled_dot_product_attention(
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
                attn_mask=None,
                dropout_p=self.attn_drop,
                scale=self.scale,
            ).permute(0, 2, 1, 3)

        # [B, N_q, H, C/H] -> [B, N_q, C]
        x = x.reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        act_fn = GEGLU(dim, inner_dim)
        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            x = module(x)
        return x


class MixtureOfExpertsWrapper(nn.Module):
    def __init__(
        self,
        experts: Callable[[], nn.Module],
        num_experts: int,
        dim: int,
    ):
        super().__init__()
        self.shared_expert = experts()
        self.experts = nn.ModuleList([experts() for _ in range(num_experts)])
        self.num_experts = num_experts
        self.routing_network = nn.Linear(dim, num_experts)
        # Store the last expert weights for auxiliary loss calculation
        self.expert_logits = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gating_logits = self.routing_network(x)
        shared_expert_eval = self.shared_expert(x)

        self.expert_logits = gating_logits

        weights = torch.softmax(gating_logits, dim=-1)

        experts_eval = torch.stack([expert(x) for expert in self.experts], -1)

        return (
            shared_expert_eval + (experts_eval * weights.unsqueeze(-2)).sum(dim=-1)
        ) * 0.5

    def get_expert_logits(self) -> Optional[torch.Tensor]:
        """Returns the expert weights from the last forward pass for auxiliary loss calculation."""
        return self.expert_logits


class BasicBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kv_dim: Optional[int] = None,
        num_heads: int = 16,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ff_drop: float = 0.0,
        memory_efficient_attention: bool = True,
        num_experts: Optional[int] = None,
        num_cross_attentions: int = 1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(
            dim,
            kv_dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            memory_efficient_attention=memory_efficient_attention,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.num_cross_attentions = num_cross_attentions

        if num_cross_attentions > 0:
            # Register new cross attentions to self.attn(i+1)
            for i in range(num_cross_attentions):
                setattr(
                    self,
                    f"attn{i+2}",
                    CrossAttention(
                        dim,
                        kv_dim=kv_dim,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        memory_efficient_attention=memory_efficient_attention,
                    ),
                )
        self.norm3 = nn.LayerNorm(dim)

        def ff():
            return FeedForward(dim, dropout=ff_drop)

        if num_experts is not None:
            self.ff = MixtureOfExpertsWrapper(ff, num_experts, dim)
        else:
            self.ff = ff()

    def forward(self, z, x):
        z_norm = self.norm1(z)
        z = z + self.attn1(z_norm, z_norm)
        # TODO: do we need to have the second attention when x is None?
        z_norm = self.norm2(z)
        if self.num_cross_attentions > 1:
            assert isinstance(x, list) or x is None

        for i in range(self.num_cross_attentions):
            z = z + getattr(self, f"attn{i+2}")(
                z_norm,
                (x[i] if isinstance(x, list) else x) if x is not None else z_norm,
            )

        z_norm = self.norm3(z)
        z = z + self.ff(z_norm)
        return z


class FuseBlock(nn.Module):
    """
    Fuse X in to Z with cross attention
    """

    def __init__(
        self,
        dim_z: int,
        dim_x: int,
        num_heads: int = 16,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ff_drop: float = 0.0,
        norm_x_input: bool = True,
        memory_efficient_attention: bool = True,
    ):
        super().__init__()
        self.norm_x_input = norm_x_input
        if self.norm_x_input:
            self.norm_x = nn.LayerNorm(dim_x)
        self.attn = CrossAttention(
            dim_z,
            kv_dim=dim_x,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            memory_efficient_attention=memory_efficient_attention,
        )
        self.norm_z1 = nn.LayerNorm(dim_z)
        self.norm_z2 = nn.LayerNorm(dim_z)
        self.ff = FeedForward(dim_z, dropout=ff_drop)

    def forward(self, z, x):
        # TODO: do we need to normalize x?
        z = z + self.attn(self.norm_z1(z), self.norm_x(x) if self.norm_x_input else x)
        z = z + self.ff(self.norm_z2(z))
        return z


@torch.no_grad()
def get_triplane_attention_mask(res):
    N = 3 * res * res
    attn_mask = torch.zeros(3, res, res, 3, res, res)

    i, j = torch.meshgrid(torch.arange(res), torch.arange(res))

    attn_mask[0, i, j, 1, i, :] = 1.0
    attn_mask[0, i, j, 2, j, :] = 1.0
    attn_mask[1, i, j, 0, i, :] = 1.0
    attn_mask[1, i, j, 2, :, j] = 1.0
    attn_mask[2, i, j, 0, :, i] = 1.0
    attn_mask[2, i, j, 1, :, j] = 1.0
    attn_mask = attn_mask.bool()

    attn_bias = torch.empty_like(attn_mask, dtype=torch.float)
    attn_bias.masked_fill_(attn_mask, 0.0)
    attn_bias.masked_fill_(~attn_mask, float("-inf"))

    return attn_bias.reshape(N, N)


class TriplaneAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        resolution: int,
        num_heads: int = 16,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        full_attention: bool = False,
        memory_efficient_attention: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.resolution = resolution
        self.full_attention = full_attention
        self.attn_mask = (
            get_triplane_attention_mask(resolution) if not full_attention else None
        )
        self.memory_efficient_attention = memory_efficient_attention

    def forward(self, x):
        B, N, C = x.shape
        # [B, N, C] -> [B, N, H, C/H]
        q = self.wq(x).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads)

        # detokenize the planes
        assert N == self.resolution**2 * 3
        attn_bias = (
            self.attn_mask.to(q)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(B, self.num_heads, -1, -1)
            if not self.full_attention
            else None
        )

        if self.memory_efficient_attention:
            # memory-efficient attention
            x = xformers.ops.memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=attn_bias,
                p=self.attn_drop,
                op=None,
                scale=self.scale,
            )
        else:
            # full attention
            x = torch.nn.functional.scaled_dot_product_attention(
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
                attn_mask=attn_bias,
                dropout_p=self.attn_drop,
                scale=self.scale,
            ).permute(0, 2, 1, 3)

        # [B, N_q, H, C/H] -> [B, N_q, C]
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TriplaneAttnBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        resolution: int,
        num_heads: int = 16,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ff_drop: float = 0.0,
        full_attention: bool = False,
        memory_efficient_attention: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = TriplaneAttention(
            dim,
            resolution,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            full_attention=full_attention,
            memory_efficient_attention=memory_efficient_attention,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = TriplaneAttention(
            dim,
            resolution,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            full_attention=full_attention,
            memory_efficient_attention=memory_efficient_attention,
        )
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=ff_drop)

    def forward(self, z):
        z_norm = self.norm1(z)
        z = z + self.attn1(z_norm)
        z_norm = self.norm2(z)
        z = z + self.attn2(z_norm)
        z_norm = self.norm3(z)
        z = z + self.ff(z_norm)
        return z


class TwoStreamBlock(nn.Module):
    def __init__(
        self,
        dim_latent: int,
        dim_input: int,
        num_basic_blocks: int = 4,
        num_heads: int = 16,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ff_drop: float = 0.0,
        triplane_attention: bool = False,
        triplane_resolution: int = 32,
        full_attention: bool = False,
        memory_efficient_attention: bool = True,
        gradient_checkpointing: bool = False,
        norm_x_input: bool = True,
        dim_cross: Optional[int] = None,
        num_experts: Optional[int] = None,
        num_cross_attentions: int = 1,
    ):
        super().__init__()

        self.gradient_checkpointing = gradient_checkpointing

        # Define the fuse block that fuse the input into the latent
        self.fuse_block_in = FuseBlock(
            dim_latent,
            dim_input,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ff_drop=ff_drop,
            norm_x_input=norm_x_input,
            memory_efficient_attention=memory_efficient_attention,
        )

        # Define the transformer block that process the latent
        self.transformer_block = nn.ModuleList(
            [
                BasicBlock(
                    dim_latent,
                    kv_dim=dim_cross,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop,
                    ff_drop=ff_drop,
                    memory_efficient_attention=memory_efficient_attention,
                    num_experts=num_experts,
                    num_cross_attentions=num_cross_attentions,
                )
                for _ in range(num_basic_blocks)
            ]
        )

        # Define the fuse block that fuse the latent into the input
        self.fuse_block_out = FuseBlock(
            dim_input,
            dim_latent,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ff_drop=ff_drop,
            norm_x_input=norm_x_input,
            memory_efficient_attention=memory_efficient_attention,
        )

        self.triplane_attention = None
        if triplane_attention:
            self.triplane_attention = TriplaneAttnBlock(
                dim_input,
                resolution=triplane_resolution,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                ff_drop=ff_drop,
                full_attention=full_attention,
                memory_efficient_attention=memory_efficient_attention,
            )

    def forward(self, latent, input, cross_input):
        if self.training and self.gradient_checkpointing:
            latent = torch.utils.checkpoint.checkpoint(
                self.fuse_block_in, latent, input, use_reentrant=False
            )
            for block in self.transformer_block:
                latent = torch.utils.checkpoint.checkpoint(
                    block, latent, cross_input, use_reentrant=False
                )
            input = torch.utils.checkpoint.checkpoint(
                self.fuse_block_out, input, latent, use_reentrant=False
            )
            if self.triplane_attention is not None:
                input = torch.utils.checkpoint.checkpoint(
                    self.triplane_attention, input, use_reentrant=False
                )
        else:
            latent = self.fuse_block_in(latent, input)
            for block in self.transformer_block:
                latent = block(latent, cross_input)
            input = self.fuse_block_out(input, latent)
            if self.triplane_attention is not None:
                input = self.triplane_attention(input)
        return latent, input


class TwoStreamInterleaveTransformer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        num_attention_heads: int = 16
        attention_head_dim: int = 88
        raw_triplane_channels: int = 1024
        triplane_channels: int = 1024
        raw_image_channels: int = 1024
        num_latents: int = 1024
        num_blocks: int = 6
        num_basic_blocks: int = 4
        dropout: float = 0.0
        latent_init_std: float = 0.02
        norm_num_groups: int = 32
        attention_bias: bool = False
        triplane_attention: bool = False
        triplane_resolution: int = 32
        triplane_full_attention: bool = False
        enable_memory_efficient_attention: bool = True
        gradient_checkpointing: bool = False
        norm_x_input: bool = False
        cross_attention_dim: int = 1024
        mix_latent: bool = True
        mix_latent_max_tokens: int = -1
        mix_latents_self_attention: bool = False
        num_experts: Optional[int] = None
        start_experts_from_block: int = 0
        num_cross_attentions: int = 1
        output_key: FieldName = Names.TOKEN

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.gradient_checkpointing = self.cfg.gradient_checkpointing
        self.mix_latent = self.cfg.mix_latent

        # Define the dimensions
        self.num_attention_heads = self.cfg.num_attention_heads
        self.attention_head_dim = self.cfg.attention_head_dim
        self.num_latents = self.cfg.num_latents
        self.latent_dim = self.num_attention_heads * self.attention_head_dim

        # Define input layers
        if self.cfg.norm_num_groups > 0:
            self.norm_triplane = torch.nn.GroupNorm(
                num_groups=self.cfg.norm_num_groups,
                num_channels=self.cfg.raw_triplane_channels,
                eps=1e-6,
                affine=True,
            )
        else:
            self.norm_triplane = nn.LayerNorm(self.cfg.raw_triplane_channels)
        self.proj_triplane = nn.Linear(
            self.cfg.raw_triplane_channels, self.cfg.triplane_channels
        )
        if self.mix_latent:
            self.norm_image = nn.LayerNorm(self.cfg.raw_image_channels)
            self.proj_image = nn.Linear(self.cfg.raw_image_channels, self.latent_dim)
        self.norm_latent = nn.LayerNorm(self.latent_dim)
        self.proj_latent = nn.Linear(self.latent_dim, self.latent_dim)

        # Define the latents
        self.latent_init = nn.Parameter(
            torch.zeros(1, self.num_latents, self.latent_dim)
        )
        nn.init.normal_(self.latent_init, std=self.cfg.latent_init_std)

        if self.cfg.mix_latents_self_attention:
            self.mix_latents_self_attention = CrossAttention(
                self.latent_dim,
                kv_dim=self.latent_dim,
                num_heads=self.num_attention_heads,
                qkv_bias=self.cfg.attention_bias,
                attn_drop=self.cfg.dropout,
                proj_drop=self.cfg.dropout,
                memory_efficient_attention=self.cfg.enable_memory_efficient_attention,
            )
        else:
            self.mix_latents_self_attention = None

        # Define the transformer blocks
        self.main_blocks = nn.ModuleList(
            [
                TwoStreamBlock(
                    self.latent_dim,
                    self.cfg.triplane_channels,
                    num_basic_blocks=self.cfg.num_basic_blocks,
                    num_heads=self.num_attention_heads,
                    qkv_bias=self.cfg.attention_bias,
                    proj_drop=self.cfg.dropout,
                    ff_drop=self.cfg.dropout,
                    triplane_attention=self.cfg.triplane_attention,
                    triplane_resolution=self.cfg.triplane_resolution,
                    full_attention=self.cfg.triplane_full_attention,
                    memory_efficient_attention=self.cfg.enable_memory_efficient_attention,
                    gradient_checkpointing=self.gradient_checkpointing,
                    norm_x_input=self.cfg.norm_x_input,
                    dim_cross=self.cfg.cross_attention_dim,
                    num_experts=(
                        self.cfg.num_experts
                        if i >= self.cfg.start_experts_from_block
                        else None
                    ),
                    num_cross_attentions=self.cfg.num_cross_attentions,
                )
                for i in range(self.cfg.num_blocks)
            ]
        )

        # 4. Define output layers
        self.proj_out = nn.Linear(
            self.cfg.triplane_channels, self.cfg.raw_triplane_channels
        )

    def forward(self, outputs, **kwargs):
        hidden_states = outputs[Names.CONDITION.add_suffix("input")]
        encoder_hidden_states = outputs[Names.CONDITION.add_suffix("cross")]
        # hidden_states: [B, triplane_dim, N_triplane] is triplane tokens
        # encoder_hidden_states: [B, N_image, image_dim] is the image tokens, potentially a list
        batch_dimensions = hidden_states.shape[:-2]
        if len(batch_dimensions) > 1:
            hidden_states = hidden_states.view(-1, *hidden_states.shape[-2:])
            if isinstance(encoder_hidden_states, list):
                encoder_hidden_states = [
                    e.reshape(hidden_states.shape[0], -1, e.shape[-1])
                    for e in encoder_hidden_states
                ]
            else:
                encoder_hidden_states = encoder_hidden_states.view(
                    hidden_states.shape[0], -1, encoder_hidden_states.shape[-1]
                )

        if isinstance(self.norm_triplane, nn.GroupNorm):
            triplane_tokens = self.norm_triplane(hidden_states)
            triplane_tokens = triplane_tokens.permute(
                0, 2, 1
            )  # [B, N_triplane, triplane_dim]
        elif isinstance(self.norm_triplane, nn.LayerNorm):
            triplane_tokens = self.norm_triplane(hidden_states.permute(0, 2, 1))
        else:
            raise ValueError("Unknown normalization layer")
        triplane_tokens = self.proj_triplane(triplane_tokens)
        if self.mix_latent:
            image_tokens = self.norm_image(
                encoder_hidden_states[0]
                if isinstance(encoder_hidden_states, list)
                else encoder_hidden_states
            )  # [B, N_image, image_dim]
            image_tokens = self.proj_image(image_tokens)
        init_latents = self.latent_init.expand(
            hidden_states.shape[0], -1, -1
        )  # [B, N_latent_init, latent_dim]
        init_latents = self.norm_latent(init_latents)
        init_latents = self.proj_latent(init_latents)
        if self.mix_latent:
            if self.cfg.mix_latents_self_attention:
                image_tokens = self.mix_latents_self_attention(
                    image_tokens, image_tokens
                )
            if self.cfg.mix_latent_max_tokens > 0:
                image_tokens = image_tokens[:, : self.cfg.mix_latent_max_tokens, :]
            latent_tokens = torch.cat(
                [image_tokens, init_latents], dim=1
            )  # [B, N_latent, latent_dim]
        else:
            latent_tokens = init_latents

        # forward the main blocks
        expert_logits = {}
        for i, block in enumerate(self.main_blocks):
            latent_tokens, triplane_tokens = block(
                latent_tokens, triplane_tokens, encoder_hidden_states
            )

            for j, bblock in enumerate(block.transformer_block):
                if hasattr(bblock.ff, "get_expert_logits"):
                    expert_logits[
                        Names.EXPERT_LOGITS.add_suffix(f"block_{i}_basicblock_{j}")
                    ] = bblock.ff.get_expert_logits()

        # project the triplane tokens back to the original dimension
        triplane_tokens = self.proj_out(triplane_tokens).permute(0, 2, 1).contiguous()
        triplane_tokens = triplane_tokens + hidden_states
        return {
            self.cfg.output_key: triplane_tokens.view(
                *batch_dimensions, *triplane_tokens.shape[-2:]
            )
        } | expert_logits

    def consumed_keys(self):
        return super().consumed_keys() | {
            Names.CONDITION.add_suffix("input"),
            Names.CONDITION.add_suffix("cross"),
        }

    def produced_keys(self):
        new_keys = {self.cfg.output_key}
        for i, block in enumerate(self.main_blocks):
            for j, bblock in enumerate(block.transformer_block):
                if hasattr(bblock.ff, "get_expert_logits"):
                    new_keys.add(
                        Names.EXPERT_LOGITS.add_suffix(f"block_{i}_basicblock_{j}")
                    )
        return super().produced_keys().union(new_keys)
