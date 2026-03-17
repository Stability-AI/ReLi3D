import torch
import torch.nn as nn
import torch.nn.functional as F

from .typing import Dict, Float, Optional, Tensor, Tuple


class RotaryPositionalEmbedding(nn.Module):
    """Module for applying Rotary Position Embeddings (RoPE) to transformer inputs."""

    def __init__(self, frequency: float = 10000.0):
        """Initialize the RoPE layer.
        Args:
            frequency (float): Base frequency for the embeddings. Default: 10000.0
        """
        super().__init__()
        self.base_frequency = frequency
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precomputes the frequency components for rotary position embeddings.

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            device: Target device for computations.
            dtype: Data type for the computed tensors.

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)

            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # Compute and cache frequency components
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, x: Float[Tensor, "B N D"], positions: Optional[torch.Tensor] = None
    ) -> Float[Tensor, "B N D"]:
        """Applies rotary position embeddings to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_length, dim).
            positions: Optional position indices. If None, uses sequential positions.

        Returns:
            Tensor: Tensor with rotary embeddings applied.
        """
        if positions is None:
            positions = torch.arange(x.size(1), device=x.device)

        # Get or compute frequency components
        max_pos = int(positions.max().item()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(
            x.size(-1), max_pos, x.device, x.dtype
        )

        # Embed positions with frequency components
        cos = F.embedding(positions.long(), cos_comp)
        sin = F.embedding(positions.long(), sin_comp)

        # Apply rotation
        return (x * cos) + (self._rotate_features(x) * sin)


class RotaryPositionalEmbedding2D(nn.Module):
    """Module for applying 2D Rotary Position Embeddings (RoPE) to transformer inputs."""

    def __init__(self, frequency: float = 10000.0):
        """Initialize the 2D RoPE layer.
        Args:
            frequency (float): Base frequency for the embeddings. Default: 10000.0
        """
        super().__init__()
        self.base_frequency = frequency
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)

            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # Compute and cache frequency components
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cos_comp: torch.Tensor,
        sin_comp: torch.Tensor,
    ) -> torch.Tensor:
        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)
        sin = F.embedding(positions, sin_comp)

        # Apply rotation
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(
        self, x: Float[Tensor, "B N D"], positions: Float[Tensor, "B N 2"]
    ) -> Float[Tensor, "B N D"]:
        """Applies 2D rotary position embeddings to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_length, dim).
            positions: Position tensor of shape (batch_size, seq_length, 2)
                     containing (height, width) coordinates.

        Returns:
            Tensor: Tensor with 2D rotary embeddings applied.
        """
        assert (
            x.size(-1) % 4 == 0
        ), "Feature dimension must be divisible by 4 for 2D RoPE"

        # Split features for vertical and horizontal components
        x_h, x_w = x.chunk(2, dim=-1)

        # Get position indices and compute frequencies
        h_pos, w_pos = positions[..., 0], positions[..., 1]
        max_h, max_w = int(h_pos.max().item()) + 1, int(w_pos.max().item()) + 1

        # Get frequency components for both dimensions
        cos_h, sin_h = self._compute_frequency_components(
            x_h.size(-1), max_h, x.device, x.dtype
        )
        cos_w, sin_w = self._compute_frequency_components(
            x_w.size(-1), max_w, x.device, x.dtype
        )

        # Apply RoPE separately for each dimension
        x_h_out = self._apply_1d_rope(x_h, h_pos.long(), cos_h, sin_h)
        x_w_out = self._apply_1d_rope(x_w, w_pos.long(), cos_w, sin_w)

        return torch.cat([x_h_out, x_w_out], dim=-1)
