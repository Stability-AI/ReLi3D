import numpy as np
import torch

from .ops import normalize


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    assert d6.shape[-1] == 6, "Input tensor must have shape (..., 6)"

    x_raw, y_raw = d6[..., :3], d6[..., 3:]

    x = normalize(x_raw, dim=-1)
    y = normalize(y_raw - (x * y_raw).sum(dim=-1, keepdim=True) * x, dim=-1)
    z = torch.cross(x, y, dim=-1)

    return torch.stack((x, y, z), dim=-1)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    assert matrix.shape[-2:] == (3, 3), "Input tensor must have shape (..., 3, 3)"
    rot = torch.cat([matrix[..., 0], matrix[..., 1]], dim=-1)
    return rot.view(*matrix.shape[:-2], 6)


def convert3x4_4x4(mat: torch.Tensor) -> torch.Tensor:
    """
    Converts a 3x4 matrix to a 4x4 matrix by adding a row [0, 0, 0, 1].

    Args:
        mat (torch.Tensor): Input tensor of shape (N, 3, 4) or (3, 4).

    Returns:
        torch.Tensor: Output tensor of shape (N, 4, 4) or (4, 4).
    """
    assert mat.shape[-2:] == (3, 4), "Input tensor must have shape (..., 3, 4)"

    if isinstance(mat, torch.Tensor):
        if len(mat.shape) == 3:
            addition = torch.tensor([0, 0, 0, 1], dtype=torch.float32).expand(
                mat.shape[0], 1, 4
            )
            output = torch.cat([mat, addition], dim=1)  # (N, 4, 4)
        else:
            addition = torch.tensor([[0, 0, 0, 1]], dtype=mat.dtype)
            output = torch.cat([mat, addition], dim=0)  # (4, 4)
    else:
        if len(mat.shape) == 3:
            output = np.concatenate(
                [mat, np.zeros_like(mat[:, 0:1])], axis=1
            )  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate(
                [mat, np.array([[0, 0, 0, 1]], dtype=mat.dtype)], axis=0
            )  # (4, 4)
            output[3, 3] = 1.0

    return output


def build_4x4_matrix(rotation_6d, t):
    assert rotation_6d.shape[-1] == 6, "rotation_6d must have shape (N, 6)"
    assert t.shape[-1] == 3, "t must have shape (N, 3)"

    R = rotation_6d_to_matrix(rotation_6d)
    c2w = torch.cat([R, t[..., None]], dim=-1)
    c2w = convert3x4_4x4(c2w)

    return c2w


def extract_6d_t_from_4x4(c2w):
    assert c2w.shape[-2:] == (4, 4), "c2w must have shape (N, 4, 4)"

    R = c2w[:, :3, :3]
    t = c2w[:, :3, 3]

    return matrix_to_rotation_6d(R), t
