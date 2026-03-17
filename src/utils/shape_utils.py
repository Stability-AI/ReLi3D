from dataclasses import dataclass

from src.constants import FieldName, Names, OutputsType
from src.utils.typing import Optional


@dataclass
class ShapeCheckResult:
    has_batch_dim: bool
    has_view_dim: bool
    has_other_dims: bool
    has_known_dims: bool


def check_shape(
    outputs: OutputsType, name: FieldName, known_dims: Optional[int] = None
) -> ShapeCheckResult:
    # Get expected dimensions from outputs
    batch_dim = outputs.get(Names.BATCH_SIZE)
    view_dim = outputs.get(Names.VIEW_SIZE.cond if name.is_cond else Names.VIEW_SIZE)

    # Get actual shape of the tensor
    tensor_shape = outputs[name].shape
    current_dim_idx = 0

    # Add a quick check if the shape only has known_dims. Then it might not have a batch or view dimension
    if known_dims is not None:
        if len(tensor_shape) == known_dims:
            return ShapeCheckResult(
                has_batch_dim=False,
                has_view_dim=False,
                has_other_dims=False,
                has_known_dims=True,
            )
        if len(tensor_shape) < known_dims:
            return ShapeCheckResult(
                has_batch_dim=False,
                has_view_dim=False,
                has_other_dims=False,
                has_known_dims=False,
            )

    # Check batch dimension if specified
    has_batch_dim = False
    if batch_dim is not None and tensor_shape[current_dim_idx] == batch_dim:
        has_batch_dim = True
        current_dim_idx += 1

    # Check if we have known dimensions left
    remaining_dims = len(tensor_shape) - current_dim_idx
    if known_dims is not None and remaining_dims == known_dims:
        return ShapeCheckResult(
            has_batch_dim=has_batch_dim,
            has_view_dim=False,
            has_other_dims=False,
            has_known_dims=True,
        )

    # Check view dimension if specified
    has_view_dim = False
    if view_dim is not None and tensor_shape[current_dim_idx] == view_dim:
        has_view_dim = True
        current_dim_idx += 1

    # Check remaining dimensions if known_dims is specified
    remaining_dims = len(tensor_shape) - current_dim_idx
    has_known_dims = False
    has_other_dims = False
    if known_dims is not None:
        has_known_dims = True if remaining_dims >= known_dims else False
        has_other_dims = True if (remaining_dims - known_dims) > 0 else False
    else:
        has_other_dims = True if remaining_dims > 0 else False

    return ShapeCheckResult(
        has_batch_dim=has_batch_dim,
        has_view_dim=has_view_dim,
        has_other_dims=has_other_dims,
        has_known_dims=has_known_dims,
    )
