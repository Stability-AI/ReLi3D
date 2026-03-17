import torch

from src.constants import Names, OutputsType
from src.utils.ops import normalize
from src.utils.typing import Literal


def octahedral_encode_position(position):
    """Encodes a 3D position into octahedral space for a stable coordinate frame."""
    p = position / (
        torch.abs(position[..., 0])
        + torch.abs(position[..., 1])
        + torch.abs(position[..., 2])
        + 1e-8
    ).unsqueeze(-1)
    oct_x = p[..., 0] + p[..., 1]
    oct_y = p[..., 1] - p[..., 0]
    return torch.stack([oct_x, oct_y], dim=-1)


def octahedral_decode_position(octahedral):
    """Decodes 2D octahedral coordinates back into a stable 3D frame vector."""
    o = torch.cat(
        [
            octahedral,
            1 - torch.abs(octahedral[..., 0:1]) - torch.abs(octahedral[..., 1:2]),
        ],
        dim=-1,
    )
    frame_vector = o / (torch.norm(o, dim=-1, keepdim=True) + 1e-8)
    return frame_vector


def compute_global_tangent_bitangent(position, up_axis=torch.tensor([0.0, 1.0, 0.0])):
    """Constructs a stable tangent-bitangent frame from global position."""
    oct_coords = octahedral_encode_position(position)

    # Generate tangent using a perpendicular vector in octahedral space
    tangent_2d = torch.stack([-oct_coords[..., 1], oct_coords[..., 0]], dim=-1)
    tangent = octahedral_decode_position(tangent_2d)

    # Compute bitangent using cross-product with a fixed global up-axis
    bitangent = torch.cross(up_axis.expand_as(tangent), tangent, dim=-1)

    # Normalize
    tangent = tangent / (torch.norm(tangent, dim=-1, keepdim=True) + 1e-8)
    bitangent = bitangent / (torch.norm(bitangent, dim=-1, keepdim=True) + 1e-8)

    return tangent, bitangent


def process_normal(
    outputs: OutputsType,
    normal_type: Literal[
        "world", "residual", "bump", "radial_bump", "octahedral_bump", "geometry"
    ] = "world",
    radial_up_axis: Literal["x", "y", "z"] = "z",
    slerp_normal_steps: int = 0,
    use_ray_samples: bool = False,
) -> OutputsType:
    suffix = "ray-samples" if use_ray_samples else ""

    shading_normal = None
    tangent = None
    bitangent = None
    if normal_type == "world":
        shading_normal = outputs[Names.SURFACE_NORMAL.add_suffix(suffix)]
    elif normal_type == "residual":
        shading_normal = normalize(
            outputs[Names.SURFACE_NORMAL.add_suffix(suffix)]
            + outputs[Names.GEOMETRY_NORMAL.add_suffix(suffix)],
            dim=-1,
        )
    elif (
        normal_type == "bump"
        or normal_type == "radial_bump"
        or normal_type == "octahedral_bump"
    ):
        if normal_type == "radial_bump":
            with torch.no_grad():
                positions = outputs[Names.POSITION.add_suffix(suffix)]
                if radial_up_axis == "z":
                    tangent = torch.stack(
                        [
                            -positions[..., 1],  # -y
                            positions[..., 0],  # x
                            torch.zeros_like(positions[..., 0]),
                        ],
                        -1,
                    )
                elif radial_up_axis == "x":
                    tangent = torch.stack(
                        [
                            torch.zeros_like(positions[..., 0]),
                            -positions[..., 2],  # -z
                            positions[..., 1],  # y
                        ],
                        -1,
                    )
                else:  # y-up
                    tangent = torch.stack(
                        [
                            positions[..., 2],  # -z
                            torch.zeros_like(positions[..., 1]),
                            -positions[..., 0],  # x
                        ],
                        -1,
                    )
                tangent = torch.cross(
                    outputs[Names.GEOMETRY_NORMAL.add_suffix(suffix)],
                    torch.cross(
                        tangent,
                        outputs[Names.GEOMETRY_NORMAL.add_suffix(suffix)],
                        dim=-1,
                    ),
                    dim=-1,
                )
                tangent = normalize(tangent)
                bitangent = torch.cross(
                    outputs[Names.GEOMETRY_NORMAL.add_suffix(suffix)], tangent, dim=-1
                )
                bitangent = normalize(bitangent)
        elif normal_type == "octahedral_bump":
            with torch.no_grad():
                positions = outputs[Names.POSITION.add_suffix(suffix)]
                up_axis = torch.tensor(
                    [0.0, 1.0, 0.0], device=positions.device
                )  # Global y-up axis
                tangent, bitangent = compute_global_tangent_bitangent(
                    positions, up_axis
                )

        shading_normal = normalize(
            tangent * outputs[Names.SURFACE_NORMAL.add_suffix(suffix)][..., 0:1]
            + bitangent * outputs[Names.SURFACE_NORMAL.add_suffix(suffix)][..., 1:2]
            + outputs[Names.GEOMETRY_NORMAL.add_suffix(suffix)].detach()
            * outputs[Names.SURFACE_NORMAL.add_suffix(suffix)][..., 2:3]
        )
    elif normal_type == "geometry":
        shading_normal = outputs[Names.GEOMETRY_NORMAL.add_suffix(suffix)]

    if slerp_normal_steps > 0:
        global_step = outputs[Names.GLOBAL_STEP]
        t = min(max(global_step / slerp_normal_steps, 0.0), 1.0)
        shading_normal = normalize(
            torch.lerp(
                outputs[Names.GEOMETRY_NORMAL.add_suffix(suffix)], shading_normal, t
            )
        )

    out = {}
    out[Names.SHADING_NORMAL.add_suffix(suffix)] = shading_normal
    if tangent is not None:
        out[Names.TANGENT.add_suffix(suffix)] = tangent
    if bitangent is not None:
        out[Names.BITANGENT.add_suffix(suffix)] = bitangent
    return out
