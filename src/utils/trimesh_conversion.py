import numpy as np
import torch
import trimesh
from PIL import Image

from src.utils.typing import Any, Float, Int, Optional

# def uint8_to_float32(x: Int[Tensor, "*B H W C"]) -> Float[Tensor, "*B H W C"]:
#     return (x.float() + 0.5) / 256.0


def float32_to_uint8(
    x: Float[np.ndarray, "*B H W C"],
    dither: bool = True,
    dither_mask: Optional[Float[np.ndarray, "*B H W C"]] = None,
    dither_strength: float = 1.0,
) -> Int[np.ndarray, "*B H W C"]:
    if dither:
        dither = (
            dither_strength * np.random.rand(*x[..., :1].shape).astype(np.float32) - 0.5
        )
        if dither_mask is not None:
            dither = dither * dither_mask
        return np.clip(np.floor((256.0 * x + dither)), 0, 255).astype(np.uint8)
    return np.clip(np.floor((256.0 * x)), 0, 255).astype(torch.uint8)


def convert_data(data):
    if data is None:
        return None
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        if data.dtype in [torch.float16, torch.bfloat16]:
            data = data.float()
        return data.detach().cpu().numpy()
    elif isinstance(data, list):
        return [convert_data(d) for d in data]
    elif isinstance(data, dict):
        return {k: convert_data(v) for k, v in data.items()}
    else:
        raise TypeError(
            "Data must be in type numpy.ndarray, torch.Tensor, list or dict, getting",
            type(data),
        )


def ensure_single_last_channel(x):
    if len(x.shape) == 4:
        x = x[0]
    if len(x.shape) == 2:
        x = np.expand_dims(x, axis=-1)
    return x[:, :, :1]


def to_trimesh(
    generated_mesh: dict[str, Any],
    align_coordinate_system: bool = False,
    fix_inversion: bool = False,
) -> trimesh.Trimesh:
    mesh = generated_mesh["mesh"]
    basecolor, metallic, roughness, bump = (None, None, None, None)
    material = None

    if "map_Kd" in generated_mesh:
        basecolor = generated_mesh["map_Kd"]
    if "map_Pm" in generated_mesh:
        metallic = generated_mesh["map_Pm"]
    if "map_Pr" in generated_mesh:
        roughness = generated_mesh["map_Pr"]
    if "map_Bump" in generated_mesh:
        bump = generated_mesh["map_Bump"]

    vertices = convert_data(mesh.v_pos)
    faces = convert_data(mesh.t_pos_idx)
    v_tex = convert_data(mesh.v_tex)
    v_nrm = convert_data(mesh.v_nrm)

    # Handle per-face UVs

    # ---- weld: map every position to a unique index -------------
    unique_pos, weld_index = np.unique(
        np.round(vertices, 6), axis=0, return_inverse=True
    )  # tolerance 1e-6

    # ---- accumulate face normals on the welded vertices --------
    weld_nrm = np.zeros_like(unique_pos)
    for tri in faces:
        n = np.cross(
            vertices[tri[1]] - vertices[tri[0]], vertices[tri[2]] - vertices[tri[0]]
        )
        for vi in tri:
            weld_nrm[weld_index[vi]] += n

    weld_nrm /= np.linalg.norm(weld_nrm, axis=1, keepdims=True)  # normalise

    # ---- look the normals back up with the original indices ----
    v_nrm = weld_nrm[weld_index]  # smooth normals, one per original vertex

    # ---- now duplicate *positions, normals and uvs* together ---
    individual_vertices = vertices[faces].reshape(-1, 3)
    individual_v_nrm = v_nrm[faces].reshape(-1, 3)
    individual_uv = v_tex[faces].reshape(-1, 2)
    individual_faces = np.arange(individual_vertices.shape[0]).reshape(-1, 3)

    # Basecolor always required
    # Convert to uint8 and PIL image
    if basecolor is not None:
        basecolor = Image.fromarray(float32_to_uint8(convert_data(basecolor))).convert(
            "RGB"
        )
        basecolor.format = "JPEG"

        if metallic is None and roughness is None and bump is None:
            material = trimesh.visual.material.SimpleMaterial(
                image=basecolor,
                diffuse=np.array([255, 255, 255, 255], dtype=np.uint8),
                specular=np.array([0, 0, 0, 255], dtype=np.uint8),
                ambient=np.array([0, 0, 0, 255], dtype=np.uint8),
                glossiness=0.0,
            )
        else:
            if metallic is not None:
                if len(metallic.shape) == 1:
                    metallic = np.full((1, 1, 1), metallic.detach().cpu().item())
                else:
                    metallic = ensure_single_last_channel(convert_data(metallic))
            else:
                metallic = np.full((1, 1, 1), 0.0)
            if roughness is not None:
                if len(roughness.shape) == 1:
                    roughness = np.full((1, 1, 1), roughness.detach().cpu().item())
                else:
                    roughness = ensure_single_last_channel(convert_data(roughness))
            else:
                roughness = np.full((1, 1, 1), 0.9)

            normal_texture = None
            if bump is not None:
                bump_np = convert_data(bump)
                bump_up = np.ones_like(bump_np)
                bump_up[..., :2] = 0.5
                bump_up[..., 2:] = 1
                bumpImg = Image.fromarray(
                    float32_to_uint8(
                        bump_np,
                        dither=True,
                        dither_mask=np.all(
                            bump_np == bump_up, axis=-1, keepdims=True
                        ).astype(np.float32),
                    )
                ).convert("RGB")
                bumpImg.format = "JPEG"  # OR test PNG. Jpeg is tricky
                normal_texture = bumpImg

            if roughness.size != 1:
                zero = np.zeros_like(roughness)
                metallic_roughness = np.concatenate(
                    [zero, roughness, metallic], axis=-1
                )

                metallic_roughness = Image.fromarray(
                    float32_to_uint8(metallic_roughness)
                ).convert("RGB")
                metallic_roughness.format = "JPEG"
                material = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=basecolor,
                    roughnessFactor=1,
                    metallicFactor=1,
                    metallicRoughnessTexture=metallic_roughness,
                    normalTexture=normal_texture,
                )
            else:
                material = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=basecolor,
                    roughnessFactor=roughness.squeeze().item(),
                    metallicFactor=metallic.squeeze().item(),
                    normalTexture=normal_texture,
                )

    final_mesh = trimesh.Trimesh(
        vertices=individual_vertices,
        faces=individual_faces,
        vertex_normals=individual_v_nrm,
        visual=trimesh.visual.texture.TextureVisuals(
            uv=individual_uv, material=material
        )
        if material is not None
        else None,
        process=False,
    )

    assert (
        final_mesh.vertices.shape[0]
        == final_mesh.vertex_normals.shape[0]
        == final_mesh.visual.uv.shape[0]
    )

    # Apply rotations to align coordinate systems
    if align_coordinate_system:
        rot = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
        final_mesh.apply_transform(rot)
        final_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
        )

    # flip normals if required
    if fix_inversion:
        trimesh.repair.fix_inversion(final_mesh)
        final_mesh.invert()

    return final_mesh
