import abc
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import trimesh
from jaxtyping import Float
from texture_baker import TextureBaker
from torch import Tensor

from src.constants import FieldName, Names, OutputsType
from src.models.isosurface import (
    FlexicubesHelper,
    MarchingCubeHelper,
    MarchingTetrahedraHelper,
)
from src.models.materials.base import BaseMaterial
from src.models.mesh import Mesh
from src.models.networks import MultiHeadMLP
from src.utils.dilation_fill import dilate_fill
from src.utils.ops import get_activation, scale_tensor
from src.utils.trimesh_conversion import to_trimesh
from src.utils.typing import Dict, List, Optional, Tuple, Union

from ..abstract_object_representation import AbstractObjectRepresentation


def uv_padding(arr, bake_mask, bake_resolution):
    if arr.ndim == 1:
        return arr
    return (
        dilate_fill(
            arr.permute(2, 0, 1)[None, ...].contiguous(),
            bake_mask.unsqueeze(0).unsqueeze(0),
            iterations=bake_resolution // 150,
        )
        .squeeze(0)
        .permute(1, 2, 0)
        .contiguous()
    )


class AbstractVolumetricRepresentation(AbstractObjectRepresentation, abc.ABC):
    @dataclass
    class Config(AbstractObjectRepresentation.Config):
        shape_key: FieldName = Names.DENSITY
        isosurface_method: str = (
            "marching_cubes"  # marching_tetrahedra or marching_cubes, flexicubes
        )
        isosurface_resolution: int = 128
        isosurface_threshold: float = 10.0
        isosurface_subdivisions: int = 0

        flexicubes_weight_scale: float = 0.5
        flexicubes_qef_reg_scale: float = 1e-3

        radius: float = 1.0

        use_deformation: bool = False
        deformation_key: FieldName = Names.VERTEX_OFFSET
        additional_vertex_keys: Dict[str, FieldName] = field(default_factory=dict)
        additional_indices_keys: Dict[str, FieldName] = field(default_factory=dict)

        indices_merging_method: str = "mean"  # mean or mlp
        indices_merging_mlp: dict = field(default_factory=dict)

        normal_type: Optional[str] = (
            None  # "finite_difference" or "finite_difference_laplacian"
        )
        finite_difference_normal_eps: float = 0.01

        shape_activation: Optional[str] = "softplus"
        density_bias: Union[float, str] = (
            "blob_magic3d"  # blob_magic3d or blob_dreamfusion or number
        )
        density_blob_scale: float = 10.0
        density_blob_std: float = 0.5

        sdf_bias: Union[float, str] = "sphere"  # sphere or number
        sdf_sphere_radius: float = 1.0

    cfg: Config

    def consumed_keys(self):
        return (
            super()
            .consumed_keys()
            .union(
                {self.cfg.shape_key, self.cfg.deformation_key}
                if not self.cfg.use_deformation
                else {self.cfg.shape_key}
            )
        )

    def shape_keys(self):
        potential_keys = (
            {
                self.cfg.shape_key,
            }
            | (
                set()
                if not self.cfg.use_deformation
                else {
                    self.cfg.deformation_key,
                }
            )
            | set(self.cfg.additional_vertex_keys.values())
            | set(self.cfg.additional_indices_keys.values())
        )
        return {v if isinstance(v, Names) else Names(v) for v in potential_keys}

    def configure(self) -> None:
        if self.cfg.isosurface_method == "marching_tetrahedra":
            # Get path of file
            self.isosurface_helper = MarchingTetrahedraHelper(
                self.cfg.isosurface_resolution,
                loop_subdivisions=self.cfg.isosurface_subdivisions,
            )
        elif self.cfg.isosurface_method == "marching_cubes":
            self.isosurface_helper = MarchingCubeHelper(
                self.cfg.isosurface_resolution,
                loop_subdivisions=self.cfg.isosurface_subdivisions,
            )
        elif self.cfg.isosurface_method == "flexicubes":
            self.isosurface_helper = FlexicubesHelper(
                self.cfg.isosurface_resolution,
                loop_subdivisions=self.cfg.isosurface_subdivisions,
                weight_scale=self.cfg.flexicubes_weight_scale,
                qef_reg_scale=self.cfg.flexicubes_qef_reg_scale,
            )
        else:
            raise NotImplementedError(
                f"Isosurface method {self.cfg.isosurface_method} not implemented"
            )

        # If it is DENSITY it should be -1
        if self.cfg.shape_key.matches(Names.DENSITY):
            self.gradient_direction = -1.0
        # If it is SDF it should be 1
        elif self.cfg.shape_key.matches(Names.SDF):
            self.gradient_direction = 1.0
        else:
            raise ValueError(
                f"Only {str(Names.DENSITY)} and {str(Names.SDF)} are supported for gradient direction. "
                f"Received {self.cfg.shape_key}."
            )

        if self.cfg.indices_merging_method == "mlp":
            self.cfg.indices_merging_mlp["in_channels"] = (
                self.isosurface_helper.grid_indices.shape[1]
                * self.cfg.indices_merging_mlp["in_channels"]
            )
            self.indices_merging_mlp = MultiHeadMLP(
                MultiHeadMLP.Config(**self.cfg.indices_merging_mlp)
            )

        self.radius = self.cfg.radius
        self.bbox: Float[Tensor, "2 3"]
        self.register_buffer(
            "bbox",
            torch.as_tensor(
                [
                    [-self.cfg.radius] * 3,
                    [self.cfg.radius] * 3,
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def get_normals(
        self, outputs: OutputsType, positions: Float[Tensor, "*B N 3"]
    ) -> Float[Tensor, "*B N 3"]:
        eps = self.cfg.finite_difference_normal_eps
        if self.cfg.normal_type == "finite_difference":
            offsets: Float[Tensor, "3 3"] = torch.as_tensor(
                [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]],
                dtype=positions.dtype,
                device=positions.device,
            )
            points_offset: Float[Tensor, "*B 3 3"] = (
                positions[..., None, :] + offsets
            ).clamp(-self.cfg.radius, self.cfg.radius)
            shape_key_shifted: Float[Tensor, "*B 3 1"] = self.forward(
                outputs, points_offset.view(-1, 3), skip_normal=True
            )[Names(self.cfg.shape_key)].view(-1, 3, 1)
            shape_key = self.forward(outputs, positions, skip_normal=True)[
                Names(self.cfg.shape_key)
            ]

            normal = (
                self.gradient_direction
                * (shape_key_shifted[..., 0::1, 0] - shape_key)
                / eps
            )
            return normal
        elif self.cfg.normal_type == "finite_difference_laplacian":
            offsets: Float[Tensor, "6 3"] = torch.as_tensor(
                [
                    [eps, 0.0, 0.0],
                    [-eps, 0.0, 0.0],
                    [0.0, eps, 0.0],
                    [0.0, -eps, 0.0],
                    [0.0, 0.0, eps],
                    [0.0, 0.0, -eps],
                ],
                dtype=positions.dtype,
                device=positions.device,
            )
            positions_shifted: Float[Tensor, "*B 6 3"] = (
                positions[..., None, :] + offsets
            ).clamp(min=-self.cfg.radius, max=self.cfg.radius)
            shape_key_shifted: Float[Tensor, "*B 6 1"] = self.forward(
                outputs, positions_shifted.view(-1, 3), skip_normal=True
            )[Names(self.cfg.shape_key)].view(-1, 6, 1)

            normal = (
                self.gradient_direction
                * 0.5
                * (shape_key_shifted[..., 0::2, 0] - shape_key_shifted[..., 1::2, 0])
                / eps
            )
            return normal
        else:
            raise ValueError(f"Normal type {self.cfg.normal_type} not implemented")

    def get_shape(
        self,
        outputs: OutputsType,
        positions: Float[Tensor, "*B N 3"],
    ) -> OutputsType:
        shape_definition = outputs[self.cfg.shape_key]
        if self.cfg.shape_key.matches(Names.DENSITY):
            if self.cfg.density_bias == "blob_magic3d":
                density_bias = (
                    self.cfg.density_blob_scale
                    * (
                        1
                        - torch.sqrt((positions**2).sum(dim=-1))
                        / self.cfg.density_blob_std
                    )[..., None]
                )
            elif self.cfg.density_bias == "blob_dreamfusion":
                density_bias = (
                    self.cfg.density_blob_scale
                    * torch.exp(
                        -0.5 * (positions**2).sum(dim=-1) / self.cfg.density_blob_std**2
                    )[..., None]
                )
            elif isinstance(self.cfg.density_bias, float):
                density_bias = self.cfg.density_bias
            else:
                raise ValueError(
                    f"Density bias {self.cfg.density_bias} not implemented"
                )

            shape_definition = shape_definition + density_bias
            outputs[Names.DENSITY.raw] = shape_definition
            outputs[Names.DENSITY] = get_activation(self.cfg.shape_activation)(
                shape_definition
            )

        elif self.cfg.shape_key.matches(Names.SDF):
            if self.cfg.sdf_bias == "sphere":
                activated_sdf = get_activation(self.cfg.shape_activation)(
                    shape_definition
                )
                sphere_sdf = (positions**2).sum(dim=-1) - self.cfg.sdf_sphere_radius**2
                outputs[Names.SDF] = activated_sdf + sphere_sdf
            elif isinstance(self.cfg.sdf_bias, float):
                outputs[Names.SDF] = get_activation(self.cfg.shape_activation)(
                    shape_definition + self.cfg.sdf_bias
                )
            else:
                raise ValueError(f"SDF bias {self.cfg.sdf_bias} not implemented")

        return outputs

    @abc.abstractmethod
    def forward_impl(
        self,
        outputs: OutputsType,
        positions: Float[Tensor, "*B N 3"],
        include: List[FieldName] = None,
        exclude: List[FieldName] = None,
    ) -> OutputsType:
        pass

    def produced_keys(self):
        return (
            super()
            .produced_keys()
            .union({self.cfg.shape_key, self.cfg.shape_key.raw, Names.GEOMETRY_NORMAL})
        )

    def forward(
        self,
        outputs: OutputsType,
        positions: Float[Tensor, "*B N 3"],
        skip_normal: bool = False,
        include: List[FieldName] = None,
        exclude: List[FieldName] = None,
    ) -> OutputsType:
        ret = {
            k if isinstance(k, Names) else Names(k): v
            for k, v in self.forward_impl(outputs, positions, include, exclude).items()
        }
        if self.cfg.shape_key in ret:
            ret.update(self.get_shape(ret, positions))
            if self.cfg.normal_type is not None and not skip_normal:
                ret[Names.GEOMETRY_NORMAL] = self.get_normals(outputs, positions)
        return ret

    def get_mesh(self, outputs: OutputsType) -> Tuple[List[Mesh], OutputsType]:
        # Disable autocast - flexicubes is unstable with it
        with torch.autocast(device_type=self.device.type, enabled=False):
            grid_vertices = scale_tensor(
                self.isosurface_helper.grid_vertices.to(self.device),
                self.isosurface_helper.points_range,
                self.bbox,
            )
            start = 0
            if Names.BATCH_SIZE in outputs:
                grid_vertices = grid_vertices.unsqueeze(start).repeat_interleave(
                    outputs[Names.BATCH_SIZE], dim=start
                )

            queried = self.forward(
                outputs,
                grid_vertices,
                include=self.shape_keys(),
            )
            shape_definition = queried[self.cfg.shape_key]
            deforms = None
            if self.cfg.use_deformation and self.cfg.deformation_key in queried:
                deforms = queried[self.cfg.deformation_key]

            if self.cfg.additional_indices_keys != {}:
                grid_indices = self.isosurface_helper.grid_indices
                for v in self.shape_keys():
                    if v not in self.cfg.additional_indices_keys.values():
                        continue

                    # queried[v]: shape [B, N, D]
                    grid_indexed_values = queried[
                        v if isinstance(v, Names) else Names(v)
                    ]  # per-vertex features

                    flat_indices = grid_indices.reshape(-1)  # shape [NF * M]

                    # Gather values at cube vertices
                    # grid_indexed_values: [B, N, D] => [B, NF*M, D] using gather
                    gathered = torch.index_select(
                        grid_indexed_values, dim=1, index=flat_indices
                    )
                    gathered = gathered.reshape(
                        grid_indexed_values.shape[0],
                        grid_indices.shape[0],
                        grid_indices.shape[1],
                        -1,
                    )  # [B, NF, M, D]

                    if self.cfg.indices_merging_method == "mean":
                        merged = gathered.mean(dim=2)  # [B, NF, D]
                    elif self.cfg.indices_merging_method == "mlp":
                        merged = self.indices_merging_mlp(
                            {
                                Names.TOKEN: gathered.reshape(
                                    grid_indexed_values.shape[0],
                                    grid_indices.shape[0],
                                    -1,
                                )
                            },
                            include=[v],
                        )[v]  # [B, NF, D]
                    else:
                        raise ValueError(
                            f"Indices merging method {self.cfg.indices_merging_method} not implemented"
                        )

                    queried[Names(v).add_suffix("merged")] = merged

            meshes = []
            detach_key_replace = {}
            for i in range(shape_definition.shape[0]):
                mesh: Mesh = self.isosurface_helper(
                    self.gradient_direction
                    * (shape_definition[i] - self.cfg.isosurface_threshold).view(-1, 1),
                    deforms[i] if deforms is not None else None,
                    **{
                        k: queried[v if isinstance(v, Names) else Names(v)][i]
                        for k, v in (self.cfg.additional_vertex_keys.items())
                    },
                    **{
                        k: queried[
                            (v if isinstance(v, Names) else Names(v)).add_suffix(
                                "merged"
                            )
                        ][i]
                        for k, v in (self.cfg.additional_indices_keys.items())
                    },
                )
                mesh.v_pos = scale_tensor(
                    mesh.v_pos,
                    self.isosurface_helper.points_range,
                    self.bbox,
                )
                meshes.append(mesh)

                # Check if the mesh has an empty surface
                zero_surface = mesh.extras["zero_surface"]
                if zero_surface:
                    # Detach all gradients where it is true
                    for k, v in queried.items():
                        if isinstance(v, torch.Tensor) and k in self.shape_keys():
                            detach_key_replace[k] = detach_key_replace.get(k, []) + [i]

            for k, indices in detach_key_replace.items():
                values = list(queried[k].unbind(dim=0))
                for index in indices:
                    values[index] = values[index].detach()
                queried[k] = torch.stack(values)

            return meshes, queried

    def get_textured_trimesh(
        self,
        outputs: OutputsType,
        material: BaseMaterial,
        texture_resolution: int = 1024,
        remesh: Optional[str] = None,
        vertex_count: int = -1,
    ) -> List[trimesh.Trimesh]:
        with torch.no_grad():
            with (
                torch.autocast(device_type=self.device.type, enabled=False)
                if "cuda" in self.device.type
                else nullcontext()
            ):
                meshes, queried = self.get_mesh(outputs)
                baker = TextureBaker()
                timeshes = []
                for i, mesh in enumerate(meshes):
                    if remesh == "quad":
                        mesh = mesh.quad_remesh(quad_vertex_count=vertex_count)
                    elif remesh == "triangle":
                        mesh = mesh.triangle_remesh(triangle_vertex_count=vertex_count)

                    mesh.unwrap_uv()

                    # Build textures
                    rast = baker.rasterize(
                        mesh.v_tex, mesh.t_pos_idx, texture_resolution
                    )
                    bake_mask = baker.get_mask(rast)

                    pos_bake = baker.interpolate(
                        mesh.v_pos,
                        rast,
                        mesh.t_pos_idx,
                    )
                    gb_pos = pos_bake[bake_mask]

                    nrm = baker.interpolate(
                        mesh.v_nrm,
                        rast,
                        mesh.t_pos_idx,
                    )
                    gb_nrm = F.normalize(nrm[bake_mask], dim=-1)

                    # Get texture
                    queried = self(
                        {
                            k: v[i] if isinstance(v, torch.Tensor) else v
                            for k, v in outputs.items()
                        },
                        gb_pos,
                    )
                    queried[Names.GEOMETRY_NORMAL] = gb_nrm
                    queried[Names.POSITION] = gb_pos

                    queried.update(material.export(queried))

                    material_keys = {
                        Names.BASECOLOR: "map_Kd",
                        Names.ROUGHNESS: "map_Pr",
                        Names.METALLIC: "map_Pm",
                        Names.SURFACE_NORMAL: "map_Bump",
                    }

                    has_bump = Names.SHADING_NORMAL in queried
                    if has_bump:
                        tng = baker.interpolate(
                            mesh.v_tng,
                            rast,
                            mesh.t_pos_idx,
                        )
                        gb_tng = tng[bake_mask]
                        gb_tng = F.normalize(gb_tng, dim=-1)
                        gb_btng = F.normalize(
                            torch.cross(gb_nrm, gb_tng, dim=-1), dim=-1
                        )
                        normal = queried[Names.SHADING_NORMAL]

                        tangent_matrix = torch.stack([gb_tng, gb_btng, gb_nrm], dim=-1)
                        normal_tangent = torch.bmm(
                            tangent_matrix.transpose(1, 2), normal.unsqueeze(-1)
                        ).squeeze(-1)

                        # Convert from [-1,1] to [0,1] range for storage
                        normal_tangent = (normal_tangent * 0.5 + 0.5).clamp(0, 1)

                        queried[Names.SURFACE_NORMAL] = normal_tangent

                    export = {}
                    for k, v in queried.items():
                        if k in material_keys:
                            f = torch.zeros(
                                texture_resolution,
                                texture_resolution,
                                v.shape[-1],
                                dtype=v.dtype,
                                device=v.device,
                            )
                            f[bake_mask] = v.view(-1, v.shape[-1])
                            export[material_keys[k]] = uv_padding(
                                f, bake_mask, texture_resolution
                            )

                    export["mesh"] = mesh

                    timeshes.append(to_trimesh(export))

        return timeshes


class AbstractNeuralVolumetricRepresentation(AbstractVolumetricRepresentation):
    @dataclass
    class Config(AbstractVolumetricRepresentation.Config):
        multi_head_mlp: MultiHeadMLP.Config = field(default_factory=MultiHeadMLP.Config)

    cfg: Config

    def configure(self, feature_dimension: int = 1288):
        super().configure()
        self.cfg.multi_head_mlp.in_channels = feature_dimension
        self.net = MultiHeadMLP(self.cfg.multi_head_mlp)

        # Check if self.net has self.shape_key as a head defined
        if str(self.cfg.shape_key) not in self.net.keys():
            raise ValueError(
                f"Shape key {self.cfg.shape_key} is not defined as a head in the MLP."
            )

        # Check that the shape key in the MLP has no activation function
        head_config = self.net.get_head_config(self.cfg.shape_key)
        if head_config.output_activation is not None:
            raise ValueError(
                f"Shape key {self.cfg.shape_key} in the MLP has an activation function. "
                "This should be handled by the object representation."
            )

    def produced_keys(self):
        return (
            super()
            .produced_keys()
            .union({Names(head.name) for head in self.cfg.multi_head_mlp.heads})
        )
