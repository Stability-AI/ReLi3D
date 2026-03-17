import os

import numpy as np
import torch
import torch.nn as nn
try:
    from torchmcubes import marching_cubes as torch_marching_cubes
except ImportError:
    torch_marching_cubes = None

import mcubes

import src
from src.models.mesh import Mesh
from src.models.mesh_subdivision import LoopSubdivision
from src.utils.flexicubes import FlexiCubes
from src.utils.typing import Float, Integer, Optional, Tensor, Tuple


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    def __init__(
        self, loop_subdivisions: int = 0, points_range: Tuple[float, float] = (0, 1)
    ) -> None:
        super().__init__()
        self.points_range = points_range
        self.loop_subdivisions = loop_subdivisions
        self.loop_subdiv = LoopSubdivision()

    @property
    def grid_vertices(self) -> Float[Tensor, "N 3"]:
        raise NotImplementedError

    @property
    def grid_indices(self) -> Integer[Tensor, "N 8"]:
        raise NotImplementedError

    @property
    def center_indices(self) -> Integer[Tensor, "N"]:
        raise NotImplementedError

    @property
    def boundary_indices(self) -> Integer[Tensor, "N"]:
        raise NotImplementedError

    @property
    def requires_instance_per_batch(self) -> bool:
        return False

    def zero_level_geometry(self, level: Float[Tensor, "N 1"]) -> Float[Tensor, ""]:
        # Check if all values are positive or negative
        pos_shape = torch.sum((level.squeeze(dim=-1) > 0).int())
        neg_shape = torch.sum((level.squeeze(dim=-1) < 0).int())
        return torch.bitwise_or(pos_shape == 0, neg_shape == 0)

    def check_empty_surface(
        self, level: Float[Tensor, "N 1"]
    ) -> Tuple[Float[Tensor, "N 1"], Float[Tensor, ""], Float[Tensor, ""]]:
        """Check if surface is empty (all positive or all negative) and return regularization loss."""
        sdf_reg_loss = torch.zeros(
            level.shape[0], device=level.device, dtype=torch.float32
        )

        # Check if all values are positive or negative
        zero_surface = self.zero_level_geometry(level).item()

        if zero_surface:
            # Create update values for empty surfaces
            update_level = torch.zeros_like(level)
            max_level = level.max()
            min_level = level.min()

            # Update center and boundary points
            update_level[self.center_indices] += 1.0 - min_level  # greater than zero
            update_level[self.boundary_indices] += -1 - max_level  # smaller than zero

            # Regularization to push SDF to have different signs
            sdf_reg_loss = torch.abs(level).mean()

            # Apply updates to empty surfaces
            level = level + update_level

        return level, sdf_reg_loss, zero_surface

    def forward_impl(
        self,
        level: Float[Tensor, "N 1"],
        deformation: Optional[Float[Tensor, "N 3"]] = None,
        **kwargs,
    ) -> Mesh:
        raise NotImplementedError

    def forward(
        self,
        level: Float[Tensor, "N 1"],
        deformation: Optional[Float[Tensor, "N 3"]] = None,
        **kwargs,
    ) -> Mesh:
        # Check for empty surfaces and get regularization loss
        level, empty_regularization, zero_surface = self.check_empty_surface(level)

        mesh = self.forward_impl(level, deformation, **kwargs)
        for _ in range(self.loop_subdivisions):
            mesh = self.loop_subdiv(mesh)

        mesh.extras["zero_surface"] = zero_surface

        # Detach all gradients where it is true
        if zero_surface:
            for k, v in mesh.extras.items():
                if isinstance(v, torch.Tensor):
                    mesh.extras[k] = v.detach()

        # Add empty surface regularization to mesh extras
        mesh.extras["empty_regularization"] = empty_regularization

        return mesh


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.resolution = resolution

        x, y, z = (
            torch.linspace(*self.points_range, self.resolution),
            torch.linspace(*self.points_range, self.resolution),
            torch.linspace(*self.points_range, self.resolution),
        )
        x, y, z = torch.meshgrid(x, y, z, indexing="ij")
        verts = torch.cat(
            [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
        ).reshape(-1, 3)

        self._grid_vertices: Float[Tensor, "..."]
        self.register_buffer(
            "_grid_vertices",
            verts,
            persistent=False,
        )

        center_indices, boundary_indices = self.get_center_boundary_index(
            self.resolution
        )
        self.register_buffer("_center_indices", center_indices, persistent=False)
        self.register_buffer("_boundary_indices", boundary_indices, persistent=False)

    @property
    def center_indices(self) -> Integer[Tensor, "N"]:
        return self._center_indices

    @property
    def boundary_indices(self) -> Integer[Tensor, "N"]:
        return self._boundary_indices

    @property
    def grid_vertices(self) -> Float[Tensor, "N3 3"]:
        return self._grid_vertices

    def forward_impl(
        self,
        level: Float[Tensor, "N3 1"],
        deformation: Optional[Float[Tensor, "N3 3"]] = None,
        **kwargs,
    ) -> Mesh:
        if deformation is not None:
            src.warn(
                f"{self.__class__.__name__} does not support deformation. Ignoring."
            )
        level = -level.view(self.resolution, self.resolution, self.resolution)
        if torch_marching_cubes is not None:
            v_pos, t_pos_idx = torch_marching_cubes(level.detach(), 0.0)
        else:
            # CPU fallback when torchmcubes is unavailable.
            v_pos_np, t_pos_idx_np = mcubes.marching_cubes(
                level.detach().cpu().numpy(), 0.0
            )
            v_pos = torch.from_numpy(v_pos_np).to(level.device, dtype=torch.float32)
            t_pos_idx = torch.from_numpy(t_pos_idx_np.astype(np.int64)).to(level.device)

        v_pos = v_pos[..., [2, 1, 0]]

        v_pos = (v_pos / (self.resolution - 1.0)) * (
            self.points_range[1] - self.points_range[0]
        ) + self.points_range[0]
        return Mesh(v_pos=v_pos, t_pos_idx=t_pos_idx)

    def zero_level_geometry(self, level: Float[Tensor, "N 1"]) -> Float[Tensor, ""]:
        sdf_nxnxn = level.reshape(
            (
                self.resolution,
                self.resolution,
                self.resolution,
            )
        )
        sdf_less_boundary = sdf_nxnxn[1:-1, 1:-1, 1:-1].reshape(-1)
        pos_shape = torch.sum((sdf_less_boundary > 0).int())
        neg_shape = torch.sum((sdf_less_boundary < 0).int())
        return torch.bitwise_or(pos_shape == 0, neg_shape == 0)

    def get_center_boundary_index(self, grid_res):
        v = torch.zeros((grid_res + 1, grid_res + 1, grid_res + 1), dtype=torch.bool)
        v[grid_res // 2 + 1, grid_res // 2 + 1, grid_res // 2 + 1] = True
        center_indices = torch.nonzero(v.reshape(-1))

        v[grid_res // 2 + 1, grid_res // 2 + 1, grid_res // 2 + 1] = False
        v[:2, ...] = True
        v[-2:, ...] = True
        v[:, :2, ...] = True
        v[:, -2:, ...] = True
        v[:, :, :2] = True
        v[:, :, -2:] = True
        boundary_indices = torch.nonzero(v.reshape(-1))
        return center_indices, boundary_indices


class MarchingTetrahedraHelper(IsosurfaceHelper):
    def __init__(self, resolution: int, tets_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.resolution = resolution
        if tets_path is None:
            self.tets_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "load",
                "tets",
                f"{resolution}_tets.npz",
            )
        else:
            self.tets_path = tets_path

        self.triangle_table: Float[Tensor, "..."]
        self.register_buffer(
            "triangle_table",
            torch.as_tensor(
                [
                    [-1, -1, -1, -1, -1, -1],
                    [1, 0, 2, -1, -1, -1],
                    [4, 0, 3, -1, -1, -1],
                    [1, 4, 2, 1, 3, 4],
                    [3, 1, 5, -1, -1, -1],
                    [2, 3, 0, 2, 5, 3],
                    [1, 4, 0, 1, 5, 4],
                    [4, 2, 5, -1, -1, -1],
                    [4, 5, 2, -1, -1, -1],
                    [4, 1, 0, 4, 5, 1],
                    [3, 2, 0, 3, 5, 2],
                    [1, 3, 5, -1, -1, -1],
                    [4, 1, 2, 4, 3, 1],
                    [3, 0, 4, -1, -1, -1],
                    [2, 0, 1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.num_triangles_table: Integer[Tensor, "..."]
        self.register_buffer(
            "num_triangles_table",
            torch.as_tensor(
                [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long
            ),
            persistent=False,
        )
        self.base_tet_edges: Integer[Tensor, "..."]
        self.register_buffer(
            "base_tet_edges",
            torch.as_tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long),
            persistent=False,
        )

        tets = np.load(self.tets_path)
        self._grid_vertices: Float[Tensor, "..."]
        self.register_buffer(
            "_grid_vertices",
            torch.from_numpy(tets["vertices"]).float(),
            persistent=False,
        )
        self.indices: Integer[Tensor, "..."]
        self.register_buffer(
            "indices", torch.from_numpy(tets["indices"]).long(), persistent=False
        )

        self._all_edges: Optional[Integer[Tensor, "Ne 2"]] = None

        center_indices, boundary_indices = self.get_center_boundary_index(
            self._grid_vertices
        )
        self.register_buffer("_center_indices", center_indices, persistent=False)
        self.register_buffer("_boundary_indices", boundary_indices, persistent=False)

    def get_center_boundary_index(self, verts):
        magn = torch.sum(verts**2, dim=-1)

        center_idx = torch.argmin(magn)
        boundary_neg = verts == verts.max()
        boundary_pos = verts == verts.min()

        boundary = torch.bitwise_or(boundary_pos, boundary_neg)
        boundary = torch.sum(boundary.float(), dim=-1)

        boundary_idx = torch.nonzero(boundary)
        return center_idx, boundary_idx.squeeze(dim=-1)

    def normalize_grid_deformation(
        self, grid_vertex_offsets: Float[Tensor, "Nv 3"]
    ) -> Float[Tensor, "Nv 3"]:
        return (
            (self.points_range[1] - self.points_range[0])
            / self.resolution  # half tet size is approximately 1 / self.resolution
            * torch.tanh(grid_vertex_offsets)
        )  # FIXME: hard-coded activation

    @property
    def grid_vertices(self) -> Float[Tensor, "Nv 3"]:
        return self._grid_vertices

    @property
    def grid_indices(self) -> Integer[Tensor, "N 8"]:
        return self.indices

    @property
    def center_indices(self) -> Integer[Tensor, "N"]:
        return self._center_indices

    @property
    def boundary_indices(self) -> Integer[Tensor, "N"]:
        return self._boundary_indices

    @property
    def all_edges(self) -> Integer[Tensor, "Ne 2"]:
        if self._all_edges is None:
            # compute edges on GPU, or it would be VERY SLOW (basically due to the unique operation)
            edges = torch.tensor(
                [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                dtype=torch.long,
                device=self.indices.device,
            )
            _all_edges = self.indices[:, edges].reshape(-1, 2)
            _all_edges_sorted = torch.sort(_all_edges, dim=1)[0]
            _all_edges = torch.unique(_all_edges_sorted, dim=0)
            self._all_edges = _all_edges
        return self._all_edges

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1 - order, dim=1)

        return torch.stack([a, b], -1)

    def _forward(self, pos_nx3, sdf_n, tet_fx4):
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(-1, 2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = (
                torch.ones(
                    (unique_edges.shape[0]), dtype=torch.long, device=pos_nx3.device
                )
                * -1
            )
            mapping[mask_edges] = torch.arange(
                mask_edges.sum(), dtype=torch.long, device=pos_nx3.device
            )
            idx_map = mapping[idx_map]  # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim=True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=pos_nx3.device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat(
            (
                torch.gather(
                    input=idx_map[num_triangles == 1],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 1]][:, :3],
                ).reshape(-1, 3),
                torch.gather(
                    input=idx_map[num_triangles == 2],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 2]][:, :6],
                ).reshape(-1, 3),
            ),
            dim=0,
        )

        return verts, faces

    def forward_impl(
        self,
        level: Float[Tensor, "N3 1"],
        deformation: Optional[Float[Tensor, "N3 3"]] = None,
        **kwargs,
    ) -> Mesh:
        if deformation is not None:
            grid_vertices = self.grid_vertices + self.normalize_grid_deformation(
                deformation
            )
        else:
            grid_vertices = self.grid_vertices

        v_pos, t_pos_idx = self._forward(grid_vertices, level, self.indices)

        mesh = Mesh(
            v_pos=v_pos,
            t_pos_idx=t_pos_idx,
            # extras
            grid_vertices=grid_vertices,
            tet_edges=self.all_edges,
            grid_level=level,
            grid_deformation=deformation,
            all_edges=self.all_edges,
        )

        return mesh


class FlexicubesHelper(IsosurfaceHelper):
    def __init__(
        self,
        resolution: int,
        qef_reg_scale: float = 1e-3,
        weight_scale: float = 0.99,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.points_range = (-0.5, 0.5)

        self.resolution = resolution
        self.flexicubes = FlexiCubes(
            qef_reg_scale=qef_reg_scale,
            weight_scale=weight_scale,
        )

        # Build the grid vertices and cubes
        verts, cubes = self.flexicubes.construct_voxel_grid(resolution)
        self.register_buffer("_grid_vertices", verts, persistent=False)
        self.register_buffer("_cube_indices", cubes, persistent=False)

        all_edges = cubes[:, self.flexicubes.cube_edges].reshape(-1, 2)
        self.register_buffer(
            "_all_edges", torch.unique(all_edges, dim=0), persistent=False
        )

        center_indices, boundary_indices = self.get_center_boundary_index(resolution)
        self.register_buffer("_center_indices", center_indices, persistent=False)
        self.register_buffer("_boundary_indices", boundary_indices, persistent=False)

    @property
    def center_indices(self):
        return self._center_indices

    @property
    def boundary_indices(self):
        return self._boundary_indices

    @property
    def grid_vertices(self):
        return self._grid_vertices

    @property
    def grid_indices(self):
        return self._cube_indices

    @property
    def all_edges(self):
        return self._all_edges

    def normalize_grid_deformation(
        self, grid_vertex_offsets: Float[Tensor, "Nv 3"]
    ) -> Float[Tensor, "Nv 3"]:
        return (
            (self.points_range[1] - self.points_range[0] - 1e-8)
            / (self.resolution * 2)
            * torch.tanh(grid_vertex_offsets)
        )

    def zero_level_geometry(self, level: Float[Tensor, "N 1"]) -> Float[Tensor, ""]:
        sdf_nxnxn = level.reshape(
            (
                self.resolution + 1,
                self.resolution + 1,
                self.resolution + 1,
            )
        )
        sdf_less_boundary = sdf_nxnxn[1:-1, 1:-1, 1:-1].reshape(-1)
        pos_shape = torch.sum((sdf_less_boundary > 0).int())
        neg_shape = torch.sum((sdf_less_boundary < 0).int())
        return torch.bitwise_or(pos_shape == 0, neg_shape == 0)

    def forward_impl(
        self,
        level: torch.Tensor,  # (N 1)
        deformation: torch.Tensor = None,
        weight_n: Optional[Float[Tensor, "N 21"]] = None,
        training: bool = False,
    ):
        # level: (N, 1) where N = num grid vertices
        # deformation: (N, 3) or None

        verts = self.grid_vertices
        if deformation is not None:
            verts = verts + self.normalize_grid_deformation(deformation)

        # FlexiCubes expects (N, 3) verts, (N,) level, (F, 8) cubes, resolution
        # Output: vertices, faces, L_dev
        vertices, faces, reg_loss = self.flexicubes(
            verts,
            level.squeeze(-1),
            self.grid_indices,
            self.resolution,
            beta_fx12=weight_n[:, :12] if weight_n is not None else None,
            alpha_fx8=weight_n[:, 12:20] if weight_n is not None else None,
            gamma_f=weight_n[:, 20] if weight_n is not None else None,
            training=training,
        )

        # Output as Mesh
        mesh = Mesh(
            v_pos=vertices,
            t_pos_idx=faces,
            grid_vertices=verts,
            grid_level=level,
            grid_deformation=deformation,
            weight_n=weight_n,
            all_edges=self.all_edges,
            reg_loss=reg_loss,
        )
        return mesh

    def get_center_boundary_index(self, grid_res):
        v = torch.zeros((grid_res + 1, grid_res + 1, grid_res + 1), dtype=torch.bool)
        v[grid_res // 2 + 1, grid_res // 2 + 1, grid_res // 2 + 1] = True
        center_indices = torch.nonzero(v.reshape(-1))

        v[grid_res // 2 + 1, grid_res // 2 + 1, grid_res // 2 + 1] = False
        v[:2, ...] = True
        v[-2:, ...] = True
        v[:, :2, ...] = True
        v[:, -2:, ...] = True
        v[:, :, :2] = True
        v[:, :, -2:] = True
        boundary_indices = torch.nonzero(v.reshape(-1))
        return center_indices, boundary_indices
