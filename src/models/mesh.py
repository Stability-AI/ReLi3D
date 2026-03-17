from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from jaxtyping import Float, Int64
from torch import Tensor

from src.utils.ops import dot
from src.utils.typing import Any, Dict, Optional

try:
    from uv_unwrapper import Unwrapper
except ImportError:
    import logging

    logging.warning(
        "Could not import uv_unwrapper. Please install it via `pip install ./native/uv_unwrapper/`"
    )
    # Exit early to avoid further errors
    raise ImportError("uv_unwrapper not found")

try:
    import gpytoolbox

    TRIANGLE_REMESH_AVAILABLE = True
except ImportError:
    TRIANGLE_REMESH_AVAILABLE = False
    import logging

    logging.warning(
        "Could not import gpytoolbox. Triangle remeshing functionality will be disabled. "
        "Install via `pip install gpytoolbox`"
    )

try:
    import pynim

    QUAD_REMESH_AVAILABLE = True
except ImportError:
    QUAD_REMESH_AVAILABLE = False
    import logging

    logging.warning(
        "Could not import pynim. Quad remeshing functionality will be disabled. "
        "Install via `pip install git+https://github.com/vork/PyNanoInstantMeshes.git@v0.0.3`"
    )


class Mesh:
    def __init__(
        self, v_pos: Float[Tensor, "Nv 3"], t_pos_idx: Int64[Tensor, "Nf 3"], **kwargs
    ) -> None:
        self.v_pos: Float[Tensor, "Nv 3"] = v_pos
        self.t_pos_idx: Int64[Tensor, "Nf 3"] = t_pos_idx.long()
        self._v_nrm: Optional[Float[Tensor, "Nv 3"]] = None
        self._v_tng: Optional[Float[Tensor, "Nv 3"]] = None
        self._v_tex: Optional[Float[Tensor, "Nt 3"]] = None
        self._edges: Optional[Int64[Tensor, "Ne 2"]] = None
        self._half_edges: Optional[Int64[Tensor, "Nhe 4"]] = None
        self.extras: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.add_extra(k, v)

        self.unwrapper = Unwrapper()

    def add_extra(self, k, v) -> None:
        self.extras[k] = v

    @property
    def requires_grad(self):
        return self.v_pos.requires_grad

    @property
    def v_nrm(self):
        if self._v_nrm is None:
            self._v_nrm = self._compute_vertex_normal()
        return self._v_nrm

    @property
    def v_tng(self):
        if self._v_tng is None:
            self._v_tng = self._compute_vertex_tangent()
        return self._v_tng

    @property
    def v_tex(self):
        if self._v_tex is None:
            self.unwrap_uv()
        return self._v_tex

    @property
    def edges(self):
        if self._edges is None:
            self._edges = self._compute_edges()
        return self._edges

    @property
    def halfedges(self) -> Int64[Tensor, "Nhe 4"]:
        """
        Return existing half-edges if available; otherwise compute them.
        """
        if self._half_edges is None:
            self._half_edges = self._compute_halfedges()
        return self._half_edges

    def _compute_vertex_normal(self):
        i0 = self.t_pos_idx[:, 0]
        i1 = self.t_pos_idx[:, 1]
        i2 = self.t_pos_idx[:, 2]

        v0 = self.v_pos[i0, :]
        v1 = self.v_pos[i1, :]
        v2 = self.v_pos[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

        # Splat face normals to vertices
        v_nrm = torch.zeros_like(self.v_pos)
        v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        v_nrm = torch.where(
            dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
        )
        v_nrm = F.normalize(v_nrm, dim=1)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_nrm))

        return v_nrm

    def _compute_vertex_tangent(self):
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(0, 3):
            pos[i] = self.v_pos[self.t_pos_idx[:, i]]
            tex[i] = self.v_tex[self.t_pos_idx[:, i]]
            # t_nrm_idx is always the same as t_pos_idx
            vn_idx[i] = self.t_pos_idx[:, i]

        tangents = torch.zeros_like(self.v_nrm)
        tansum = torch.zeros_like(self.v_nrm)

        # Compute tangent space for each triangle
        duv1 = tex[1] - tex[0]
        duv2 = tex[2] - tex[0]
        dpos1 = pos[1] - pos[0]
        dpos2 = pos[2] - pos[0]

        tng_nom = dpos1 * duv2[..., 1:2] - dpos2 * duv1[..., 1:2]

        denom = duv1[..., 0:1] * duv2[..., 1:2] - duv1[..., 1:2] * duv2[..., 0:1]

        # Avoid division by zero for degenerated texture coordinates
        denom_safe = denom.clip(1e-6)
        tang = tng_nom / denom_safe

        # Update all 3 vertices
        for i in range(0, 3):
            idx = vn_idx[i][:, None].repeat(1, 3)
            tangents.scatter_add_(0, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
            tansum.scatter_add_(
                0, idx, torch.ones_like(tang)
            )  # tansum[n_i] = tansum[n_i] + 1
        # Also normalize it. Here we do not normalize the individual triangles first so larger area
        # triangles influence the tangent space more
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = F.normalize(tangents, dim=1)
        tangents = F.normalize(tangents - dot(tangents, self.v_nrm) * self.v_nrm)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))

        return tangents

    def quad_remesh(
        self,
        quad_vertex_count: int = -1,
        quad_rosy: int = 4,
        quad_crease_angle: float = -1.0,
        quad_smooth_iter: int = 2,
        quad_align_to_boundaries: bool = False,
    ) -> Mesh:
        if not QUAD_REMESH_AVAILABLE:
            raise ImportError("Quad remeshing requires pynim to be installed")
        if quad_vertex_count < 0:
            quad_vertex_count = self.v_pos.shape[0]
        v_pos = self.v_pos.detach().cpu().numpy().astype(np.float32)
        t_pos_idx = self.t_pos_idx.detach().cpu().numpy().astype(np.uint32)

        new_vert, new_faces = pynim.remesh(
            v_pos,
            t_pos_idx,
            quad_vertex_count // 4,
            rosy=quad_rosy,
            posy=4,
            creaseAngle=quad_crease_angle,
            align_to_boundaries=quad_align_to_boundaries,
            smooth_iter=quad_smooth_iter,
            deterministic=False,
        )

        # Briefly load in trimesh
        mesh = trimesh.Trimesh(vertices=new_vert, faces=new_faces.astype(np.int32))

        v_pos = torch.from_numpy(mesh.vertices).to(self.v_pos).contiguous()
        t_pos_idx = torch.from_numpy(mesh.faces).to(self.t_pos_idx).contiguous()

        # Create new mesh
        return Mesh(v_pos, t_pos_idx)

    def triangle_remesh(
        self,
        triangle_average_edge_length_multiplier: Optional[float] = None,
        triangle_remesh_steps: int = 10,
        triangle_vertex_count=-1,
    ):
        if not TRIANGLE_REMESH_AVAILABLE:
            raise ImportError("Triangle remeshing requires gpytoolbox to be installed")
        if triangle_vertex_count > 0:
            reduction = triangle_vertex_count / self.v_pos.shape[0]
            print("Triangle reduction:", reduction)
            v_pos = self.v_pos.detach().cpu().numpy().astype(np.float32)
            t_pos_idx = self.t_pos_idx.detach().cpu().numpy().astype(np.int32)
            if reduction > 1.0:
                subdivide_iters = int(math.ceil(math.log(reduction) / math.log(2)))
                print("Subdivide iters:", subdivide_iters)
                v_pos, t_pos_idx = gpytoolbox.subdivide(
                    v_pos,
                    t_pos_idx,
                    iters=subdivide_iters,
                )
                reduction = triangle_vertex_count / v_pos.shape[0]

            # Simplify
            points_out, faces_out, _, _ = gpytoolbox.decimate(
                v_pos,
                t_pos_idx,
                face_ratio=reduction,
            )

            # Convert back to torch
            self.v_pos = torch.from_numpy(points_out).to(self.v_pos)
            self.t_pos_idx = torch.from_numpy(faces_out).to(self.t_pos_idx)
            self._edges = None
            triangle_average_edge_length_multiplier = None

        edges = self.edges
        if triangle_average_edge_length_multiplier is None:
            h = None
        else:
            h = float(
                torch.linalg.norm(
                    self.v_pos[edges[:, 0]] - self.v_pos[edges[:, 1]], dim=1
                )
                .mean()
                .item()
                * triangle_average_edge_length_multiplier
            )

        # Convert to numpy
        v_pos = self.v_pos.detach().cpu().numpy().astype(np.float64)
        t_pos_idx = self.t_pos_idx.detach().cpu().numpy().astype(np.int32)

        # Remesh
        v_remesh, f_remesh = gpytoolbox.remesh_botsch(
            v_pos,
            t_pos_idx,
            triangle_remesh_steps,
            h,
        )

        # Convert back to torch
        v_pos = torch.from_numpy(v_remesh).to(self.v_pos).contiguous()
        t_pos_idx = torch.from_numpy(f_remesh).to(self.t_pos_idx).contiguous()

        # Create new mesh
        return Mesh(v_pos, t_pos_idx)

    @torch.no_grad()
    def unwrap_uv(
        self,
        island_padding: float = 0.02,
    ) -> Mesh:
        uv, indices = self.unwrapper(
            self.v_pos, self.v_nrm, self.t_pos_idx, island_padding
        )

        # Do store per vertex UVs.
        # This means we need to duplicate some vertices at the seams
        individual_vertices = self.v_pos[self.t_pos_idx].reshape(-1, 3)
        individual_faces = torch.arange(
            individual_vertices.shape[0],
            device=individual_vertices.device,
            dtype=self.t_pos_idx.dtype,
        ).reshape(-1, 3)
        uv_flat = uv[indices].reshape((-1, 2))
        # uv_flat[:, 1] = 1 - uv_flat[:, 1]

        self.v_pos = individual_vertices
        self.t_pos_idx = individual_faces
        self._v_tex = uv_flat
        self._v_nrm = self._compute_vertex_normal()
        self._v_tng = self._compute_vertex_tangent()

        return self

    def _compute_edges(self):
        # Compute edges
        edges = torch.cat(
            [
                self.t_pos_idx[:, [0, 1]],
                self.t_pos_idx[:, [1, 2]],
                self.t_pos_idx[:, [2, 0]],
            ],
            dim=0,
        )
        edges = edges.sort()[0]
        edges = torch.unique(edges, dim=0)
        return edges

    def _compute_halfedges(self) -> Int64[Tensor, "Nhe 4"]:
        """
        Construct the half-edge data structure from the triangular faces.
        Returns tensor of shape [Nhe, 4] where each row is:
        [face_idx, end_vertex, next_halfedge_idx, twin_halfedge_idx]

        Note: twin_halfedge_idx will be -1 for boundary edges
        """
        faces = self.t_pos_idx
        num_faces = faces.shape[0]
        device = faces.device

        # Create initial halfedges [face_idx, end_vertex, next_he, twin_he]
        halfedges = torch.zeros((num_faces * 3, 4), dtype=torch.long, device=device)

        # Assign face indices and next halfedge indices
        halfedges[:, 0] = torch.arange(num_faces, device=device).repeat_interleave(3)
        halfedges[:, 2] = torch.arange(num_faces * 3, device=device)
        halfedges[:, 2] = (halfedges[:, 2] + 1) % 3 + (halfedges[:, 2] // 3) * 3

        # Assign end vertices
        halfedges[0::3, 1] = faces[:, 1].flatten()  # v0->v1
        halfedges[1::3, 1] = faces[:, 2].flatten()  # v1->v2
        halfedges[2::3, 1] = faces[:, 0].flatten()  # v2->v0

        # Initialize twins to -1
        halfedges[:, 3] = -1

        # Build start vertices
        start_vertices = torch.zeros_like(halfedges[:, 1])
        start_vertices[0::3] = faces[:, 0].flatten()  # v0->v1
        start_vertices[1::3] = faces[:, 1].flatten()  # v1->v2
        start_vertices[2::3] = faces[:, 2].flatten()  # v2->v0

        # Create edge to halfedge index mapping
        edges = torch.stack([start_vertices, halfedges[:, 1]], dim=1)
        edge_dict = {tuple(edge.tolist()): idx for idx, edge in enumerate(edges)}

        # Find twins
        for i, (start_v, end_v) in enumerate(edges):
            twin_edge = (end_v.item(), start_v.item())
            if twin_edge in edge_dict:
                halfedges[i, 3] = edge_dict[twin_edge]

        return halfedges

    def normal_consistency(self) -> Float[Tensor, "Ne"]:
        """Calculates the cosine similarity between both vertices of each edge."""
        edge_nrm: Float[Tensor, "Ne 2 3"] = self.v_nrm[self.edges]
        nc = 1.0 - torch.cosine_similarity(edge_nrm[:, 0], edge_nrm[:, 1], dim=-1)
        return nc

    def _laplacian_uniform(self):
        # from stable-dreamfusion
        # https://github.com/ashawkey/stable-dreamfusion/blob/8fb3613e9e4cd1ded1066b46e80ca801dfb9fd06/nerf/renderer.py#L224
        verts, faces = self.v_pos, self.t_pos_idx

        V = verts.shape[0]

        # Neighbor indices
        ii = faces[:, [1, 2, 0]].flatten()
        jj = faces[:, [2, 0, 1]].flatten()
        adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(
            dim=1
        )
        adj_values = torch.ones(adj.shape[1]).to(verts)

        # Diagonal indices
        diag_idx = adj[0]

        # Build the sparse matrix
        idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
        values = torch.cat((-adj_values, adj_values))

        # The coalesce operation sums the duplicate indices, resulting in the
        # correct diagonal
        return torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()

    def laplacian(self) -> Float[Tensor, ""]:
        with torch.no_grad():
            L = self._laplacian_uniform()
        loss = L.mm(self.v_pos)
        loss = loss.norm(dim=1)
        return loss
