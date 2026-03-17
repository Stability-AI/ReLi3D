import torch
import torch.nn as nn

from src.models.mesh import Mesh


class LoopSubdivision(nn.Module):
    """
    One iteration of Loop subdivision in a more vectorized manner,
    for closed manifold meshes (no boundary).
    """

    def forward(self, mesh: Mesh) -> Mesh:
        """
        mesh: Mesh to subdivide

        Returns:
          new_mesh: Mesh with subdivided vertices and faces
        """
        # --------------------------------------------------------
        # 1) Extract old mesh data
        # --------------------------------------------------------
        old_v_pos = mesh.v_pos  # shape (N, 3)
        old_t_idx = mesh.t_pos_idx  # shape (F, 3)
        halfedges = mesh.halfedges  # shape (3F, 4) or None

        device = old_v_pos.device
        num_verts = old_v_pos.shape[0]
        num_faces = old_t_idx.shape[0]

        # Trigger halfedge computation if not given
        he = halfedges  # shape [3F, 4], columns: [face_idx, end_v, next_he, twin_he]
        he_count = he.shape[0]

        # --------------------------------------------------------
        # 2) Build adjacency for each vertex (neighbors)
        #    so we can do the Loop vertex update
        # --------------------------------------------------------
        # We reconstruct "start_v" for each halfedge the same way as in many halfedge computations.
        prev_index = torch.arange(he_count, device=device)
        # The 'previous' halfedge index in the same face
        mod3_mask = prev_index % 3 == 0
        prev_index[mod3_mask] += 2
        prev_index[~mod3_mask] -= 1

        start_v = he[prev_index, 1]  # The vertex at the start of this halfedge
        end_v = he[:, 1]

        # Collect neighbors per vertex
        vertex_neighbors = [[] for _ in range(num_verts)]
        for i in range(he_count):
            sv = start_v[i].item()
            ev = end_v[i].item()
            vertex_neighbors[sv].append(ev)

        # Convert to unique tensors
        for v in range(num_verts):
            if len(vertex_neighbors[v]) == 0:
                # Handle isolated or error cases
                vertex_neighbors[v] = torch.empty((0,), device=device, dtype=torch.long)
            else:
                vertex_neighbors[v] = torch.unique(
                    torch.tensor(vertex_neighbors[v], device=device, dtype=torch.long)
                )

        # --------------------------------------------------------
        # 3) Update old vertex positions (Loop weighting)
        # --------------------------------------------------------
        new_old_positions = old_v_pos.clone()  # shape (N,3)
        for v in range(num_verts):
            neighs = vertex_neighbors[v]
            n = neighs.shape[0]

            if n < 3:
                # For boundary or degenerate valence < 3,
                # do a simple local averaging
                if n > 0:
                    new_old_positions[v] = 0.75 * old_v_pos[v] + 0.25 * old_v_pos[
                        neighs
                    ].mean(dim=0)
                continue

            # Standard Loop formula for interior vertex
            beta = 3.0 / (8.0 * n)  # This is now the same for all n >= 3

            sum_neighbors = old_v_pos[neighs].sum(dim=0)
            new_old_positions[v] = (1.0 - n * beta) * old_v_pos[
                v
            ] + beta * sum_neighbors

        # --------------------------------------------------------
        # 4) Compute edge midpoints + fix them via Loop interior rule
        #    BEFORE concatenating
        # --------------------------------------------------------
        # (a) Identify unique edges
        candidate_edges = torch.stack(
            [
                old_t_idx[:, [0, 1]],
                old_t_idx[:, [1, 2]],
                old_t_idx[:, [2, 0]],
            ],
            dim=1,
        ).reshape(-1, 2)  # shape (3F,2)

        sorted_edges = torch.sort(candidate_edges, dim=1)[0]  # canonical orientation
        unique_edges, inverse_edges = torch.unique(
            sorted_edges, return_inverse=True, dim=0
        )
        # unique_edges: (E,2); inverse_edges: (3F,)

        # (b) Create midpoints as the naive 0.5*(A+B)
        eA = unique_edges[:, 0]  # (E,)
        eB = unique_edges[:, 1]  # (E,)
        posA = old_v_pos[eA]  # shape (E,3)
        posB = old_v_pos[eB]  # shape (E,3)
        midpoints = 0.5 * (posA + posB)  # shape (E,3)

        # (c) For each edge, see if it's interior (2 adjacent faces)
        #     and apply the 3/8–1/8 rule
        for e_idx in range(unique_edges.shape[0]):
            vA = eA[e_idx]
            vB = eB[e_idx]

            # Find which faces contain this edge
            #   We'll do a mask on candidate_edges
            mask = (candidate_edges == vA).any(dim=1) & (candidate_edges == vB).any(
                dim=1
            )
            face_indices = torch.div(torch.where(mask)[0], 3, rounding_mode="floor")

            # For a closed manifold interior edge, face_indices should have length 2
            # (the two adjacent triangles).
            # We'll just do the standard Loop midpoint update if we see exactly 2 faces.
            # Otherwise, we leave it at the naive midpoint.
            if face_indices.shape[0] == 2:
                f1, f2 = face_indices
                # Opposite vertices in each face
                opp_v1 = old_t_idx[f1][
                    ~torch.isin(old_t_idx[f1], torch.tensor([vA, vB], device=device))
                ]
                opp_v2 = old_t_idx[f2][
                    ~torch.isin(old_t_idx[f2], torch.tensor([vA, vB], device=device))
                ]
                midpoints[e_idx] = (3.0 / 8.0) * (old_v_pos[vA] + old_v_pos[vB]) + (
                    1.0 / 8.0
                ) * (old_v_pos[opp_v1] + old_v_pos[opp_v2])

        # --------------------------------------------------------
        # 5) Build final vertex positions
        #    (Now that midpoints[] is corrected, do the concat)
        # --------------------------------------------------------
        edge_vertex_offset = num_verts
        new_v_pos = torch.cat([new_old_positions, midpoints], dim=0)  # shape (N+E,3)

        # --------------------------------------------------------
        # 6) Construct the new faces (4 per old triangle)
        # --------------------------------------------------------
        new_faces = []
        for f_idx in range(num_faces):
            v1 = old_t_idx[f_idx, 0].item()
            v2 = old_t_idx[f_idx, 1].item()
            v3 = old_t_idx[f_idx, 2].item()

            base = f_idx * 3
            e1_idx = inverse_edges[base + 0].item()  # midpoint for (v1,v2)
            e2_idx = inverse_edges[base + 1].item()  # midpoint for (v2,v3)
            e3_idx = inverse_edges[base + 2].item()  # midpoint for (v3,v1)

            m12 = edge_vertex_offset + e1_idx
            m23 = edge_vertex_offset + e2_idx
            m31 = edge_vertex_offset + e3_idx

            # The 4 sub-triangles
            new_faces.extend(
                [[v1, m12, m31], [m12, v2, m23], [m23, v3, m31], [m12, m23, m31]]
            )

        new_t_idx = torch.tensor(new_faces, device=device, dtype=torch.long)

        # Build and return the new subdivided mesh
        return Mesh(v_pos=new_v_pos, t_pos_idx=new_t_idx)
