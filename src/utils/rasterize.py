import abc

try:
    import drtk
except ImportError:
    drtk = None

import nvdiffrast.torch as dr
import torch

from src.utils.typing import Float, Integer, Optional, Tensor, Tuple, Union


class AbstractRasterizerContext(abc.ABC):
    @abc.abstractmethod
    def vertex_transform(
        self,
        verts: Float[Tensor, "Nv 3"],
        w2c_matrix: Float[Tensor, "B 4 4"],
        projection_matrix: Float[Tensor, "B 4 4"],
        resolution: Union[int, Tuple[int, int]],
    ) -> Union[Float[Tensor, "B Nv 3"], Float[Tensor, "B Nv 4"]]:
        pass

    @abc.abstractmethod
    def rasterize(
        self,
        pos: Union[Float[Tensor, "Nv 3"], Float[Tensor, "B Nv 4"]],
        tri: Integer[Tensor, "Nf 3"],
        resolution: Union[int, Tuple[int, int]],
        **kwargs,
    ) -> Tuple[Float[Tensor, "B H W 4"], Optional[Float[Tensor, "B H W 4"]]]:
        pass

    @abc.abstractmethod
    def rasterize_one(
        self,
        pos: Union[Float[Tensor, "Nv 3"], Float[Tensor, "Nv 4"]],
        tri: Integer[Tensor, "Nf 3"],
        resolution: Union[int, Tuple[int, int]],
        **kwargs,
    ) -> Tuple[Float[Tensor, "B H W 4"], Optional[Float[Tensor, "B H W 4"]]]:
        pass

    @abc.abstractmethod
    def post_process(
        self,
        color: Float[Tensor, "B H W C"],
        rast: Float[Tensor, "B H W 4"],
        pos: Union[Float[Tensor, "B Nv 3"], Float[Tensor, "B Nv 4"]],
        tri: Integer[Tensor, "Nf 3"],
    ) -> Float[Tensor, "B H W C"]:
        pass

    @abc.abstractmethod
    def interpolate(
        self,
        attr: Float[Tensor, "B Nv C"],
        rast: Float[Tensor, "B H W 4"],
        tri: Integer[Tensor, "Nf 3"],
        **kwargs,
    ) -> Tuple[Float[Tensor, "B H W C"], Optional[Float[Tensor, "B H W C"]]]:
        pass

    @abc.abstractmethod
    def interpolate_one(
        self,
        attr: Float[Tensor, "Nv C"],
        rast: Float[Tensor, "B H W 4"],
        tri: Integer[Tensor, "Nf 3"],
        **kwargs,
    ) -> Tuple[Float[Tensor, "B H W C"], Optional[Float[Tensor, "B H W C"]]]:
        pass

    @abc.abstractmethod
    def get_mask(self, rast: Float[Tensor, "B H W 4"]) -> Float[Tensor, "B H W"]:
        pass


class NVDiffRasterizerContext(AbstractRasterizerContext):
    def __init__(self, context_type: str, device: torch.device) -> None:
        self.device = device
        self.ctx = self.initialize_context(context_type, device)

    def initialize_context(
        self, context_type: str, device: torch.device
    ) -> Union[dr.RasterizeGLContext, dr.RasterizeCudaContext]:
        if context_type == "gl":
            return dr.RasterizeGLContext(device=device)
        elif context_type == "cuda":
            return dr.RasterizeCudaContext(device=device)
        else:
            raise ValueError(f"Unknown rasterizer context type: {context_type}")

    def vertex_transform(
        self,
        verts: Float[Tensor, "Nv 3"],
        w2c_matrix: Float[Tensor, "B 4 4"],
        projection_matrix: Float[Tensor, "B 4 4"],
        resolution: Union[int, Tuple[int, int]],
    ) -> Float[Tensor, "B Nv 4"]:
        mvp_mtx = projection_matrix @ w2c_matrix
        verts_homo = torch.nn.functional.pad(verts, (0, 1), mode="constant", value=1.0)
        return torch.matmul(verts_homo, mvp_mtx.permute(0, 2, 1))

    def rasterize(
        self,
        pos: Float[Tensor, "B Nv 4"],
        tri: Integer[Tensor, "Nf 3"],
        resolution: Union[int, Tuple[int, int]],
        ranges: Optional[Float[Tensor, "B 2"]] = None,
    ):
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        # rasterize in instance mode (single topology)
        return dr.rasterize(
            self.ctx, pos.float(), tri.int(), resolution, ranges=ranges, grad_db=True
        )

    def rasterize_one(
        self,
        pos: Float[Tensor, "Nv 4"],
        tri: Integer[Tensor, "Nf 3"],
        resolution: Union[int, Tuple[int, int]],
    ):
        # rasterize one single mesh under a single viewpoint
        rast, rast_db = self.rasterize(pos[None, ...], tri, resolution)
        return rast[0], rast_db[0]

    def post_process(
        self,
        color: Float[Tensor, "B H W C"],
        rast: Float[Tensor, "B H W 4"],
        pos: Float[Tensor, "B Nv 4"],
        tri: Integer[Tensor, "Nf 3"],
    ) -> Float[Tensor, "B H W C"]:
        return dr.antialias(color.float(), rast, pos.float(), tri.int())

    def interpolate(
        self,
        attr: Float[Tensor, "B Nv C"],
        rast: Float[Tensor, "B H W 4"],
        tri: Integer[Tensor, "Nf 3"],
        rast_db=None,
        diff_attrs=None,
    ) -> Float[Tensor, "B H W C"]:
        return dr.interpolate(
            attr.float(), rast, tri.int(), rast_db=rast_db, diff_attrs=diff_attrs
        )

    def interpolate_one(
        self,
        attr: Float[Tensor, "Nv C"],
        rast: Float[Tensor, "B H W 4"],
        tri: Integer[Tensor, "Nf 3"],
        rast_db=None,
        diff_attrs=None,
    ) -> Float[Tensor, "B H W C"]:
        return self.interpolate(attr[None, ...], rast, tri, rast_db, diff_attrs)

    def get_mask(self, rast: Float[Tensor, "B H W 4"]) -> Float[Tensor, "B H W"]:
        return rast[..., 3:] > 0


class EdgeGradEstimatorFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        v_pix: torch.Tensor,
        v_pix_img: torch.Tensor,
        vi: torch.Tensor,
        img: torch.Tensor,
        index_img: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(v_pix, img, index_img, vi)
        return img

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[1]:
            return None, None, None, grad_output, None

        v_pix, img, index_img, vi = ctx.saved_tensors
        # Permute img and grad_output from B H W C to B C H W
        v_pix = v_pix.float()
        img_permuted = img.permute(0, 3, 1, 2).contiguous().float()
        grad_output_permuted = grad_output.permute(0, 3, 1, 2).contiguous().float()

        # Call the CUDA function
        grad_v_pix_img = torch.ops.edge_grad_ext.edge_grad_estimator_backward(
            v_pix, img_permuted, index_img, vi.int(), grad_output_permuted
        )

        return None, grad_v_pix_img, None, grad_output, None


class DRTKRasterizerContext(AbstractRasterizerContext):
    def __init__(self) -> None:
        if drtk is None:
            raise ImportError(
                "drtk is not installed. Use the nvdiffrast rasterizer or install DRTK."
            )

    def vertex_transform(
        self,
        verts: Float[Tensor, "Nv 3"],
        w2c_matrix: Float[Tensor, "B 4 4"],
        projection_matrix: Float[Tensor, "B 4 4"],
        resolution: Union[int, Tuple[int, int]],
    ) -> Float[Tensor, "B Nv 3"]:
        if isinstance(resolution, int):
            resolution = (resolution, resolution)

        verts_homo = torch.nn.functional.pad(verts, (0, 1), mode="constant", value=1.0)
        verts_camera = torch.matmul(verts_homo, w2c_matrix.permute(0, 2, 1))
        z_component = -verts_camera[..., 2]

        verts_clip = torch.matmul(verts_camera, projection_matrix.permute(0, 2, 1))

        pos_ndc = verts_clip[..., :3] / verts_clip[..., 3:]
        x_image = ((pos_ndc[..., 0] + 1) * (resolution[1] / 2)) - 0.5
        y_image = ((pos_ndc[..., 1] + 1) * (resolution[0] / 2)) - 0.5

        return torch.stack((x_image, y_image, z_component), -1)

    def rasterize(
        self,
        pos: Float[Tensor, "B Nv 4"],
        tri: Integer[Tensor, "Nf 3"],
        resolution: Union[int, Tuple[int, int]],
    ):
        if isinstance(resolution, int):
            resolution = (resolution, resolution)

        index_img = drtk.rasterize(
            pos.float(), tri.int(), height=resolution[0], width=resolution[1]
        )
        depth_img, bary_img = drtk.render(pos.float(), tri.int(), index_img)

        rast = torch.cat([bary_img, index_img.float().unsqueeze(1)], 1)
        return rast.permute(0, 2, 3, 1), None

    def rasterize_one(
        self,
        pos: Float[Tensor, "Nv 4"],
        tri: Integer[Tensor, "Nf 3"],
        resolution: Union[int, Tuple[int, int]],
    ):
        # rasterize one single mesh under a single viewpoint
        rast, _ = self.rasterize(pos[None, ...], tri, resolution)
        return rast[0], None

    def post_process(
        self,
        color: Float[Tensor, "B H W C"],
        rast: Float[Tensor, "B H W 4"],
        pos: Float[Tensor, "B Nv 3"],
        tri: Integer[Tensor, "Nf 3"],
    ) -> Float[Tensor, "B H W C"]:
        # Prepare inputs
        bary_img = rast[..., :3]
        index_img = rast[..., 3].int()
        img = color

        v_pix = pos.float()
        vi = tri.int()

        v_pix_img = drtk.interpolate(
            v_pix, vi, index_img, bary_img.permute(0, 3, 1, 2).detach()
        )

        grad_enhanced = EdgeGradEstimatorFunction.apply(
            v_pix, v_pix_img, vi, img, index_img
        )

        return grad_enhanced

    def interpolate(
        self,
        attr: Float[Tensor, "B Nv C"],
        rast: Float[Tensor, "B H W 4"],
        tri: Integer[Tensor, "Nf 3"],
    ) -> Float[Tensor, "B H W C"]:
        bary_img = rast[..., :3].permute(0, 3, 1, 2)
        index_img = rast[..., 3].int()

        if attr.shape[0] != index_img.shape[0]:
            attr = attr.repeat_interleave(index_img.shape[0], 0)

        return (
            drtk.interpolate(attr.float(), tri.int(), index_img, bary_img).permute(
                0, 2, 3, 1
            ),
            None,
        )

    def interpolate_one(
        self,
        attr: Float[Tensor, "Nv C"],
        rast: Float[Tensor, "B H W 4"],
        tri: Integer[Tensor, "Nf 3"],
    ) -> Float[Tensor, "B H W C"]:
        return self.interpolate(attr[None, ...], rast, tri)

    def get_mask(self, rast: Float[Tensor, "B H W 4"]) -> Float[Tensor, "B H W"]:
        return rast[..., 3:] >= 0
