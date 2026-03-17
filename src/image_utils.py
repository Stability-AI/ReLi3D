from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as torchvision_F


def get_1d_bounds(arr):
    nz = np.flatnonzero(arr)
    return nz[0], nz[-1]


def get_bbox_from_mask(mask, thr=0.5):
    masks_for_box = (mask > thr).astype(np.float32)
    assert masks_for_box.sum() > 0, "Empty mask!"
    x0, x1 = get_1d_bounds(masks_for_box.sum(axis=-2))
    y0, y1 = get_1d_bounds(masks_for_box.sum(axis=-1))
    return x0, y0, x1, y1


def resize_foreground(
    image: Union[Image.Image, np.ndarray],
    ratio: float,
    out_size=None,
    threshold=0.5,
) -> Image:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image, mode="RGBA")
    assert image.mode == "RGBA"
    # Get bounding box
    mask_np = np.array(image)[:, :, -1]
    if mask_np.dtype == np.uint8:
        mask_np = mask_np / 255
    assert np.all(mask_np <= 1) and np.all(mask_np >= 0), "Mask is not between 0 and 1!"
    x1, y1, x2, y2 = get_bbox_from_mask(mask_np, thr=threshold)
    h, w = y2 - y1, x2 - x1
    yc, xc = (y1 + y2) / 2, (x1 + x2) / 2
    scale = max(h, w) / ratio

    new_image = torchvision_F.crop(
        image,
        top=int(yc - scale / 2),
        left=int(xc - scale / 2),
        height=int(scale),
        width=int(scale),
    )
    if out_size is not None:
        new_image = new_image.resize(out_size)

    return new_image


def load_image(
    path: Union[str, bytes, torch.Tensor],
    is_rgb: bool = True,
    float_output: bool = True,
    resize_size: Optional[Tuple[int, int]] = None,
    to_torch: bool = True,
) -> np.ndarray:
    if isinstance(path, torch.Tensor) and path.ndim == 1 and path.dtype == torch.uint8:
        path = path.numpy()
    if isinstance(path, bytes):
        path = np.frombuffer(path, np.uint8)

    if isinstance(path, np.ndarray):
        image = cv2.imdecode(path, cv2.IMREAD_UNCHANGED)
    else:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if image is None:
        return None

    # Handle EXR files which might be single channel
    if len(image.shape) == 2:
        image = image[..., None]

    # Handle alpha, RGB, hdr, going from int types to float 0-1 if requested
    alpha_present = image.shape[-1] == 4
    alpha = None
    if alpha_present:
        alpha = image[..., -1]
        image = image[..., :3]
    if is_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 2:
            image = image[..., None]

    image_dtype = image.dtype
    if float_output and image_dtype not in [np.float16, np.float32, np.float64]:
        max_val = np.iinfo(image_dtype).max
        image = image.astype(np.float32) / max_val
        assert image.min() >= 0 and image.max() <= 1, "Image is not between 0 and 1!"

    if alpha is not None:
        image = np.concatenate([image, alpha[..., None]], axis=-1)

    if resize_size is not None:
        image = cv2.resize(image, resize_size)

    if to_torch:
        image = torch.from_numpy(image)
        if float_output:
            image = image.float()

    return image
