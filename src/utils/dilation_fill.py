import torch
import torch.nn.functional as F


def dilate_fill(img, mask, iterations=10):
    oldMask = mask.float()
    oldImg = img

    mask_kernel = torch.ones(
        (1, 1, 3, 3),
        dtype=oldMask.dtype,
        device=oldMask.device,
    )

    for i in range(iterations):
        newMask = torch.nn.functional.max_pool2d(oldMask, 3, 1, 1)

        # Fill the extension with mean color of old valid regions
        img_unfold = F.unfold(oldImg, (3, 3)).view(1, img.shape[1], 3 * 3, -1)
        mask_unfold = F.unfold(oldMask, (3, 3)).view(1, 1, 3 * 3, -1)
        new_mask_unfold = F.unfold(newMask, (3, 3)).view(1, 1, 3 * 3, -1)

        # Average color of the valid region
        mean_color = (img_unfold.sum(dim=2) / mask_unfold.sum(dim=2).clip(1)).unsqueeze(
            2
        )
        # Extend it to the new region
        fill_color = (mean_color * new_mask_unfold).view(1, img.shape[1] * 3 * 3, -1)

        mask_conv = F.conv2d(
            newMask, mask_kernel, padding=1
        )  # Get the sum for each kernel patch
        newImg = F.fold(
            fill_color, (img.shape[-2], img.shape[-1]), (3, 3)
        ) / mask_conv.clamp(1)

        diffMask = newMask - oldMask

        oldMask = newMask
        oldImg = torch.lerp(oldImg, newImg, diffMask)
        # diffMask * newImg + (1 - diffMask) * oldImg

    return oldImg
