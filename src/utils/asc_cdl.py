import warnings
import xml.etree.ElementTree as ET

import torch
from jaxtyping import Float


def load_asc_cdl(cdl_path: str, device: torch.device = torch.device("cpu")) -> dict:
    """
    Loads ASC CDL parameters from an XML file.

    Parameters:
        cdl_path (str): Path to the ASC CDL XML file

    Returns:
        Dict:
            slope, offset, power, and saturation values as torch tensors
    """
    try:
        tree = ET.parse(cdl_path)
        root = tree.getroot()
    except Exception as e:
        raise ValueError(f"Error loading ASC CDL from {cdl_path}: {e}")

    # Extract SOP values
    sop_node = root.find(".//SOPNode")
    slope = torch.tensor(
        [float(x) for x in sop_node.find("Slope").text.split()], device=device
    )
    offset = torch.tensor(
        [float(x) for x in sop_node.find("Offset").text.split()], device=device
    )
    power = torch.tensor(
        [float(x) for x in sop_node.find("Power").text.split()], device=device
    )

    # Extract Saturation value
    sat_node = root.find(".//SatNode")
    saturation = torch.tensor(float(sat_node.find("Saturation").text), device=device)

    return {"slope": slope, "offset": offset, "power": power, "saturation": saturation}


def save_asc_cdl(cdl_dict: dict, cdl_path: str):
    """
    Saves ASC CDL parameters to an XML file.

    Parameters:
        cdl_dict (dict): Dictionary containing slope, offset, power, and saturation values
    """
    root = ET.Element("ASC_CDL")
    sop_node = ET.SubElement(root, "SOPNode")
    ET.SubElement(sop_node, "Slope").text = " ".join(
        str(x) for x in cdl_dict["slope"].cpu().numpy()
    )
    ET.SubElement(sop_node, "Offset").text = " ".join(
        str(x) for x in cdl_dict["offset"].cpu().numpy()
    )
    ET.SubElement(sop_node, "Power").text = " ".join(
        str(x) for x in cdl_dict["power"].cpu().numpy()
    )
    sat_node = ET.SubElement(root, "SatNode")
    ET.SubElement(sat_node, "Saturation").text = str(
        cdl_dict["saturation"].cpu().numpy()
    )

    tree = ET.ElementTree(root)
    try:
        tree.write(cdl_path)
    except Exception as e:
        raise ValueError(f"Error saving ASC CDL to {cdl_path}: {e}")


def apply_sop(
    img: Float[torch.Tensor, "*B C H W"],
    slope: Float[torch.Tensor, "*B C"],
    offset: Float[torch.Tensor, "*B C"],
    power: Float[torch.Tensor, "*B C"],
    clamp: bool = True,
) -> Float[torch.Tensor, "*B C H W"]:
    """
    Applies Slope, Offset, and Power adjustments.

    Parameters:
        img (torch.Tensor): Input image tensor (*B, C, H, W)
        slope (torch.Tensor): Slope per channel (*B, C)
        offset (torch.Tensor): Offset per channel (*B, C)
        power (torch.Tensor): Power per channel (*B, C)

    Returns:
        torch.Tensor: Image after SOP adjustments.
    """
    so = img * slope.unsqueeze(-1).unsqueeze(-1) + offset.unsqueeze(-1).unsqueeze(-1)
    if clamp:
        so = torch.clamp(so, min=0.0, max=1.0)
    return torch.where(
        so > 1e-7, torch.pow(so.clamp(min=1e-7), power.unsqueeze(-1).unsqueeze(-1)), so
    )


def apply_saturation(
    img: Float[torch.Tensor, "*B C H W"],
    saturation: Float[torch.Tensor, "*B"],
) -> Float[torch.Tensor, "*B C H W"]:
    """
    Applies saturation adjustment.

    Parameters:
        img (torch.Tensor): Image tensor (*B, C, H, W)
        saturation (torch.Tensor): Saturation factor (*B)

    Returns:
        torch.Tensor: Image after saturation adjustment.
    """
    # Calculate luminance using Rec. 709 coefficients
    lum = (
        0.2126 * img[..., 0, :, :]
        + 0.7152 * img[..., 1, :, :]
        + 0.0722 * img[..., 2, :, :]
    )
    lum = lum.unsqueeze(-3)  # Add channel dimension
    return lum + (img - lum) * saturation.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def asc_cdl_forward(
    img: Float[torch.Tensor, "*B C H W"],
    slope: Float[torch.Tensor, "*B C"],
    offset: Float[torch.Tensor, "*B C"],
    power: Float[torch.Tensor, "*B C"],
    saturation: Float[torch.Tensor, "*B"],
    clamp: bool = True,
) -> Float[torch.Tensor, "*B C H W"]:
    """
    Applies ASC CDL transformation in Fwd or FwdNoClamp mode.

    Parameters:
        img (torch.Tensor): Input image tensor (*B, C, H, W)
        slope (torch.Tensor): Slope per channel (*B, C)
        offset (torch.Tensor): Offset per channel (*B, C)
        power (torch.Tensor): Power per channel (*B, C)
        saturation (torch.Tensor): Saturation factor (*B)
        clamp (bool): If True, clamps output to [0, 1] (Fwd mode). If False, no clamping (FwdNoClamp mode).

    Returns:
        torch.Tensor: Transformed image tensor.
    """
    # Add warning if saturation, slope, power are below 0
    if (saturation < 0).any():
        warnings.warn("Saturation is below 0, this will result in a color shift.")
    if (slope < 0).any():
        warnings.warn("Slope is below 0, this will result in a color shift.")
    if (power < 0).any():
        warnings.warn("Power is below 0, this will result in a color shift.")

    img_batch_dim = img.shape[:-3]
    # Check if slope, offset, power, saturation have the same batch dimension
    # If they do not have any batch dimensions, add a single batch dimensions
    if slope.ndim == 1:
        slope = slope.view(*[1] * len(img_batch_dim), *slope.shape)
    if offset.ndim == 1:
        offset = offset.view(*[1] * len(img_batch_dim), *offset.shape)
    if power.ndim == 1:
        power = power.view(*[1] * len(img_batch_dim), *power.shape)
    if saturation.ndim == 0:
        saturation = saturation.view(*[1] * len(img_batch_dim), *saturation.shape)

    # Now check that the lengths are matching
    assert slope.ndim == len(img_batch_dim) + 1
    assert offset.ndim == len(img_batch_dim) + 1
    assert power.ndim == len(img_batch_dim) + 1
    assert saturation.ndim == len(img_batch_dim)

    # Apply Slope, Offset, and Power adjustments
    img = apply_sop(img, slope, offset, power, clamp=clamp)
    # print("img after sop", img.min(), img.max())
    # Apply Saturation adjustment
    img = apply_saturation(img, saturation)
    # print("img after saturation", img.min(), img.max())
    # Clamp if in Fwd mode
    if clamp:
        img = torch.clamp(img, 0.0, 1.0)
    return img


def inverse_saturation(
    img: Float[torch.Tensor, "*B C H W"],
    saturation: Float[torch.Tensor, "*B"],
) -> Float[torch.Tensor, "*B C H W"]:
    """
    Reverts saturation adjustment.

    Parameters:
        img (torch.Tensor): Image tensor (*B, C, H, W)
        saturation (torch.Tensor): Saturation factor (*B)

    Returns:
        torch.Tensor: Image after reversing saturation adjustment.
    """
    # Calculate luminance using Rec. 709 coefficients
    lum = (
        0.2126 * img[..., 0, :, :]
        + 0.7152 * img[..., 1, :, :]
        + 0.0722 * img[..., 2, :, :]
    )
    lum = lum.unsqueeze(-3)  # Add channel dimension
    return lum + (img - lum) / saturation.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def asc_cdl_reverse(
    img: Float[torch.Tensor, "*B C H W"],
    slope: Float[torch.Tensor, "*B C"],
    offset: Float[torch.Tensor, "*B C"],
    power: Float[torch.Tensor, "*B C"],
    saturation: Float[torch.Tensor, "*B"],
    clamp: bool = True,
) -> Float[torch.Tensor, "*B C H W"]:
    """
    Applies reverse ASC CDL transformation.

    Parameters:
        img (torch.Tensor): Transformed image tensor (*B, C, H, W)
        slope (torch.Tensor): Slope per channel (*B, C)
        offset (torch.Tensor): Offset per channel (*B, C)
        power (torch.Tensor): Power per channel (*B, C)
        saturation (torch.Tensor): Saturation factor (*B)
        clamp (bool): If True, clamps output to [0, 1].

    Returns:
        torch.Tensor: Recovered input image tensor.
    """
    # Add warning if saturation, slope, power are below 0
    if (saturation < 0).any():
        warnings.warn("Saturation is below 0, this will result in a color shift.")
    if (slope < 0).any():
        warnings.warn("Slope is below 0, this will result in a color shift.")
    if (power < 0).any():
        warnings.warn("Power is below 0, this will result in a color shift.")

    img_batch_dim = img.shape[:-3]
    # Check if slope, offset, power, saturation have the same batch dimension
    # If they do not have any batch dimensions, add a single batch dimensions
    if slope.ndim == 1:
        slope = slope.view(*[1] * len(img_batch_dim), *slope.shape)
    if offset.ndim == 1:
        offset = offset.view(*[1] * len(img_batch_dim), *offset.shape)
    if power.ndim == 1:
        power = power.view(*[1] * len(img_batch_dim), *power.shape)
    if saturation.ndim == 0:
        saturation = saturation.view(*[1] * len(img_batch_dim), *saturation.shape)

    # Now check that the lengths are matching
    assert slope.ndim == len(img_batch_dim) + 1
    assert offset.ndim == len(img_batch_dim) + 1
    assert power.ndim == len(img_batch_dim) + 1
    assert saturation.ndim == len(img_batch_dim)

    # Inverse Saturation adjustment
    img = inverse_saturation(img, saturation)
    # Inverse SOP adjustments
    if clamp:
        img = torch.clamp(img, 0.0, 1.0)
    img = torch.where(
        img > 1e-7, torch.pow(img, 1 / power.unsqueeze(-1).unsqueeze(-1)), img
    )
    img = (img - offset.unsqueeze(-1).unsqueeze(-1)) / slope.unsqueeze(-1).unsqueeze(-1)
    # Clamp if specified
    if clamp:
        img = torch.clamp(img, 0.0, 1.0)
    return img
