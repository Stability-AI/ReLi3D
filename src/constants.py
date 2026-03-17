from __future__ import annotations

import abc

import numpy as np
import torch

from src.utils.typing import Any, Dict, Tensor, Tuple, Union


# Abstract class for normalization
class Normalization(abc.ABC):
    @abc.abstractmethod
    def normalize(
        self, data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        pass


# Concrete normalization classes
class MinMaxNormalization(Normalization):
    def normalize(
        self, data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        torch_to = None
        if isinstance(data, torch.Tensor):
            torch_to = {"dtype": data.dtype, "device": data.device}
            data = data.detach().cpu().numpy()
        normed = (data - np.min(data)) / (np.max(data) - np.min(data))
        if torch_to is not None:
            normed = torch.from_numpy(normed).to(**torch_to)
        return normed


class UnitNormalization(Normalization):
    def normalize(self, data: np.ndarray) -> np.ndarray:
        torch_to = None
        if isinstance(data, torch.Tensor):
            torch_to = {"dtype": data.dtype, "device": data.device}
            data = data.detach().float().cpu().numpy()
        norms = np.linalg.norm(data, axis=-1, keepdims=True)
        norms[norms == 0] = 1
        normed = data / norms
        if torch_to is not None:
            normed = torch.from_numpy(normed).to(**torch_to)
        return normed


# Abstract class for logging transform
class LoggingTransform(abc.ABC):
    @abc.abstractmethod
    def transform(
        self, data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        pass


class ValueTransform(LoggingTransform):
    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value

    def transform(
        self, data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        torch_to = None
        if isinstance(data, torch.Tensor):
            torch_to = {"dtype": data.dtype, "device": data.device}
            data = data.detach().float().cpu().numpy()
        ret = (data - self.min_value) / (self.max_value - self.min_value)
        if torch_to is not None:
            ret = torch.from_numpy(ret).to(**torch_to)
        return ret


class InversedValueTransform(LoggingTransform):
    def transform(
        self, data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        torch_to = None
        if isinstance(data, torch.Tensor):
            torch_to = {"dtype": data.dtype, "device": data.device}
            data = data.detach().cpu().numpy()
        ret = 1 / data
        if torch_to is not None:
            ret = torch.from_numpy(ret).to(**torch_to)
        return ret


# Known suffix-to-flag mapping
SUFFIX_FLAGS = {
    "coarse": "is_coarse",
    "fine": "is_fine",
    "rays": "is_rays",
    "ray-samples": "is_raysample",
    "integration-samples": "is_integration_sample",
    "cond": "is_cond",
    "raw": None,  # 'raw' doesn't correspond to a known flag
}


class FieldName:
    def __init__(
        self,
        value: str,
        default_value: Union[float, None] = None,
        empty_indicator_value: Union[float, None] = None,
        normalization: Union[None, UnitNormalization, MinMaxNormalization] = None,
        value_transform: Union[None, ValueTransform, InversedValueTransform] = None,
    ):
        self._value_ = value
        self.default_value = default_value
        self.empty_indicator_value = empty_indicator_value

        self.normalization = normalization
        self.value_transform = value_transform

        # Default flags
        self.is_tonemappable = False
        self.is_image = False
        self.is_rays = False
        self.is_raysample = False
        self.is_integration_sample = False
        self.is_coarse = False
        self.is_fine = False
        self.is_cond = False
        self.is_meta_data = False
        self.is_camera = False

    def add_suffix(self, suffix: str):
        if suffix == "":
            return self
        parts = self._value_.split("_")
        base = parts[0]
        current_suffixes = parts[1:]
        new_suffixes = current_suffixes + [suffix]
        # Ensure canonical ordering
        new_suffixes = sorted(new_suffixes)
        new_value = base if not new_suffixes else f"{base}_{'_'.join(new_suffixes)}"

        new_obj = self.__class__(
            new_value,
            default_value=self.default_value,
            empty_indicator_value=self.empty_indicator_value,
            normalization=self.normalization,
            value_transform=self.value_transform,
        )

        # Copy current flags
        new_obj.is_tonemappable = self.is_tonemappable
        new_obj.is_image = self.is_image
        new_obj.is_rays = self.is_rays
        new_obj.is_raysample = self.is_raysample
        new_obj.is_integration_sample = self.is_integration_sample
        new_obj.is_coarse = self.is_coarse
        new_obj.is_fine = self.is_fine
        new_obj.is_cond = self.is_cond
        new_obj.is_meta_data = self.is_meta_data
        new_obj.is_camera = self.is_camera
        # If the suffix corresponds to a known flag, set it
        if suffix in SUFFIX_FLAGS and SUFFIX_FLAGS[suffix] is not None:
            setattr(new_obj, SUFFIX_FLAGS[suffix], True)

        # If it's rays or raysamples, it's not an image
        if new_obj.is_rays or new_obj.is_raysample:
            new_obj.is_image = False

        return new_obj

    def pop_suffix(self) -> Tuple["FieldName", str]:
        parts = self._value_.split("_")
        if len(parts) == 1:
            # No suffix to pop
            return self, ""

        suffix = parts[-1]
        base = parts[0]
        remaining_suffixes = parts[1:-1]
        new_value = (
            base if not remaining_suffixes else f"{base}_{'_'.join(remaining_suffixes)}"
        )

        new_obj = self.__class__(
            new_value,
            default_value=self.default_value,
            empty_indicator_value=self.empty_indicator_value,
            normalization=self.normalization,
            value_transform=self.value_transform,
        )

        # Copy flags
        new_obj.is_tonemappable = self.is_tonemappable
        new_obj.is_image = self.is_image
        new_obj.is_rays = self.is_rays
        new_obj.is_raysample = self.is_raysample
        new_obj.is_integration_sample = self.is_integration_sample
        new_obj.is_coarse = self.is_coarse
        new_obj.is_fine = self.is_fine
        new_obj.is_cond = self.is_cond
        new_obj.is_meta_data = self.is_meta_data
        new_obj.is_camera = self.is_camera

        # If the popped suffix had a known flag, revert it
        if suffix in SUFFIX_FLAGS and SUFFIX_FLAGS[suffix] is not None:
            setattr(new_obj, SUFFIX_FLAGS[suffix], False)

        # Now re-derive flags from scratch to ensure image-like logic is correct
        # We can use Names.from_flag_obj for this, as previously defined
        final_obj = Names.from_flag_obj(new_obj)
        return final_obj, suffix

    def remove_suffix(self, suffix: str):
        parts = self._value_.split("_")
        base = parts[0]
        remaining_suffixes = parts[1:]
        if suffix in remaining_suffixes:
            remaining_suffixes.remove(suffix)
        new_value = (
            base if not remaining_suffixes else f"{base}_{'_'.join(remaining_suffixes)}"
        )
        return Names.from_flag_obj(self.__class__(new_value))

    def normalize(
        self, data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        if self.normalization is not None:
            return self.normalization.normalize(data)
        return data

    def transform(
        self, data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        if self.value_transform is not None:
            return self.value_transform.transform(data)
        return data

    def fill_empty(
        self, data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        if self.empty_indicator_value is not None and self.default_value is not None:
            torch_to = None
            if isinstance(data, torch.Tensor):
                torch_to = {"dtype": data.dtype, "device": data.device}
                if data.dtype == torch.bfloat16:
                    data = data.float()
                data = data.detach().cpu().numpy()

            data[data == self.empty_indicator_value] = self.default_value

            if torch_to is not None:
                data = torch.from_numpy(data).to(**torch_to)
        return data

    def __repr__(self) -> str:
        flags = {
            "is_tonemappable": self.is_tonemappable,
            "is_image": self.is_image,
            "is_rays": self.is_rays,
            "is_raysample": self.is_raysample,
            "is_integration_sample": self.is_integration_sample,
            "is_coarse": self.is_coarse,
            "is_fine": self.is_fine,
            "is_cond": self.is_cond,
            "is_meta_data": self.is_meta_data,
            "is_camera": self.is_camera,
        }
        # Only show flags that are True to reduce clutter
        active_flags = [f for f, v in flags.items() if v]

        default_value_str = (
            repr(self.default_value) if self.default_value is not None else "None"
        )
        empty_indicator_value_str = (
            repr(self.empty_indicator_value)
            if self.empty_indicator_value is not None
            else "None"
        )

        normalization_str = (
            repr(self.normalization) if self.normalization is not None else "None"
        )
        value_transform_str = (
            repr(self.value_transform) if self.value_transform is not None else "None"
        )

        return (
            f"{self.__class__.__name__}(\n"
            f"  value={self._value_!r},\n"
            f"  default_value={default_value_str},\n"
            f"  empty_indicator_value={empty_indicator_value_str},\n"
            f"  flags={active_flags},\n"
            f"  normalization={normalization_str},\n"
            f"  value_transform={value_transform_str}\n"
            f")"
        )

    def __str__(self) -> str:
        # Return just the canonical value string
        return self._value_

    def __eq__(self, other: Any) -> bool:
        # Check if other is a string
        if isinstance(other, str):
            return self._value_ == other
        # Check if other is not a FieldName
        if not isinstance(other, FieldName):
            return False
        # Otherwise safe to compare
        return self._value_ == other._value_

    def __hash__(self) -> int:
        return hash(self._value_)

    def contains(self, suffix: str) -> bool:
        return suffix in self._value_.split("_")

    def matches(self, other: FieldName) -> bool:
        """Whether both fields represent the same fields."""
        return self is other or repr(self) == repr(other)

    # Helper property methods for adding known suffixes
    @property
    def coarse(self):
        return self.add_suffix("coarse")

    @property
    def fine(self):
        return self.add_suffix("fine")

    @property
    def rays(self):
        return self.add_suffix("rays")

    @property
    def ray_samples(self):
        return self.add_suffix("ray-samples")

    @property
    def ray_accumulated(self):
        if not self.contains("ray-samples"):
            raise ValueError(
                "Trying to ray-accumulate a FieldName that does not represent ray samples."
            )
        return self.remove_suffix("ray-samples")

    @property
    def integration_samples(self):
        return self.add_suffix("integration-samples")

    @property
    def raw(self):
        return self.add_suffix("raw")

    @property
    def cond(self):
        return self.add_suffix("cond")


class Names:
    # Base keys
    IMAGE = FieldName("image", default_value=0, empty_indicator_value=-1)
    OPACITY = FieldName("opacity", default_value=0, empty_indicator_value=-1)
    FOREGROUND = FieldName("foreground", default_value=0, empty_indicator_value=-1)
    BACKGROUND = FieldName("background", default_value=0, empty_indicator_value=-1)
    CONDITION = FieldName("condition", default_value=0, empty_indicator_value=-1)
    TOKEN = FieldName("token")

    # Diffusion keys
    TIMESTEPS = FieldName("timesteps")
    NOISE = FieldName("noise")
    LATENTS = FieldName("latents")

    #  Geometry keys
    SURFACE_NORMAL = FieldName(
        "surface-normal", default_value=0, empty_indicator_value=-2
    )
    SHADING_NORMAL = FieldName(
        "shading-normal", default_value=0, empty_indicator_value=-2
    )
    GEOMETRY_NORMAL = FieldName(
        "geometry-normal", default_value=0, empty_indicator_value=-2
    )
    VERTEX_OFFSET = FieldName("vertex-offset")
    REFLECTED_VIEW_DIRECTION = FieldName("reflected-view-direction")

    # Camera keys
    VIEW_INDEX = FieldName("view-index")
    CAMERA_EMBEDDING = FieldName("camera-embedding")
    CAMERA_TO_WORLD = FieldName("camera-to-world")
    WORLD_TO_CAMERA = FieldName("world-to-camera")
    PROJECTION_MATRIX = FieldName("projection-matrix")
    CAMERA_POSITION = FieldName("camera-position")
    INTRINSICS = FieldName("intrinsics")
    INTRINSICS_NORMED = FieldName("intrinsics-normed")

    #  Ray keys
    DIRECTION = FieldName(
        "direction"
    )  # World-coordinate ray from camera origin to pixel.
    ORIGIN = FieldName("origin")  # World-coordinate camera location.
    PLUCKER_RAYS = FieldName("plucker-rays")

    # NeRF Keys
    DISTANCE = FieldName("distance")
    LOCATION = FieldName("location")
    NORMALIZED_DISTANCE = FieldName("normalized-distance")
    NORMALIZED_LOCATION = FieldName("normalized-location")
    WEIGHTS = FieldName("weights")
    RAY_INDICES = FieldName("ray-indices")

    # Volumetric keys
    DENSITY = FieldName("density")
    SDF = FieldName("sdf")
    FLEXICUBES_WEIGHT = FieldName("flexicubes-weight")

    # Mixture of Experts keys
    EXPERT_LOGITS = FieldName("expert-logits")

    #  Light keys
    LIGHT_DIRECTION = FieldName("light-direction")
    VIEW_DIRECTION = FieldName("view-direction")

    #  Rendering keys
    DEPTH = FieldName("depth")
    Z_DEPTH = FieldName("z-depth")
    POSITION = FieldName("position")
    TANGENT = FieldName("tangent")
    BITANGENT = FieldName("bitangent")
    AMBIENT_OCCLUSION = FieldName("ambient-occlusion")
    RADIANCE = FieldName("radiance")
    IRRADIANCE = FieldName("irradiance")
    TONEMAPPED_IRRADIANCE = FieldName("tonemapped-irradiance")
    VISIBILITY = FieldName("visibility")
    VISIBLE_RAYS = FieldName("visible-rays")

    # Environment Keys
    ENV_MAP = FieldName("env-map")
    ENV_SAMPLING_DISTRIBUTION = FieldName("env-sampling-distribution")
    SH_COEFFICIENTS = FieldName("sh-coefficients")
    SG_AMPLITUDES = FieldName("sg-amplitudes")
    RENI_LATENT = FieldName("reni-latent", empty_indicator_value=-10)
    ILLUMINATION_ROTATION = FieldName(
        "illumination-rotation", empty_indicator_value=-10
    )
    ILLUMINATION_Z_ROTATION_RADS = FieldName(
        "illumination-z-rotation-rads", empty_indicator_value=0
    )
    ILLUMINATION_IDX = FieldName("illumination-idx", empty_indicator_value=-10)
    ILLUMINATION_STRENGTH = FieldName(
        "illumination-strength", empty_indicator_value=-10
    )

    # PBR Keys
    BASECOLOR = FieldName("basecolor", default_value=0, empty_indicator_value=-1)
    DIFFUSE = FieldName("diffuse", default_value=0, empty_indicator_value=-1)
    SPECULAR = FieldName("specular", default_value=0, empty_indicator_value=-1)
    METALLIC = FieldName("metallic", default_value=0, empty_indicator_value=-1)
    ROUGHNESS = FieldName("roughness", default_value=0, empty_indicator_value=-1)

    # Light keys
    LIGHT_SOURCE_COLOR = FieldName("light-color")
    LIGHT_SOURCE_INTENSITY = FieldName("light-intensity")
    LIGHT_SOURCE_POWER = FieldName("light-power")
    LIGHT_SOURCE_SPOT_OPENING_ANGLE_OUTER_OFFSET = FieldName(
        "light-spot-opening-angle-outer-offset"
    )
    LIGHT_SOURCE_SPOT_OPENING_ANGLE_INNER = FieldName("light-spot-opening-angle-inner")
    LIGHT_SOURCE_DIRECTION = FieldName("light-direction")
    LIGHT_SOURCE_POSITION = FieldName("light-position")

    # Representation keys
    MESH = FieldName("mesh")
    TRIPLANE = FieldName("triplane")
    VOXEL_GRID = FieldName("voxel-grid")
    GAUSSIAN_SPLAT = FieldName("gaussian-splat")
    OCCUPANCY_GRID = FieldName("occupancy-grid")
    POINT_CLOUD = FieldName("point-cloud")

    # ASC CDL keys
    CDL_SLOPE = FieldName("cdl-slope")
    CDL_OFFSET = FieldName("cdl-offset")
    CDL_POWER = FieldName("cdl-power")
    CDL_SATURATION = FieldName("cdl-saturation")

    # Meta Data keys
    BATCH_SIZE = FieldName("batch-size")
    VIEW_SIZE = FieldName("view-size")
    RAY_SIZE = FieldName("ray-size")
    GLOBAL_STEP = FieldName("global-step")
    HEIGHT = FieldName("height")
    WIDTH = FieldName("width")
    INVALID_BATCH_SENTINEL = FieldName("batch-invalidation-sentinel")

    # Object keys
    OBJECT_UID = FieldName("object-uid")

    # Dataset keys
    DATASET_NAME = FieldName("dataset-name")
    DATASET_TYPE = FieldName("dataset-type")

    TONEMAPPABLE = {"env-map", "radiance", "irradiance"}
    IMAGE_LIKE = {
        "image",
        "opacity",
        "foreground",
        "background",
        "basecolor",
        "specular",
        "metallic",
        "roughness",
        "radiance",
        "irradiance",
        "tonemapped-irradiance",
        "env-map",
        "surface-normal",
        "shading-normal",
        "geometry-normal",
        "ambient-occlusion",
        "tangent",
        "bitangent",
    }
    META_DATA_LIKE = {
        "batch-size",
        "view-size",
        "ray-size",
        "ray-indices",
        "global-step",
        "height",
        "width",
        "object-uid",
        "dataset-name",
        "dataset-type",
    }
    CAMERA_LIKE = {
        "intrinsics",
        "intrinsics-normed",
        "camera-embedding",
        "camera-to-world",
        "world-to-camera",
        "projection-matrix",
        "camera-position",
    }

    BASE_TEMPLATES: Dict[
        str, FieldName
    ] = {}  # Will be populated after class definition

    def __new__(cls, value: str) -> FieldName:
        if isinstance(value, Names) or isinstance(value, FieldName):
            value = str(value)
        parts = value.split("_")
        base = parts[0]
        suffixes = parts[1:]

        if base not in cls.BASE_TEMPLATES:
            raise ValueError(f"Unknown base name '{base}'.")

        suffixes = sorted(suffixes)
        canonical_value = base if not suffixes else f"{base}_{'_'.join(suffixes)}"
        template = cls.BASE_TEMPLATES[base]

        obj = FieldName(
            canonical_value,
            empty_indicator_value=template.empty_indicator_value,
            default_value=template.default_value,
            normalization=template.normalization,
            value_transform=template.value_transform,
        )

        # Set base flags
        if base in cls.TONEMAPPABLE:
            obj.is_tonemappable = True
        if base in cls.IMAGE_LIKE:
            obj.is_image = True
        if base in cls.META_DATA_LIKE:
            obj.is_meta_data = True
        if base in cls.CAMERA_LIKE:
            obj.is_camera = True

        for sfx in suffixes:
            if sfx in SUFFIX_FLAGS and SUFFIX_FLAGS[sfx] is not None:
                setattr(obj, SUFFIX_FLAGS[sfx], True)

        if obj.is_rays or obj.is_raysample:
            obj.is_image = False

        return obj

    @classmethod
    def from_flag_obj(cls, flag_obj: FieldName) -> FieldName:
        """Rebuild a FieldName object to ensure consistent flags after suffix operations."""
        # Reparse to apply IMAGE_LIKE logic again if needed
        parts = flag_obj._value_.split("_")
        base = parts[0]
        suffixes = parts[1:]

        # Sort suffixes to ensure canonical form (should already be canonical)
        suffixes = sorted(suffixes)
        canonical_value = base if not suffixes else f"{base}_{'_'.join(suffixes)}"

        # Start from a template again
        template = cls.BASE_TEMPLATES[base]
        new_obj = FieldName(
            canonical_value,
            empty_indicator_value=template.empty_indicator_value,
            default_value=template.default_value,
            normalization=template.normalization,
            value_transform=template.value_transform,
        )

        # Base category flags
        if base in cls.TONEMAPPABLE:
            new_obj.is_tonemappable = True
        if base in cls.IMAGE_LIKE:
            new_obj.is_image = True
        if base in cls.META_DATA_LIKE:
            new_obj.is_meta_data = True
        if base in cls.CAMERA_LIKE:
            new_obj.is_camera = True

        # Reapply suffix flags
        for sfx in suffixes:
            if sfx in SUFFIX_FLAGS and SUFFIX_FLAGS[sfx] is not None:
                setattr(new_obj, SUFFIX_FLAGS[sfx], True)

        # If rays or raysample present, no image
        if new_obj.is_rays or new_obj.is_raysample:
            new_obj.is_image = False

        return new_obj


# After the class is defined, populate BASE_TEMPLATES by inspecting Names' attributes
for name, value in vars(Names).items():
    # Check if the attribute is a FieldName instance
    if isinstance(value, FieldName):
        # The base name is the first part of the value
        base = value._value_.split("_")[0]
        Names.BASE_TEMPLATES[base] = value

# After the class definition and BASE_TEMPLATES population
# Initialize flags for all predefined FieldName instances
for name, value in vars(Names).items():
    if isinstance(value, FieldName):
        base = value._value_
        if base in Names.TONEMAPPABLE:
            value.is_tonemappable = True
        if base in Names.IMAGE_LIKE:
            value.is_image = True
        if base in Names.META_DATA_LIKE:
            value.is_meta_data = True
        if base in Names.CAMERA_LIKE:
            value.is_camera = True

# Set transformations
Names.SURFACE_NORMAL.normalization = UnitNormalization()
Names.SURFACE_NORMAL.value_transform = ValueTransform(-1, 1)
Names.SHADING_NORMAL.normalization = UnitNormalization()
Names.SHADING_NORMAL.value_transform = ValueTransform(-1, 1)
Names.GEOMETRY_NORMAL.normalization = UnitNormalization()
Names.GEOMETRY_NORMAL.value_transform = ValueTransform(-1, 1)
Names.TANGENT.normalization = UnitNormalization()
Names.TANGENT.value_transform = ValueTransform(-1, 1)
Names.BITANGENT.normalization = UnitNormalization()
Names.BITANGENT.value_transform = ValueTransform(-1, 1)
Names.DEPTH.value_transform = InversedValueTransform()
Names.POSITION.normalization = MinMaxNormalization()

# LRM types
LossTermsType = Dict[str, Tensor]
OutputsType = Dict[Names, Any]
