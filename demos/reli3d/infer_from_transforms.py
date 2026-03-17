#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.constants import Names
from src.data.reli3d_mapper import ReLi3DMapper
from src.systems.feed_forward_system import FeedForwardSystem
from src.utils.config import instantiate_config
from src.utils.misc import load_module_weights

DEFAULT_HF_REPO_ID = "StabilityLabs/ReLi3D"
DEFAULT_HF_REVISION = "main"
DEFAULT_HF_CONFIG_FILENAME = "config.yaml"
DEFAULT_HF_CHECKPOINT_FILENAME = "reli3d_final.ckpt"


def _default_config_path() -> Path:
    config_yaml = Path("artifacts/model/config.yaml")
    raw_yaml = Path("artifacts/model/raw.yaml")
    if config_yaml.exists():
        return config_yaml
    if raw_yaml.exists():
        return raw_yaml
    return config_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ReLi3D inference from object folders with transforms.json"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("demo_files/objects"),
        help="Root folder containing object directories with transforms.json.",
    )
    parser.add_argument(
        "--objects",
        nargs="*",
        default=None,
        help="Optional object directory names. Defaults to all objects in input-root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Where to write inference outputs.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Model config path. Defaults to artifacts/model/config.yaml (or artifacts/model/raw.yaml if present).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Model checkpoint path. If omitted, uses RELI3D_CHECKPOINT or "
            "artifacts/model/reli3d_final.ckpt."
        ),
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=-1,
        help="If >0, use only the first N views from transforms.json.",
    )
    parser.add_argument(
        "--remesh",
        choices=["none", "triangle", "quad"],
        default="none",
        help="Remeshing mode passed to model.get_mesh.",
    )
    parser.add_argument(
        "--vertex-count",
        type=int,
        default=-1,
        help="Target vertex count for remeshing. Ignored for --remesh none.",
    )
    parser.add_argument(
        "--texture-size",
        type=int,
        default=1024,
        help="Texture resolution for mesh export.",
    )
    parser.add_argument(
        "--isosurface-threshold",
        type=float,
        default=None,
        help="Optional override for object_representation isosurface threshold.",
    )
    parser.add_argument(
        "--force-rasterizer",
        choices=["drtk", "nvdiffrast"],
        default=None,
        help="Optional override for renderer.rasterizer.",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=DEFAULT_HF_REPO_ID,
        help="Hugging Face model repo id used for auto-download.",
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        default=DEFAULT_HF_REVISION,
        help="Hugging Face repo revision used for auto-download.",
    )
    parser.add_argument(
        "--hf-config-filename",
        type=str,
        default=DEFAULT_HF_CONFIG_FILENAME,
        help="Config filename in the Hugging Face repo.",
    )
    parser.add_argument(
        "--hf-checkpoint-filename",
        type=str,
        default=DEFAULT_HF_CHECKPOINT_FILENAME,
        help="Checkpoint filename in the Hugging Face repo.",
    )
    parser.add_argument(
        "--disable-hf-download",
        action="store_true",
        help="Disable automatic download of missing config/checkpoint from Hugging Face.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing object outputs.",
    )
    return parser.parse_args()


def _load_system_cfg(config_path: Path) -> FeedForwardSystem.Config:
    cfg = OmegaConf.load(config_path)
    if "system" in cfg:
        system_cfg = cfg.system
    elif "main_module" in cfg and "system" in cfg.main_module:
        system_cfg = cfg.main_module.system
    else:
        raise ValueError(
            f"Config at {config_path} must contain either `system` or `main_module.system`."
        )

    return instantiate_config(
        FeedForwardSystem.Config,
        OmegaConf.to_container(system_cfg, resolve=True),
    )


def _resolve_checkpoint(cli_checkpoint: Path | None) -> Tuple[Path, bool]:
    if cli_checkpoint is not None:
        return cli_checkpoint.expanduser().resolve(), False

    env_ckpt = os.environ.get("RELI3D_CHECKPOINT")
    if env_ckpt:
        return Path(env_ckpt).expanduser().resolve(), False

    default_ckpt = Path("artifacts/model/reli3d_final.ckpt")
    return default_ckpt.resolve(), True


def _ensure_fov_xy(fov_value: Any) -> List[float]:
    if isinstance(fov_value, (int, float)):
        val = float(fov_value)
        return [val, val]
    if isinstance(fov_value, list):
        if len(fov_value) == 1:
            val = float(fov_value[0])
            return [val, val]
        if len(fov_value) >= 2:
            return [float(fov_value[0]), float(fov_value[1])]
    raise ValueError(f"Unsupported camera_fov format: {fov_value}")

def _download_hf_file(
    repo_id: str,
    revision: str,
    repo_filename: str,
    output_path: Path,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency `huggingface_hub`. Install with `pip install huggingface_hub`."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _download(token: str | bool | None) -> Path:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                filename=repo_filename,
                local_dir=str(output_path.parent),
                token=token,
            )
        )

    def _is_auth_error(exc: Exception) -> bool:
        message = str(exc).lower()
        auth_markers = [
            "401",
            "invalid username or password",
            "repository not found",
            "unauthorized",
        ]
        return any(marker in message for marker in auth_markers)

    def _token_from_git_credential() -> str:
        try:
            proc = subprocess.run(
                ["git", "credential", "fill"],
                input="protocol=https\nhost=huggingface.co\n\n",
                text=True,
                capture_output=True,
                check=True,
            )
        except Exception:
            return ""

        for line in proc.stdout.splitlines():
            if line.startswith("password="):
                return line.split("=", 1)[1].strip()
        return ""

    try:
        downloaded = _download(token=None)
    except Exception as exc:
        if not _is_auth_error(exc):
            raise

        git_token = _token_from_git_credential()
        if git_token:
            print("HF auth failed with current token context. Retrying with git credential token...")
            try:
                downloaded = _download(token=git_token)
            except Exception as exc_git:
                if not _is_auth_error(exc_git):
                    raise
                print("Git credential auth failed. Retrying without auth token...")
                downloaded = _download(token=False)
        else:
            print("HF auth failed and no git credential token found. Retrying without auth token...")
            downloaded = _download(token=False)

    output_path = output_path.resolve()
    downloaded = downloaded.resolve()

    if downloaded != output_path:
        shutil.copy2(downloaded, output_path)

    return output_path



def _ensure_model_assets(
    args: argparse.Namespace,
    config_path: Path,
    checkpoint_path: Path,
    default_config_requested: bool,
    default_checkpoint_requested: bool,
) -> Tuple[Path, Path]:
    model_dir = (REPO_ROOT / "artifacts/model").resolve()

    if config_path.exists() and checkpoint_path.exists():
        return config_path, checkpoint_path

    if args.disable_hf_download:
        missing = []
        if not config_path.exists():
            missing.append(str(config_path))
        if not checkpoint_path.exists():
            missing.append(str(checkpoint_path))
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required model assets: {missing_str}."
        )

    if not config_path.exists():
        if not default_config_requested:
            raise FileNotFoundError(
                f"Config not found: {config_path}. Pass an existing --config or use default artifacts path for HF auto-download."
            )
        target_config = model_dir / args.hf_config_filename
        print(
            f"Config missing locally. Downloading `{args.hf_config_filename}` from {args.hf_repo_id}@{args.hf_revision}..."
        )
        config_path = _download_hf_file(
            repo_id=args.hf_repo_id,
            revision=args.hf_revision,
            repo_filename=args.hf_config_filename,
            output_path=target_config,
        )

    if not checkpoint_path.exists():
        if not default_checkpoint_requested:
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. Pass an existing --checkpoint or use default artifacts path for HF auto-download."
            )
        target_ckpt = model_dir / args.hf_checkpoint_filename
        print(
            f"Checkpoint missing locally. Downloading `{args.hf_checkpoint_filename}` from {args.hf_repo_id}@{args.hf_revision}..."
        )
        checkpoint_path = _download_hf_file(
            repo_id=args.hf_repo_id,
            revision=args.hf_revision,
            repo_filename=args.hf_checkpoint_filename,
            output_path=target_ckpt,
        )

    return config_path.resolve(), checkpoint_path.resolve()


def _build_sft_dict(
    object_uid: str,
    images: List[Image.Image],
    c2ws: List[np.ndarray],
    fovs: List[List[float]],
    principal_points: List[List[float]],
) -> Dict[str, torch.Tensor]:
    sft_dict: Dict[str, torch.Tensor] = {
        "object_uid": torch.tensor(np.frombuffer(object_uid.encode("utf-8"), dtype=np.uint8))
    }

    for i, (image, c2w, fov, principal_point) in enumerate(
        zip(images, c2ws, fovs, principal_points)
    ):
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        rgb = rgba[..., :3]
        alpha = rgba[..., 3]

        ok, rgb_buf = cv2.imencode(".jpg", rgb[..., ::-1])
        if not ok:
            raise RuntimeError("JPEG encoding failed for RGB image")

        mask_bgr = cv2.merge([alpha, alpha, alpha])
        ok, mask_buf = cv2.imencode(".jpg", mask_bgr)
        if not ok:
            raise RuntimeError("JPEG encoding failed for mask image")

        sft_dict[f"rgb_{i:04d}"] = torch.from_numpy(rgb_buf.copy())
        sft_dict[f"metallicroughmask_{i:04d}"] = torch.from_numpy(mask_buf.copy())
        sft_dict[f"c2w_{i:04d}"] = torch.tensor(c2w, dtype=torch.float32)
        sft_dict[f"fov_rad_{i:04d}"] = torch.tensor(fov, dtype=torch.float32)
        sft_dict[f"principal_point_{i:04d}"] = torch.tensor(principal_point, dtype=torch.float32)

    return sft_dict


def _create_batch_from_sft(
    sft_dict: Dict[str, torch.Tensor],
    device: torch.device,
    num_views: int,
) -> Dict[Any, Any]:
    mapper_cfg = {
        "num_views_input": num_views,
        "num_views_output": num_views,
        "train_input_views": list(range(num_views)),
        "train_sup_views": "random",
        "cond_height": 512,
        "cond_width": 512,
        "eval_height": 512,
        "eval_width": 512,
        "binarize_mask": True,
        "dataset_is_repaired": True,
        "add_pose_noise": False,
        "pose_noise_std_trans": 0.00,
        "pose_noise_std_rot": 0.00,
    }
    mapper = ReLi3DMapper(cfg=mapper_cfg, sft_key="safetensors", split="test")
    mapper_batch = mapper(
        {"safetensors": sft_dict, "dataset_name": "custom", "dataset_type": "pbr"}
    )

    batch_elem: Dict[Any, Any] = {
        Names.IMAGE.cond: mapper_batch[Names.IMAGE.cond],
        Names.IMAGE.add_suffix("mask").cond: mapper_batch[Names.IMAGE.add_suffix("mask").cond],
        Names.OPACITY.cond: mapper_batch[Names.OPACITY.cond],
        Names.CAMERA_TO_WORLD.cond: mapper_batch[Names.CAMERA_TO_WORLD.cond],
        Names.CAMERA_POSITION.cond: mapper_batch[Names.CAMERA_POSITION.cond],
        Names.INTRINSICS.cond: mapper_batch[Names.INTRINSICS.cond],
        Names.INTRINSICS_NORMED.cond: mapper_batch[Names.INTRINSICS_NORMED.cond],
        Names.VIEW_SIZE: mapper_batch[Names.VIEW_SIZE],
        Names.VIEW_SIZE.cond: mapper_batch[Names.VIEW_SIZE.cond],
    }

    bg_key = Names.IMAGE.add_suffix("bg").cond
    if bg_key in mapper_batch:
        batch_elem[bg_key] = mapper_batch[bg_key]

    batch = {k: v.unsqueeze(0).to(device) for k, v in batch_elem.items()}
    batch[Names.BATCH_SIZE] = 1
    return batch


def _load_object_data(object_dir: Path, num_views: int) -> Dict[str, Any]:
    transforms_path = object_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing transforms.json in {object_dir}")

    with transforms_path.open("r", encoding="utf-8") as f:
        transforms = json.load(f)

    frames = sorted(transforms["frames"], key=lambda x: x.get("view_index", 0))
    if num_views > 0:
        frames = frames[:num_views]
    if not frames:
        raise ValueError(f"No frames available in {transforms_path}")

    images: List[Image.Image] = []
    c2ws: List[np.ndarray] = []
    fovs: List[List[float]] = []
    principal_points: List[List[float]] = []

    for frame in frames:
        image_path = object_dir / frame["file_path"]
        image = Image.open(image_path).convert("RGBA")

        c2w_key = "transform_matrix" if "transform_matrix" in frame else "camera_transform"
        c2w = np.asarray(frame[c2w_key], dtype=np.float32)

        fov = _ensure_fov_xy(frame.get("camera_fov", frame.get("fov_rad", 0.7)))

        principal = frame.get("camera_principal_point")
        if principal is None:
            w, h = image.size
            principal = [0.5 * w, 0.5 * h]

        images.append(image)
        c2ws.append(c2w)
        fovs.append([float(fov[0]), float(fov[1])])
        principal_points.append([float(principal[0]), float(principal[1])])

    return {
        "object_uid": transforms.get("object_uid", object_dir.name),
        "images": images,
        "c2ws": c2ws,
        "fovs": fovs,
        "principal_points": principal_points,
    }


def _list_object_dirs(input_root: Path, objects: List[str] | None) -> List[Path]:
    if objects:
        object_dirs = [input_root / name for name in objects]
    else:
        object_dirs = [
            d
            for d in sorted(input_root.iterdir())
            if d.is_dir() and (d / "transforms.json").exists()
        ]

    missing = [d for d in object_dirs if not d.exists()]
    if missing:
        missing_str = ", ".join(str(m) for m in missing)
        raise FileNotFoundError(f"Missing object directories: {missing_str}")

    return object_dirs


def _public_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return resolved.name


def _configure_rasterizer(cfg: FeedForwardSystem.Config, forced: str | None) -> None:
    if not isinstance(cfg.renderer, dict):
        return

    if forced is not None:
        cfg.renderer["rasterizer"] = forced
        if forced == "nvdiffrast":
            cfg.renderer.setdefault("context_type", "cuda")
        return

    rasterizer = cfg.renderer.get("rasterizer")
    if rasterizer == "drtk" and importlib.util.find_spec("drtk") is None:
        print("drtk is unavailable, falling back to nvdiffrast.")
        cfg.renderer["rasterizer"] = "nvdiffrast"
        cfg.renderer.setdefault("context_type", "cuda")


def main() -> None:
    args = parse_args()

    artifact_config = (REPO_ROOT / "artifacts/model/config.yaml").resolve()
    artifact_raw = (REPO_ROOT / "artifacts/model/raw.yaml").resolve()

    config_path = args.config.expanduser().resolve()
    checkpoint_path, default_checkpoint_requested = _resolve_checkpoint(args.checkpoint)

    default_config_requested = config_path in {artifact_config, artifact_raw}

    config_path, checkpoint_path = _ensure_model_assets(
        args=args,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        default_config_requested=default_config_requested,
        default_checkpoint_requested=default_checkpoint_requested,
    )

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}."
        )

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading config: {config_path}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Device: {device}")

    cfg = _load_system_cfg(config_path)
    _configure_rasterizer(cfg, args.force_rasterizer)

    model = FeedForwardSystem(cfg)
    state_dict = load_module_weights(str(checkpoint_path), module_name="system", map_location="cpu")[0]
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    object_dirs = _list_object_dirs(input_root, args.objects)
    print(f"Found {len(object_dirs)} object(s)")

    for object_dir in object_dirs:
        out_dir = output_root / object_dir.name
        if out_dir.exists() and not args.overwrite:
            print(f"Skipping {object_dir.name}: output exists at {out_dir} (use --overwrite)")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running object: {object_dir.name}")
        object_data = _load_object_data(object_dir, args.num_views)

        sft_dict = _build_sft_dict(
            object_uid=object_data["object_uid"],
            images=object_data["images"],
            c2ws=object_data["c2ws"],
            fovs=object_data["fovs"],
            principal_points=object_data["principal_points"],
        )

        batch = _create_batch_from_sft(
            sft_dict=sft_dict,
            device=device,
            num_views=len(object_data["images"]),
        )

        original_threshold = None
        if args.isosurface_threshold is not None:
            original_threshold = model.object_representation.cfg.isosurface_threshold
            model.object_representation.cfg.isosurface_threshold = args.isosurface_threshold

        try:
            start = time.time()
            with torch.no_grad(), (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            ):
                mesh_list, global_dict = model.get_mesh(
                    batch,
                    texture_resolution=args.texture_size,
                    remesh=args.remesh,
                    vertex_count=args.vertex_count if args.vertex_count > 0 else None,
                )
            runtime = time.time() - start
        finally:
            if original_threshold is not None:
                model.object_representation.cfg.isosurface_threshold = original_threshold

        mesh = mesh_list[-1]
        mesh_path = out_dir / "mesh.glb"
        mesh.export(mesh_path, file_type="glb", include_normals=True)

        env_map_path = out_dir / "illumination.hdr"
        if Names.ENV_MAP in global_dict:
            env_map = global_dict[Names.ENV_MAP][0].detach().cpu().numpy()
            cv2.imwrite(str(env_map_path), env_map[..., ::-1])

        with (out_dir / "run_info.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "object": object_dir.name,
                    "object_uid": object_data["object_uid"],
                    "num_views": len(object_data["images"]),
                    "runtime_sec": runtime,
                    "config": _public_path(config_path),
                    "checkpoint": _public_path(checkpoint_path),
                    "mesh": "mesh.glb",
                    "illumination": "illumination.hdr" if env_map_path.exists() else None,
                },
                f,
                indent=2,
            )

        print(f"Finished {object_dir.name}: {mesh_path}")


if __name__ == "__main__":
    main()
