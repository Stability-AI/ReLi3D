#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ReLi3D model assets from Hugging Face")
    parser.add_argument("--repo-id", type=str, default="StabilityLabs/ReLi3D")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--config-filename", type=str, default="config.yaml")
    parser.add_argument("--checkpoint-filename", type=str, default="reli3d_final.ckpt")
    parser.add_argument(
        "--skip-config",
        action="store_true",
        help="Skip downloading the config file.",
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Skip downloading the checkpoint file.",
    )
    return parser.parse_args()


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


def download_file(repo_id: str, revision: str, filename: str, output_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency `huggingface_hub`. Install with `pip install huggingface_hub`."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    def _download(token: Optional[Union[str, bool]]) -> Path:
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            filename=filename,
            local_dir=str(output_dir),
            token=token,
        )
        return Path(path).resolve()

    try:
        return _download(token=None)
    except Exception as exc:
        if not _is_auth_error(exc):
            raise

        git_token = _token_from_git_credential()
        if git_token:
            print(f"Auth retry for {filename}: using git credential token...")
            try:
                return _download(token=git_token)
            except Exception as exc_git:
                if not _is_auth_error(exc_git):
                    raise

        print(f"Auth retry for {filename}: trying unauthenticated access...")
        return _download(token=False)


def main() -> None:
    args = parse_args()

    if args.skip_config and args.skip_checkpoint:
        raise ValueError("Both --skip-config and --skip-checkpoint were set. Nothing to download.")

    if not args.skip_config:
        config_path = download_file(
            repo_id=args.repo_id,
            revision=args.revision,
            filename=args.config_filename,
            output_dir=args.output_dir,
        )
        print(f"Downloaded config: {config_path}")

    if not args.skip_checkpoint:
        checkpoint_path = download_file(
            repo_id=args.repo_id,
            revision=args.revision,
            filename=args.checkpoint_filename,
            output_dir=args.output_dir,
        )
        print(f"Downloaded checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
