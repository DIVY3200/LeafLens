"""Publish the web app to a Hugging Face Space (Docker SDK).

The CI/CD workflow runs this on every push to main. It can also be run by hand:

    python deploy/deploy_space.py your-username/leaflens --dry-run   # list what would be uploaded
    HF_TOKEN=hf_... python deploy/deploy_space.py your-username/leaflens

The Space is created on the first run. Hugging Face then builds the Dockerfile and serves the
app at https://huggingface.co/spaces/<your-username>/leaflens.
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    "Dockerfile",
    "requirements-deploy.txt",
    "app.py",
    "plant_disease",
    "templates",
    "static",
    "models/leaflens_mobilenetv2.keras",
    "models/leaflens_mobilenetv2.json",
]
# Paths this script owns in the Space; anything there that is no longer in FILES gets removed.
MANAGED = ["Dockerfile", "requirements-deploy.txt", "app.py", "plant_disease/**", "templates/**", "static/**", "models/**"]


def stage(folder):
    """Copy what the Space needs into `folder`, with the Space's own README (its config lives there)."""
    for name in FILES:
        source, target = ROOT / name, folder / name
        if not source.exists():
            sys.exit(f"Missing {name}; is the trained model committed?")
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    shutil.copy2(ROOT / "deploy" / "space_README.md", folder / "README.md")
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("space", nargs="?", default=os.environ.get("HF_SPACE"), help="<username>/<space-name>")
    parser.add_argument("--message", default="Deploy from GitHub")
    parser.add_argument("--dry-run", action="store_true", help="only list the files that would be uploaded")
    args = parser.parse_args()
    if not args.space or "/" not in args.space:
        sys.exit("Give the Space as <username>/<space-name>, or set HF_SPACE.")

    with tempfile.TemporaryDirectory() as tmp:
        files = stage(Path(tmp))
        size = sum((Path(tmp) / f).stat().st_size for f in files) / 1e6
        print(f"{len(files)} files ({size:.1f} MB) for https://huggingface.co/spaces/{args.space}")
        if args.dry_run:
            print("\n".join(f"  {f}" for f in files))
            return

        token = os.environ.get("HF_TOKEN")
        if not token:
            sys.exit("Set HF_TOKEN to a Hugging Face access token with write permission.")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(folder_path=tmp, repo_id=args.space, repo_type="space",
                          commit_message=args.message, delete_patterns=MANAGED)
        print(f"Uploaded. Hugging Face is now building the app: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
