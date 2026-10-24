"""Publish the web app to a Hugging Face Space (Docker SDK).

The CI/CD workflow runs this on every push to main. It can also be run by hand:

    python deploy/deploy_space.py your-username/leaflens --dry-run   # list what would be uploaded
    HF_TOKEN=hf_... python deploy/deploy_space.py your-username/leaflens

The Space is created on the first run. Hugging Face then builds the Dockerfile and serves the
app at https://huggingface.co/spaces/<your-username>/leaflens.
"""
import argparse
import os
import re
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


def space_id(value):
    """'<user>/<space>' from that form or a pasted Space URL, e.g. https://huggingface.co/spaces/<user>/<space>."""
    value = (value or "").strip().rstrip("/")
    value = re.sub(r"^(https?://)?(www\.)?huggingface\.co/", "", value)
    value = re.sub(r"^spaces/", "", value)
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", value):
        fail(f"HF_SPACE should look like <hf-username>/<space-name>, got {value!r}.")
    return value


def fail(message):
    """Stop with `message`. On GitHub Actions it also shows on the run's summary page."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::error title=Deploy to Hugging Face failed::{message}", flush=True)
    sys.exit(message)


def check_token(api, space):
    """Fail early, in plain words, if the token can't publish to `space`."""
    try:
        me = api.whoami()
    except Exception:
        fail("Hugging Face rejected the HF_TOKEN secret: it is mistyped, expired or revoked. "
             "Create a new access token with the Write role and replace the secret.")
    owner = space.split("/")[0].lower()
    accounts = {me.get("name", "")} | {org.get("name", "") for org in me.get("orgs") or []}
    if owner not in {a.lower() for a in accounts}:
        fail(f"The HF_TOKEN token belongs to {me.get('name')!r}, but the Space would be created under {owner!r}. "
             "Use a token from that account, or change HF_SPACE.")
    role = ((me.get("auth") or {}).get("accessToken") or {}).get("role")
    if role == "read":
        fail("The HF_TOKEN token is read-only. Create an access token with the Write role and replace the secret.")


def stage(folder):
    """Copy what the Space needs into `folder`, with the Space's own README (its config lives there)."""
    for name in FILES:
        source, target = ROOT / name, folder / name
        if not source.exists():
            fail(f"Missing {name}; is the trained model committed?")
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
    space = space_id(args.space)
    url = f"https://huggingface.co/spaces/{space}"
    if os.environ.get("GITHUB_OUTPUT"):  # lets the workflow link the deployment to the Space
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"url={url}\n")

    with tempfile.TemporaryDirectory() as tmp:
        files = stage(Path(tmp))
        size = sum((Path(tmp) / f).stat().st_size for f in files) / 1e6
        print(f"{len(files)} files ({size:.1f} MB) for {url}")
        if args.dry_run:
            print("\n".join(f"  {f}" for f in files))
            return

        token = (os.environ.get("HF_TOKEN") or "").strip()
        if not token:
            fail("The HF_TOKEN secret is missing or empty. In the GitHub repository open Settings > Secrets and "
                 "variables > Actions, choose the Secrets tab (not Variables) and add a repository secret named HF_TOKEN.")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        check_token(api, space)
        try:
            api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
            api.upload_folder(folder_path=tmp, repo_id=space, repo_type="space",
                              commit_message=args.message, delete_patterns=MANAGED)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            fail(f"Hugging Face refused the upload ({status or type(exc).__name__}): {str(exc).splitlines()[0]}")
        print(f"Uploaded. Hugging Face is now building the app: {url}")


if __name__ == "__main__":
    main()
