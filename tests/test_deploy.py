import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("deploy_space", ROOT / "deploy" / "deploy_space.py")
deploy_space = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy_space)


@pytest.mark.parametrize("value", [
    "divy3200/leaflens",
    " divy3200/leaflens/ ",
    "https://huggingface.co/divy3200/leaflens",
    "https://huggingface.co/spaces/divy3200/leaflens",
    "huggingface.co/spaces/divy3200/leaflens",
])
def test_space_id_accepts_names_and_pasted_urls(value):
    assert deploy_space.space_id(value) == "divy3200/leaflens"


@pytest.mark.parametrize("value", ["", "leaflens", "divy3200/leaf lens", "https://github.com/DIVY3200/LeafLens"])
def test_space_id_rejects_anything_else(value):
    with pytest.raises(SystemExit):
        deploy_space.space_id(value)


def test_the_space_gets_the_app_and_its_own_readme(tmp_path):
    if not (ROOT / "models" / "leaflens_mobilenetv2.keras").exists():
        pytest.skip("the trained model is not in this checkout")
    files = deploy_space.stage(tmp_path)
    assert {"Dockerfile", "README.md", "app.py", "requirements-deploy.txt", "models/leaflens_mobilenetv2.keras"} <= set(files)
    assert not any("__pycache__" in f or f.startswith(("tests/", "Dataset/")) for f in files)
    assert "sdk: docker" in (tmp_path / "README.md").read_text(encoding="utf-8")
