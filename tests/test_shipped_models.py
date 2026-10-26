"""The model files in the repository load, match their cards, and have a guide for every class."""
from pathlib import Path

import pytest

from app import default_model
from plant_disease.classifier import PlantDiseaseClassifier
from plant_disease.knowledge import KNOWLEDGE

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "models/leaflens_mobilenetv2.tflite": dict(image_size=(224, 224), classes=47),
    "models/leaflens_mobilenetv2.keras": dict(image_size=(224, 224), classes=47),
    "trained_plant_disease_model.keras": dict(image_size=(128, 128), classes=38),
}


def require(*names):
    missing = [n for n in names if not (ROOT / n).exists()]
    if missing:
        pytest.skip(f"not in this checkout: {', '.join(missing)}")


@pytest.mark.parametrize("name", MODELS)
def test_shipped_model_loads_and_matches_its_card(name, photo):
    require(name)
    clf = PlantDiseaseClassifier(ROOT / name)
    assert clf.image_size == MODELS[name]["image_size"]
    assert len(clf.classes) == MODELS[name]["classes"]
    assert all(label in KNOWLEDGE for label in clf.classes)
    assert {"plantvillage", "fieldplant"} <= set(clf.card["metrics"])  # the numbers the app shows
    assert 0 <= clf.predict(photo())["prediction"]["confidence"] <= 1


def test_the_deployed_tflite_model_matches_the_trained_keras_model(photo):
    require("models/leaflens_mobilenetv2.keras", "models/leaflens_mobilenetv2.tflite")
    trained = PlantDiseaseClassifier(ROOT / "models/leaflens_mobilenetv2.keras")
    deployed = PlantDiseaseClassifier(ROOT / "models/leaflens_mobilenetv2.tflite")
    for seed in range(4):
        image = photo(f"leaf{seed}.jpg", size=(400, 300), seed=seed)
        expected = {m["label"]: m["confidence"] for m in trained.predict(image, top_k=47)["top"]}
        actual = {m["label"]: m["confidence"] for m in deployed.predict(image, top_k=47)["top"]}
        assert actual == pytest.approx(expected, abs=1e-3)


def test_the_app_serves_the_tflite_model():
    require("models/leaflens_mobilenetv2.tflite")
    assert default_model().name == "leaflens_mobilenetv2.tflite"
