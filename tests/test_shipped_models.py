"""The model files in the repository load, match their cards, and have a guide for every class."""
from pathlib import Path

import pytest

from plant_disease.classifier import PlantDiseaseClassifier
from plant_disease.knowledge import KNOWLEDGE

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "models/leaflens_mobilenetv2.keras": dict(image_size=(224, 224), classes=47),
    "trained_plant_disease_model.keras": dict(image_size=(128, 128), classes=38),
}


@pytest.mark.parametrize("name", MODELS)
def test_shipped_model_loads_and_matches_its_card(name, photo):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} is not in this checkout")
    clf = PlantDiseaseClassifier(str(path))
    assert clf.image_size == MODELS[name]["image_size"]
    assert len(clf.classes) == MODELS[name]["classes"]
    assert all(label in KNOWLEDGE for label in clf.classes)
    assert {"plantvillage", "fieldplant"} <= set(clf.card["metrics"])  # the numbers the app shows
    assert 0 <= clf.predict(photo())["prediction"]["confidence"] <= 1
