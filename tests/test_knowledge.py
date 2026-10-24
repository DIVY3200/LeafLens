import pytest

from plant_disease.knowledge import FIELDPLANT, KNOWLEDGE, PLANTVILLAGE_CLASSES, describe, supported_plants

KEYS = {"plant", "name", "type", "cause", "summary", "symptoms", "treatment", "prevention"}


def test_original_model_class_order_is_unchanged():
    assert len(PLANTVILLAGE_CLASSES) == 38
    assert PLANTVILLAGE_CLASSES == sorted(PLANTVILLAGE_CLASSES)
    assert not set(FIELDPLANT) & set(PLANTVILLAGE_CLASSES)


@pytest.mark.parametrize("label", sorted(KNOWLEDGE))
def test_every_guide_is_complete(label):
    entry = KNOWLEDGE[label]
    assert set(entry) == KEYS
    assert entry["summary"] and entry["prevention"]
    if entry["type"] is not None:  # a disease or disorder, not "healthy"
        assert entry["cause"] and entry["symptoms"] and entry["treatment"]


def test_labels_without_a_guide_get_a_safe_fallback():
    entry = describe("Corn_(maize)___Purple_discoloration")
    assert (entry["plant"], entry["name"], entry["type"]) == ("Corn (maize)", "Purple discoloration", "unknown")
    assert describe("Okra___healthy")["type"] is None


def test_supported_plants_list_diseases_before_healthy():
    plants = {p["name"]: p["conditions"] for p in supported_plants()}
    assert len(plants) == 14
    assert plants["Apple"][-1] == {"name": "Healthy", "healthy": True}
    assert not any(c["healthy"] for c in plants["Apple"][:-1])
