import json
import shutil

import numpy as np
import pytest

from plant_disease.classifier import CONFIDENCE_THRESHOLD, PlantDiseaseClassifier, model_card, resize_bilinear
from plant_disease.knowledge import PLANTVILLAGE_CLASSES

from conftest import CARD_CLASSES


@pytest.fixture(scope="module")
def classifier(carded_model):
    return PlantDiseaseClassifier(carded_model)


@pytest.mark.integration
def test_a_model_without_a_card_uses_the_original_38_classes(legacy_model):
    clf = PlantDiseaseClassifier(legacy_model)
    assert clf.classes == PLANTVILLAGE_CLASSES
    assert clf.card == {}


@pytest.mark.integration
def test_the_card_supplies_class_names(classifier, carded_model):
    assert classifier.classes == CARD_CLASSES
    assert model_card(carded_model)["name"] == "Tiny test model"


@pytest.mark.integration
def test_a_card_that_disagrees_with_the_model_is_rejected(tmp_path, legacy_model):
    shutil.copy(legacy_model, tmp_path / "broken.keras")
    (tmp_path / "broken.json").write_text(json.dumps({"classes": ["A___b", "C___d"]}))
    with pytest.raises(ValueError, match="predicts 38 classes"):
        PlantDiseaseClassifier(str(tmp_path / "broken.keras"))


@pytest.mark.integration
def test_photos_are_centre_cropped_to_the_model_input(classifier, photo):
    pixels = classifier.preprocess(photo(size=(640, 360)))
    assert pixels.shape == (32, 32, 3)
    assert pixels.dtype == np.float32
    assert 0 <= pixels.min() and pixels.max() <= 255


@pytest.mark.integration
def test_phone_photos_are_turned_upright_using_exif(classifier, photo):
    # Stored with the dark half on the left, tagged "rotate 90 degrees clockwise to display".
    pixels = classifier.preprocess(photo(size=(200, 200), split=True, exif_orientation=6))
    top, bottom = pixels[:4].mean(), pixels[-4:].mean()
    left, right = pixels[:, :4].mean(), pixels[:, -4:].mean()
    assert top < 50 and bottom > 200  # the dark half is now on top
    assert abs(left - right) < 20


@pytest.mark.integration
def test_predictions_are_ranked_and_come_with_a_guide(classifier, photo):
    result = classifier.predict(photo(), top_k=3)
    confidences = [match["confidence"] for match in result["top"]]
    assert confidences == sorted(confidences, reverse=True)
    assert sum(confidences) == pytest.approx(1, abs=0.01)
    prediction = result["prediction"]
    assert prediction["label"] == result["top"][0]["label"]
    assert set(prediction["info"]) == {"type", "cause", "summary", "symptoms", "treatment", "prevention"}
    assert result["uncertain"] == (prediction["confidence"] < CONFIDENCE_THRESHOLD)
    assert result["inference_ms"] >= 0


@pytest.mark.unit
def test_numpy_resize_matches_the_tensorflow_resize_used_in_training():
    tf = pytest.importorskip("tensorflow")
    pixels = np.random.default_rng(0).uniform(0, 255, (256, 256, 3)).astype(np.float32)
    for size in [(224, 224), (128, 128), (299, 299), (32, 32)]:
        expected = tf.image.resize(pixels, size, method="bilinear").numpy()
        np.testing.assert_allclose(resize_bilinear(pixels, size), expected, atol=1e-3)


@pytest.mark.integration
def test_the_tflite_export_gives_the_same_answers(carded_model, lite_model, photo):
    lite = PlantDiseaseClassifier(lite_model)
    assert lite.classes == CARD_CLASSES  # the .json card is shared by file name
    assert lite.image_size == (32, 32)
    expected = PlantDiseaseClassifier(carded_model).predict(photo(), top_k=3)
    actual = lite.predict(photo(), top_k=3)
    assert {m["label"]: m["confidence"] for m in actual["top"]} == pytest.approx(
        {m["label"]: m["confidence"] for m in expected["top"]}, abs=1e-3)
