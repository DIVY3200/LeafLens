"""Shared fixtures: tiny stand-in models and generated photos, so tests never need the datasets."""
import json
import os

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

CARD_CLASSES = ["Cassava___Mosaic_disease", "Corn_(maize)___Stripe", "Tomato___healthy"]


def save_tiny_model(path, num_classes, size=32):
    import keras

    inputs = keras.Input((size, size, 3))
    x = keras.layers.Rescaling(1 / 255)(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    keras.Model(inputs, outputs).save(path)


@pytest.fixture(scope="session")
def legacy_model(tmp_path_factory):
    """A 38-class model without a card, like the original CNN."""
    path = tmp_path_factory.mktemp("legacy") / "legacy.keras"
    save_tiny_model(str(path), 38)
    return str(path)


@pytest.fixture(scope="session")
def carded_model(tmp_path_factory):
    """A 3-class model with a JSON card, like train_transfer.py writes."""
    folder = tmp_path_factory.mktemp("carded")
    save_tiny_model(str(folder / "carded.keras"), len(CARD_CLASSES))
    card = {
        "name": "Tiny test model",
        "classes": CARD_CLASSES,
        "trained_on": ["PlantVillage", "FieldPlant"],
        "metrics": {
            "plantvillage": {"accuracy": 0.9, "n": 1200, "skipped_unknown_classes": 0},
            "fieldplant": {"accuracy": 0.8, "n": 50, "skipped_unknown_classes": 0},
        },
    }
    (folder / "carded.json").write_text(json.dumps(card), encoding="utf-8")
    return str(folder / "carded.keras")


@pytest.fixture
def photo(tmp_path):
    """Make a JPEG: random noise by default, or left half dark / right half bright with split=True."""

    def make(name="leaf.jpg", size=(300, 200), split=False, exif_orientation=None):
        width, height = size
        if split:
            pixels = np.zeros((height, width, 3), np.uint8)
            pixels[:, width // 2:] = 255
        else:
            pixels = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
        path = tmp_path / name
        exif = Image.Exif()
        if exif_orientation:
            exif[0x0112] = exif_orientation
        Image.fromarray(pixels).save(path, "JPEG", exif=exif)
        return path

    return make
