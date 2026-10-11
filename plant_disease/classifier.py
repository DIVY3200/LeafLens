"""Load a trained Keras model once and classify leaf photos with it."""
import json
import os
import threading
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image, ImageOps

try:
    import tensorflow as tf
except ImportError as exc:  # most often: running on a Python version TensorFlow doesn't support yet
    raise ImportError(
        "TensorFlow is not installed for this Python interpreter. TensorFlow supports "
        "Python 3.9-3.13, so start the app with one of those, e.g. `py -3.10 app.py`."
    ) from exc

from . import layers  # noqa: F401  registers custom layers before models are loaded
from .knowledge import PLANTVILLAGE_CLASSES, describe

# Below this top-1 probability the photo is reported as "uncertain". Chosen from
# the bundled test images: correct predictions sat mostly above 0.85, while
# random noise and non-leaf photos scored 0.33-0.54.
CONFIDENCE_THRESHOLD = 0.6

# The dataset's images are 256x256. Normalising every photo to that size first
# means the final bilinear resize to the model's input matches training exactly.
SOURCE_SIZE = 256


def model_card(model_path):
    """The JSON written next to a model by train_transfer.py / evaluate.py ({} if there is none)."""
    try:
        with open(os.path.splitext(model_path)[0] + ".json", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _public(label, confidence):
    entry = describe(label)
    return {
        "label": label,
        "plant": entry["plant"],
        "name": entry["name"],
        "healthy": entry["type"] is None,
        "confidence": round(float(confidence), 4),
    }


class PlantDiseaseClassifier:
    def __init__(self, model_path, threshold=CONFIDENCE_THRESHOLD):
        self.model = tf.keras.models.load_model(model_path)
        self.card = model_card(model_path)
        self.classes = self.card.get("classes") or PLANTVILLAGE_CLASSES  # models without a card: the original 38
        self.image_size = tuple(self.model.input_shape[1:3])
        self.threshold = threshold
        self._lock = threading.Lock()

        outputs = self.model.output_shape[-1]
        if outputs != len(self.classes):
            raise ValueError(f"{model_path} predicts {outputs} classes but its class list has {len(self.classes)}")

        # Compiled once: calling a deep model like MobileNetV2 eagerly costs ~5x more per photo.
        self._forward = tf.function(lambda batch: self.model(batch, training=False),
                                    input_signature=[tf.TensorSpec([None, *self.image_size, 3], tf.float32)])
        self._run(np.zeros((1, *self.image_size, 3), np.float32))  # warm-up so the first real request is fast

    def _run(self, batch):
        with self._lock:
            return self._forward(batch).numpy()

    def preprocess(self, source):
        """Open an image (path or file object) and turn it into the model's input array."""
        image = Image.open(source)
        image = ImageOps.exif_transpose(image).convert("RGB")  # phone photos are often stored rotated

        # Centre-crop to a square so the leaf isn't squashed when resized.
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        if side != SOURCE_SIZE:
            image = image.resize((SOURCE_SIZE, SOURCE_SIZE), Image.LANCZOS)

        # Same resize the training pipelines apply (bilinear, raw 0-255 values; models rescale internally).
        pixels = np.asarray(image, dtype=np.float32)
        return tf.image.resize(pixels, self.image_size, method="bilinear").numpy()

    def predict(self, source, top_k=3):
        pixels = self.preprocess(source)

        start = time.perf_counter()
        probabilities = self._run(pixels[None])[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        ranked = np.argsort(probabilities)[::-1][:top_k]
        top = [_public(self.classes[i], probabilities[i]) for i in ranked]
        best = top[0]
        info = describe(best["label"])

        return {
            "prediction": {
                **best,
                "info": {key: info[key] for key in ("type", "cause", "summary", "symptoms", "treatment", "prevention")},
            },
            "top": top,
            "uncertain": best["confidence"] < self.threshold,
            "threshold": self.threshold,
            "inference_ms": round(elapsed_ms, 1),
        }
