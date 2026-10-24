"""Load a trained model once and classify leaf photos with it.

Two model formats work:
  .keras   the training format; needs TensorFlow
  .tflite  the same network converted by export_tflite.py; needs only the small LiteRT
           runtime, so the deployed server fits a free 512 MB instance
"""
import json
import os
import threading
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image, ImageOps

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


def resize_bilinear(pixels, size):
    """Resize an HxWx3 float array exactly like tf.image.resize(..., "bilinear"), which training used.

    Half-pixel centres, no antialiasing; plain NumPy so the server doesn't need TensorFlow.
    """
    def weights(in_size, out_size):
        position = (np.arange(out_size, dtype=np.float32) + 0.5) * np.float32(in_size / out_size) - 0.5
        floor = np.floor(position)
        lower = np.maximum(floor, 0).astype(np.int64)
        upper = np.minimum(np.ceil(position), in_size - 1).astype(np.int64)
        return lower, upper, position - floor

    top, bottom, dy = weights(pixels.shape[0], size[0])
    left, right, dx = weights(pixels.shape[1], size[1])
    dx = dx[None, :, None]
    upper = pixels[top][:, left] + (pixels[top][:, right] - pixels[top][:, left]) * dx
    lower = pixels[bottom][:, left] + (pixels[bottom][:, right] - pixels[bottom][:, left]) * dx
    return upper + (lower - upper) * dy[:, None, None]


class _KerasModel:
    def __init__(self, path):
        try:
            import tensorflow as tf
        except ImportError as exc:  # most often: a Python version TensorFlow doesn't support yet
            raise ImportError(
                "Loading a .keras model needs TensorFlow, which is not installed for this Python interpreter. "
                "TensorFlow supports Python 3.9-3.13, e.g. `py -3.10 app.py`; or use the .tflite model."
            ) from exc
        from . import layers  # noqa: F401  registers custom layers before the model is loaded

        model = tf.keras.models.load_model(path)
        self.input_size = tuple(model.input_shape[1:3])
        self.num_classes = model.output_shape[-1]
        # Compiled once: calling a deep model like MobileNetV2 eagerly costs ~5x more per photo.
        self._forward = tf.function(lambda batch: model(batch, training=False),
                                    input_signature=[tf.TensorSpec([None, *self.input_size, 3], tf.float32)])

    def __call__(self, batch):
        return self._forward(batch).numpy()


class _LiteModel:
    def __init__(self, path):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:  # TensorFlow ships the same interpreter
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

        self._interpreter = Interpreter(model_path=path)
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]
        self.input_size = tuple(int(v) for v in self._input["shape"][1:3])
        self.num_classes = int(self._output["shape"][-1])

    def __call__(self, batch):
        results = []
        for image in batch:  # the converted model takes one image at a time
            self._interpreter.set_tensor(self._input["index"], np.ascontiguousarray(image[None], np.float32))
            self._interpreter.invoke()
            results.append(self._interpreter.get_tensor(self._output["index"])[0].copy())
        return np.array(results)


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
        model_path = str(model_path)
        self._model = _LiteModel(model_path) if model_path.endswith(".tflite") else _KerasModel(model_path)
        self.card = model_card(model_path)
        self.classes = self.card.get("classes") or PLANTVILLAGE_CLASSES  # models without a card: the original 38
        self.image_size = self._model.input_size
        self.threshold = threshold
        self._lock = threading.Lock()

        if self._model.num_classes != len(self.classes):
            raise ValueError(f"{model_path} predicts {self._model.num_classes} classes "
                             f"but its class list has {len(self.classes)}")

        self._run(np.zeros((1, *self.image_size, 3), np.float32))  # warm-up so the first real request is fast

    def _run(self, batch):
        with self._lock:
            return self._model(batch)

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

        # Raw 0-255 values; the models rescale internally.
        return resize_bilinear(np.asarray(image, dtype=np.float32), self.image_size)

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
