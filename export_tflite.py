"""Convert a trained .keras model to TensorFlow Lite for the deployed server.

The .tflite file runs on the small LiteRT runtime instead of full TensorFlow, so the web
server needs about a tenth of the memory and fits a free 512 MB instance. The conversion is
checked before it is saved: on random photos, both models must give the same answers.

    python export_tflite.py                                    # models/leaflens_mobilenetv2.keras
    python export_tflite.py --model models/other.keras
"""
import argparse
import contextlib
import io
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

from plant_disease import layers  # noqa: F401  registers custom layers


def convert(model):
    with contextlib.redirect_stdout(io.StringIO()):  # Keras prints every captured variable while exporting
        return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def max_difference(model, tflite_bytes, samples=8):
    """Largest probability difference between the two models on random 0-255 images (and whether top-1 agrees)."""
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    size = tuple(inp["shape"][1:3])
    images = np.random.default_rng(0).uniform(0, 255, (samples, *size, 3)).astype(np.float32)
    expected = model(images, training=False).numpy()
    actual = []
    for image in images:
        interpreter.set_tensor(inp["index"], image[None])
        interpreter.invoke()
        actual.append(interpreter.get_tensor(out["index"])[0])
    actual = np.array(actual)
    return float(np.abs(expected - actual).max()), bool((expected.argmax(1) == actual.argmax(1)).all())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="models/leaflens_mobilenetv2.keras")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="largest allowed probability difference")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    tflite_bytes = convert(model)
    difference, same_top1 = max_difference(model, tflite_bytes)
    if difference > args.tolerance or not same_top1:
        raise SystemExit(f"Conversion changed the predictions (max difference {difference:.2e}); not saved.")

    out = os.path.splitext(args.model)[0] + ".tflite"
    with open(out, "wb") as f:
        f.write(tflite_bytes)
    print(f"saved {out} ({len(tflite_bytes) / 1e6:.1f} MB); max probability difference {difference:.1e}, "
          "same top answer on every test image. It uses the same .json card as the .keras model.")


if __name__ == "__main__":
    main()
