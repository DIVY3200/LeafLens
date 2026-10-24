"""LeafLens: a web app that detects plant diseases from camera or uploaded leaf photos.

Run:  py -3.10 app.py                          -> http://localhost:5000
      py -3.10 app.py --host 0.0.0.0 --https   (use the live camera from a phone on the same Wi-Fi)
      py -3.10 app.py --model path/to/model.keras

By default it uses the field-trained model from train_transfer.py when that exists (its
TensorFlow Lite export first, which needs no TensorFlow), and falls back to the original
PlantVillage CNN otherwise.
"""
import argparse
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import UnidentifiedImageError

from plant_disease.classifier import PlantDiseaseClassifier
from plant_disease.knowledge import supported_plants

ROOT = Path(__file__).resolve().parent
MODEL_CHOICES = [
    ROOT / "models" / "leaflens_mobilenetv2.tflite",  # trained on PlantVillage + FieldPlant, runs on LiteRT
    ROOT / "models" / "leaflens_mobilenetv2.keras",   # the same model in Keras format
    ROOT / "trained_plant_disease_model.keras",       # original CNN, PlantVillage only
]
SOURCE_WORDS = {"PlantVillage": "lab", "FieldPlant": "field"}


def default_model():
    return next(path for path in MODEL_CHOICES if path.exists())


def create_app(model_path=None):
    model_path = Path(model_path or default_model())
    classifier = PlantDiseaseClassifier(str(model_path))
    card = classifier.card
    metrics = card.get("metrics", {})
    trained_on = card.get("trained_on", ["PlantVillage"])
    page = dict(
        plants=supported_plants(classifier.classes),
        class_count=len(classifier.classes),
        model_name=card.get("name", "Custom CNN (PlantVillage only)"),
        trained_on=" and ".join(SOURCE_WORDS.get(s, s) for s in trained_on),
        datasets=" and ".join(trained_on),
        lab_accuracy=metrics.get("plantvillage"),
        field_accuracy=metrics.get("fieldplant"),
    )

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB per upload
    app.json.sort_keys = False

    @app.get("/")
    def index():
        return render_template("index.html", **page)

    @app.post("/api/predict")
    def predict():
        image = request.files.get("image")
        if image is None or not image.filename:
            return jsonify(error="Send a leaf photo in the 'image' form field."), 400
        try:
            return jsonify(classifier.predict(image.stream))
        except (UnidentifiedImageError, OSError):
            return jsonify(error="That file couldn't be read as an image. Try a JPG or PNG photo."), 400

    @app.get("/api/health")
    def health():
        commit = os.environ.get("RENDER_GIT_COMMIT", "")[:7] or None  # set by Render: which commit is live
        return jsonify(status="ok", model=page["model_name"], file=model_path.name, commit=commit,
                       classes=len(classifier.classes), input_size=classifier.image_size, metrics=metrics)

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error="That image is larger than 10 MB. Try a smaller photo."), 413

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LeafLens plant disease web app.")
    parser.add_argument("--model", help="a .keras model (default: the field-trained model if present)")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open the app to other devices on your network")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--https", action="store_true",
                        help="serve over HTTPS with a self-signed certificate; phone browsers only allow the camera on HTTPS")
    args = parser.parse_args()

    create_app(args.model).run(host=args.host, port=args.port, ssl_context="adhoc" if args.https else None)
