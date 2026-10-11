"""Measure a model on lab (PlantVillage) and field (FieldPlant) photos separately.

This is the paper's cross-dataset test (Tables 3 and 4). For example, the original CNN was trained
only on PlantVillage, so every FieldPlant photo is new to it:

    python evaluate.py --model trained_plant_disease_model.keras --plantvillage Dataset/old/valid --leak-free-against Dataset/old/train --split all --save

A model trained by train_transfer.py is evaluated on the held-out test split:

    python evaluate.py --model models/leaflens_mobilenetv2.keras

Field photos whose condition a model has never been taught (e.g. cassava for a
PlantVillage-only model) are skipped and reported, not counted as wrong.
"""
import argparse
import collections
import datetime
import json
import os
import random

import numpy as np

from plant_disease.datasets import (has_equivalent, is_correct, plantvillage_leaf_id, read_fieldplant_manifest,
                                    scan_plantvillage)


def image_dataset(paths, image_size, batch_size=64):
    import tensorflow as tf

    def load(path):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return tf.image.resize(image, (image_size, image_size))  # bilinear on raw 0-255 values, like the app

    return (tf.data.Dataset.from_tensor_slices(paths)
            .map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE))


def evaluate_items(model, classes, items, batch_size=64):
    """Accuracy per domain and overall, plus the predictions (for reports)."""
    image_size = model.input_shape[1]
    known = [it for it in items if has_equivalent(it["label"], classes)]
    skipped = collections.Counter(it["domain"] for it in items if not has_equivalent(it["label"], classes))
    if not known:
        return {}, [], []
    probabilities = model.predict(image_dataset([it["path"] for it in known], image_size, batch_size), verbose=0)
    predictions = [classes[i] for i in probabilities.argmax(axis=1)]

    metrics = {}
    for domain in sorted({it["domain"] for it in known}) + ["overall"]:
        index = [i for i, it in enumerate(known) if domain == "overall" or it["domain"] == domain]
        correct = sum(is_correct(known[i]["label"], predictions[i]) for i in index)
        metrics[domain] = {"accuracy": round(correct / len(index), 4), "n": len(index),
                           "skipped_unknown_classes": sum(skipped.values()) if domain == "overall" else skipped[domain]}
    return metrics, known, predictions


def write_reports(known, predictions, report_dir):
    """Classification report and a field confusion matrix (like Fig. 15 of the paper)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix

    os.makedirs(report_dir, exist_ok=True)
    # Count an equivalent prediction (tomato blight vs early/late blight) as the true label.
    y_true = [it["label"] for it in known]
    y_pred = [t if is_correct(t, p) else p for t, p in zip(y_true, predictions)]
    labels = sorted(set(y_true))
    with open(os.path.join(report_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    field = [i for i, it in enumerate(known) if it["domain"] == "fieldplant"]
    if field:
        t = [y_true[i] for i in field]
        p = [y_pred[i] for i in field]
        rows = sorted(set(t))
        columns = sorted(set(t) | set(p))
        matrix = confusion_matrix(t, p, labels=columns)[[columns.index(r) for r in rows]]
        fig, ax = plt.subplots(figsize=(max(8, len(columns) * 0.45), max(6, len(rows) * 0.4)))
        ax.imshow(matrix, cmap="Greens")
        ax.set_xticks(range(len(columns)), [c.replace("___", " · ") for c in columns], rotation=70, ha="right", fontsize=7)
        ax.set_yticks(range(len(rows)), [r.replace("___", " · ") for r in rows], fontsize=7)
        for (r, c), v in np.ndenumerate(matrix):
            if v:
                ax.text(c, r, v, ha="center", va="center", fontsize=6, color="white" if v > matrix.max() / 2 else "black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True (field leaves)")
        ax.set_title("Confusion matrix on FieldPlant")
        fig.tight_layout()
        fig.savefig(os.path.join(report_dir, "confusion_fieldplant.png"), dpi=130)
        plt.close(fig)


def load_model(model_path):
    import tensorflow as tf
    from plant_disease import layers  # noqa: F401  registers custom layers
    from plant_disease.classifier import model_card
    from plant_disease.knowledge import PLANTVILLAGE_CLASSES

    model = tf.keras.models.load_model(model_path)
    card = model_card(model_path)
    return model, card, card.get("classes") or PLANTVILLAGE_CLASSES


def save_card(model_path, card):
    with open(os.path.splitext(model_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)


def print_metrics(metrics):
    for domain, m in metrics.items():
        extra = f"  ({m['skipped_unknown_classes']} skipped: classes the model was never taught)" if m["skipped_unknown_classes"] else ""
        print(f"  {domain:13s} {m['accuracy']:7.2%}  on {m['n']:,} leaves{extra}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="trained_plant_disease_model.keras")
    parser.add_argument("--plantvillage", nargs="*", default=["Dataset/old/train", "Dataset/old/valid"], help="PlantVillage class-folder roots")
    parser.add_argument("--fieldplant", default="data/fieldplant", help="output of prepare_fieldplant.py")
    parser.add_argument("--split", choices=["test", "all"], default="test",
                        help="'all' only makes sense for images the model never trained on")
    parser.add_argument("--leak-free-against", metavar="DIR",
                        help="drop PlantVillage images whose original leaf also appears in DIR (e.g. the model's train folder)")
    parser.add_argument("--limit", type=int, help="evaluate at most this many leaves per domain (random sample)")
    parser.add_argument("--save", action="store_true", help="store the metrics in the model's JSON card (shown by the app)")
    args = parser.parse_args()

    items = scan_plantvillage(args.plantvillage) if args.plantvillage else []
    if args.leak_free_against:
        seen = {plantvillage_leaf_id(n) for d in os.listdir(args.leak_free_against)
                for n in os.listdir(os.path.join(args.leak_free_against, d))}
        before = len(items)
        items = [it for it in items if plantvillage_leaf_id(os.path.basename(it["path"])) not in seen]
        print(f"leak-free: dropped {before - len(items):,} PlantVillage images whose leaf appears in {args.leak_free_against}/")
    if os.path.exists(os.path.join(args.fieldplant, "manifest.csv")):
        items += read_fieldplant_manifest(args.fieldplant)
    else:
        print(f"no prepared FieldPlant images at {args.fieldplant} (run prepare_fieldplant.py); lab leaves only")
    if args.split == "test":
        items = [it for it in items if it["split"] == "test"]
    if args.limit:
        rng = random.Random(0)
        by_domain = collections.defaultdict(list)
        for it in items:
            by_domain[it["domain"]].append(it)
        items = [it for group in by_domain.values() for it in rng.sample(group, min(args.limit, len(group)))]

    model, card, classes = load_model(args.model)
    print(f"evaluating {args.model} on {len(items):,} leaves ({args.split} split)...")
    metrics, known, predictions = evaluate_items(model, classes, items)
    print_metrics(metrics)

    report_dir = os.path.join("reports", os.path.splitext(os.path.basename(args.model))[0])
    if known:
        write_reports(known, predictions, report_dir)
        print(f"reports written to {report_dir}/")
    if args.save:
        card.setdefault("name", "Custom CNN (PlantVillage only)")
        card.setdefault("classes", classes)
        card.setdefault("trained_on", ["PlantVillage"])
        card["metrics"] = metrics
        card["evaluated"] = {"split": args.split, "leak_free_against": args.leak_free_against,
                             "date": datetime.date.today().isoformat()}
        save_card(args.model, card)
        print(f"metrics saved to {os.path.splitext(args.model)[0]}.json")


if __name__ == "__main__":
    main()
