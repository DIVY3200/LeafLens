"""Train a field-ready classifier with transfer learning, following the FieldPlant paper.

Moupojou et al. (IEEE Access, 2023) found that models trained only on PlantVillage lab
photos collapse on field photos, and that ImageNet-pretrained MobileNet did best on field
data. This script reproduces their recipe:

* backbone pretrained on ImageNet: MobileNetV2 (default), VGG16, InceptionV3 or InceptionResNetV2
* phase 1: frozen backbone, new classifier head, learning rate 0.001
* phase 2: fine-tune the top of the backbone, learning rate 0.0001
* sparse categorical cross-entropy, images resized to each backbone's input size

On top of that, it trains on PlantVillage *and* FieldPlant together, adds
field-style augmentation and class weights (FieldPlant is very imbalanced), and splits by
leaf/photo group so no copy of a test leaf is ever seen in training.

    python prepare_fieldplant.py Dataset/FieldPlant        # once: photos -> data/fieldplant
    python train_transfer.py                               # MobileNetV2 on lab + field leaves
    python train_transfer.py --train-on plantvillage       # the paper's PV-only baseline

The best epoch so far is checkpointed in reports/<model>/, so if a run is interrupted (the
laptop sleeps, the terminal closes) its result isn't lost: re-run the same command with
--finalize to test and save that checkpoint without training again.
"""
import argparse
import collections
import csv
import datetime
import os
import random
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import keras

from evaluate import evaluate_items, print_metrics, save_card, write_reports
from plant_disease.datasets import read_fieldplant_manifest, scan_plantvillage
from plant_disease.layers import CaffePreprocess

BACKBONES = {
    # name: (constructor, default input size, ImageNet input convention)
    "mobilenetv2": (keras.applications.MobileNetV2, 224, "tf"),
    "vgg16": (keras.applications.VGG16, 224, "caffe"),
    "inceptionv3": (keras.applications.InceptionV3, 299, "tf"),
    "inceptionresnetv2": (keras.applications.InceptionResNetV2, 299, "tf"),
}
DISPLAY = {"mobilenetv2": "MobileNetV2", "vgg16": "VGG16", "inceptionv3": "InceptionV3",
           "inceptionresnetv2": "InceptionResNetV2"}


# ---------------------------------------------------------------------- data

def collect(args):
    """Train/val/test item lists and the class list.

    Training and validation use only the --train-on source(s); the test split always holds
    both datasets when available, so a lab-only model is also scored on field leaves
    (the paper's cross-dataset test).
    """
    rng = random.Random(args.seed)
    has_field = os.path.exists(os.path.join(args.fieldplant, "manifest.csv"))
    if args.train_on in ("both", "fieldplant") and not has_field:
        sys.exit(f"No prepared FieldPlant images in {args.fieldplant}. Run: python prepare_fieldplant.py Dataset/FieldPlant")
    items = scan_plantvillage(args.plantvillage) + (read_fieldplant_manifest(args.fieldplant) if has_field else [])
    sources = {"both": {"plantvillage", "fieldplant"}, "plantvillage": {"plantvillage"}, "fieldplant": {"fieldplant"}}[args.train_on]

    splits = {name: [it for it in items if it["split"] == name and (name == "test" or it["domain"] in sources)]
              for name in ("train", "val", "test")}

    # PlantVillage has ~1,800 images per class but FieldPlant only a few hundred at best, so
    # PlantVillage is capped per class to keep training time and the domain balance sensible.
    caps = {"train": args.pv_per_class, "val": max(1, args.pv_per_class // 5), "test": None}
    for name, cap in caps.items():
        if cap:
            by_class = collections.defaultdict(list)
            for it in splits[name]:
                (by_class[it["label"]] if it["domain"] == "plantvillage" else by_class[None]).append(it)
            field = by_class.pop(None, [])
            splits[name] = field + [it for group in by_class.values() for it in rng.sample(group, min(cap, len(group)))]

    # A class needs enough training examples; FieldPlant-only classes with fewer are dropped.
    train_counts = collections.Counter(it["label"] for it in splits["train"])
    pv_labels = {it["label"] for it in splits["train"] if it["domain"] == "plantvillage"}
    classes = sorted(label for label, n in train_counts.items() if label in pv_labels or n >= args.min_field_samples)
    dropped = sorted((label, n) for label, n in train_counts.items() if label not in classes)
    for name in splits:
        splits[name] = [it for it in splits[name] if it["label"] in classes or name == "test"]
    return splits, classes, dropped


def augmenter():
    """Field-style variation: framing, rotation, lighting. Applied to training batches only."""
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.15),
        keras.layers.RandomZoom((-0.25, 0.1)),
        keras.layers.RandomTranslation(0.1, 0.1),
        keras.layers.RandomBrightness(0.25, value_range=(0, 255)),
        keras.layers.RandomContrast(0.25),
    ], name="augment")


def make_dataset(items, class_index, image_size, batch_size, training, seed):
    paths = [it["path"] for it in items]
    labels = [class_index[it["label"]] for it in items]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(items), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return tf.image.resize(image, (image_size, image_size)), label  # raw 0-255, like the app sends

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    if training:
        augment = augmenter()
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def class_weights(items, class_index, mode):
    if mode == "none":
        return None
    counts = collections.Counter(class_index[it["label"]] for it in items)
    total, k = sum(counts.values()), len(class_index)
    raw = {c: total / (k * n) for c, n in counts.items()}
    if mode == "sqrt":
        raw = {c: w ** 0.5 for c, w in raw.items()}
    mean = sum(raw.values()) / len(raw)
    return {c: w / mean for c, w in raw.items()}


# --------------------------------------------------------------------- model

def build_model(backbone, num_classes, image_size):
    constructor, _, convention = BACKBONES[backbone]
    inputs = keras.Input((image_size, image_size, 3), name="rgb_0_255")
    x = keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs) if convention == "tf" else CaffePreprocess()(inputs)
    base = constructor(include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3), pooling="avg")
    base.trainable = False
    x = base(x, training=False)  # keep BatchNorm in inference mode, also while fine-tuning
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classes")(x)
    return keras.Model(inputs, outputs, name=f"leaflens_{backbone}"), base


def unfreeze_top(base, fraction):
    base.trainable = True
    cutoff = int(len(base.layers) * (1 - fraction))
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= cutoff and not isinstance(layer, keras.layers.BatchNormalization)


def fit(model, train_ds, val_ds, epochs, learning_rate, weights, checkpoint, log_path, initial_epoch=0):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True),
        keras.callbacks.CSVLogger(log_path),  # written every epoch, so the curves survive an interrupted run
    ]
    return model.fit(train_ds, validation_data=val_ds, epochs=initial_epoch + epochs, initial_epoch=initial_epoch,
                     class_weight=weights, callbacks=callbacks, verbose=1 if sys.stdout.isatty() else 2)


def read_history(log_paths):
    """Epoch rows from each phase's CSV log (phases that never started are left out)."""
    phases = []
    for path in log_paths:
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as f:
                phases.append([{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)])
    return phases


def training_summary(phases, finalized):
    """Epoch counts and the epoch the saved weights come from (the last phase's best), for the model card."""
    if not phases:
        return {"finalized_from_checkpoint": finalized}
    best = min(phases[-1], key=lambda row: row["val_loss"])
    return {
        "epochs_phase1": len(phases[0]),
        "epochs_phase2": len(phases[1]) if len(phases) > 1 else 0,
        "best_epoch": int(best["epoch"]) + 1,
        "best_val_accuracy": round(best["val_accuracy"], 4),
        "best_val_loss": round(best["val_loss"], 4),
        "finalized_from_checkpoint": finalized,
    }


def plot_history(phases, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for phase in phases for row in phase]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        ax.plot([r[metric] for r in rows], label=f"training {metric}")
        ax.plot([r[f"val_{metric}"] for r in rows], label=f"validation {metric}")
        if len(phases) > 1:
            ax.axvline(len(phases[0]) - 0.5, color="green", linestyle="--", label="start fine-tuning")
        ax.set_ylabel(metric)
        ax.legend()
    axes[1].set_xlabel("epoch")
    fig.suptitle("Training and validation (compare Fig. 14 of the FieldPlant paper)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backbone", choices=BACKBONES, default="mobilenetv2")
    parser.add_argument("--train-on", choices=["both", "plantvillage", "fieldplant"], default="both")
    parser.add_argument("--plantvillage", nargs="+", default=["Dataset/old/train", "Dataset/old/valid"], help="PlantVillage class-folder roots")
    parser.add_argument("--fieldplant", default="data/fieldplant")
    parser.add_argument("--image-size", type=int, help="default: the backbone's ImageNet size (224 or 299)")
    parser.add_argument("--pv-per-class", type=int, default=600, help="PlantVillage training images per class")
    parser.add_argument("--min-field-samples", type=int, default=40,
                        help="drop FieldPlant-only classes with fewer training leaves than this")
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--head-lr", type=float, default=1e-3, help="paper: 0.001")
    parser.add_argument("--finetune-lr", type=float, default=1e-4, help="paper: 0.0001")
    parser.add_argument("--unfreeze", type=float, default=0.3, help="fraction of backbone layers fine-tuned in phase 2")
    parser.add_argument("--class-weights", choices=["sqrt", "balanced", "none"], default="sqrt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", help="default: models/leaflens_<backbone>.keras")
    parser.add_argument("--finalize", action="store_true",
                        help="don't train: finish an interrupted run from its best checkpoint "
                             "(pass the same options as that run)")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # Keras progress bars use characters some Windows consoles can't encode
    keras.utils.set_random_seed(args.seed)
    image_size = args.image_size or BACKBONES[args.backbone][1]
    out = args.out or os.path.join("models", f"leaflens_{args.backbone}.keras")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    report_dir = os.path.join("reports", os.path.splitext(os.path.basename(out))[0])
    os.makedirs(report_dir, exist_ok=True)

    splits, classes, dropped = collect(args)
    class_index = {c: i for i, c in enumerate(classes)}
    for name, items in splits.items():
        domains = collections.Counter(it["domain"] for it in items)
        print(f"{name:5s} {len(items):6,} leaves  " + "  ".join(f"{d}: {n:,}" for d, n in sorted(domains.items())))
    print(f"{len(classes)} classes" + (f"; dropped (too few field examples): {dropped}" if dropped else ""))

    checkpoint = os.path.join(report_dir, "checkpoint.keras")  # best epoch so far, updated during training
    phase_logs = [os.path.join(report_dir, f"history_phase{n}.csv") for n in (1, 2)]

    if args.finalize:
        if not os.path.exists(checkpoint):
            sys.exit(f"No checkpoint at {checkpoint}, so there is no interrupted run to finalize.")
        model = keras.models.load_model(checkpoint)
        if model.output_shape[-1] != len(classes):
            sys.exit(f"The checkpoint predicts {model.output_shape[-1]} classes but these options give {len(classes)}; "
                     "pass the same options as the interrupted run.")
        print(f"\nfinalizing the best checkpoint of the interrupted run: {checkpoint}")
    else:
        for log in phase_logs:
            if os.path.exists(log):
                os.remove(log)
        train_ds = make_dataset(splits["train"], class_index, image_size, args.batch_size, True, args.seed)
        val_ds = make_dataset(splits["val"], class_index, image_size, args.batch_size, False, args.seed)
        weights = class_weights(splits["train"], class_index, args.class_weights)
        model, base = build_model(args.backbone, len(classes), image_size)

        print(f"\nphase 1: training the new head on a frozen {DISPLAY[args.backbone]} (lr {args.head_lr})")
        head = fit(model, train_ds, val_ds, args.head_epochs, args.head_lr, weights, checkpoint, phase_logs[0])
        if args.finetune_epochs:
            print(f"\nphase 2: fine-tuning the top {args.unfreeze:.0%} of the backbone (lr {args.finetune_lr})")
            unfreeze_top(base, args.unfreeze)
            fit(model, train_ds, val_ds, args.finetune_epochs, args.finetune_lr, weights, checkpoint, phase_logs[1],
                initial_epoch=len(head.history["loss"]))

    model.save(out)
    phases = read_history(phase_logs)
    if phases:
        plot_history(phases, os.path.join(report_dir, "history.png"))

    print("\ntest results (leaves never seen in training):")
    metrics, known, predictions = evaluate_items(model, classes, splits["test"], args.batch_size)
    print_metrics(metrics)
    if known:
        write_reports(known, predictions, report_dir)

    trained_on = {"both": ["PlantVillage", "FieldPlant"], "plantvillage": ["PlantVillage"],
                  "fieldplant": ["FieldPlant"]}[args.train_on]
    save_card(out, {
        "name": f"{DISPLAY[args.backbone]} ({' + '.join(trained_on)})",
        "backbone": args.backbone,
        "image_size": image_size,
        "classes": classes,
        "trained_on": trained_on,
        "train_counts": dict(sorted(collections.Counter(it["label"] for it in splits["train"]).items())),
        "metrics": metrics,
        "evaluated": {"split": "test", "date": datetime.date.today().isoformat()},
        "training": training_summary(phases, args.finalize),
        "hyperparameters": {k: v for k, v in vars(args).items()
                            if k not in ("plantvillage", "fieldplant", "out", "finalize")},
    })
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
    print(f"\nsaved {out} and its card {os.path.splitext(out)[0]}.json; plots in {report_dir}/")


if __name__ == "__main__":
    main()
