"""Turn a FieldPlant download into 256x256 training images plus a manifest.csv.

Two download formats are recognised automatically:

  Class folders (whole photos)              Roboflow YOLO export (leaf boxes)
    FieldPlant/train/<class>/*.jpg            fieldplant/data.yaml
    FieldPlant/val/<class>/*.jpg              fieldplant/train/images/*.jpg
    FieldPlant/test/<class>/*.jpg             fieldplant/train/labels/*.txt

Whole photos are centre-cropped to a square, exactly like the app's camera frame: the
paper's "raw images" setting (Table 3). YOLO exports are cut into one image per annotated
leaf: the paper's Cropped-FieldPlant (Table 4). Crop modes for YOLO exports:

  square (default)  the leaf box grown into a square, keeping the real background around it
  box               exactly the annotated box, stretched to a square, as in the paper

The download's own train/val/test folders are ignored: train_transfer.py re-splits by
photo group so burst shots of the same plant never straddle train and test.

    python prepare_fieldplant.py Dataset/FieldPlant
    python prepare_fieldplant.py path/to/yolo_export --crop-mode box --out data/fieldplant_box
"""
import argparse
import ast
import collections
import csv
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor

from PIL import Image, ImageOps

from plant_disease.datasets import FIELDPLANT_TO_LABEL, IMAGE_EXTENSIONS, ROBOFLOW_SUFFIX, fieldplant_photo_group

SIZE = 256


def short_name(stem):
    """Roboflow names end in "_jpg.rf.<32 hex>"; keep names short so paths stay under Windows' 260-character limit."""
    return f"{ROBOFLOW_SUFFIX.sub('', stem)[:40]}_{hashlib.md5(stem.encode()).hexdigest()[:8]}"


def save(image, out_dir, label, name, fp_class, stem, photo):
    relative = f"{label}/{name}.jpg"
    os.makedirs(os.path.join(out_dir, label), exist_ok=True)
    image.save(os.path.join(out_dir, relative), quality=92)
    return dict(path=relative, label=label, fieldplant_class=fp_class, group=fieldplant_photo_group(stem), photo=photo)


# ------------------------------------------------------------- whole photos

def find_class_folder_photos(root):
    """(image path, FieldPlant class) for every photo in every split/class folder."""
    photos = []
    for split in sorted(os.listdir(root)):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for fp_class in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, fp_class)
            if os.path.isdir(class_dir):
                photos += [(os.path.join(class_dir, n), fp_class) for n in sorted(os.listdir(class_dir))
                           if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS]
    return photos


def square_photo(job):
    image_path, fp_class, out_dir = job
    stem = os.path.splitext(os.path.basename(image_path))[0]
    image = Image.open(image_path)
    image.draft("RGB", (SIZE, SIZE))  # fast JPEG decode at the smallest scale that still covers 256 px
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    square = image.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.LANCZOS)
    row = save(square, out_dir, FIELDPLANT_TO_LABEL[fp_class], short_name(stem), fp_class, stem,
               os.path.basename(image_path))
    return [row], collections.Counter()


# --------------------------------------------------------------- leaf crops

def read_class_names(root):
    text = open(os.path.join(root, "data.yaml"), encoding="utf-8").read()
    try:
        import yaml
        names = yaml.safe_load(text)["names"]
    except ImportError:  # Roboflow writes names as a one-line list
        line = next(row for row in text.splitlines() if row.startswith("names:"))
        names = ast.literal_eval(line.split(":", 1)[1].strip())
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def find_yolo_photos(root):
    """(image path, label path) for every image in every split folder of the export."""
    pairs = []
    for split in sorted(os.listdir(root)):
        images = os.path.join(root, split, "images")
        if not os.path.isdir(images):
            continue
        for name in sorted(os.listdir(images)):
            stem, ext = os.path.splitext(name)
            if ext.lower() in IMAGE_EXTENSIONS:
                pairs.append((os.path.join(images, name), os.path.join(root, split, "labels", stem + ".txt")))
    return pairs


def square_box(cx, cy, w, h, width, height, margin):
    """The leaf box grown to a square (plus a margin), shifted to stay inside the photo."""
    side = min(max(w, h) * (1 + margin), width, height)
    left = min(max(cx - side / 2, 0), width - side)
    top = min(max(cy - side / 2, 0), height - side)
    return left, top, left + side, top + side


def crop_leaves(job):
    image_path, label_path, class_names, out_dir, mode, margin, min_side = job
    stem = os.path.splitext(os.path.basename(image_path))[0]
    boxes = []
    for line in open(label_path, encoding="utf-8"):
        parts = line.split()
        if len(parts) == 5:  # skip polygon rows or blank lines
            boxes.append((class_names[int(parts[0])], *map(float, parts[1:])))
    if not boxes:
        return [], collections.Counter({"photos without leaves": 1})

    image = Image.open(image_path)
    full_w, full_h = image.size
    limiting = max if mode == "square" else min  # the crop dimension that must keep >= SIZE pixels
    smallest = min(limiting(w * full_w, h * full_h) for _, _, _, w, h in boxes)
    scale = min(1.0, max(SIZE / smallest, 1 / 8))
    image.draft("RGB", (int(full_w * scale), int(full_h * scale)))  # fast JPEG decode at reduced size
    image = image.convert("RGB")
    width, height = image.size

    rows, skipped = [], collections.Counter()
    for n, (fp_class, cx, cy, w, h) in enumerate(boxes):
        if min(w * full_w, h * full_h) < min_side:
            skipped["boxes too small"] += 1
            continue
        if mode == "square":
            box = square_box(cx * width, cy * height, w * width, h * height, width, height, margin)
        else:
            box = (max((cx - w / 2) * width, 0), max((cy - h / 2) * height, 0),
                   min((cx + w / 2) * width, width), min((cy + h / 2) * height, height))
        crop = image.crop(tuple(round(v) for v in box)).resize((SIZE, SIZE), Image.LANCZOS)
        rows.append(save(crop, out_dir, FIELDPLANT_TO_LABEL[fp_class], f"{short_name(stem)}_{n}", fp_class, stem,
                         os.path.basename(image_path)))
    return rows, skipped


# --------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("fieldplant", help="the extracted FieldPlant download")
    parser.add_argument("--out", default="data/fieldplant")
    parser.add_argument("--crop-mode", choices=["square", "box"], default="square", help="YOLO exports only")
    parser.add_argument("--margin", type=float, default=0.05, help="square mode: extra context around each leaf box")
    parser.add_argument("--min-side", type=int, default=48, help="skip leaf boxes smaller than this many pixels")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    skipped = collections.Counter()
    if os.path.exists(os.path.join(args.fieldplant, "data.yaml")):
        class_names = read_class_names(args.fieldplant)
        pairs = find_yolo_photos(args.fieldplant)
        skipped["photos without a label file"] = sum(not os.path.exists(lab) for _, lab in pairs)
        worker, kind = crop_leaves, f"leaf crops ({args.crop_mode})"
        jobs = [(img, lab, class_names, args.out, args.crop_mode, args.margin, args.min_side)
                for img, lab in pairs if os.path.exists(lab)]
    else:
        photos = find_class_folder_photos(args.fieldplant)
        class_names = sorted({c for _, c in photos})
        worker, kind = square_photo, "whole photos (centre square)"
        jobs = [(img, c, args.out) for img, c in photos]
    unknown = set(class_names) - set(FIELDPLANT_TO_LABEL)
    if unknown:
        raise SystemExit(f"Unknown FieldPlant classes {sorted(unknown)}; add them to FIELDPLANT_TO_LABEL.")
    if not jobs:
        raise SystemExit(f"No FieldPlant photos found in {args.fieldplant}")
    print(f"{len(jobs)} photos -> {kind}, using {args.workers} workers...")

    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for done, (job_rows, job_skipped) in enumerate(pool.map(worker, jobs, chunksize=8), 1):
            rows += job_rows
            skipped += job_skipped
            if done % 1000 == 0:
                print(f"  {done}/{len(jobs)} photos")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "fieldplant_class", "group", "photo"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["path"]))

    counts = collections.Counter(r["fieldplant_class"] for r in rows)
    print(f"\n{len(rows)} images from {len({r['photo'] for r in rows})} photos -> {args.out}")
    for name in class_names:
        print(f"  {counts[name]:5d}  {name:28s} -> {FIELDPLANT_TO_LABEL[name]}")
    for reason, n in skipped.items():
        if n:
            print(f"skipped: {n} {reason}")


if __name__ == "__main__":
    main()
