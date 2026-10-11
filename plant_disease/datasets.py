"""Dataset helpers shared by prepare_fieldplant.py, train_transfer.py and evaluate.py.

Two sources are combined, following Moupojou et al., "FieldPlant: A Dataset of Field
Plant Images for Plant Disease Detection and Classification With Deep Learning"
(IEEE Access, 2023):

* PlantVillage: lab photos, one leaf on a plain background (the "New Plant Diseases
  Dataset" already in train/ and valid/).
* FieldPlant: phone photos taken in plantations, with every leaf boxed by plant
  pathologists. prepare_fieldplant.py turns a download into 256x256 images: whole photos
  (class-folder downloads) or one image per leaf box (YOLO exports, the paper's C-FP).

Splits are assigned by hashing a *group* key rather than the file, so augmented copies of
one PlantVillage leaf, or burst shots of one field plant, never end up on both sides of
the train/test boundary.
"""
import csv
import hashlib
import os
import re

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# FieldPlant classes that describe the same condition as a PlantVillage class are merged
# into it (the pairing used for the cross-dataset test in Fig. 15 of the paper). The rest
# become new classes, named in PlantVillage's "Plant___Condition" style.
FIELDPLANT_TO_LABEL = {
    "Cassava Bacterial Blight": "Cassava___Bacterial_blight",
    "Cassava Brown Leaf Spot": "Cassava___Brown_leaf_spot",
    "Cassava Healthy": "Cassava___healthy",
    "Cassava Mosaic": "Cassava___Mosaic_disease",
    "Cassava Root Rot": "Cassava___Root_rot",
    "Corn Brown Spots": "Corn_(maize)___Brown_spot",
    "Corn Charcoal": "Corn_(maize)___Charcoal",
    "Corn Chlorotic Leaf Spot": "Corn_(maize)___Chlorotic_leaf_spot",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn Healthy": "Corn_(maize)___healthy",
    "Corn Insects Damages": "Corn_(maize)___Insect_damage",
    "Corn Mildew": "Corn_(maize)___Downy_mildew",
    "Corn Purple Discoloration": "Corn_(maize)___Purple_discoloration",
    "Corn Smut": "Corn_(maize)___Smut",
    "Corn Streak": "Corn_(maize)___Streak_virus",
    "Corn Stripe": "Corn_(maize)___Stripe",
    "Corn Violet Decoloration": "Corn_(maize)___Violet_discoloration",
    "Corn Yellow Spots": "Corn_(maize)___Yellow_spots",
    "Corn Yellowing": "Corn_(maize)___Yellowing",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Tomato Brown Spots": "Tomato___Brown_spots",
    "Tomato bacterial wilt": "Tomato___Bacterial_wilt",
    "Tomato blight leaf": "Tomato___Blight",
    "Tomato healthy": "Tomato___healthy",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}

# The paper lists these as non-leaf classes (roots and cobs), so they are left out of a leaf classifier.
NON_LEAF_CLASSES = {"Cassava Root Rot", "Corn Charcoal"}

# FieldPlant labels tomato blight without saying early or late, so a prediction of either
# counts as correct for it (and vice versa).
_EQUIVALENT = {
    frozenset({"Tomato___Blight", "Tomato___Early_blight"}),
    frozenset({"Tomato___Blight", "Tomato___Late_blight"}),
}


def is_correct(true_label, predicted_label):
    return true_label == predicted_label or frozenset({true_label, predicted_label}) in _EQUIVALENT


def has_equivalent(label, classes):
    """True if `label` (or a label counted as equivalent) is one of `classes`."""
    return label in classes or any(is_correct(label, c) for c in classes)


# ----------------------------------------------------------------- grouping

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
_AUGMENT_SUFFIX = re.compile(r"(_(flipLR|flipTB|\d+deg\w*|new30degFlipLR|newGRR|change_\w+|flip\w*))+$", re.I)
ROBOFLOW_SUFFIX = re.compile(r"_(jpe?g|png)\.rf\.[0-9a-f]+$", re.I)
_TIMESTAMP = re.compile(r"(\d{8})_?(\d{4})\d{2}")


def plantvillage_leaf_id(filename):
    """The original leaf a (possibly flipped or rotated) PlantVillage file was made from."""
    match = _UUID.match(filename)
    if match:
        return match.group(0).lower()
    return _AUGMENT_SUFFIX.sub("", os.path.splitext(filename)[0]).strip().lower()


def fieldplant_photo_group(photo_stem):
    """Group key for a FieldPlant photo. Shots taken in the same minute are usually the same plant."""
    base = ROBOFLOW_SUFFIX.sub("", photo_stem)
    match = _TIMESTAMP.search(base)
    return match.group(1) + match.group(2) if match else base.lower()


def split_of(group, val_fraction=0.1, test_fraction=0.1):
    """Deterministic train/val/test assignment from a group key (stable across runs and machines)."""
    position = int(hashlib.md5(group.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if position < test_fraction:
        return "test"
    if position < test_fraction + val_fraction:
        return "val"
    return "train"


# ------------------------------------------------------------------ sources

def scan_plantvillage(roots):
    """Items for every image under each root's class folders (e.g. ["train", "valid"])."""
    items = []
    for root in roots:
        for label in sorted(os.listdir(root)):
            folder = os.path.join(root, label)
            if not os.path.isdir(folder):
                continue
            for name in sorted(os.listdir(folder)):
                if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                    group = "pv:" + plantvillage_leaf_id(name)
                    items.append(dict(path=os.path.join(folder, name), label=label, domain="plantvillage",
                                      group=group, split=split_of(group)))
    return items


def read_fieldplant_manifest(data_dir, include_non_leaf=False):
    """Items for the images written by prepare_fieldplant.py (whole photos or leaf crops)."""
    items = []
    with open(os.path.join(data_dir, "manifest.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["fieldplant_class"] in NON_LEAF_CLASSES and not include_non_leaf:
                continue
            group = "fp:" + row["group"]
            items.append(dict(path=os.path.join(data_dir, row["path"]), label=row["label"], domain="fieldplant",
                              group=group, split=split_of(group)))
    return items
