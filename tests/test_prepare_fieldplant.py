import csv
import sys

import pytest
from PIL import Image

import prepare_fieldplant


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["prepare_fieldplant.py", *map(str, args), "--workers", "1"])
    prepare_fieldplant.main()


def read_manifest(folder):
    with open(folder / "manifest.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_class_folder_download_becomes_256px_squares(tmp_path, monkeypatch):
    source = tmp_path / "FieldPlant"
    for split, fp_class, name, size in [
        ("train", "Cassava Mosaic", "IMG20221216154033_jpg.rf.aaa.jpg", (400, 300)),
        ("test", "Tomato blight leaf", "Tomate_Mildiou-19-_jpg.rf.bbb.jpg", (300, 500)),
    ]:
        (source / split / fp_class).mkdir(parents=True)
        Image.new("RGB", size, "green").save(source / split / fp_class / name)

    out = tmp_path / "out"
    run(monkeypatch, source, "--out", out)
    rows = read_manifest(out)
    assert {r["label"] for r in rows} == {"Cassava___Mosaic_disease", "Tomato___Blight"}
    assert {r["group"] for r in rows} == {"202212161540", "tomate_mildiou-19-"}
    for row in rows:
        assert Image.open(out / row["path"]).size == (256, 256)


@pytest.mark.parametrize("mode", ["square", "box"])
def test_yolo_export_is_cut_into_one_image_per_leaf(tmp_path, monkeypatch, mode):
    source = tmp_path / "yolo"
    (source / "train" / "images").mkdir(parents=True)
    (source / "train" / "labels").mkdir(parents=True)
    (source / "data.yaml").write_text("names: ['Corn Healthy', 'Corn leaf blight']\n")
    Image.new("RGB", (1000, 800), "green").save(source / "train" / "images" / "photo.jpg")
    # Two real leaves and one 10-pixel box, which is too small to classify.
    (source / "train" / "labels" / "photo.txt").write_text("0 0.25 0.5 0.3 0.6\n1 0.75 0.5 0.2 0.4\n1 0.5 0.5 0.01 0.01\n")

    out = tmp_path / "out"
    run(monkeypatch, source, "--out", out, "--crop-mode", mode)
    rows = read_manifest(out)
    assert sorted(r["label"] for r in rows) == ["Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy"]
    for row in rows:
        assert Image.open(out / row["path"]).size == (256, 256)


def test_square_crops_stay_inside_the_photo():
    left, top, right, bottom = prepare_fieldplant.square_box(cx=10, cy=10, w=50, h=20, width=100, height=80, margin=0)
    assert (left, top) == (0, 0)
    assert right - left == bottom - top == 50
