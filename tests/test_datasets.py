import pytest

from plant_disease.datasets import (FIELDPLANT_TO_LABEL, fieldplant_photo_group, has_equivalent, is_correct,
                                    plantvillage_leaf_id, read_fieldplant_manifest, scan_plantvillage, split_of)
from plant_disease.knowledge import PLANTVILLAGE_CLASSES

pytestmark = pytest.mark.unit

UUID = "0114f7b3-7f56-4bef-aa47-7a82da142d4b"


def test_augmented_plantvillage_copies_share_a_leaf_id():
    copies = [f"{UUID}___RS_Erly.B 7437.JPG", f"{UUID}___RS_Erly.B 7437_180deg.JPG", f"{UUID}___RS_Erly.B 7437_flipTB.JPG"]
    assert len({plantvillage_leaf_id(name) for name in copies}) == 1
    assert plantvillage_leaf_id("RS_Rust 1563.JPG") == plantvillage_leaf_id("RS_Rust 1563_flipLR.JPG")
    assert plantvillage_leaf_id("RS_Rust 1563.JPG") != plantvillage_leaf_id("RS_Rust 1564.JPG")


def test_burst_shots_share_a_photo_group():
    first = fieldplant_photo_group("IMG20221216154033_jpg.rf.0250abb6e07766776b79a780abc10781")
    same_minute = fieldplant_photo_group("IMG20221216154034_jpg.rf.0238d2426e93b78989834853807a7568")
    next_minute = fieldplant_photo_group("IMG20221216154107_jpg.rf.a2a47ad43118c1e415302a71d29cf0c9")
    assert first == same_minute != next_minute
    assert fieldplant_photo_group("IMG_20221118_080520_360_jpg.rf.e4aec6954634d69ce5d8af7553b9a636") == "202211180805"
    assert fieldplant_photo_group("Mais_Brulure_-140-_jpg.rf.9d30cfe0f517bcbbcfdc081982eff736") == "mais_brulure_-140-"


def test_split_is_deterministic_and_roughly_80_10_10():
    groups = [f"group-{i}" for i in range(5000)]
    splits = [split_of(g) for g in groups]
    assert splits == [split_of(g) for g in groups]
    share = {name: splits.count(name) / len(splits) for name in ("train", "val", "test")}
    assert 0.77 < share["train"] < 0.83
    assert 0.08 < share["val"] < 0.12
    assert 0.08 < share["test"] < 0.12


def test_tomato_blight_counts_as_early_or_late_blight_only():
    assert is_correct("Tomato___Blight", "Tomato___Early_blight")
    assert is_correct("Tomato___Late_blight", "Tomato___Blight")
    assert not is_correct("Tomato___Early_blight", "Tomato___Late_blight")
    assert has_equivalent("Tomato___Blight", PLANTVILLAGE_CLASSES)
    assert not has_equivalent("Cassava___Mosaic_disease", PLANTVILLAGE_CLASSES)


def test_seven_fieldplant_classes_merge_into_plantvillage_ones():
    merged = {fp for fp, label in FIELDPLANT_TO_LABEL.items() if label in PLANTVILLAGE_CLASSES}
    assert merged == {"Corn Gray leaf spot", "Corn Healthy", "Corn leaf blight", "Corn rust leaf",
                      "Tomato healthy", "Tomato leaf mosaic virus", "Tomato leaf yellow virus"}


def test_scan_keeps_every_copy_of_a_leaf_in_one_split(tmp_path):
    folder = tmp_path / "Apple___healthy"
    folder.mkdir()
    for suffix in ("", "_flipLR", "_180deg"):
        (folder / f"{UUID}___RS_HL 1{suffix}.JPG").write_bytes(b"")
    (folder / "notes.txt").write_text("not an image")

    items = scan_plantvillage([str(tmp_path)])
    assert len(items) == 3
    assert {it["label"] for it in items} == {"Apple___healthy"}
    assert len({it["split"] for it in items}) == 1


def test_manifest_leaves_out_non_leaf_classes(tmp_path):
    (tmp_path / "manifest.csv").write_text(
        "path,label,fieldplant_class,group,photo\n"
        "Cassava___Mosaic_disease/a.jpg,Cassava___Mosaic_disease,Cassava Mosaic,g1,a.jpg\n"
        "Cassava___Root_rot/b.jpg,Cassava___Root_rot,Cassava Root Rot,g2,b.jpg\n",
        encoding="utf-8",
    )
    items = read_fieldplant_manifest(str(tmp_path))
    assert [it["label"] for it in items] == ["Cassava___Mosaic_disease"]
    assert items[0]["domain"] == "fieldplant"
    assert len(read_fieldplant_manifest(str(tmp_path), include_non_leaf=True)) == 2
