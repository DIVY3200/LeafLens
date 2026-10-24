import argparse
import collections

import pytest

import train_transfer


@pytest.fixture
def datasets(tmp_path):
    """PlantVillage: 2 classes x 30 leaves. FieldPlant: a common class (60 photos) and a rare one (3 photos)."""
    pv = tmp_path / "pv"
    for label in ("Apple___healthy", "Tomato___healthy"):
        (pv / label).mkdir(parents=True)
        for i in range(30):
            (pv / label / f"{i:08x}-0000-0000-0000-000000000000___leaf.JPG").write_bytes(b"")
    fp = tmp_path / "fp"
    fp.mkdir()
    rows = ["path,label,fieldplant_class,group,photo"]
    rows += [f"m/{i}.jpg,Cassava___Mosaic_disease,Cassava Mosaic,mosaic{i},{i}.jpg" for i in range(60)]
    rows += [f"s/{i}.jpg,Corn_(maize)___Smut,Corn Smut,smut{i},s{i}.jpg" for i in range(3)]
    (fp / "manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(pv), str(fp)


def options(datasets, **overrides):
    pv, fp = datasets
    values = dict(plantvillage=[pv], fieldplant=fp, train_on="both", pv_per_class=5, min_field_samples=10, seed=0)
    return argparse.Namespace(**{**values, **overrides})


def test_plantvillage_is_capped_and_rare_field_classes_are_dropped(datasets):
    splits, classes, dropped = train_transfer.collect(options(datasets))
    per_class = collections.Counter(it["label"] for it in splits["train"] if it["domain"] == "plantvillage")
    assert max(per_class.values()) <= 5
    assert classes == ["Apple___healthy", "Cassava___Mosaic_disease", "Tomato___healthy"]
    assert [label for label, _ in dropped] == ["Corn_(maize)___Smut"]


def test_no_group_is_split_across_train_val_and_test(datasets):
    splits, _, _ = train_transfer.collect(options(datasets, pv_per_class=1000))
    seen = {}
    for name, items in splits.items():
        for it in items:
            assert seen.setdefault(it["group"], name) == name


def test_a_lab_only_model_is_still_tested_on_field_photos(datasets):
    splits, classes, _ = train_transfer.collect(options(datasets, train_on="plantvillage"))
    assert {it["domain"] for it in splits["train"]} == {"plantvillage"}
    assert "fieldplant" in {it["domain"] for it in splits["test"]}
    assert classes == ["Apple___healthy", "Tomato___healthy"]


def test_class_weights_favour_rare_classes():
    items = [{"label": "common"}] * 90 + [{"label": "rare"}] * 10
    index = {"common": 0, "rare": 1}
    weights = train_transfer.class_weights(items, index, "sqrt")
    assert weights[1] > weights[0]
    assert (weights[0] + weights[1]) / 2 == pytest.approx(1)
    assert train_transfer.class_weights(items, index, "none") is None


def test_history_survives_as_csv_and_reports_the_last_phases_best_epoch(tmp_path):
    logs = [tmp_path / "history_phase1.csv", tmp_path / "history_phase2.csv"]
    logs[0].write_text("epoch,accuracy,loss,val_accuracy,val_loss\n0,0.7,0.9,0.80,0.50\n")
    logs[1].write_text("epoch,accuracy,loss,val_accuracy,val_loss\n1,0.8,0.5,0.90,0.30\n2,0.9,0.4,0.91,0.40\n")

    phases = train_transfer.read_history([str(p) for p in logs] + [str(tmp_path / "missing.csv")])
    assert [len(p) for p in phases] == [1, 2]
    assert train_transfer.training_summary(phases, finalized=True) == {
        "epochs_phase1": 1, "epochs_phase2": 2, "best_epoch": 2,
        "best_val_accuracy": 0.9, "best_val_loss": 0.3, "finalized_from_checkpoint": True,
    }
