import io
import tarfile

import numpy as np
import pytest

from kite_data.prediction_audit import convert_prediction, metric_delta
from kite_data.sample_audit import import_archive

pytestmark = pytest.mark.no_assets


def example():
    raw = {"hand_id": np.array("hand"), "split": np.array("val"),
           "joints_pred": np.array([[.1, .2], [.3, .4]], dtype=np.float32),
           "wrist_pos_pred": np.zeros((2, 3), dtype=np.float32),
           "wrist_quat_pred": np.array([[1., 0, 0, 0], [1., 0, 0, 0]])}
    return raw, {"hand_id": "hand", "split": "val", "examples": 2}, ["j0", "j1"]


def test_import_preserves_values_and_joint_order():
    raw, record, names = example()
    converted = convert_prediction(raw, record, names)
    for key, value in converted.items():
        np.testing.assert_array_equal(value, raw[key + "_pred"])


@pytest.mark.parametrize("change", ["hand", "split", "rows", "columns", "nan", "quaternion"])
def test_import_rejects_mismatched_or_invalid_prediction(change):
    raw, record, names = example()
    if change in ("hand", "split"):
        raw["hand_id" if change == "hand" else "split"] = np.array("wrong")
    elif change == "rows":
        raw["joints_pred"] = raw["joints_pred"][:1]
    elif change == "columns":
        raw["joints_pred"] = raw["joints_pred"][:, :1]
    elif change == "nan":
        raw["wrist_pos_pred"][0, 0] = np.nan
    else:
        raw["wrist_quat_pred"][0] = 0
    with pytest.raises(ValueError):
        convert_prediction(raw, record, names)


def test_crosscheck_detects_permuted_rows_and_success_mismatch():
    raw = {"model_contact_mm": np.array([5., 10.]), "model_success": np.array([0., 1.])}
    row = {"mean_position_mm": "10", "contact_success": "True"}
    assert metric_delta(row, raw, "model", 1, "contact_mm") == (0, True)
    assert not metric_delta(row, raw, "model", 0, "contact_mm")[1]
    assert not metric_delta(row, raw, "model", 0, "success")[1]


def test_prediction_archive_accepts_expected_root_and_rejects_other_root(tmp_path):
    archive = tmp_path / "predictions.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("leap_sample_v1_predictions/README.md")
        member.size = 2
        tar.addfile(member, io.BytesIO(b"ok"))
    with pytest.raises(ValueError):
        import_archive(archive, tmp_path / "wrong")
    root = import_archive(archive, tmp_path / "correct", "leap_sample_v1_predictions")
    assert (root / "README.md").read_text() == "ok"
