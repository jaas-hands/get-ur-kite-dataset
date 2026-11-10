import numpy as np
import pytest

from kite_data.eval_io import export_predictions, load_predictions, read_npz

pytestmark = pytest.mark.no_assets


@pytest.fixture
def exported(tmp_path):
    record = {"examples": 3, "hand_id": "hand", "split": "val", "sha256": "data-hash", "hand_sha256": "hand-hash"}
    pred = {"joints": np.arange(6).reshape(3, 2), "wrist_pos": np.zeros((3, 3)),
            "wrist_quat": np.tile([1., 0, 0, 0], (3, 1))}
    path = tmp_path / "hand.npz"
    export_predictions(path, pred, record, ["a", "b"])
    return path, record, pred


def test_shuffled_predictions_align_by_id(exported):
    path, record, expected = exported
    data = read_npz(path)
    for key in ("example_id", "joints", "wrist_pos", "wrist_quat"):
        data[key] = data[key][[2, 0, 1]]
    np.savez(path, **data)
    actual = load_predictions(path, record, ["a", "b"])
    assert all(np.array_equal(actual[k], expected[k]) for k in expected)


@pytest.mark.parametrize("key,value", [("example_id", [0, 0, 2]), ("example_id", [0., 1., 2.]),
    ("dataset_sha256", "wrong"), ("hand_sha256", "wrong"), ("dof_names", ["b", "a"]),
    ("position_unit", "mm"), ("quaternion_order", "xyzw"), ("pose_type", "residual"),
    ("split", "test_geometry"), ("frame", "palm")])
def test_mismatched_handoff_is_rejected(exported, key, value):
    path, record, _ = exported
    data = read_npz(path)
    data[key] = value
    np.savez(path, **data)
    with pytest.raises(ValueError):
        load_predictions(path, record, ["a", "b"])


def test_export_does_not_overwrite(exported):
    path, record, pred = exported
    with pytest.raises(FileExistsError):
        export_predictions(path, pred, record, ["a", "b"])
