import json
import numpy as np
import pytest

from kite_data.eval_io import read_npz
from kite_data.evaluate import main


@pytest.fixture(scope="module")
def sample_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("evaluation")
    main(["sample", "--out", str(root / "data"), "--hands", str(root / "hands"),
          "--hands-count", "1", "--count", "8"])
    return root


def test_real_hand_evaluation_and_prediction_roundtrip(sample_data):
    root = sample_data
    common = ["--data", str(root / "data"), "--hands", str(root / "hands")]
    main(["run", *common, "--baseline", "goal", "--run-id", "goal", "--out", str(root / "goal"), "--plot-count", "1"])
    main(["run", *common, "--baseline", "init", "--run-id", "init", "--out", str(root / "init"), "--plot-count", "0"])
    goal = json.loads((root / "goal" / "summary.json").read_text())[0]
    initial = json.loads((root / "init" / "summary.json").read_text())[0]
    assert goal["mean_position_mm"] < 1e-3
    assert goal["mean_normal_deg"] < .01
    assert goal["contact_success_pct"] == 100
    assert initial["mean_position_mm"] > goal["mean_position_mm"]
    main(["export", *common, "--out", str(root / "predictions")])
    main(["run", *common, "--predictions", str(root / "predictions"), "--run-id", "roundtrip",
          "--out", str(root / "roundtrip"), "--plot-count", "0"])
    assert json.loads((root / "roundtrip" / "summary.json").read_text())[0] == initial
    main(["compare", "--runs", f"goal={root / 'goal'}", f"init={root / 'init'}", f"init={root / 'roundtrip'}",
          "--out", str(root / "comparison")])
    for name in ("report.md", "methods.csv", "comparison.png", "comparison.json"):
        assert (root / "comparison" / name).stat().st_size > 0
    manifest = json.loads((root / "goal" / "manifest.json").read_text())
    assert manifest["dataset"][0]["hand_files"]["hand.xml"]
    assert manifest["code"]["source_sha256"]["eval_metrics.py"]
    assert list((root / "goal" / "poses").rglob("*.png"))


def test_failures_are_clear_and_do_not_overwrite(sample_data, tmp_path):
    root = sample_data
    common = ["--data", str(root / "data"), "--hands", str(root / "hands")]
    out = tmp_path / "existing"
    out.mkdir()
    (out / "keep.txt").write_text("keep")
    with pytest.raises(SystemExit) as exc:
        main(["export", *common, "--out", str(out)])
    assert exc.value.code == 2 and (out / "keep.txt").read_text() == "keep"
    with pytest.raises(SystemExit):
        main(["run", *common, "--baseline", "goal", "--run-id", "empty", "--splits", "missing", "--out", str(tmp_path / "empty")])
    with pytest.raises(SystemExit):
        main(["run", *common, "--predictions", str(tmp_path / "absent"), "--run-id", "missing", "--out", str(tmp_path / "bad")])


def test_compare_rejects_different_protocols(tmp_path):
    for name, threshold in [("a", 10), ("b", 20)]:
        out = tmp_path / name
        out.mkdir()
        (out / "manifest.json").write_text(json.dumps({"protocol": {"threshold": threshold}, "dataset": []}))
        (out / "summary.json").write_text("[]")
    with pytest.raises(SystemExit):
        main(["compare", "--runs", f"a={tmp_path / 'a'}", f"b={tmp_path / 'b'}", "--out", str(tmp_path / "result")])


@pytest.mark.no_assets
def test_split_selection_and_aggregation(tmp_path):
    from kite_data.eval_io import discover
    from kite_data.eval_metrics import ERRORS, RATES
    from kite_data.eval_report import grouped_summary, split_summary

    with pytest.raises(ValueError, match="only once"):
        discover(tmp_path, ["val", "val"])
    rows = []
    for split, hand, n, error in [("val", "a", 1, 0), ("val", "b", 3, 20), ("test_geometry", "c", 2, 5)]:
        for _ in range(n):
            row = {"split": split, "hand_id": hand, **{k: error for k in ERRORS}, **{k: False for k in RATES}}
            rows.append(row)
    result = {r["split"]: r for r in split_summary(grouped_summary(rows, ("split", "hand_id")))}
    assert result["val"]["mean_position_mm"] == 10
    assert result["test_geometry"]["mean_position_mm"] == 5


def test_invalid_model_output_is_reported(sample_data, tmp_path):
    root = sample_data
    common = ["--data", str(root / "data"), "--hands", str(root / "hands")]
    preds = tmp_path / "predictions"
    main(["export", *common, "--out", str(preds)])
    path = next(preds.rglob("*.npz"))
    data = read_npz(path)
    data["wrist_quat"][0] = 0
    np.savez_compressed(path, **data)
    out = tmp_path / "invalid"
    main(["run", *common, "--predictions", str(preds), "--run-id", "invalid", "--out", str(out), "--plot-count", "1"])
    result = json.loads((out / "summary.json").read_text())[0]
    assert result["invalid"] == 1 and result["invalid_prediction_pct"] == 12.5
