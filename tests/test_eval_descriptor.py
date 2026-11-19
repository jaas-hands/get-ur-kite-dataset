import copy
import json

import numpy as np
import pytest

from kite_data.eval_geometry import DescriptorHand
from kite_data.eval_metrics import baseline, evaluate_hand, summarize
from kite_data.evaluate import main
from kite_data.sample import HandSampler

pytestmark = pytest.mark.no_assets


def geometry():
    # A 1 m arm rotates about z; its fixed tip is another 1 m along x.
    desc = {"id": "toy", "n_dof": 1, "dof_names": ["hinge"],
            "links": [{"name": n} for n in ("palm", "arm", "tip")],
            "joints": [
                {"name": "hinge", "parent_link": "palm", "child_link": "arm", "locked": False,
                 "dof": 0, "lower": -2, "upper": 2, "axis": [0, 0, 1],
                 "origin_xyz": [1, 0, 0], "origin_quat_wxyz": [1, 0, 0, 0]},
                {"name": "fixed", "parent_link": "arm", "child_link": "tip", "locked": True,
                 "dof": -1, "axis": [0, 0, 1], "origin_xyz": [1, 0, 0],
                 "origin_quat_wxyz": [1, 0, 0, 0]}]}
    tpl = {"pos": np.zeros((2, 3)), "normal": np.tile([1., 0, 0], (2, 1)),
           "link": np.array([2, 2]), "region": np.zeros(2, int)}
    tpl["pos"][1, 2] = .1
    return desc, tpl


def test_descriptor_fk_has_known_rotation_translation_and_fixed_link():
    desc, _ = geometry()
    hand = DescriptorHand(desc)
    wrist = np.eye(4)
    wrist[:3, 3] = [3, 4, 5]
    result = hand.fk([[0], [np.pi / 2]], wrist)["tip"]
    assert np.allclose(result[:, :3, 3], [[5, 4, 5], [4, 5, 5]])
    assert np.allclose(result[1, :3, 0], [0, 1, 0])
    assert hand.link_dof_ancestors("tip") == [0]
    bad = copy.deepcopy(desc)
    bad["dof_names"] = ["wrong"]
    with pytest.raises(ValueError, match="joint order"):
        DescriptorHand(bad)


def test_contact_only_cli_without_xml_and_prediction_roundtrip(tmp_path):
    desc, tpl = geometry()
    hand_dir = tmp_path / "hands" / "toy"
    hand_dir.mkdir(parents=True)
    (hand_dir / "hand.json").write_text(json.dumps(desc))
    np.savez(hand_dir / "template.npz", **tpl)
    sampler = HandSampler(DescriptorHand(desc), tpl, check_collisions=False)
    ex = {"hand_id": "toy", "split": "val", "k": np.array([2]),
          "template_idx": np.array([[0, 1, -1, -1, -1, -1]])}
    for tag in ("goal", "init"):
        ex["joints_" + tag] = np.array([[2.]])
        ex["wrist_pos_" + tag] = np.zeros((1, 3))
        ex["wrist_quat_" + tag] = np.array([[1., 0, 0, 0]])
    ex["contact_pos"], ex["contact_normal"] = sampler.contact_points(
        ex["joints_goal"], ex["wrist_pos_goal"], np.eye(3)[None], ex["template_idx"])
    pred = baseline(ex, "goal")
    pred["joints"] = np.array([[3.]])
    raw = summarize(evaluate_hand(sampler, ex, pred))
    clamped = summarize(evaluate_hand(sampler, ex, pred, joint_policy="clamp"))
    assert raw["contact_success_pct"] == 0
    assert clamped["contact_success_pct"] == 100
    assert clamped["joint_limit_violation_pct"] == 100
    assert clamped["self_collision_pct"] is None and clamped["valid_contact_success_pct"] is None
    pred["joints"][0, 0] = np.inf
    assert evaluate_hand(sampler, ex, pred, joint_policy="clamp")[0]["invalid_prediction"]
    with pytest.raises(ValueError, match="cannot generate"):
        sampler.contact_joints(1, np.random.default_rng(0))
    data_dir = tmp_path / "examples" / "val"
    data_dir.mkdir(parents=True)
    np.savez(data_dir / "toy.npz", **ex)
    common = ["--data", str(data_dir.parent), "--hands", str(hand_dir.parent), "--contact-only"]
    main(["export", *common, "--baseline", "goal", "--out", str(tmp_path / "preds")])
    for label, source in [("goal", ["--baseline", "goal"]), ("roundtrip", ["--predictions", str(tmp_path / "preds")])]:
        main(["run", *common, *source, "--run-id", label, "--out", str(tmp_path / label),
              "--joint-policy", "clamp", "--plot-count", "1"])
    summary = json.loads((tmp_path / "goal" / "summary.json").read_text())
    assert summary == json.loads((tmp_path / "roundtrip" / "summary.json").read_text())
    assert summary[0]["contact_success_pct"] == 100
    assert summary[0]["valid_contact_success_pct"] is None
    assert "NOT performed" in (tmp_path / "goal" / "report.md").read_text()
    main(["compare", "--runs", f"goal={tmp_path / 'goal'}", f"roundtrip={tmp_path / 'roundtrip'}",
          "--out", str(tmp_path / "compare")])
