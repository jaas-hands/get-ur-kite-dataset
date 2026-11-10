from types import SimpleNamespace

import numpy as np
import pytest

from kite_data.eval_metrics import (Thresholds, average_hands, baseline, evaluate_hand,
                                   normal_error_deg, summarize)

pytestmark = pytest.mark.no_assets


class Sampler:
    """Two independently moving points with known positions and collision behavior."""

    hand = SimpleNamespace(n_dof=2)
    lower, upper = np.array([-1., -1.]), np.array([1., 1.])
    tpl_pos = np.zeros((2, 3))
    collider = SimpleNamespace(collides=lambda q: bool(q[0] > .5))

    def contact_points(self, q, pos, rotation, tidx):
        points = np.zeros((*tidx.shape, 3))
        points[..., 0] = q[np.arange(len(q))[:, None], np.maximum(tidx, 0)]
        points = np.einsum("bij,bsj->bsi", rotation, points) + pos[:, None]
        normals = np.broadcast_to(rotation[:, None, :, 0], points.shape).copy()
        return points, normals


@pytest.fixture
def case():
    ex = {"template_idx": np.array([[0, 1, -1, -1, -1, -1]] * 3), "k": np.full(3, 2),
          "contact_pos": np.zeros((3, 6, 3)), "contact_normal": np.tile([1., 0, 0], (3, 6, 1))}
    for tag in ("goal", "init"):
        ex[f"joints_{tag}"] = np.zeros((3, 2))
        ex[f"wrist_pos_{tag}"] = np.zeros((3, 3))
        ex[f"wrist_quat_{tag}"] = np.tile([1., 0, 0, 0], (3, 1))
    ex["wrist_pos_init"][:, 0] = .02
    return Sampler(), ex


def test_goal_and_no_motion(case):
    sampler, ex = case
    goal = evaluate_hand(sampler, ex, baseline(ex, "goal"))
    initial = evaluate_hand(sampler, ex, baseline(ex, "init"))
    assert all(r["mean_position_mm"] == 0 and r["valid_contact_success"] for r in goal)
    assert all(r["mean_position_mm"] == 20 and not r["contact_success"] for r in initial)


def test_known_translation_and_normal_angles(case):
    sampler, ex = case
    pred = baseline(ex, "goal")
    pred["wrist_pos"][:, 1] = .01
    rows = evaluate_hand(sampler, ex, pred)
    assert all(r["mean_position_mm"] == 10 and r["mean_normal_deg"] == 0 for r in rows)
    assert np.allclose(normal_error_deg([[2, 0, 0], [0, 3, 0], [-1, 0, 0]], [[1, 0, 0]] * 3), [0, 90, 180])


def test_success_requires_every_contact(case):
    sampler, ex = case
    pred = baseline(ex, "goal")
    pred["joints"][:, 0] = .019
    row = evaluate_hand(sampler, ex, pred)[0]
    assert row["mean_position_mm"] == 9.5
    assert not row["contact_success"]


def test_padded_contacts_and_joints_are_ignored(case):
    sampler, ex = case
    pred = baseline(ex, "goal")
    pred["joints"] = np.pad(pred["joints"], ((0, 0), (0, 14)), constant_values=np.nan)
    ex["contact_pos"][:, 2:] = np.nan
    ex["contact_normal"][:, 2:] = np.nan
    assert all(r["valid_contact_success"] for r in evaluate_hand(sampler, ex, pred))


@pytest.mark.parametrize("key,value", [("wrist_pos", np.nan), ("joints", np.inf), ("wrist_quat", 0)])
def test_invalid_predictions_remain_in_success_denominator(case, key, value):
    sampler, ex = case
    pred = baseline(ex, "goal")
    pred[key][0] = value
    rows = evaluate_hand(sampler, ex, pred)
    assert rows[0]["invalid_prediction"] and rows[0]["mean_position_mm"] is None
    assert rows[0]["self_collision"] is None
    result = summarize(rows)
    assert result["invalid"] == 1 and result["evaluated"] == 2
    assert result["valid_contact_success_pct"] == pytest.approx(200 / 3)


def test_limits_and_collision_are_separate(case):
    sampler, ex = case
    pred = baseline(ex, "goal")
    pred["joints"][0, 0] = .6
    pred["joints"][1, 1] = 1.1
    rows = evaluate_hand(sampler, ex, pred, Thresholds(position_mm=2000))
    assert rows[0]["self_collision"] and not rows[0]["joint_limit_violation"]
    assert rows[1]["joint_limit_violation"] and not rows[1]["self_collision"]
    assert rows[0]["contact_success"] and not rows[0]["valid_contact_success"]


def test_quaternion_sign_and_batch_size(case):
    sampler, ex = case
    pred = baseline(ex, "goal")
    pred["wrist_quat"] *= -1
    assert evaluate_hand(sampler, ex, pred, batch_size=1) == evaluate_hand(sampler, ex, pred, batch_size=7)


def test_bad_targets_fail_loudly(case):
    sampler, ex = case
    ex["k"][0] = 3
    with pytest.raises(ValueError, match="counts"):
        evaluate_hand(sampler, ex, baseline(ex, "goal"))
    ex["k"][0] = 2
    ex["contact_normal"][0, 0] = 0
    with pytest.raises(ValueError, match="nonzero"):
        evaluate_hand(sampler, ex, baseline(ex, "goal"))


def test_aggregation_weights_hands_not_examples(case):
    sampler, ex = case
    a = summarize(evaluate_hand(sampler, ex, baseline(ex, "goal"))[:1])
    b = summarize(evaluate_hand(sampler, ex, baseline(ex, "init")))
    result = average_hands([a, b])
    assert result["examples"] == 4
    assert result["mean_position_mm"] == 10  # pooled examples would give 15
    assert result["contact_success_pct"] == 50


def test_all_invalid_hand_does_not_disappear(case):
    sampler, ex = case
    good = summarize(evaluate_hand(sampler, ex, baseline(ex, "goal")))
    pred = baseline(ex, "goal")
    pred["joints"][:] = np.nan
    bad = summarize(evaluate_hand(sampler, ex, pred))
    result = average_hands([good, bad])
    assert result["mean_position_mm"] is None
    assert result["invalid_prediction_pct"] == 50
    assert result["valid_contact_success_pct"] == 50


@pytest.mark.parametrize("kwargs", [{"position_mm": -1}, {"normal_deg": 181}, {"joint_tolerance_rad": np.nan}])
def test_bad_thresholds(kwargs):
    with pytest.raises(ValueError):
        Thresholds(**kwargs)
