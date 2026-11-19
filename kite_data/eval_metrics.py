"""Contact metrics for absolute wrist poses and joint angles. No model required."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Thresholds:
    """Proposed defaults; agree on these before running final test comparisons."""

    position_mm: float = 10.0
    normal_deg: float = 15.0
    joint_tolerance_rad: float = 1e-6

    def __post_init__(self):
        values = (self.position_mm, self.normal_deg, self.joint_tolerance_rad)
        if not all(np.isfinite(v) and v >= 0 for v in values) or self.normal_deg > 180:
            raise ValueError("thresholds must be finite and nonnegative; normal_deg must be <= 180")


def normal_error_deg(actual, target):
    """Unsigned angle between outward normals; zero vectors are invalid."""
    a, b = np.asarray(actual, float), np.asarray(target, float)
    an, bn = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    if np.any(an == 0) or np.any(bn == 0):
        raise ValueError("contact normals must be nonzero")
    dot = np.sum((a / an[..., None]) * (b / bn[..., None]), axis=-1)
    return np.rad2deg(np.arccos(np.clip(dot, -1, 1)))


def baseline(examples, name):
    if name not in ("goal", "init"):
        raise ValueError("baseline must be goal or init")
    return {key: np.asarray(examples[f"{key}_{name}"], float)
            for key in ("joints", "wrist_pos", "wrist_quat")}


def validate_examples(sampler, ex):
    """Reject malformed targets rather than scoring them as model failures."""
    tidx = np.asarray(ex["template_idx"])
    if tidx.ndim != 2 or not np.issubdtype(tidx.dtype, np.integer):
        raise ValueError("template_idx must be a 2D integer array")
    n, slots = tidx.shape
    if n == 0 or slots != 6:
        raise ValueError("examples must contain at least one row with six contact slots")
    valid = tidx >= 0
    k = np.asarray(ex["k"])
    if (k.shape != (n,) or not np.array_equal(valid.sum(1), k)
            or np.any((k < 2) | (k > 6))
            or np.any(tidx < -1) or np.any(tidx >= len(sampler.tpl_pos))):
        raise ValueError("contact counts or template indices are invalid")
    for key in ("contact_pos", "contact_normal"):
        a = np.asarray(ex[key])
        if a.shape != (n, slots, 3) or not np.isfinite(a[valid]).all():
            raise ValueError(f"{key} must contain finite valid contacts with shape (n, 6, 3)")
    if np.any(np.linalg.norm(ex["contact_normal"][valid], axis=-1) < 1e-12):
        raise ValueError("target contact normals must be nonzero")
    return valid


def evaluate_hand(sampler, ex, prediction, thresholds=Thresholds(), batch_size=512, joint_policy="raw"):
    """Return one row per example, in dataset order; padded joint columns are ignored.

    Invalid predictions fail both success tests. Their errors and physical checks are
    unknown (None), with an explicit invalid flag, rather than silently discarded.
    """
    from .sample import _matrix

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if joint_policy not in ("raw", "clamp"):
        raise ValueError("joint_policy must be raw or clamp")
    valid = validate_examples(sampler, ex)
    n, nd = len(valid), sampler.hand.n_dof
    q = np.asarray(prediction["joints"], float)
    pos = np.asarray(prediction["wrist_pos"], float)
    quat = np.asarray(prediction["wrist_quat"], float)
    if q.shape not in ((n, nd), (n, 16)) or pos.shape != (n, 3) or quat.shape != (n, 4):
        raise ValueError(f"predictions need joints (n, {nd}) or (n, 16), wrist_pos (n, 3), wrist_quat (n, 4)")
    q = q[:, :nd]
    scored_q = np.clip(q, sampler.lower, sampler.upper) if joint_policy == "clamp" else q
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.linalg.norm(quat, axis=1)
    good = (np.isfinite(q).all(1) & np.isfinite(pos).all(1) & np.isfinite(quat).all(1)
            & (np.abs(norms - 1) <= 1e-3))
    rows = [{"example_id": i, "k": int(ex["k"][i]), "n_dof": nd,
             "mean_position_mm": None, "max_position_mm": None,
             "mean_normal_deg": None, "max_normal_deg": None,
             "contact_success": False, "joint_limit_violation": None,
             "self_collision": None, "invalid_prediction": True,
             "valid_contact_success": False if sampler.collider is not None else None} for i in range(n)]
    indices = np.flatnonzero(good)
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start + batch_size]
        with np.errstate(over="ignore", invalid="ignore"):
            x, normals = sampler.contact_points(scored_q[ids], pos[ids], _matrix(quat[ids]), ex["template_idx"][ids])
            errors = np.linalg.norm(x - ex["contact_pos"][ids], axis=-1) * 1000
        for j, i in enumerate(ids):
            mask = valid[i]
            pe = errors[j, mask]
            ne = normal_error_deg(normals[j, mask], ex["contact_normal"][i, mask])
            if not (np.isfinite(pe).all() and np.isfinite(ne).all()):
                continue
            limit = bool(np.any((q[i] < sampler.lower - thresholds.joint_tolerance_rad)
                                | (q[i] > sampler.upper + thresholds.joint_tolerance_rad)))
            collision = bool(sampler.collider.collides(scored_q[i])) if sampler.collider is not None else None
            success = bool(np.all(pe <= thresholds.position_mm) and np.all(ne <= thresholds.normal_deg))
            rows[i].update(mean_position_mm=float(pe.mean()), max_position_mm=float(pe.max()),
                           mean_normal_deg=float(ne.mean()), max_normal_deg=float(ne.max()),
                           contact_success=success, joint_limit_violation=limit,
                           self_collision=collision, invalid_prediction=False,
                           valid_contact_success=(success and not limit and not collision) if collision is not None else None)
    return rows


ERRORS = ("mean_position_mm", "max_position_mm", "mean_normal_deg", "max_normal_deg")
RATES = ("contact_success", "valid_contact_success", "invalid_prediction",
         "joint_limit_violation", "self_collision")


def summarize(rows):
    """Summary within one hand (or one diagnostic group of that hand)."""
    finite = [r for r in rows if not r["invalid_prediction"]]
    out = {"examples": len(rows), "evaluated": len(finite), "invalid": len(rows) - len(finite)}
    for key in ERRORS:
        out[key] = float(np.mean([r[key] for r in finite])) if finite else None
    for key in RATES:
        denominator = finite if key in ("joint_limit_violation", "self_collision") else rows
        values = [r[key] for r in denominator]
        out[f"{key}_pct"] = float(np.mean(values) * 100) if values and all(v is not None for v in values) else None
    return out


def average_hands(summaries):
    """Equal weight per hand. Undefined hand metrics stay undefined at split level."""
    out = {key: sum(s[key] for s in summaries) for key in ("examples", "evaluated", "invalid")}
    out["hands"] = len(summaries)
    for key in (*ERRORS, *(f"{k}_pct" for k in RATES)):
        vals = [s[key] for s in summaries]
        out[key] = float(np.mean(vals)) if vals and all(v is not None for v in vals) else None
    return out
