"""Prediction handoff and provenance for evaluation runs."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from . import config

POSE_KEYS = ("joints", "wrist_pos", "wrist_quat")
CONVENTIONS = {"schema_version": 1, "pose_type": "absolute", "frame": "world",
               "position_unit": "metres", "joint_unit": "radians", "quaternion_order": "wxyz"}


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hand_files(hand_dir):
    return {name: sha256(hand_dir / name) for name in ("hand.json", "template.npz", "hand.xml")}


def hand_digest(hashes):
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def source_record(path, root, hands_dir):
    data = read_npz(path)
    hand_id, split = str(data["hand_id"]), str(data["split"])
    relative = path.relative_to(root)
    if relative.as_posix() != f"{split}/{hand_id}.npz":
        raise ValueError(f"{path}: file path must match its split and hand_id")
    hashes = hand_files(hands_dir / hand_id)
    return data, {"path": relative.as_posix(), "sha256": sha256(path),
                  "hand_id": hand_id, "split": split, "examples": len(data["k"]),
                  "hand_files": hashes, "hand_sha256": hand_digest(hashes)}


def discover(root, splits):
    if len(set(splits)) != len(splits):
        raise ValueError("each split must be selected only once")
    paths = []
    for split in splits:
        if split in (".", "..") or "/" in split or "\\" in split:
            raise ValueError("split names must be directory names")
        files = sorted((root / split).glob("*.npz"))
        if not files:
            raise ValueError(f"no examples in {root / split}; get a sample or run the sample command")
        paths.extend(files)
    return paths


def export_predictions(path, prediction, record, dof_names):
    """Write an interchange file. Use this from a model's inference script."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    n = record["examples"]
    np.savez_compressed(path, **{k: prediction[k] for k in POSE_KEYS},
                        example_id=np.arange(n, dtype=np.int64), hand_id=record["hand_id"],
                        split=record["split"], dataset_sha256=record["sha256"],
                        hand_sha256=record["hand_sha256"], dof_names=np.array(dof_names), **CONVENTIONS)


def load_predictions(path, record, dof_names):
    """Require a complete one-to-one handoff; row order may differ."""
    p = read_npz(path)
    expected = {**CONVENTIONS, "hand_id": record["hand_id"], "split": record["split"],
                "dataset_sha256": record["sha256"], "hand_sha256": record["hand_sha256"]}
    for key, value in expected.items():
        if key not in p or np.asarray(p[key]).shape != () or p[key].item() != value:
            raise ValueError(f"{path}: {key} does not match the dataset/conventions")
    if "dof_names" not in p or not np.array_equal(p["dof_names"], dof_names):
        raise ValueError(f"{path}: dof_names must match the hand's canonical joint order")
    ids = np.asarray(p.get("example_id", []))
    n = record["examples"]
    if (ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer)
            or not np.array_equal(np.sort(ids), np.arange(n))):
        raise ValueError(f"{path}: example_id must contain every dataset row exactly once")
    order = np.argsort(ids)
    for key in POSE_KEYS:
        if key not in p or p[key].ndim != 2 or len(p[key]) != n:
            raise ValueError(f"{path}: {key} must have one row per example")
    return {key: p[key][order] for key in POSE_KEYS}


def code_version():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=config.REPO_ROOT, text=True).strip()
    try:
        revision, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        revision, status = None, None
    # Includes uncommitted evaluator code, which a commit ID alone cannot identify.
    files = sorted((config.REPO_ROOT / "kite_data").glob("*.py"))
    return {"commit": revision, "dirty": bool(status) if status is not None else None,
            "source_sha256": {p.name: sha256(p) for p in files}}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def new_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"{path} is not empty; choose a new output directory")
    path.mkdir(parents=True, exist_ok=True)
    return path
