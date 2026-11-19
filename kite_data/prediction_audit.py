"""Import Anushka's sample predictions, score both models, and cross-check metrics."""

import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np

from . import config
from .eval_io import (POSE_KEYS, code_version, discover, export_predictions, new_directory,
                      read_npz, sha256, source_record, write_json)
from .sample_audit import audit_sample, import_archive

MODELS = ("get_joint_tokens", "shared_slot_baseline")
# Position: 1 micrometre. Angles: allow float32 acos rounding near parallel normals.
# Discrete successes/violations must agree exactly, regardless of numeric tolerance.
METRICS = {"contact_mm": ("mean_position_mm", .001),
           "normal_deg": ("mean_normal_deg", .05),
           "success": ("contact_success", 0),
           "limit_violation": ("joint_limit_violation", 0)}


def convert_prediction(raw, record, names):
    """Use the producer's documented row order; do not silently remap joint columns."""
    for key in ("hand_id", "split"):
        if key not in raw or np.asarray(raw[key]).shape != () or str(raw[key]) != record[key]:
            raise ValueError(f"prediction {key} differs from sample")
    n = record["examples"]
    prediction = {}
    for key, width in zip(POSE_KEYS, (len(names), 3, 4)):
        a = np.asarray(raw[key + "_pred"])
        if a.shape != (n, width) or not np.isfinite(a).all():
            raise ValueError(f"invalid {key}_pred shape or values")
        prediction[key] = a
    if np.any(np.abs(np.linalg.norm(prediction["wrist_quat"], axis=1) - 1) > 1e-3):
        raise ValueError("predicted quaternions must have unit norm")
    return prediction


def read_rows(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    indexed = {(r["split"], r["hand_id"], int(r["example_id"])): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate evaluated example IDs")
    return indexed


def metric_delta(row, raw, prefix, index, metric):
    field, tolerance = METRICS[metric]
    value = float(raw[prefix + "_" + metric][index])
    actual = row[field]
    if metric in ("success", "limit_violation"):
        if actual not in ("True", "False") or value not in (0, 1):
            raise ValueError("missing or invalid binary metric")
        actual = float(actual == "True")
    else:
        actual = float(actual)
    delta = abs(actual - value)
    return delta, bool(np.isfinite(delta) and delta <= tolerance)


def cross_check(source, out, records):
    initial = read_rows(out / "init/examples.csv")
    checks = []
    for model in MODELS:
        rows = read_rows(out / model / "examples.csv")
        for record in records:
            raw = read_npz(source / model / record["path"])
            for i in range(record["examples"]):
                key = (record["split"], record["hand_id"], i)
                for prefix, evaluated in (("model", rows), ("noop", initial)):
                    for metric in METRICS:
                        delta, ok = metric_delta(evaluated[key], raw, prefix, i, metric)
                        checks.append({"model": model, "split": key[0], "hand_id": key[1],
                                       "example_id": i, "prediction": prefix, "metric": metric,
                                       "abs_delta": delta, "passed": ok})
    with (out / "crosscheck.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    goals = json.loads((out / "goal/summary.json").read_text())
    result = {
        "metric_checks": len(checks), "metric_failures": sum(not r["passed"] for r in checks),
        "max_abs_delta": {k: max(r["abs_delta"] for r in checks if r["metric"] == k) for k in METRICS},
        "absolute_tolerances": {k: v[1] for k, v in METRICS.items()},
        "goal_pass": all(r["mean_position_mm"] < .001 and r["mean_normal_deg"] < .01
                         and r["contact_success_pct"] == 100 for r in goals),
        "scope": "contact/normal error, contact success and raw joint-limit flags; wrist metrics and collisions not cross-checked",
        "alignment": "producer-documented row order plus per-row no-motion agreement; original prediction files have no dataset hashes or row IDs",
    }
    write_json(out / "verification.json", result)
    if result["metric_failures"] or not result["goal_pass"]:
        raise ValueError(f"cross-check failed; inspect {out / 'crosscheck.csv'}")
    print(json.dumps(result, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = new_directory(args.out.resolve())
    sample = import_archive(args.sample, out / "sample_input")
    source = import_archive(args.predictions, out / "prediction_input", "leap_sample_v1_predictions")
    manifest, audit = audit_sample(sample)
    data, hands = sample / "examples/kite_code", sample / "hands"
    files = discover(data, list(manifest["hands"]))
    records = [source_record(p, data, hands, contact_only=True)[1] for p in files]
    write_json(out / "input_audit.json", {**audit, "code": code_version(),
               "archives": {str(p.resolve()): sha256(p) for p in (args.sample, args.predictions)},
               "source_prediction_files": {str(p.relative_to(source)): sha256(p)
                                           for p in sorted(source.rglob("*")) if p.is_file()}})
    for model in MODELS:
        expected = {r["path"] for r in records}
        actual = {str(p.relative_to(source / model)) for p in (source / model).rglob("*.npz")}
        if expected != actual:
            raise ValueError(f"{model}: prediction files do not match sample files")
        for record in records:
            names = json.loads((hands / record["hand_id"] / "hand.json").read_text())["dof_names"]
            raw = read_npz(source / model / record["path"])
            prediction = convert_prediction(raw, record, names)
            for prefix in ("model", "noop"):
                for metric in METRICS:
                    a = raw[prefix + "_" + metric]
                    if a.shape != (record["examples"],) or not np.isfinite(a).all():
                        raise ValueError("missing or invalid supplied metric rows")
            export_predictions(out / "adapted" / model / record["path"], prediction, record, names)
        summary = json.loads((source / model / "summary.json").read_text())
        if summary["data"] != "leap_sample_v1" or summary["frame"] != "world":
            raise ValueError("unsupported prediction data/frame")
        write_json(out / f"{model}_metadata.json", {
            "producer_summary": summary, "provenance_status": "producer-supplied; checkpoint not provided or rerun",
            "prediction_archive_sha256": sha256(args.predictions),
            "row_mapping": "README specifies row i matches original sample row i; IDs/hashes added during import",
            "sample_scope": "first 32 rows per LEAP hand; training rows are in-sample; not a full benchmark",
        })
    common = ["--data", str(data), "--hands", str(hands), "--contact-only",
              "--splits", *manifest["hands"], "--joint-policy", "clamp", "--joint-tolerance-rad", "0.001"]
    commands = []

    def run(*arguments):
        command = [sys.executable, "-m", "kite_data.evaluate", *arguments]
        commands.append(shlex.join(command))
        (out / "commands.sh").write_text("#!/bin/sh\nset -eu\ncd " + shlex.quote(str(config.REPO_ROOT))
            + "\n# Recorded commands; use fresh output folders to rerun.\n" + "\n".join(commands) + "\n")
        with (out / "run.log").open("a") as log:
            log.write("\n$ " + commands[-1] + "\n")
            log.flush()
            subprocess.run(command, cwd=config.REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print("Completed:", arguments[0], arguments[-1], flush=True)

    for name in ("goal", "init"):
        run("run", *common, "--baseline", name, "--plot-count", "0", "--run-id", name, "--out", str(out / name))
    for model in MODELS:
        run("run", *common, "--predictions", str(out / "adapted" / model), "--plot-count", "2",
            "--metadata", str(out / f"{model}_metadata.json"), "--run-id", model, "--out", str(out / model))
    cross_check(source, out, records)
    run("compare", "--runs", f"GET={out / MODELS[0]}", f"Shared-slot={out / MODELS[1]}",
        f"No-motion={out / 'init'}", "--out", str(out / "comparison"))


if __name__ == "__main__":
    main()
