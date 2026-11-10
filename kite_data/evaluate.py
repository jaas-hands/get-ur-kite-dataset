"""Evaluate contact realization: sample, export, run, and compare. See docs/EVALUATION.md."""

import argparse
import csv
import importlib.metadata
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config
from .eval_io import (CONVENTIONS, code_version, discover, export_predictions,
                      load_predictions, new_directory, sha256,
                      source_record, write_json)
from .eval_metrics import Thresholds, baseline, evaluate_hand
from .eval_report import (diagnostic_plot, distribution_plot, grouped_summary,
                          pose_plot, pyplot, result_table, select_examples,
                          split_summary, write_csv, write_report)

DEFAULT_DATA = config.DATA_DIR / "examples" / "eval_smoke"


def load_sampler(hand_dir):
    from .collision import Collider
    from .qa import load_hand
    from .sample import HandSampler

    hand, template = load_hand(hand_dir)
    desc = json.loads((hand_dir / "hand.json").read_text())
    if desc["dof_names"] != [j.name for j in hand.dof_joints]:
        raise ValueError(f"{hand_dir}: joint order differs from this generator")
    collider = Collider(hand, (hand_dir / "hand.xml").read_text())
    return HandSampler(hand, template, collider), template, desc["dof_names"]


def sample(args):
    from .generate import generate_one
    from .splits import make_splits

    if args.count < 1 or args.hands_count < 1:
        raise ValueError("count and hands-count must be positive")
    splits = make_splits()
    ids = splits["hands"]["val"]
    # Spread across validation topologies before taking more geometry variants.
    ids = sorted(ids, key=lambda x: (int(x.rsplit("_g", 1)[1]), x))
    if args.hands_count > len(ids):
        raise ValueError(f"only {len(ids)} validation hands are available")
    out = new_directory(args.out)
    records = []
    for hand_id in ids[:args.hands_count]:
        print(f"Generating {args.count} validation examples for {hand_id}", flush=True)
        generate_one((hand_id, "val", args.count, args.seed, "kite", str(args.hands), str(out)))
        _, record = source_record(out / "val" / f"{hand_id}.npz", out, args.hands)
        records.append(record)
    write_json(out / "sample.json", {"purpose": "validation smoke sample, not final test results",
                                    "seed": args.seed, "split_rules": splits["rules"],
                                    "split_seed": splits["seed"], "files": records, "code": code_version()})
    print(f"Sample ready: {out}")


def export(args):
    out = new_directory(args.out)
    records = []
    for path in discover(args.data, args.splits):
        ex, record = source_record(path, args.data, args.hands)
        names = json.loads((args.hands / record["hand_id"] / "hand.json").read_text())["dof_names"]
        export_predictions(out / record["path"], baseline(ex, args.baseline), record, names)
        records.append(record)
    write_json(out / "export.json", {"source": f"{args.baseline} baseline; not model predictions",
                                    "conventions": CONVENTIONS, "dataset": records})
    print(f"Example prediction files: {out}. Replace only the three pose arrays with model outputs.")


def run(args):
    thresholds = Thresholds(args.position_mm, args.normal_deg, args.joint_tolerance_rad)
    if args.batch_size < 1 or args.plot_count < 0 or args.plot_count > 4:
        raise ValueError("batch-size must be positive and plot-count between 0 and 4")
    paths = discover(args.data, args.splits)
    if args.predictions:
        expected = {p.relative_to(args.data) for p in paths}
        present = {p.relative_to(args.predictions) for s in args.splits for p in (args.predictions / s).glob("*.npz")}
        if present != expected:
            raise ValueError(f"prediction files differ from data: missing={expected - present}, extra={present - expected}")
    metadata = json.loads(args.metadata.read_text()) if args.metadata else {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    out = new_directory(args.out)
    records, hand_rows, diagnostic_rows, display_rows, selections = [], [], [], [], []
    with (out / "examples.csv").open("w", newline="") as stream:
        writer = None
        for path in paths:
            ex, record = source_record(path, args.data, args.hands)
            sampler, template, names = load_sampler(args.hands / record["hand_id"])
            pred_path = args.predictions / record["path"] if args.predictions else None
            pred = load_predictions(pred_path, record, names) if pred_path else baseline(ex, args.baseline)
            rows = evaluate_hand(sampler, ex, pred, thresholds, args.batch_size)
            initial = rows if args.baseline == "init" else evaluate_hand(sampler, ex, baseline(ex, "init"), thresholds, args.batch_size)
            if any(r["invalid_prediction"] for r in initial):
                raise ValueError(f"{path}: dataset contains an invalid starting pose")
            for row, before in zip(rows, initial):
                row.update(run_id=args.run_id, split=record["split"], hand_id=record["hand_id"],
                           initial_position_mm=before["mean_position_mm"])
                row["improvement_mm"] = (before["mean_position_mm"] - row["mean_position_mm"]
                                         if row["mean_position_mm"] is not None else None)
                row["initial_error_bin"] = ("<10 mm" if before["mean_position_mm"] < 10 else
                                            "10–25 mm" if before["mean_position_mm"] < 25 else ">=25 mm")
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            hand_rows.extend(grouped_summary(rows, ("split", "hand_id")))
            for key in ("n_dof", "k", "initial_error_bin"):
                for group in grouped_summary(rows, ("split", "hand_id", key)):
                    value = group.pop(key)
                    diagnostic_rows.append({"group_by": key, "group": value, **group})
            display_rows.extend(rows[::max(1, (len(rows) + 49) // 50)])
            ids = select_examples(rows, args.plot_count)
            if ids:
                image = Path("poses") / record["split"] / f"{record['hand_id']}.png"
                pose_plot(out / image, sampler, template, ex, pred, rows, ids)
                selections.extend({"split": record["split"], "hand_id": record["hand_id"],
                                   "example_id": i, "selection": "first" if j == 0 else "worst_remaining",
                                   "image": str(image)} for j, i in enumerate(ids))
            records.append({**record, "prediction_sha256": sha256(pred_path) if pred_path else None})
            print(f"{record['split']}/{record['hand_id']}: {len(rows)} examples", flush=True)
    splits = split_summary(hand_rows)
    manifest = {"schema_version": 1, "run_id": args.run_id, "baseline": args.baseline,
                "created_at": datetime.now(timezone.utc).isoformat(), "data_root": str(args.data.resolve()),
                "hands_root": str(args.hands.resolve()), "dataset": records, "code": code_version(),
                "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "mujoco", "trimesh")},
                "model": metadata, "protocol": {"thresholds": asdict(thresholds),
                    "threshold_status": "proposed; confirm with team", "conventions": CONVENTIONS,
                    "matching": "labelled_surface_points", "aggregation": "contacts -> examples -> equal-weight hands",
                    "collision": "stored hand.xml: boxes and fingertip pads; hand only",
                    "penetration_tolerance_m": config.PENETRATION_TOL, "quaternion_norm_tolerance": 1e-3,
                    "invalid_predictions": "fail success; errors/physical checks undefined",
                    "error_denominator": "valid predictions; undefined split mean if any hand has none"},
                "plot_selection": "first row, then invalid/worst position error; stable ties by row index"}
    write_csv(out / "hands.csv", hand_rows)
    write_csv(out / "splits.csv", splits)
    write_csv(out / "diagnostics.csv", diagnostic_rows)
    if selections:
        write_csv(out / "plot_selection.csv", selections)
    distribution_plot(out, hand_rows)
    diagnostic_plot(out, display_rows)
    write_json(out / "manifest.json", manifest)
    write_json(out / "summary.json", splits)
    write_report(out, manifest, splits, args.plot_count)
    print(f"Report: {out / 'report.md'}")


def compare(args):
    entries, reference, paths_seen = [], None, set()
    for item in args.runs:
        label, separator, directory = item.partition("=")
        if not separator or not label or not directory:
            raise ValueError("runs must be LABEL=RUN_DIRECTORY")
        path = Path(directory).resolve()
        if path in paths_seen:
            raise ValueError("the same run directory cannot be counted twice")
        paths_seen.add(path)
        manifest = json.loads((path / "manifest.json").read_text())
        signature = {"protocol": manifest["protocol"], "dataset": [
            {k: v for k, v in r.items() if k != "prediction_sha256"} for r in manifest["dataset"]]}
        if reference is not None and signature != reference:
            raise ValueError("runs use different data, hand assets, or metric settings; evaluate them consistently first")
        reference = signature
        summary = json.loads((path / "summary.json").read_text())
        entries.append((label, path, manifest, summary))
    out = new_directory(args.out)
    rows = [{"method": label, "run_id": m["run_id"], **s} for label, _, m, summaries in entries for s in summaries]
    write_csv(out / "runs.csv", rows)
    methods = sorted({e[0] for e in entries})
    splits = sorted({r["split"] for r in rows})
    combined = []
    for split in splits:
        for method in methods:
            group = [r for r in rows if r["split"] == split and r["method"] == method]
            result = {"method": method, "split": split, "runs": len(group)}
            for key in ("mean_position_mm", "mean_normal_deg", "valid_contact_success_pct", "invalid_prediction_pct"):
                values = [r[key] for r in group]
                known = all(v is not None for v in values)
                result[key] = float(np.mean(values)) if known else None
                result[key + "_sd"] = float(np.std(values, ddof=1)) if known and len(values) > 1 else None
            combined.append(result)
    write_csv(out / "methods.csv", combined)
    plt = pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, key, title in zip(axes, ("mean_position_mm", "mean_normal_deg", "valid_contact_success_pct"),
                             ("Position error (mm)", "Normal error (deg)", "Valid contact success (%)")):
        for i, method in enumerate(methods):
            group = [r for r in combined if r["method"] == method]
            x = np.arange(len(splits)) + (i - (len(methods) - 1) / 2) * .15
            y = [np.nan if r[key] is None else r[key] for r in group]
            sd = [0 if r[key + "_sd"] is None else r[key + "_sd"] for r in group]
            ax.errorbar(x, y, yerr=sd, fmt="o", capsize=3, label=method)
        ax.set_xticks(range(len(splits)), [s.removeprefix("test_") for s in splits], rotation=25)
        ax.set_xlim(-.5, len(splits) - .5)
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.suptitle("Run means ± sample SD where multiple runs are supplied")
    fig.savefig(out / "comparison.png", dpi=140)
    plt.close(fig)
    lines = ["# Run comparison", "", "Data, hand assets, and metric settings match across these runs.",
             "Training budgets, encoder choices, and seed independence still need review in each run's metadata.",
             "Repeated labels group runs; they do not establish independent training seeds. One run has no estimated seed variation.", ""]
    for label, path, manifest, summaries in entries:
        lines += [f"## {label}: {manifest['run_id']}", "", result_table(summaries), "", f"Source: {path}", ""]
    lines += ["`methods.csv` reports means and sample SD across supplied runs. These are descriptive comparisons, not significance tests.",
              "Goal is a sanity reference; init is no motion. Neither is a trained decoder."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    write_json(out / "comparison.json", {"runs": [{"label": label, "path": str(path),
                "manifest_sha256": sha256(path / "manifest.json"), "model": m["model"]}
                for label, path, m, _ in entries], "protocol": reference["protocol"]})
    print(f"Comparison: {out / 'report.md'}")


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample", help="generate a small validation sample, sequentially on CPU")
    s.add_argument("--hands-count", type=int, default=2)
    s.add_argument("--count", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, default=DEFAULT_DATA)
    s.add_argument("--hands", type=Path, default=config.DATA_DIR / "hands")
    s.set_defaults(func=sample)
    for name, help_text, function in (("run", "evaluate stored poses or exported predictions", run),
                                      ("export", "write baseline prediction files as a teammate handoff example", export)):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="directory containing split subdirectories")
        p.add_argument("--hands", type=Path, default=config.DATA_DIR / "hands")
        p.add_argument("--splits", nargs="+", default=["val"])
        p.add_argument("--out", type=Path, required=True)
        p.set_defaults(func=function)
        if name == "export":
            p.add_argument("--baseline", choices=["init", "goal"], default="init")
        else:
            choice = p.add_mutually_exclusive_group(required=True)
            choice.add_argument("--baseline", choices=["init", "goal"])
            choice.add_argument("--predictions", type=Path, help="directory mirroring the data's split/hand files")
            p.add_argument("--run-id", required=True)
            p.add_argument("--metadata", type=Path, help="JSON recording checkpoint, encoder, graph setting, seed, budget, selection rule")
            p.add_argument("--position-mm", type=float, default=10)
            p.add_argument("--normal-deg", type=float, default=15)
            p.add_argument("--joint-tolerance-rad", type=float, default=1e-6)
            p.add_argument("--batch-size", type=int, default=512)
            p.add_argument("--plot-count", type=int, default=2, help="0–4 pose panels per hand; use 0 for large evaluations")
    c = sub.add_parser("compare", help="compare runs on identical data and metric settings")
    c.add_argument("--runs", nargs="+", required=True, help="LABEL=DIRECTORY; repeat a label to group runs")
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(func=compare)
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        ap.exit(2, f"Evaluation stopped: {exc}\n")


if __name__ == "__main__":
    main()
