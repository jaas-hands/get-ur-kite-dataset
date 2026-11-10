"""Tables, plots, and concise summaries for evaluation runs."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .eval_metrics import average_hands, summarize


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_summary(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return [{**dict(zip(keys, key)), **summarize(group)} for key, group in sorted(groups.items())]


def split_summary(hand_rows):
    groups = defaultdict(list)
    for row in hand_rows:
        groups[row["split"]].append(row)
    return [{"split": split, **average_hands(group)} for split, group in sorted(groups.items())]


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


def result_table(rows, label="Split"):
    lines = [f"| {label} | Position (mm) | Normal (deg) | Contact success (%) | Valid success (%) | Invalid (%) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        vals = [row["split"], *(fmt(row[k]) for k in ("mean_position_mm", "mean_normal_deg",
                "contact_success_pct", "valid_contact_success_pct", "invalid_prediction_pct"))]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(out, manifest, summaries, plot_count):
    t = manifest["protocol"]["thresholds"]
    lines = [f"# {manifest['run_id']}", "", result_table(summaries), "",
             "Each split weights its hands equally. Errors average contacts within an example, then examples within a hand.",
             "Errors describe valid predictions only; success rates include every example. Invalid predictions count as failures.",
             "If any hand has no valid predictions, its split error is n/a. Collision/limit rates use only evaluated predictions; see counts in the CSVs.", "",
             f"Contact success: every selected point within {t['position_mm']:g} mm and {t['normal_deg']:g} degrees.",
             "Valid success also requires no joint-limit violation or hand self-collision. These thresholds are proposed defaults, not an approved protocol.",
             "Matching uses the labelled surface points. It does not credit a different finger reaching the same target.",
             "Collision checks use the dataset's approximate boxes/pads, without an arm or objects.", "",
             "## Files", "", "- `examples.csv`: all examples, including invalid predictions.",
             "- `hands.csv` / `splits.csv`: per-hand results and equal-hand split averages.",
             "- `diagnostics.csv`: per-hand groups by DoF, contact count, and initial-error bin.",
             "- `manifest.json`: input hashes, conventions, settings, software versions, and supplied model metadata.",
             "- `distribution.png`: per-hand contact errors and valid success.",
             "- `diagnostics.png`: error versus initial error and contact count."]
    if plot_count:
        lines += [f"- `poses/`: {plot_count} examples selected per hand (first row and worst remaining position errors).",
                  "  Selection is recorded in `plot_selection.csv`. Invalid rows rank first and are labelled instead of drawn."]
    if manifest["baseline"] == "goal":
        lines += ["", "Goal poses are a reconstruction check, not a trained-model result. Mixed goal poses can still self-collide."]
    elif manifest["baseline"] == "init":
        lines += ["", "This is the no-motion baseline. A learned decoder should improve on it."]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def distribution_plot(out, hands):
    plt = pyplot()
    splits = sorted({r["split"] for r in hands})
    fig, axes = plt.subplots(1, 3, figsize=(max(10, len(splits) * 2), 4), layout="constrained")
    for ax, key, title in zip(axes, ("mean_position_mm", "mean_normal_deg", "valid_contact_success_pct"),
                              ("Position error (mm)", "Normal error (deg)", "Valid contact success (%)")):
        for x, split in enumerate(splits):
            vals = [r[key] for r in hands if r["split"] == split and r[key] is not None]
            ax.scatter(np.full(len(vals), x) + np.linspace(-.12, .12, len(vals)), vals, alpha=.65, s=20)
            if vals:
                ax.scatter(x, np.mean(vals), marker="_", s=300, color="black")
        ax.set_xticks(range(len(splits)), [s.removeprefix("test_") for s in splits], rotation=25)
        ax.set_xlim(-.5, len(splits) - .5)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Each dot is one hand; black marks show means over hands with defined metrics")
    fig.savefig(out / "distribution.png", dpi=140)
    plt.close(fig)


def diagnostic_plot(out, rows):
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for split in sorted({r["split"] for r in rows}):
        valid = [r for r in rows if r["split"] == split and not r["invalid_prediction"]]
        # Deterministic display sample; full observations remain in examples.csv.
        valid = valid[::max(1, (len(valid) + 1999) // 2000)]
        for ax, key in zip(axes, ("initial_position_mm", "k")):
            ax.scatter([r[key] for r in valid], [r["mean_position_mm"] for r in valid],
                       s=8, alpha=.35, label=split)
            ax.set_ylabel("Predicted contact error (mm)")
    axes[0].set_xlabel("Initial contact error (mm)")
    axes[1].set_xlabel("Number of contacts")
    axes[1].set_xticks(range(2, 7))
    axes[1].legend(fontsize=8)
    fig.savefig(out / "diagnostics.png", dpi=140)
    plt.close(fig)


def select_examples(rows, count):
    if count <= 0:
        return []
    # Use the same first example across runs; remaining panels deliberately show failures.
    ranked = sorted(range(1, len(rows)), key=lambda i: (
        not rows[i]["invalid_prediction"], -(rows[i]["mean_position_mm"] or 0), i))
    return [0, *ranked[:count - 1]]


def pose_plot(path, sampler, template, ex, prediction, rows, ids):
    from .qa import _square_axes, world_points
    from .sample import _matrix

    plt = pyplot()
    fig = plt.figure(figsize=(5 * len(ids), 5), layout="constrained")
    for panel, i in enumerate(ids):
        ax = fig.add_subplot(1, len(ids), panel + 1, projection="3d")
        clouds = []
        poses = [("Start", "#aaaaaa", ex["joints_init"][i], ex["wrist_pos_init"][i], ex["wrist_quat_init"][i]),
                 ("Goal", "#2c9b68", ex["joints_goal"][i], ex["wrist_pos_goal"][i], ex["wrist_quat_goal"][i])]
        if not rows[i]["invalid_prediction"]:
            poses.append(("Prediction", "#386de0", prediction["joints"][i, :sampler.hand.n_dof],
                          prediction["wrist_pos"][i], prediction["wrist_quat"][i]))
        for label, color, q, pos, quat in poses:
            wrist = np.eye(4)
            wrist[:3, :3], wrist[:3, 3] = _matrix(quat[None])[0], pos
            points = world_points(sampler.hand, template, q, wrist) * 1000
            clouds.append(points)
            ax.scatter(*points[::4].T, s=1, color=color, alpha=.45, label=label)
        valid = ex["template_idx"][i] >= 0
        target = ex["contact_pos"][i, valid] * 1000
        normals = ex["contact_normal"][i, valid]
        ax.scatter(*target.T, c="black", s=25, label="Targets")
        ax.quiver(*target.T, *normals.T, length=15, color="black")
        if not rows[i]["invalid_prediction"]:
            achieved, _ = sampler.contact_points(
                prediction["joints"][i:i + 1, :sampler.hand.n_dof],
                prediction["wrist_pos"][i:i + 1], _matrix(prediction["wrist_quat"][i:i + 1]),
                ex["template_idx"][i:i + 1])
            achieved = achieved[0, valid] * 1000
            ax.scatter(*achieved.T, c="#386de0", s=25, marker="x", label="Achieved")
            for a, b in zip(achieved, target):
                ax.plot(*np.stack([a, b]).T, color="#386de0", linewidth=1)
        _square_axes(ax, np.concatenate([*clouds, target]))
        error = "invalid prediction" if rows[i]["invalid_prediction"] else f"{rows[i]['mean_position_mm']:.2f} mm"
        ax.set_title(f"Row {i}: {error}\n{'first example' if panel == 0 else 'worst remaining error'}", fontsize=10)
        ax.legend(fontsize=8, markerscale=1.5)
    fig.suptitle(sampler.hand.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
