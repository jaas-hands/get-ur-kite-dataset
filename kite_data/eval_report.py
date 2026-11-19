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
    """Write the result table, scoring rules, and links to the full output."""
    protocol = manifest["protocol"]
    thresholds = protocol["thresholds"]
    contact_only = protocol["collision"] == "not evaluated"
    purpose = {
        "goal": "Goal check: score the recorded answer. Errors should be near zero.",
        "init": "No-motion baseline: score the starting pose. This is what a decoder should improve on.",
    }.get(manifest["baseline"], "Score the supplied predictions against the requested contacts.")
    lines = [
        f"# {manifest['run_id']}", "", purpose, "", result_table(summaries), "",
        f"Success requires every contact within {thresholds['position_mm']:g} mm and "
        f"every normal within {thresholds['normal_deg']:g} degrees.",
        f"Joint scoring: {protocol.get('joint_policy', 'raw')}. Limit violations use the raw prediction.",
        "Collisions: NOT performed; self-collision and valid success are n/a, not zero."
        if contact_only else "Valid success also requires no limit violation or hand self-collision (approximate boxes/pads).",
        "", "Errors average contacts, then examples, then hands equally within each split. "
        "Invalid predictions fail success; their errors and physical checks are unavailable. "
        "A hand with no valid predictions makes its split error n/a.",
        "", "Matching uses the labelled surface points. Another finger reaching the target gets no credit. "
        "Contact success does not establish task success. Confirm the protocol before final comparisons.",
        "", "## Files", "",
        "- [examples.csv](examples.csv): every row.",
        "- [hands.csv](hands.csv) / [splits.csv](splits.csv): grouped results.",
        "- [distribution.png](distribution.png) / [diagnostics.png](diagnostics.png): errors and success plots.",
        "- [diagnostics.csv](diagnostics.csv): results by joint count, contact count, and starting error.",
        "- [manifest.json](manifest.json): inputs, settings, software versions, and code hashes.",
    ]
    if plot_count:
        lines += ["- [plot_selection.csv](plot_selection.csv): pose plot paths and selected rows "
                  "(first row, then invalid/worst errors)."]
    if manifest["baseline"] == "goal":
        lines += ["", "This checks reconstruction, not model quality. Goal poses can still self-collide."]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def distribution_plot(out, hands):
    plt = pyplot()
    splits = sorted({r["split"] for r in hands})
    fig, axes = plt.subplots(1, 3, figsize=(max(10, len(splits) * 2), 4), layout="constrained")
    success_key = "contact_success_pct" if all(r["valid_contact_success_pct"] is None for r in hands) else "valid_contact_success_pct"
    for ax, key, title in zip(axes, ("mean_position_mm", "mean_normal_deg", success_key),
                              ("Position error (mm)", "Normal error (deg)", "Contact success (%)" if success_key == "contact_success_pct" else "Valid contact success (%)")):
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
