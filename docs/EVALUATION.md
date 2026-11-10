# Evaluation and analysis

Does the predicted hand pose achieve the requested contacts? Start with that question.
This evaluator runs on CPU and uses the existing hand FK and collision model. It does
not need KITE, a trained decoder, an arm, or a GPU.

The thresholds below are proposals for team review. Results from the small sample
are software checks, not evidence of generalization.

## Start here

From this repository, activate a Python 3.10+ environment and install the dependencies:

```bash
python -m pip install -e ".[dev,fast]"
bash scripts/fetch_assets.sh
python -m pytest -q -ra
```

`fast` installs Embree for surface sampling. If it is unavailable on your machine,
install `.[dev]`; the initial template build will be slower. Tests needing hand assets
skip when the assets are missing. The metric and prediction-format tests run without them.

Generate two validation hands with 100 examples each. This reuses the existing
generator and does not change the training/test split:

```bash
python -m kite_data.evaluate sample
```

Then check the goal pose and the starting pose:

```bash
python -m kite_data.evaluate run --baseline goal --run-id goal-check \
  --out data/evaluation/goal
python -m kite_data.evaluate run --baseline init --run-id no-motion \
  --out data/evaluation/init
python -m kite_data.evaluate compare \
  --runs goal=data/evaluation/goal no-motion=data/evaluation/init \
  --out data/evaluation/baselines
```

Open `report.md` and the PNG plots in each output directory. Outputs are local and
git-ignored under `data/`. Commands refuse to overwrite a nonempty output directory;
use a new name for another run.

Expected: goal contact errors are near zero; starting-pose errors measure the work the
decoder needs to do. Goal self-collision need not be zero: the sampler's mixed goal
pose is not guaranteed collision-free. It is a diagnostic, not a reason to delete examples.

## Use a teammate's sample

Keep the generated layout:

```text
data/examples/team_sample/val/<hand_id>.npz
data/hands/<hand_id>/hand.json
data/hands/<hand_id>/template.npz
data/hands/<hand_id>/hand.xml
data/hands/<hand_id>/hand.urdf
```

Pass `--data data/examples/team_sample`. Use `--hands` if hand assets are elsewhere.
The original third-party assets are still needed to reconstruct the hand with the
existing code. Ask for the generator commit, split file, and seed/config too.

For larger runs, select splits explicitly and disable per-hand pose plots:

```bash
python -m kite_data.evaluate run --data data/examples/kite \
  --splits test_geometry test_finger_subset test_joint_count test_combined \
  --baseline init --run-id test-no-motion --plot-count 0 \
  --out data/evaluation/test-init
```

Use validation for development and checkpoint selection. Agree on the protocol before
examining final test results. A directory with only a few hands is a subset; the
manifest records exactly which files were evaluated. It does not certify a complete split.

## Model predictions

Ask the decoder owner for **absolute** wrist poses and joint angles for the same
examples. The model must not receive the goal pose as an inference input.

Export a working format example:

```bash
python -m kite_data.evaluate export --out data/predictions/format-example
```

This exports the **starting poses**, not model predictions. Each NPZ mirrors the
dataset path (`val/<hand_id>.npz`). Replace only these three arrays with inference outputs:

| Field | Shape | Meaning |
| --- | --- | --- |
| `joints` | `(n, n_dof)` or `(n, 16)` | Radians, canonical movable-joint order; trailing padding ignored |
| `wrist_pos` | `(n, 3)` | Absolute palm position in the dataset world frame, metres |
| `wrist_quat` | `(n, 4)` | Unit quaternion in **w, x, y, z** order; either sign is accepted |

Preserve the rest of the handoff fields:

| Field | Meaning |
| --- | --- |
| `example_id` | Original zero-based NPZ row index; every row exactly once |
| `hand_id`, `split` | Scalars identifying the source file |
| `dataset_sha256` | SHA-256 of that exact source NPZ |
| `hand_sha256` | Digest of the hand.json, template.npz, and hand.xml hashes |
| `dof_names` | Canonical movable-joint names from hand.json |
| `schema_version` | Integer `1` |
| `pose_type`, `frame` | `absolute`, `world` |
| `position_unit`, `joint_unit`, `quaternion_order` | `metres`, `radians`, `wxyz` |

Rows can be reordered if `example_id` moves with them. Missing/duplicate IDs, missing
or extra hand files in selected splits, wrong units, mismatched hashes, and wrong
joint order stop evaluation. Incorrect values inside otherwise valid pose arrays
are scored as invalid predictions rather than discarded.

The inference script can also call `eval_io.export_predictions` directly. For an
already exported format file, a minimal adapter is:

```python
from pathlib import Path
import numpy as np
from kite_data.eval_io import read_npz

source = Path("data/predictions/format-example/val/<hand_id>.npz")
output = Path("data/predictions/model/val")
output.mkdir(parents=True, exist_ok=True)
payload = read_npz(source)
# Fill these from inference, in the same row order as payload['example_id'].
payload.update(joints=predicted_joints,
               wrist_pos=predicted_wrist_positions,
               wrist_quat=predicted_wxyz_quaternions)
np.savez_compressed(output / source.name, **payload)
```

Record the experiment in a JSON file. For example (replace these values):

```json
{
  "checkpoint": "path or published identifier + hash",
  "encoder": "KITE checkpoint identifier + hash",
  "graph_bias": true,
  "training_seed": 0,
  "training_budget": "examples, optimizer steps, batch size",
  "checkpoint_selection": "lowest validation contact error"
}
```

```bash
python -m kite_data.evaluate run --predictions data/predictions/model \
  --run-id graph-seed0 --metadata run.json --out data/evaluation/graph-seed0
```

The evaluator stores supplied metadata without claiming it has verified training.
An omitted metadata file produces an empty model record; fill it before sharing results.

## Metrics and denominators

| Metric | Definition |
| --- | --- |
| Position error | Euclidean distance between labelled surface points and their targets, in mm |
| Normal error | Angle between normalized achieved/target outward normals, in degrees |
| Contact success | **All** valid contacts within 10 mm **and** 15 degrees |
| Joint-limit violation | Any active joint outside its limits, allowing 1e-6 rad numerical tolerance |
| Self-collision | Any penetration beyond the existing `PENETRATION_TOL` in the stored collision model |
| Valid contact success | Contact success, no limit violation, and no self-collision |
| Invalid prediction | Non-finite active pose values, quaternion norm more than 1e-3 from one, or non-finite derived errors |

Set thresholds with `--position-mm`, `--normal-deg`, and `--joint-tolerance-rad`.
No poses are clamped before scoring. Small quaternion normalization error is tolerated;
zero/non-unit quaternions beyond the stated tolerance fail. Padded contacts and joints
do not contribute.

Errors average contacts within an example, examples within a hand, and hands equally
within a split. `max_*` columns are per-example maxima; summary versions are averages
of those maxima. Keep split results separate.

Invalid predictions stay in the denominator of both success rates. Their contact
errors and collision/limit checks are unknown, written as empty CSV cells / JSON null.
Error means and physical violation rates use evaluated predictions; every summary
also reports `examples`, `evaluated`, and `invalid`. If a hand has no valid predictions,
its error and the equal-hand split error are undefined. Always inspect invalid rates
alongside errors; valid-only error means can otherwise reward missing predictions.

Matching uses the labelled surface points. A different finger reaching the same
target can still score poorly. An assignment-free metric would be a separate analysis.
Collision boxes/pads approximate the hand only; there is no arm, table, or object here.
Contact success therefore does not establish grasping or lifting success.

## Reports and comparisons

Each run writes per-example, per-hand, and per-split CSVs, plus a JSON manifest and
summary, a short Markdown report, and PNGs. `examples.csv` includes initial contact
error and improvement in mm (positive = better than no motion). `diagnostics.csv`
groups results within each hand by DoF, contact count, and initial error. The diagnostic
scatter plot uses a deterministic display sample; the CSVs retain all observations.

Pose sheets show the first example and the worst remaining position errors, with
invalid predictions first. `plot_selection.csv` records their IDs. The first example
is shared across runs; worst-case selections can differ. Use `--plot-count 0` for large
evaluations, or 1–4 for detailed inspection.

```bash
python -m kite_data.evaluate compare --runs \
  graph=data/evaluation/graph-seed0 \
  graph=data/evaluation/graph-seed1 \
  no-graph=data/evaluation/no-graph-seed0 \
  --out data/evaluation/comparison
```

Comparison refuses different data, hand assets, or metric settings. Repeated labels
group runs and report means/sample SD. It does not establish that those runs use
independent seeds or fair training budgets. Review their metadata. Aim for three seeds
if feasible; one seed is preliminary. Do not treat more examples as more training seeds.

Questions for analysis: which held-out split is hardest? Does graph information help
across seeds? Does it improve over no motion? Are errors concentrated in particular
hands, contact counts, or starting-error ranges? Include failures as well as good cases.

## Before the team approves final evaluation

- Confirm selected-point matching, thresholds, and equal-hand aggregation.
- Name the data, encoder, training, and simulation owners.
- Agree on frozen encoder, data/budget, seed set, and checkpoint-selection rule for the graph ablation.
- Check dataset conventions against released KITE code before scaling.
- Keep optional wide-noise, external hands, and closed-loop experiments separate.

First deliverable: a goal-versus-start report with pose plots and passing checks.
After that, the same evaluator accepts the team's model predictions.
