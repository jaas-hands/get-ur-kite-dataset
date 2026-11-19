# Evaluation

Check whether a hand pose reaches the requested contacts. Runs locally on CPU.

The first comparison of Anushka's supplied predictions is in [Sample results](SAMPLE_RESULTS.md).

## Run the teammate sample

From the repo root, using the existing environment:

```sh
.venv/bin/python -m kite_data.sample_audit \
  --archive /Users/janiceshen/Downloads/leap_sample_v1.tar.gz \
  --out data/evaluation/my_sample_check
```

Use a new output folder each time. The script extracts a copy, checks the inputs,
and runs goal/no-motion baselines plus a prediction export/import check.
Open `comparison/report.md`, the PNGs, and `verification.json`.

The sample contains 224 LEAP examples. The extra Allegro/Barrett records need KITE
assets and a separate adapter. Training rows are not held-out familiar-hand examples.
These first-row slices are for integration checks, not model selection or final results.

## Run the supplied model predictions

```sh
.venv/bin/python -m kite_data.prediction_audit \
  --sample /Users/janiceshen/Downloads/leap_sample_v1.tar.gz \
  --predictions /Users/janiceshen/Downloads/leap_sample_v1_predictions.tar.gz \
  --out data/evaluation/my_prediction_check
```

This imports both models, preserves their predicted poses, runs the evaluator and
compares them with no motion. It also recomputes the goal sanity check and checks
contact errors, normals, success and limit flags against the supplied per-row scores.
Use a fresh output folder. Open `comparison/report.md`, `comparison/comparison.png`
and `verification.json`; `crosscheck.csv` lists every numeric comparison.

The archive's README defines row order and joint conventions. The importer adds
our row IDs and input hashes; those were not supplied by the producer. Per-row
no-motion agreement provides a further alignment check. Checkpoint identifiers
remain producer-supplied; this command does not load or rerun checkpoints.

This run uses clamped contact scoring and **0.001 rad** limit tolerance to match
the producer, overriding the stricter default below. Collisions remain unavailable.
The shared-slot model is a KITE-inspired baseline, not official KITE or a pure
graph-off ablation. The sample is too small to establish a model ranking.

## Scoring

| Metric | Meaning |
| --- | --- |
| Position error | Distance from each labelled surface point to its target, in mm |
| Normal error | Angle between the achieved and requested surface normals, in degrees |
| Contact success | Every contact within 10 mm **and** every normal within 15° |
| Joint-limit violation | Any active joint outside its limits by more than 1e-6 rad |
| Self-collision | Penetration beyond `PENETRATION_TOL` in the hand collision model |
| Valid success | Contact success with no limit violation or self-collision |
| Invalid prediction | Nonfinite pose/error or quaternion norm more than 1e-3 from one |

- `--contact-only`: read `hand.json` and `template.npz` directly. Collision metrics
  and valid success are **n/a**, not zero. This is the mode used for Anushka's sample.
- Default mode: reconstruct the hand using the generator and score collisions with
  its stored `hand.xml`. Requires the original third-party assets.
- `--joint-policy clamp`: clip joints before contact/collision scoring, as in the
  sample README. Limit violations always use the original prediction. Default is `raw`.
- Change tolerances with `--position-mm`, `--normal-deg`, and `--joint-tolerance-rad`.

Errors average contacts within each example, examples within each hand, then hands
equally within a split. Summary `max_*` columns average the per-example maxima.
Invalid predictions fail success; their errors and physical checks are unavailable.
Error means and violation rates use evaluated rows. If any hand has no valid rows,
its split error is n/a. Check the invalid count alongside the error.

Matching follows the labelled point: a different finger reaching the target gets no
credit. Collision checks approximate the hand only. Neither metric proves task success.

## Score model predictions

Ask for absolute wrist poses and joint angles for the same examples. The model must
not receive the goal pose as an inference input.

Export a format example using your data paths and selected splits:

```sh
python -m kite_data.evaluate export --contact-only \
  --data SAMPLE/examples/kite_code --hands SAMPLE/hands --splits val \
  --out data/predictions/format-example
```

This writes starting poses, not model output. Replace only these arrays:

| Array | Shape | Convention |
| --- | --- | --- |
| `joints` | `(n, n_dof)` or `(n, 16)` | Radians; `dof_names` order; trailing padding ignored |
| `wrist_pos` | `(n, 3)` | Absolute palm position, world frame, metres |
| `wrist_quat` | `(n, 4)` | Unit quaternion in w, x, y, z order |

Keep the other fields: row IDs, hand/split, joint names, hashes, schema version and
units. Rows may be reordered only with their IDs. Every row must appear once.
Hand hashes cover JSON/template files, plus XML in collision mode.

```python
import numpy as np
from kite_data.eval_io import read_npz

payload = read_npz("format-example/val/HAND.npz")
payload.update(joints=predicted_joints,
               wrist_pos=predicted_positions,
               wrist_quat=predicted_wxyz)
np.savez_compressed("model/val/HAND.npz", **payload)
```

Record checkpoint/encoder identifiers and hashes, training seed and budget, graph
setting, and checkpoint-selection rule in `run.json`. These are supplied metadata;
the evaluator does not verify how the model was trained.

```sh
python -m kite_data.evaluate run --contact-only --joint-policy clamp \
  --data SAMPLE/examples/kite_code --hands SAMPLE/hands --splits val \
  --predictions data/predictions/model --metadata run.json \
  --run-id model-val --out data/evaluation/model-val
```

Use the same data paths and scoring options for every method. Use validation to
select checkpoints; agree on the protocol before comparing final test results.

## Files to open

| File | Use |
| --- | --- |
| `report.md`, `summary.json` | Read the result |
| `examples.csv` | Inspect each prediction; positive improvement means better than no motion |
| `hands.csv`, `splits.csv` | Compare hands and splits |
| `diagnostics.csv` | Group by joint count, contact count, or starting error |
| `poses/`, `plot_selection.csv` | Inspect the first row and worst remaining errors |
| `manifest.json` | Check input hashes, settings, and code used for that run |

The sample script also saves `input_audit.json`, `commands.sh`, `run.log`,
`verification.json`, and baseline `prediction_example/` files. Outputs under `data/`
are local and git-ignored. Choose `--plot-count 0` for large runs.

```sh
python -m kite_data.evaluate compare \
  --runs graph=data/evaluation/graph no-graph=data/evaluation/no-graph \
  --out data/evaluation/comparison
```

Different inputs or scoring rules are rejected. Repeated labels group runs and
report mean/sample SD; they do not establish independent seeds or matched budgets.

## Fresh setup and generated samples

```sh
python -m pip install -e '.[dev,fast]'
bash scripts/fetch_assets.sh
python -m pytest -q
python -m kite_data.evaluate sample
python -m kite_data.evaluate run --baseline goal --run-id goal --out data/evaluation/goal
python -m kite_data.evaluate run --baseline init --run-id init --out data/evaluation/init
```

`fast` adds Embree for faster sampling; omit it if unavailable. Asset-dependent tests
skip when assets are missing. The goal baseline should have near-zero contact error;
mixed goal poses can still collide.

## Code map

- `evaluate.py`: run/export/compare commands.
- `eval_geometry.py`: read exported joint geometry.
- `eval_metrics.py`: errors, success rates, and aggregation.
- `eval_io.py`: prediction format and input hashes.
- `eval_report.py`: tables and plots.
- `sample_audit.py`: import and check the teammate archive.
- `prediction_audit.py`: import both supplied models and cross-check their scores.

## Next steps

- Agree on the full evaluation splits and main comparison with the team. The
  224-example sample checks the workflow; it is too small to rank the models.
- Evaluate the full predictions with the same scoring settings, and include
  additional training seeds if available.
- For task evaluation, get the actual checkpoints and connect the decoder to
  Simon's rollout setup. Keep task success separate from these contact metrics.
