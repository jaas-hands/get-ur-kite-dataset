"""Check a LEAP sample and run goal, no-motion, and prediction-import baselines."""

import argparse
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tarfile

import numpy as np

from . import config
from .eval_io import code_version, new_directory, read_npz, sha256, write_json
from .eval_metrics import ERRORS
from .evaluate import load_sampler
from .sample import verify


def import_archive(archive, out, root_name="leap_sample_v1"):
    """Extract a sample into a new folder. Reject links and paths outside it."""
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if sum(m.size for m in members) > 100_000_000:
            raise ValueError("expected a small sample archive, not a full dataset")
        names = set()
        for m in members:
            p = PurePosixPath(m.name)
            if (p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != root_name
                    or not (m.isfile() or m.isdir()) or m.name in names):
                raise ValueError(f"unsupported archive entry: {m.name}")
            names.add(m.name)
        new_directory(out)
        for m in members:
            path = out / m.name
            if m.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(tar.extractfile(m).read())
    return out / root_name


def audit_sample(root):
    manifest = json.loads((root / "splits.sample.json").read_text())
    full = json.loads((root / "splits.full.json").read_text())
    expected_hash = manifest["sample"]["source"]["splits_json_sha256_16"]
    if sha256(root / "splits.full.json")[:16] != expected_hash:
        raise ValueError("full split manifest checksum does not match sample provenance")
    all_ids = [h for ids in full["hands"].values() for h in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("a hand appears in multiple full-data splits")
    checks = []
    for split, hands in manifest["hands"].items():
        for hid in hands:
            path = root / "examples" / "kite_code" / split / f"{hid}.npz"
            ex = read_npz(path)
            sampler, _, _ = load_sampler(root / "hands" / hid, contact_only=True)
            if str(ex["hand_id"]) != hid or str(ex["split"]) != split or hid not in full["hands"][split]:
                raise ValueError("example metadata or full split membership mismatch")
            if len(ex["k"]) != manifest["sample"]["files"][hid]["rows"]:
                raise ValueError("sample row count differs from manifest")
            for key, value in ex.items():
                if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                    raise ValueError(f"nonfinite sample field: {hid}/{key}")
            result = verify(sampler, ex)
            if not result["ok"]:
                raise ValueError(f"sample invariant failed: {hid}: {result}")
            # verify() returns zero for skipped collisions; this run did not check them.
            result.pop("collisions")
            checks.append({"hand_id": hid, "split": split, "rows": len(ex["k"]),
                           **result, "collision_checked": False})
    return manifest, {"checks": checks, "full_manifest_hash_matches": True,
                      "full_split_hands_disjoint": True, "full_hand_count": len(all_ids),
                      "source": manifest["sample"]["source"]}


def check_baselines(out):
    """Check goal accuracy and agreement between the baseline paths."""
    summaries = {label: json.loads((out / label / "summary.json").read_text())
                 for label in ("goal", "init", "init_raw", "roundtrip")}
    raw_delta = {key: max(abs(a[key] - b[key]) for a, b in zip(summaries["init"], summaries["init_raw"]))
                 for key in ERRORS}
    # Float32 angles at a boundary can sit a few nanoradians beyond the decimal limit.
    # Keep rates/counts exact, allow only 1e-6 mm/degree differences in aggregate errors.
    raw_matches = all((np.isclose(a[k], b[k], atol=1e-6, rtol=0) if k in ERRORS else a[k] == b[k])
                      for a, b in zip(summaries["init"], summaries["init_raw"]) for k in a)
    checks = {"goal_reconstruction_pass": all(r["mean_position_mm"] < .001 and r["mean_normal_deg"] < .01
                  and r["contact_success_pct"] == 100 for r in summaries["goal"]),
              "raw_and_clamped_baselines_match_within_1e-6": bool(raw_matches),
              "prediction_export_import_matches": summaries["init"] == summaries["roundtrip"]}
    write_json(out / "verification.json", {"checks": checks, "max_split_error_delta_raw_vs_clamped": raw_delta})
    if not all(checks.values()):
        raise ValueError(f"baseline verification failed: {checks}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new output directory; existing results are never overwritten")
    args = parser.parse_args()
    out = new_directory(args.out.resolve())
    root = import_archive(args.archive, out / "input")
    manifest, audit = audit_sample(root)
    inventory = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}
    write_json(out / "input_audit.json", {**audit, "archive": str(args.archive.resolve()),
               "archive_sha256": sha256(args.archive), "extracted_files_sha256": inventory, "code": code_version()})
    metadata = {"purpose": "teammate sample integration smoke test; not model or task performance",
                "source": audit["source"], "archive_sha256": sha256(args.archive),
                "sample_rows": "first 32 rows per LEAP hand; train rows are training data",
                "excluded": "Allegro/Barrett pilot: external KITE models/templates not supplied",
                "protocol_note": "clamp per sample README; raw joint-limit violations retained"}
    write_json(out / "metadata.json", metadata)
    common = ["--data", str(root / "examples/kite_code"), "--hands", str(root / "hands"),
              "--contact-only", "--splits", *manifest["hands"]]
    commands = []

    def run(*arguments):
        command = [sys.executable, "-m", "kite_data.evaluate", *arguments]
        commands.append(shlex.join(command))
        with (out / "commands.sh").open("w") as f:
            f.write("#!/bin/sh\nset -eu\ncd " + shlex.quote(str(config.REPO_ROOT)) + "\n")
            f.write("# Recorded commands; use fresh --out directories to rerun.\n" + "\n".join(commands) + "\n")
        with (out / "run.log").open("a") as log:
            log.write("\n$ " + commands[-1] + "\n")
            log.flush()
            result = subprocess.run(command, cwd=config.REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"command failed; see {out / 'run.log'}")
        print("Completed:", arguments[0], arguments[-1], flush=True)

    for label in ("goal", "init"):
        run("run", *common, "--baseline", label, "--joint-policy", "clamp", "--metadata", str(out / "metadata.json"),
            "--run-id", f"leap-sample-{label}", "--out", str(out / label))
    run("run", *common, "--baseline", "init", "--joint-policy", "raw", "--plot-count", "0",
        "--run-id", "leap-sample-init-raw", "--out", str(out / "init_raw"))
    run("export", *common, "--baseline", "init", "--out", str(out / "prediction_example"))
    run("run", *common, "--predictions", str(out / "prediction_example"), "--joint-policy", "clamp", "--plot-count", "0",
        "--run-id", "leap-sample-import-check", "--out", str(out / "roundtrip"))
    run("compare", "--runs", f"Goal={out / 'goal'}", f"No-motion={out / 'init'}", "--out", str(out / "comparison"))
    check_baselines(out)
    print(f"Verified baseline runs. Results: {out}")


if __name__ == "__main__":
    main()
