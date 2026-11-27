"""Build primitive-geometry hands and their KITE-recipe records (morphology-diversity test, 2026-11-19).

    python -m kite_data.build_prim hands    --n 900 --out data/prim/hands
    python -m kite_data.build_prim records  --hands data/prim/hands --out data/prim/examples/kite_code --plan plan.json
    python -m kite_data.build_prim couple   --src data/prim/hands --out data/prim_c/hands --seed 20261119

couple (2026-11-21): the same bodies with coupled joints added by prim_hand.sample_coupling (its own RNG stream). Built
from an existing, validated hands directory so that every body keeps its geometry, template and gate result:
  * a body that draws no coupling is copied byte for byte (hand.json, hand.xml, template.npz);
  * a coupled body gets a new hand.json / hand.xml; its template.npz is copied (coupling does not change the geometry
    or the zero pose). It must pass the same validity gate (zero-pose collision, >= 5% of 500 uniform samples of its
    independent coordinates collision-free); if it does not, the body is copied uncoupled and logged.
Records are then made with `records` exactly as for the source draw (same plan, seed and regime).

Hands: P0000..P{n-1}, each deterministic in (seed, index, attempt). A body is redrawn (attempt + 1) if:
  * it collides at the zero pose;
  * fewer than 5% of 500 uniform joint samples are collision-free;
  * or it hits the exclusion rule.
Exclusion rule (keeps the reserved families' topology unseen): 5 fingers with 4 joints on every finger, which is
Wuji's 5 x 4 layout and the nearest layout to Shadow (4/4/4/5/5) that this generator can produce.

Records: kite_data.sample.HandSampler, kite_code regime (KITE's released recipe), checked by sample.verify. The
seed differs per (split, hand). The plan JSON maps split name -> {"hands": [...], "n": rows per hand}.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
import zlib
from pathlib import Path

import numpy as np

from . import __version__
from .collision import Collider
from .prim_hand import COUPLING, PrimHand, load_prim, sample_coupling, sample_spec, spec_from_dict
from .sample import REGIMES, HandSampler, verify

SEED = 20261119
MAX_ATTEMPTS = 50
GATE_SAMPLES = 500
GATE_MIN_FREE = 0.05


def excluded(spec) -> bool:
    return spec.n_fingers == 5 and all(f.n_joints == 4 for f in spec.fingers)


def build_one(job) -> dict:
    index, seed, out_root = job
    reasons = []
    for attempt in range(MAX_ATTEMPTS):
        spec = sample_spec(index, seed, attempt)
        if excluded(spec):
            reasons.append("excluded_topology"); continue
        hand = PrimHand(spec)
        xml = hand.to_mjcf()
        col = Collider(hand, xml)
        zero = sorted(col.colliding_pairs(np.zeros(hand.n_dof)))
        rng = np.random.default_rng([seed, index, attempt, 7])
        free = float(col.free_mask(rng.uniform(hand.lower, hand.upper, size=(GATE_SAMPLES, hand.n_dof))).mean())
        if zero or free < GATE_MIN_FREE:
            reasons.append("zero_pose_collision" if zero else f"free_rate_{free:.3f}"); continue
        break
    else:
        raise RuntimeError(f"P{index:04d}: no valid body in {MAX_ATTEMPTS} attempts: {reasons[-5:]}")
    out = Path(out_root) / hand.id
    out.mkdir(parents=True, exist_ok=True)
    desc = hand.descriptor()
    desc["build"] = {"seed": seed, "attempt": attempt, "rejected": reasons, "gate": {"free_rate": round(free, 4), "n_samples": GATE_SAMPLES}}
    (out / "hand.json").write_text(json.dumps(desc, indent=1) + "\n")
    (out / "hand.xml").write_text(xml)
    tpl = hand.build_template(seed)
    np.savez_compressed(out / "template.npz", **tpl)
    longs = [f for f in spec.fingers if f.name != "thumb"]
    return {"id": hand.id, "topology": spec.topology, "layout": spec.layout, "n_fingers": spec.n_fingers,
            "n_long": len(longs), "has_thumb": spec.n_fingers > len(longs), "n_dof": hand.n_dof,
            "palm_w_mm": round(spec.palm_size[0] * 1000, 1), "palm_l_mm": round(spec.palm_size[1] * 1000, 1),
            "max_finger_len_mm": round(max(sum(f.link_lengths) for f in spec.fingers) * 1000, 1),
            "attempts": attempt + 1, "free_rate": round(free, 4)}


def couple_one(job) -> dict:
    import shutil
    src, seed, out_root = job
    src = Path(src)
    js = json.loads((src / "hand.json").read_text())
    spec = spec_from_dict(js["spec"])
    sc = sample_coupling(spec, seed)
    out = Path(out_root) / src.name
    out.mkdir(parents=True, exist_ok=True)
    row = {"id": src.name, "topology_src": spec.topology, "n_dof_src": spec.n_dof, "n_coupled": sc.n_coupled,
           "dropped": False, "free_rate": None, "mults": ""}
    if sc.n_coupled:
        hand = PrimHand(sc)
        xml = hand.to_mjcf()
        col = Collider(hand, xml)
        zero = sorted(col.colliding_pairs(np.zeros(hand.n_dof)))
        rng = np.random.default_rng([js["build"]["seed"], spec.index, spec.attempt, 7])
        free = float(col.free_mask(rng.uniform(hand.lower, hand.upper, size=(GATE_SAMPLES, hand.n_dof))).mean())
        row["free_rate"] = round(free, 4)
        if zero or free < GATE_MIN_FREE:
            row["dropped"] = True
        else:
            desc = hand.descriptor()
            desc["build"] = js["build"] | {"coupling": {"seed": seed, "source": str(src), "params": COUPLING,
                                                        "gate": {"free_rate": round(free, 4), "n_samples": GATE_SAMPLES}}}
            (out / "hand.json").write_text(json.dumps(desc, indent=1) + "\n")
            (out / "hand.xml").write_text(xml)
            shutil.copyfile(src / "template.npz", out / "template.npz")
            row["mults"] = " ".join(f"{f.name}:{m[1]:.3f}" for f in sc.fingers for m in (f.mimic or []) if m is not None)
            row.update(topology=sc.topology, n_dof=hand.n_dof)
            return row
    for f in ("hand.json", "hand.xml", "template.npz"):
        shutil.copyfile(src / f, out / f)
    row.update(topology=spec.topology, n_dof=spec.n_dof, n_coupled=0 if row["dropped"] else row["n_coupled"])
    return row


def gen_one(job) -> dict:
    hand_id, split, n, seed, regime, hands_dir, out_dir = job
    t0 = time.time()
    hand = load_prim(Path(hands_dir) / hand_id)
    tpl = dict(np.load(Path(hands_dir) / hand_id / "template.npz"))
    sampler = HandSampler(hand, tpl, Collider(hand, hand.to_mjcf()))
    rng = np.random.default_rng([seed, zlib.crc32(split.encode()), zlib.crc32(hand_id.encode())])
    ex = sampler.sample(n, rng, REGIMES[regime])
    check = verify(sampler, ex, check_collisions=min(n, 50))
    if not check["ok"]:
        raise RuntimeError(f"{hand_id}: invariant check failed: {check}")
    path = Path(out_dir) / split / f"{hand_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **ex, hand_id=hand_id, split=split, regime=regime, seed=seed, kite_data=__version__)
    return {"hand_id": hand_id, "split": split, "n": n, "n_dof": hand.n_dof,
            "contact_accept_rate": round(sampler.accepted / sampler.tried, 4), "mask_density": round(float(ex["mask"].mean()), 4),
            "max_contact_err_m": check["contact_pos_err"], "seconds": round(time.time() - t0, 2)}


def _pool(fn, jobs, workers, label):
    # spawn + ProcessPoolExecutor: a worker that dies (e.g. SIGILL from MuJoCo on a CPU without AVX2) raises
    # BrokenProcessPool here instead of hanging forever as multiprocessing.Pool did (2026-11-19).
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0, rows = time.time(), []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        for fut in as_completed([ex.submit(fn, j) for j in jobs]):
            rows.append(fut.result())
            if len(rows) % 50 == 0 or len(rows) == len(jobs):
                print(f"  {label} {len(rows)}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    return rows, time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["hands", "records", "couple"])
    ap.add_argument("--src", default=None, help="couple: source hands directory")
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--hands", default="data/prim/hands")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--regime", default="kite_code")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    if a.what == "hands":
        out = Path(a.out or a.hands)
        rows, el = _pool(build_one, [(i, a.seed, str(out)) for i in range(a.n)], a.workers, "hands")
        rows.sort(key=lambda r: r["id"])
        with open(out / "index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        summ = {"kite_data": __version__, "module": "kite_data.build_prim", "args": vars(a), "elapsed_s": round(el, 1),
                "n_hands": len(rows), "redrawn": sum(r["attempts"] > 1 for r in rows),
                "n_fingers": {str(k): sum(r["n_fingers"] == k for r in rows) for k in range(2, 6)},
                "layout": {k: sum(r["layout"] == k for r in rows) for k in ("flex", "abd_first", "abd_second")},
                "n_dof": {"min": min(r["n_dof"] for r in rows), "max": max(r["n_dof"] for r in rows)},
                "distinct_topologies": len({r["topology"] for r in rows})}
        (out / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
        print(json.dumps({k: v for k, v in summ.items() if k != "args"}, indent=1))
    elif a.what == "couple":
        src, out = Path(a.src), Path(a.out)
        dirs = sorted(p for p in src.iterdir() if (p / "hand.json").exists())
        rows, el = _pool(couple_one, [(str(d), a.seed, str(out)) for d in dirs], a.workers, "couple")
        rows.sort(key=lambda r: r["id"])
        keys = ["id", "topology_src", "topology", "n_dof_src", "n_dof", "n_coupled", "dropped", "free_rate", "mults"]
        with open(out / "index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows([{k: r.get(k) for k in keys} for r in rows])
        summ = {"kite_data": __version__, "module": "kite_data.build_prim couple", "args": vars(a), "params": COUPLING,
                "elapsed_s": round(el, 1), "n_hands": len(rows), "n_coupled_bodies": sum(r["n_coupled"] > 0 for r in rows),
                "n_dropped_by_gate": sum(r["dropped"] for r in rows),
                "coupled_joints_per_body": {str(k): sum(r["n_coupled"] == k for r in rows) for k in range(0, 6)},
                "distinct_topologies": len({r["topology"] for r in rows})}
        (out / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
        print(json.dumps({k: v for k, v in summ.items() if k != "args"}, indent=1))
    else:
        plan = json.loads(Path(a.plan).read_text())
        out = Path(a.out)
        jobs = [(h, s, int(p["n"]), a.seed, a.regime, a.hands, str(out)) for s, p in plan.items() for h in p["hands"]]
        rows, el = _pool(gen_one, jobs, a.workers, "records")
        rows.sort(key=lambda r: (r["split"], r["hand_id"]))
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        summ = {"regime": a.regime, "sampler": REGIMES[a.regime].to_dict(), "seed": a.seed, "elapsed_s": round(el, 1),
                "splits": {s: {"hands": sum(r["split"] == s for r in rows), "examples": sum(r["n"] for r in rows if r["split"] == s),
                               "max_contact_err_m": max(r["max_contact_err_m"] for r in rows if r["split"] == s)} for s in plan}}
        (out / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
        print(json.dumps(summ["splits"], indent=1))


if __name__ == "__main__":
    main()
