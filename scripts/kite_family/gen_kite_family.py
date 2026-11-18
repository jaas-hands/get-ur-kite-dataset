"""Generate KITE-native synthetic contact records for one KITE embodiment with KITE's own sampler.

Runs KITE's ContactPolicyProbeDataset (released conf/kite.yaml via paper_config.runtime_config, the
embodiment's overrides applied exactly as train_kinematic_decoder.py does), then exports the fixed
per-sample buffers in the common record schema (getkite.decoder.multifamily in jaas-hands/get-ur-kite):

    q2_t, q2_euler   contact (generating) base pose: metres, intrinsic-XYZ Euler (KITE's virtual joints)
    q2_joints        (N, D) contact joint values in KITE's joint order (written to coords.json)
    joint_active     (N, D) KITE's "contact-relevant" mask: a joint whose descendant links carry a contact.
                     KITE's target is q_goal = where(active, q2, q1); every coordinate is supervised.
    c_pos, c_normal, c_valid   (N, K, .) world-frame contacts at q2 (K = 6 padded slots), as psi sees them
    c_tpl            (N, K) template-point index of each contact (provenance for scoring; -1 = padding)
q1 is not stored: it is drawn by getkite.decoder.multifamily with KITE's gaussian recipe (fixed seeds for
evaluation pools, fresh draws for training).

Usage (CPU is fine for a pilot):
    python scripts/kite_family/gen_kite_family.py --robot allegro_right --n 2000 --seed 1 --out DIR/allegro_right_train.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

def _find_kite() -> Path:
    """KITE checkout: $KITE_ROOT, else the get-ur-kite repo's submodule or a KITE clone next to this repo."""
    if os.environ.get("KITE_ROOT"):
        return Path(os.environ["KITE_ROOT"]).expanduser().resolve()
    ws = Path(__file__).resolve().parents[3]
    for c in (ws / "get-ur-kite" / "third_party" / "KITE", ws / "KITE"):
        if (c / "networks").is_dir():
            return c.resolve()
    sys.exit("set KITE_ROOT to a KITE checkout with its runtime assets (python download_assets.py --only runtime)")


KITE = _find_kite()
sys.path.insert(0, str(KITE))
sys.path.insert(0, str(Path(__file__).resolve().parent / "shims"))  # pure-torch pytorch3d.ops.sample_farthest_points
os.chdir(KITE)  # KITE resolves assets relative to its root


def build(robot: str, n: int, seed: int, workdir: Path):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    import types
    for m in ("wandb",):  # training-only imports of train_kinematic_decoder; unused by data generation
        sys.modules.setdefault(m, types.ModuleType(m))
    import train_kinematic_decoder as T
    from contact_policy_probe import _build_probe_dataset, _build_probe_dataset_common_kwargs
    from paper_config import runtime_config

    with initialize_config_dir(config_dir=str(KITE / "conf"), version_base="1.2"):
        cfg = compose(config_name="kite")
    cfg = runtime_config(cfg)
    with open_dict(cfg):
        cfg.seed = seed
        cfg.probe.train_samples = n
        cfg.dataset.generation_device = "cpu"
        cfg.dataset.load_saved_buffers = False
        cfg.dataset.save_buffers = False
        cfg.dataset.build_batch_size = min(int(cfg.dataset.build_batch_size), 1024)
    T._maybe_materialize_contact_candidate_hparams(cfg, manipulator_names=[robot], save_dir=str(workdir))
    overrides = OmegaConf.select(cfg, "co_training.overrides", default=None)
    mcfg = T._build_per_manipulator_cfg(cfg, manipulator_name=robot, overrides_cfg=overrides)
    kw = _build_probe_dataset_common_kwargs(mcfg)
    kw["seed"] = seed
    kw["hand_points_fk_device"] = "cpu"
    kw["epoch_cache_enabled"] = False
    ds, _ = _build_probe_dataset(mcfg, common_kwargs=kw, save_dir=str(workdir), split="train",
                                 generation_device=torch.device("cpu"))
    return ds, mcfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / f".kite_work_{a.robot}_{a.seed}"
    work.mkdir(exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    ds, mcfg = build(a.robot, a.n, a.seed, work)

    base_t, base_r = list(ds.model.base_translation_indices), list(ds.model.base_rotation_indices)
    names = list(ds.model.pk_chain.get_joint_parameter_names())
    dof = int(ds.local_dof)
    jidx = [i for i in range(dof) if i not in base_t + base_r]
    q2 = ds.buffer_q2_padded[:, :dof].double()
    active = ds._build_joint_active_by_contact_mask_cpu(ds.buffer_surface_mask)[:, :dof]
    assert active[:, base_t + base_r].all(), "KITE always supervises the base pose"
    cc = ds.buffer_contact_cloud.double()
    valid = ds.buffer_contact_valid_mask.bool()
    tpl = ds.buffer_contact_parent_sparse_indices.long()
    K = int(valid.sum(1).max())
    rec = {
        "q2_t": q2[:, base_t].numpy(), "q2_euler": q2[:, base_r].numpy(), "q2_joints": q2[:, jidx].numpy(),
        "joint_active": active[:, jidx].numpy(),
        "c_pos": cc[:, :K, :3].numpy(), "c_normal": cc[:, :K, 3:6].numpy(), "c_valid": valid[:, :K].numpy(),
        "c_tpl": torch.where(valid[:, :K], tpl[:, :K], -1).numpy(),
    }
    meta = {
        "robot": a.robot, "seed": a.seed, "n": a.n, "kite_rev": os.popen(f"git -C {KITE} rev-parse HEAD").read().strip(),
        "coords": [names[i] for i in jidx], "base_translation": [names[i] for i in base_t],
        "base_rotation": [names[i] for i in base_r], "rotation_convention": "intrinsic_xyz_euler",
        "contact_mode": str(mcfg.probe.contact_mode), "sparse_sampling_mode": str(mcfg.probe.sparse_sampling_mode),
        "contact_count_range": list(mcfg.probe.contact_count_range),
        "template": str(getattr(ds.model, "surface_template_cache_path", "")),
        "sampler": "KITE ContactPolicyProbeDataset (conf/kite.yaml, runtime_config)",
    }
    np.savez_compressed(out, **rec, meta=json.dumps(meta))
    h = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(json.dumps(meta | {"sha256_16": h, "K": K, "active_frac": float(rec["joint_active"].mean())}))


if __name__ == "__main__":
    main()
