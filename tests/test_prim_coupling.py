"""Coupled joints in primitive bodies (kite_data.prim_hand.sample_coupling; 2026-11-21)."""
import json

import mujoco
import numpy as np

from kite_data.collision import Collider
from kite_data.prim_hand import PrimHand, sample_coupling, sample_spec, spec_from_dict

SEED = 20261119


def _coupled(n=40):
    out = []
    for i in range(n):
        s = sample_spec(i, SEED)
        c = sample_coupling(s, SEED)
        out.append((s, c))
    return out


def test_coupling_deterministic_and_leaves_other_features():
    for s, c in _coupled():
        c2 = sample_coupling(s, SEED)
        assert json.dumps(c.__dict__, default=lambda o: o.__dict__) == json.dumps(c2.__dict__, default=lambda o: o.__dict__)
        assert c.layout == s.layout and c.palm_size == s.palm_size and len(c.fingers) == len(s.fingers)
        for fs, fc in zip(s.fingers, c.fingers):
            assert fs.joint_types == fc.joint_types and fs.link_lengths == fc.link_lengths and fs.mount_xyz == fc.mount_xyz
            for k, m in enumerate(fc.mimic or [None] * fc.n_joints):
                if m is None:
                    assert tuple(fc.limits[k]) == tuple(fs.limits[k])
                else:  # last joint, follows an earlier flexion joint, limits = m x source limits
                    assert k == fc.n_joints - 1 and fc.joint_types[k] == fc.joint_types[m[0]] == "flex"
                    assert np.allclose(fc.limits[k], np.array(fs.limits[m[0]]) * m[1])
        assert c.n_dof == s.n_dof - c.n_coupled


def test_some_bodies_coupled_some_not():
    n = [c.n_coupled for _, c in _coupled()]
    assert any(x == 0 for x in n) and any(x > 0 for x in n)


def test_uncoupled_descriptor_unchanged():
    for s, c in _coupled():
        if c.n_coupled == 0:
            assert json.dumps(PrimHand(c).descriptor()) == json.dumps(PrimHand(s).descriptor())
            assert PrimHand(c).to_mjcf() == PrimHand(s).to_mjcf()


def test_coupled_fk_matches_mujoco_and_collider_sets_mimic():
    rng = np.random.default_rng(0)
    for s, c in _coupled():
        if not c.n_coupled:
            continue
        h = PrimHand(c)
        assert h.n_dof == len(h.dof_joints) == c.n_dof
        m = mujoco.MjModel.from_xml_string(h.to_mjcf()); d = mujoco.MjData(m)
        assert m.neq == c.n_coupled
        q = rng.uniform(h.lower, h.upper)
        for j in h.joints:
            d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j.name)]] = q[j.dof] if j.mimic is None else j.mimic[1] * q[j.mimic[0]]
        mujoco.mj_kinematics(m, d)
        T = h.fk(q)
        for l in h.links:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, l)
            assert np.abs(d.xpos[b] - T[l][:3, 3]).max() < 1e-8
        col = Collider(h, h.to_mjcf())
        col.colliding_pairs(q)
        for adr, src, mult, off in col.mimic:
            assert abs(col.data.qpos[adr] - (mult * q[src] + off)) < 1e-12
        # the joint mask: a coupled joint's link has its source coordinate as an ancestor
        for j in h.joints:
            if j.mimic is not None:
                assert j.mimic[0] in h.link_dof_ancestors(j.child)


def test_descriptor_roundtrip():
    for s, c in _coupled(10):
        h = PrimHand(c)
        h2 = PrimHand(spec_from_dict(h.descriptor()["spec"]))
        assert h2.spec.n_coupled == c.n_coupled and h2.to_mjcf() == h.to_mjcf()
