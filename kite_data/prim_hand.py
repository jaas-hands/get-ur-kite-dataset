"""Procedural primitive-geometry hands (morphology-diversity test, 2026-11-19).

The hands are built from primitives, not LEAP meshes: the palm is a box and each phalanx is a capsule. Variation:
  * 1-4 long fingers, plus an optional thumb (2-5 fingers in total);
  * 2-4 joints per finger;
  * a per-hand joint layout:
      "flex"        flexion joints only;
      "abd_first"   a side-to-side (abduction) joint, then flexion;
      "abd_second"  a 2-DOF knuckle: flexion, then abduction at the same point, then flexion;
  * phalanx lengths: nominal x U(0.6, 1.5) per link, x a per-finger and per-hand scale;
  * palm width and length: each x U(0.7, 1.4);
  * finger spacing and splay;
  * thumb mount height on the palm side, outward angle, opposition roll, and first-joint type (abduction or twist);
  * joint limits;
  * optional coupled joints (sample_coupling, added 2026-11-21; docs/coupling_and_wrist_proposal.md): in a coupled body,
    a finger's last flexion joint follows the previous flexion joint, q_last = m * q_prev (a URDF/MJCF "mimic" joint).

Output matches what LEAP variants write, so it plugs into the same pipeline:
  * hand.json in the descriptor format getkite.decoder.robots.leap_desc reads, plus "family": "prim";
  * template.npz (link, pos, normal, region, fps, link_names);
  * hand.xml, a collision MJCF for kite_data's Collider and for FK validation.

Records come from kite_data.sample.HandSampler (KITE's recipe; the kite_code regime) unchanged.

Frames:
  * Palm frame: z is distal (towards the fingers), y runs across the palm, x is the palmar normal.
  * Flexion is a positive rotation about the joint's local +y, which bends the finger towards +x.
"""

from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass, field

import numpy as np

from . import __version__, config
from .hand import Hand, HandJoint
from .transforms import axis_rotation, make_transform, matrix_to_quat_wxyz, matrix_to_rpy

PALM = config.PALM  # "palm_lower", so Hand.fk / link_dof_ancestors and the decoder's leap_desc work unchanged
FAMILY = "prim"
LAYOUTS = ("flex", "abd_first", "abd_second")
NOMINAL_PALM = (0.085, 0.095, 0.028)  # width (y), length (z), thickness (x), metres
NOMINAL_FINGER_LEN = 0.10
PHALANX_FRACTIONS = {1: (1.0,), 2: (0.55, 0.45), 3: (0.45, 0.30, 0.25), 4: (0.34, 0.27, 0.21, 0.18)}
RANGES = {
    "palm_scale": (0.7, 1.4),
    "hand_scale": (0.85, 1.2),
    "finger_scale": (0.8, 1.15),
    "link_scale": (0.6, 1.5),
    "radius": (0.0075, 0.0115),
    "splay_deg": (0.0, 8.0),
    "flex_upper": (1.3, 1.7),
    "abd_half": (0.25, 0.45),
    "thumb_height": (0.10, 0.60),  # fraction of palm length
    "thumb_out_deg": (35.0, 80.0),
    "thumb_roll_deg": (0.0, 90.0),
    "thumb_len": (0.65, 0.95),  # x the long-finger nominal length
}
FLEX_LOWER = -0.15
# Coupling (sample_coupling): drawn from its own RNG stream, so every other spec feature is unchanged.
COUPLING = {"p_body": 0.5, "p_finger": 0.75, "mult": (0.5, 1.2)}


@dataclass
class FingerSpec:
    name: str  # f0..f3 (long fingers) or "thumb"
    n_joints: int
    joint_types: list  # "flex" | "abd" | "twist", proximal -> distal
    same_point: list  # joint k shares its location with joint k+1 (its child is a geometry-less knuckle)
    link_lengths: list  # one per phalanx (= number of flexion joints)
    radius: float
    mount_xyz: list
    mount_rpy: list
    limits: list  # (lower, upper) per joint
    mimic: list | None = None  # per joint: None (independent) or [source joint k in this finger, multiplier]


@dataclass
class PrimSpec:
    id: str
    seed: int
    index: int
    attempt: int
    layout: str
    palm_size: list  # width, length, thickness
    fingers: list = field(default_factory=list)  # list[FingerSpec]

    @property
    def topology(self) -> str:
        """e.g. 'L3.4.2-T3-abd_first' (joints per long finger, thumb joints, layout)."""
        longs = [f.n_joints for f in self.fingers if f.name != "thumb"]
        th = [f.n_joints for f in self.fingers if f.name == "thumb"]
        nc = self.n_coupled
        return f"L{'.'.join(map(str, longs))}-T{th[0] if th else 0}-{self.layout}" + (f"-c{nc}" if nc else "")

    @property
    def n_coupled(self) -> int:
        return sum(m is not None for f in self.fingers for m in (f.mimic or []))

    @property
    def n_fingers(self) -> int:
        return len(self.fingers)

    @property
    def n_dof(self) -> int:
        """Independent joint coordinates (coupled joints excluded)."""
        return sum(f.n_joints for f in self.fingers) - self.n_coupled


def _joint_types(n: int, layout: str, thumb: bool, rng) -> tuple[list, list]:
    """Joint types proximal -> distal, and whether each joint shares its location with the next one (then its
    child is a geometry-less knuckle link; otherwise its child is a phalanx)."""
    if thumb and layout != "flex":
        first = "abd" if rng.random() < 0.5 else "twist"
        return [first] + ["flex"] * (n - 1), [True] + [False] * (n - 1)
    if layout == "flex":
        return ["flex"] * n, [False] * n
    if layout == "abd_first":
        return ["abd"] + ["flex"] * (n - 1), [True] + [False] * (n - 1)
    # abd_second: a 2-DOF knuckle (flexion, then abduction at the same point), then flexion
    return ["flex", "abd"] + ["flex"] * (n - 2), [True] + [False] * (n - 1)


def sample_spec(index: int, seed: int, attempt: int = 0) -> PrimSpec:
    """Deterministic in (seed, index, attempt). attempt > 0 redraws a body that failed the validity gate."""
    rng = np.random.default_rng([seed, index, attempt, zlib.crc32(b"prim")])
    U = lambda k: float(rng.uniform(*RANGES[k]))  # noqa: E731
    layout = LAYOUTS[int(rng.integers(len(LAYOUTS)))]
    n_long = int(rng.integers(1, 5))
    has_thumb = bool(rng.random() < 0.8) or n_long == 1
    W, L, T = NOMINAL_PALM[0] * U("palm_scale"), NOMINAL_PALM[1] * U("palm_scale"), NOMINAL_PALM[2]
    hand_scale = U("hand_scale")
    spacing = (0.80 * W) / max(n_long, 1)
    fingers = []
    ys = [0.0] if n_long == 1 else list(np.linspace(0.40 * W - spacing / 2, -0.40 * W + spacing / 2, n_long))
    splay_step = np.deg2rad(U("splay_deg"))
    for i, y in enumerate(ys):
        n = int(rng.integers(2, 5))
        types, same = _joint_types(n, layout, False, rng)
        n_phal = n - sum(same)
        flen = NOMINAL_FINGER_LEN * hand_scale * U("finger_scale")
        lens = [flen * fr * U("link_scale") for fr in PHALANX_FRACTIONS[n_phal]]
        r = min(U("radius") * hand_scale, spacing / 2 - 0.001, 0.45 * min(lens))  # phalanx > 2r: no cap overlap
        splay = splay_step * (i - (n_long - 1) / 2)  # fan out: R_x(+a) tilts z towards -y, finger 0 sits at +y
        lim = [(FLEX_LOWER, U("flex_upper")) if t == "flex" else (-(a := U("abd_half")), a) for t in types]
        fingers.append(FingerSpec(f"f{i}", n, types, same, lens, r, [0.0, float(y), L], [float(splay), 0.0, 0.0], lim))
    if has_thumb:
        n = int(rng.integers(2, 5))
        types, same = _joint_types(n, layout, True, rng)
        n_phal = n - sum(same)
        tlen = NOMINAL_FINGER_LEN * hand_scale * U("thumb_len")
        lens = [tlen * fr * U("link_scale") for fr in PHALANX_FRACTIONS[n_phal]]
        r = min(U("radius") * hand_scale * 1.1, 0.45 * min(lens))
        lim = []
        for t in types:
            if t == "flex":
                lim.append((FLEX_LOWER, U("flex_upper")))
            elif t == "twist":
                lim.append((-0.3, float(rng.uniform(0.9, 1.4))))
            else:
                lim.append((-(a := U("abd_half")), a))
        mount = [T / 2 * float(rng.uniform(0.0, 0.8)), -W / 2 - 0.25 * r, L * U("thumb_height")]
        # rotate the distal axis (z) about x towards -y by the outward angle, then roll about the thumb's own axis
        rpy = [float(np.deg2rad(U("thumb_out_deg"))), 0.0, 0.0]
        roll = float(np.deg2rad(U("thumb_roll_deg")))
        fingers.append(FingerSpec("thumb", n, types, same, lens, r, mount, rpy + [roll], lim))
    sid = f"P{index:04d}"
    return PrimSpec(sid, seed, index, attempt, layout, [W, L, T], fingers)


def sample_coupling(spec: PrimSpec, seed: int) -> PrimSpec:
    """Copy of spec with coupled joints. Deterministic in (seed, spec.index, spec.attempt) through a stream separate from
    sample_spec's, so all other features are unchanged. The body is coupled with probability p_body; then each finger
    with >= 2 flexion joints is coupled with probability p_finger: its last flexion joint (always the finger's last
    joint) follows the previous flexion joint, q = m * q_src, m ~ U(mult), offset 0, with limits m x the source's (so a
    coupled joint never leaves its range). The same draws are consumed for every finger, coupled or not."""
    import copy
    rng = np.random.default_rng([seed, spec.index, spec.attempt, zlib.crc32(b"couple")])
    out = copy.deepcopy(spec)
    body = bool(rng.random() < COUPLING["p_body"])
    for f in out.fingers:
        u, m = float(rng.random()), float(rng.uniform(*COUPLING["mult"]))
        flex = [k for k, t in enumerate(f.joint_types) if t == "flex"]
        f.mimic = [None] * f.n_joints
        if body and len(flex) >= 2 and u < COUPLING["p_finger"] and flex[-1] == f.n_joints - 1:
            k, src = flex[-1], flex[-2]
            f.mimic[k] = [src, m]
            lo, hi = f.limits[src]
            f.limits[k] = (m * lo, m * hi)
        if not any(x is not None for x in f.mimic):
            f.mimic = None
    return out


def _axis(t: str) -> np.ndarray:
    return {"flex": np.array([0.0, 1.0, 0.0]), "abd": np.array([1.0, 0.0, 0.0]), "twist": np.array([0.0, 0.0, 1.0])}[t]



def ensure_finger_tokens(P0: np.ndarray, region: np.ndarray, fps: np.ndarray, min_frac: float = 0.05) -> np.ndarray:
    """FPS keypoints are the decoder's surface tokens. Plain FPS can leave a short finger squeezed between neighbours
    with no token (14/900 bodies on 2026-11-19). For every finger region holding >= min_frac of the template and no
    token, replace the latest-picked token (lowest FPS priority) whose region keeps >= 1 other token with that finger's
    point farthest from the remaining tokens. Bodies already covered are returned unchanged."""
    fps = np.asarray(fps).copy()
    regs, counts = np.unique(region, return_counts=True)
    for r, c in zip(regs, counts):
        if r == 0 or c < min_frac * len(P0) or (region[fps] == r).any():
            continue
        for k in range(len(fps) - 1, -1, -1):
            if (region[fps] == region[fps[k]]).sum() > 1:
                break
        rest = np.delete(fps, k)
        cand = np.flatnonzero(region == r)
        d = np.sqrt(((P0[cand][:, None] - P0[rest][None]) ** 2).sum(-1)).min(1)
        fps[k] = cand[int(np.argmax(d))]
    return fps

class PrimHand(Hand):
    """kite_data.hand.Hand interface (fk, link_dof_ancestors, graph) on a primitive body. Geometry per link:
    palm -> box; phalanx -> capsule along local +z from 0 to `length`; knuckle links (after an abduction or twist
    joint that is followed by another joint at the same point) carry no geometry."""

    def __init__(self, spec: PrimSpec):  # noqa: D107  (Hand.__init__ is LEAP-specific; not called)
        self.spec, self.id = spec, spec.id
        self.links, self.joints = [PALM], []
        self.geom = {PALM: ("box", np.array([0.0, 0.0, spec.palm_size[1] / 2]),
                            np.array([spec.palm_size[2] / 2, spec.palm_size[0] / 2, spec.palm_size[1] / 2]))}
        self.region = {PALM: "palm"}
        self.finger_first_geom_link = {}
        n_indep = 0
        for f in spec.fingers:
            parent, last_dof, phal = PALM, -1, 0
            kdof = {}  # joint k in this finger -> its dof (independent joints)
            R = axis_rotation(np.array([1.0, 0.0, 0.0]), f.mount_rpy[0])
            if len(f.mount_rpy) > 3:  # thumb: then roll about its own distal axis
                R = R @ axis_rotation(np.array([0.0, 0.0, 1.0]), f.mount_rpy[3])
            origin = make_transform(f.mount_xyz, R)
            for k, (t, (lo, hi)) in enumerate(zip(f.joint_types, f.limits)):
                child = f"{f.name}_l{k}"
                j = HandJoint(f"{f.name}_j{k}", f.name, k, parent, child, origin, _axis(t), float(lo), float(hi),
                              1.0, 8.0, False, parent_dof=last_dof)
                mm = f.mimic[k] if f.mimic else None
                if mm is None:
                    j.dof = kdof[k] = n_indep
                    n_indep += 1
                    last_dof = j.dof
                else:  # coupled: follows joint mm[0] of this finger; always the finger's last joint (a leaf)
                    assert k == f.n_joints - 1 and mm[0] in kdof, f"{spec.id}: unsupported coupling {f.name} j{k}"
                    j.mimic = (kdof[mm[0]], float(mm[1]), 0.0)
                self.joints.append(j)
                self.links.append(child)
                self.region[child] = f.name
                if f.same_point[k]:
                    origin = np.eye(4)
                else:
                    ln = f.link_lengths[phal]
                    phal += 1
                    self.geom[child] = ("capsule", float(ln), float(f.radius))
                    self.finger_first_geom_link.setdefault(f.name, child)
                    origin = make_transform([0.0, 0.0, ln])
                parent = child
        self.dof_joints = [j for j in self.joints if j.mimic is None]
        self.n_dof = len(self.dof_joints)
        assert self.n_dof == spec.n_dof
        self.link_parent_joint = {j.child: j for j in self.joints}

    @property
    def topology_id(self) -> str:
        return self.spec.topology

    # ------------------------------------------------------------ collision model
    def geom_links(self) -> list:
        return [l for l in self.links if l in self.geom]

    def adjacent_geom_pairs(self) -> list:
        """Pairs of geometry-bearing links with no other geometry-bearing link between them on the kinematic path
        (they touch by construction at a joint, so they are excluded from collision checks); MuJoCo's parent filter
        alone misses pairs separated by a geometry-less knuckle link."""
        out = set()
        for l in self.geom_links():
            if l == PALM:
                continue
            p = self.link_parent_joint[l].parent
            while p != PALM and p not in self.geom:
                p = self.link_parent_joint[p].parent
            out.add(tuple(sorted((l, p))))
        return sorted(out)

    def to_mjcf(self) -> str:
        children = {l: [] for l in self.links}
        for j in self.joints:
            children[j.parent].append(j)
        f = lambda v: " ".join(f"{x:.9g}" for x in np.ravel(v))  # noqa: E731

        def body(link, ind):
            out = []
            g = self.geom.get(link)
            if g is None:
                out.append(f'{ind}<inertial pos="0 0 0" mass="0.001" diaginertia="1e-7 1e-7 1e-7"/>')
            elif g[0] == "box":
                out.append(f'{ind}<geom name="{link}_c0" type="box" pos="{f(g[1])}" size="{f(g[2])}" mass="0.3"/>')
            else:
                out.append(f'{ind}<geom name="{link}_c0" type="capsule" fromto="0 0 0 0 0 {g[1]:.9g}" size="{g[2]:.9g}" mass="0.02"/>')
            for j in children[link]:
                out.append(f'{ind}<body name="{j.child}" pos="{f(j.origin[:3, 3])}" quat="{f(matrix_to_quat_wxyz(j.origin[:3, :3]))}">')
                out.append(f'{ind}  <joint name="{j.name}" axis="{f(j.axis)}" range="{j.lower:.9g} {j.upper:.9g}"/>')
                out += body(j.child, ind + "  ")
                out.append(f"{ind}</body>")
            return out

        excl = [f'    <exclude body1="{a}" body2="{b}"/>' for a, b in self.adjacent_geom_pairs()]
        eq = [f'    <joint joint1="{j.name}" joint2="{self.dof_joints[j.mimic[0]].name}" polycoef="{j.mimic[2]:.9g} {j.mimic[1]:.9g} 0 0 0"/>'
              for j in self.joints if j.mimic is not None]
        return "\n".join([f'<mujoco model="prim_{self.id}">', '  <compiler angle="radian" autolimits="true"/>',
                          '  <default><geom contype="1" conaffinity="1"/></default>', "  <worldbody>",
                          f'    <body name="{PALM}">', *body(PALM, "      "), "    </body>", "  </worldbody>",
                          "  <contact>", *excl, "  </contact>", *(["  <equality>", *eq, "  </equality>"] if eq else []),
                          "</mujoco>", ""])

    # ------------------------------------------------------------ surface template
    def _inside(self, link_frames, link, P_world) -> np.ndarray:
        """Is each world point strictly inside link's primitive (margin 0.2 mm)?"""
        T = link_frames[link]
        Pl = (P_world - T[:3, 3]) @ T[:3, :3]
        g = self.geom[link]
        if g[0] == "box":
            return np.all(np.abs(Pl - g[1]) < g[2] - 2e-4, axis=1)
        z = np.clip(Pl[:, 2], 0.0, g[1])
        d = np.linalg.norm(Pl - np.stack([np.zeros_like(z), np.zeros_like(z), z], 1), axis=1)
        return d < g[2] - 2e-4

    def _surface(self, link, n, rng):
        g = self.geom[link]
        if g[0] == "box":
            c, h = g[1], g[2]
            areas = np.array([h[1] * h[2], h[1] * h[2], h[0] * h[2], h[0] * h[2], h[0] * h[1], h[0] * h[1]]) * 4
            face = rng.choice(6, size=n, p=areas / areas.sum())
            u = rng.uniform(-1, 1, size=(n, 3))
            ax, sgn = face // 2, np.where(face % 2 == 0, 1.0, -1.0)
            P = u * h
            P[np.arange(n), ax] = sgn * h[ax]
            N = np.zeros((n, 3)); N[np.arange(n), ax] = sgn
            return P + c, N, float(areas.sum() / n)
        L, r = g[1], g[2]
        a_cyl, a_sph = 2 * np.pi * r * L, 4 * np.pi * r * r
        cyl = rng.random(n) < a_cyl / (a_cyl + a_sph)
        th = rng.uniform(0, 2 * np.pi, n)
        N = np.stack([np.cos(th), np.sin(th), np.zeros(n)], 1)
        P = np.stack([r * np.cos(th), r * np.sin(th), rng.uniform(0, L, n)], 1)
        m = ~cyl
        v = rng.normal(size=(m.sum(), 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        cap = np.where(v[:, 2:] >= 0, L, 0.0)
        N[m], P[m] = v, v * r + np.concatenate([np.zeros((m.sum(), 2)), cap], 1)
        return P, N, float((a_cyl + a_sph) / n)

    def build_template(self, seed: int = 0) -> dict:
        """4096 area-weighted surface points (link frame) with outward normals, minus points buried in a touching
        neighbour (joint ends inside the next phalanx, finger bases inside the palm); 32 FPS indices at q = 0."""
        rng = np.random.default_rng([seed, zlib.crc32(self.id.encode())])
        T0 = self.fk(np.zeros(self.n_dof))
        adj = {}
        for a, b in self.adjacent_geom_pairs():
            adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
        pts, nrm, wts, lnk = [], [], [], []
        gl = self.geom_links()
        for li, link in enumerate(self.links):
            if link not in self.geom:
                continue
            P, N, w = self._surface(link, 20000, rng)
            Pw = P @ T0[link][:3, :3].T + T0[link][:3, 3]
            keep = np.ones(len(P), bool)
            for other in adj.get(link, []):
                keep &= ~self._inside(T0, other, Pw)
            pts.append(P[keep]); nrm.append(N[keep]); wts.append(np.full(keep.sum(), w)); lnk.append(np.full(keep.sum(), li))
        P, N, W, Lk = map(np.concatenate, (pts, nrm, wts, lnk))
        pick = np.sort(rng.choice(len(P), size=config.TEMPLATE_SIZE, replace=False, p=W / W.sum()))
        P, N, Lk = P[pick], N[pick], Lk[pick]
        names = ["palm"] + [f.name for f in self.spec.fingers]
        region = np.array([names.index(self.region[self.links[i]]) for i in Lk], dtype=np.int8)
        P0 = np.empty_like(P)
        for li in np.unique(Lk):
            m = Lk == li
            T = T0[self.links[li]]
            P0[m] = P[m] @ T[:3, :3].T + T[:3, 3]
        from .template import farthest_point_sampling
        assert set(np.unique(Lk)) == {self.links.index(l) for l in gl}, "a link lost all its template points"
        return {"link": Lk.astype(np.int16), "pos": P.astype(np.float32), "normal": N.astype(np.float32),
                "region": region, "fps": ensure_finger_tokens(P0, region, farthest_point_sampling(P0, config.FPS_SIZE)).astype(np.int16),
                "link_names": np.array(self.links), "region_names": np.array(names)}

    # ------------------------------------------------------------ descriptor (leap_desc format)
    def descriptor(self) -> dict:
        depth, has_child = {}, {j.parent_dof for j in self.joints}
        for j in self.dof_joints:
            depth[j.dof] = 1 if j.parent_dof < 0 else depth[j.parent_dof] + 1
        dep = lambda j: depth[j.dof] if j.mimic is None else (1 if j.parent_dof < 0 else depth[j.parent_dof] + 1)  # noqa: E731
        joints = [{"name": j.name, "finger": j.finger, "k": j.k, "parent_link": j.parent, "child_link": j.child,
                   "origin_xyz": np.round(j.origin[:3, 3], 9).tolist(),
                   "origin_rpy": np.round(matrix_to_rpy(j.origin[:3, :3]), 9).tolist(),
                   "origin_quat_wxyz": np.round(matrix_to_quat_wxyz(j.origin[:3, :3]), 9).tolist(),
                   "axis": j.axis.tolist(), "lower": j.lower, "upper": j.upper, "locked": False, "dof": j.dof,
                   "parent_dof": j.parent_dof, "depth": dep(j), "leaf": j.mimic is not None or j.dof not in has_child}
                  | ({"mimic": {"joint": self.dof_joints[j.mimic[0]].name, "multiplier": j.mimic[1], "offset": j.mimic[2]}}
                     if j.mimic is not None else {})
                  for j in self.joints]
        spec = asdict(self.spec)
        for f in spec["fingers"]:  # uncoupled bodies keep the pre-coupling descriptor exactly
            if f.get("mimic") is None:
                f.pop("mimic", None)
        return {"id": self.id, "family": FAMILY, "topology": self.spec.topology,
                "fingers": [f.name for f in self.spec.fingers], "n_dof": self.n_dof, "spec": spec,
                "dof_names": [j.name for j in self.dof_joints],
                "links": [{"name": l, "region": self.region[l],
                           "geom": None if l not in self.geom else [self.geom[l][0]] + np.ravel(
                               [np.ravel(x) for x in self.geom[l][1:]]).tolist(),
                           "parent_joint": None if l == PALM else self.link_parent_joint[l].name} for l in self.links],
                "joints": joints,
                "generator": {"kite_data": __version__, "module": "kite_data.prim_hand", "lock_angle": 0.0,
                              "ranges": RANGES} | ({"coupling": COUPLING} if self.spec.n_coupled else {})}


def spec_from_dict(d: dict) -> PrimSpec:
    fs = [FingerSpec(**f) for f in d["fingers"]]
    return PrimSpec(**{**d, "fingers": fs})


def load_prim(hand_dir) -> PrimHand:
    import json
    from pathlib import Path
    d = json.loads((Path(hand_dir) / "hand.json").read_text())
    return PrimHand(spec_from_dict(d["spec"]))
