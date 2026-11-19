"""Read exported LEAP kinematics without regenerating a hand or requiring meshes."""

import numpy as np

from .transforms import axis_rotation, make_transform, quat_wxyz_to_matrix


class DescriptorHand:
    """Evaluation-only hand. Stored joint origins already include locked-joint angles."""

    def __init__(self, desc):
        self.id = desc["id"]
        self.links = [x["name"] for x in desc["links"]]
        self.root = self.links[0]
        self.n_dof = desc["n_dof"]
        self.joints = desc["joints"]
        active = sorted((j for j in self.joints if not j["locked"]), key=lambda j: j["dof"])
        if ([j["dof"] for j in active] != list(range(self.n_dof))
                or [j["name"] for j in active] != desc["dof_names"]):
            raise ValueError("descriptor joint order does not match dof_names")
        self.lower = np.array([j["lower"] for j in active], float)
        self.upper = np.array([j["upper"] for j in active], float)
        if not np.isfinite([self.lower, self.upper]).all() or np.any(self.lower > self.upper):
            raise ValueError("invalid descriptor joint limits")
        self.origins = []
        seen = {self.root}
        self.parents = {}
        for j in self.joints:
            if j["parent_link"] not in seen or j["child_link"] in seen:
                raise ValueError("descriptor joints must form a tree in parent-first order")
            if (j["locked"] and j["dof"] != -1) or not np.isclose(np.linalg.norm(j["axis"]), 1):
                raise ValueError("invalid locked joint or non-unit joint axis")
            origin = make_transform(j["origin_xyz"], quat_wxyz_to_matrix(j["origin_quat_wxyz"]))
            if not np.isfinite(origin).all():
                raise ValueError("nonfinite joint origin")
            self.origins.append(origin)
            seen.add(j["child_link"])
            self.parents[j["child_link"]] = j
        if len(self.links) != len(seen) or set(self.links) != seen:
            raise ValueError("descriptor links do not match joint tree")

    def link_dof_ancestors(self, link):
        result = []
        while link != self.root:
            joint = self.parents[link]
            if not joint["locked"]:
                result.append(joint["dof"])
            link = joint["parent_link"]
        return sorted(result)

    def fk(self, q, wrist=None):
        q = np.asarray(q, float)
        shape = q.shape[:-1] + (4, 4)
        frames = {self.root: np.broadcast_to(np.eye(4) if wrist is None else wrist, shape)}
        for j, origin in zip(self.joints, self.origins):
            frame = frames[j["parent_link"]] @ origin
            if not j["locked"]:
                rotation = np.zeros(shape)
                rotation[..., :3, :3] = axis_rotation(j["axis"], q[..., j["dof"]])
                rotation[..., 3, 3] = 1
                frame = frame @ rotation
            frames[j["child_link"]] = frame
        return frames
