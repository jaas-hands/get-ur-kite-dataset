from __future__ import annotations

import torch


def sample_farthest_points(points, lengths=None, K=50, random_start_point=False):
    """points (N, P, D) -> (selected (N, K, D), indices (N, K)); mirrors pytorch3d.ops.sample_farthest_points."""
    N, P, D = points.shape
    if lengths is None:
        lengths = torch.full((N,), P, dtype=torch.long, device=points.device)
    if isinstance(K, int):
        K = torch.full((N,), K, dtype=torch.long, device=points.device)
    Kmax = int(K.max())
    idx_all = torch.full((N, Kmax), -1, dtype=torch.long, device=points.device)
    for n in range(N):
        L, k = int(lengths[n]), min(int(K[n]), int(lengths[n]))
        p = points[n, :L]
        start = int(torch.randint(L, (1,))) if random_start_point else 0
        dist = torch.full((L,), float("inf"), dtype=p.dtype, device=p.device)
        sel, last = [start], start
        for _ in range(1, k):
            dist = torch.minimum(dist, ((p - p[last]) ** 2).sum(-1))
            last = int(torch.argmax(dist))
            sel.append(last)
        idx_all[n, :k] = torch.tensor(sel, device=points.device)
    gathered = torch.where((idx_all >= 0)[..., None], points.gather(1, idx_all.clamp_min(0)[..., None].expand(-1, -1, D)), 0.0)
    return gathered, idx_all
