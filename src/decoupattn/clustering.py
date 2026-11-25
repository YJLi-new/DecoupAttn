from __future__ import annotations

import torch

from .constants import BUILD_STEP, CLUSTERS, KMEANS_ITERS

KMEANS_SEED = 2
SCHEDULES = ("frozen", "reassign", "rebuild")


def _assign_torch(xa, cen, lab32, chunk=4096):
    c2 = (cen * cen).sum(-1)
    ca = (cen.bfloat16() if xa.dtype == torch.bfloat16 else cen).float()
    for s0 in range(0, xa.shape[1], chunk):
        x = xa[:, s0:s0 + chunk].float()
        lab32[:, s0:s0 + chunk] = (c2[:, None, :] - 2.0 * torch.bmm(x, ca.transpose(1, 2))).argmin(-1).to(torch.int32)


def _seg_means_torch(x, lab, G, old):
    H, S, D = x.shape
    lab = lab.long()
    cnt = torch.zeros(H, G, device=x.device).scatter_add_(1, lab, torch.ones(H, S, device=x.device))
    sm = torch.zeros(H, G, D, device=x.device).scatter_add_(1, lab[..., None].expand(H, S, D), x)
    mean = sm / cnt.clamp(min=1)[..., None]
    return torch.where((cnt > 0)[..., None], mean, old)


@torch.no_grad()
def kmeans_keep(x, G, iters, seed=KMEANS_SEED, bf16=True, init=None):
    H, S, D = x.shape
    if init is None:
        g = torch.Generator(device=x.device).manual_seed(seed)
        idx = torch.stack([torch.randperm(S, generator=g, device=x.device)[:G] for _ in range(H)])
        C = torch.gather(x, 1, idx[..., None].expand(H, G, D)).contiguous()
    else:
        C = init.clone()
    lab32 = torch.empty(H, S, dtype=torch.int32, device=x.device)
    xa = x.bfloat16() if bf16 else x
    cuda = x.is_cuda
    if cuda:
        from .kernels import kmeans as KM
    for it in range(iters + 1):
        if cuda:
            KM.assign(xa, C, lab32)
        else:
            _assign_torch(xa, C, lab32)
        if it == iters:
            break
        C = KM.seg_stats(x, lab32, G, old=C)[0] if cuda else _seg_means_torch(x, lab32, G, C)
    labs = lab32.long()
    present = torch.zeros(H, G, dtype=torch.long, device=x.device).scatter_(1, labs, 1)
    return torch.gather(torch.cumsum(present, 1) - 1, 1, labs), C


def kmeans_tf32(k, C, iters, init):
    kh = k[0].transpose(0, 1).float().contiguous()
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        return kmeans_keep(kh, C, iters, init=init)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


class Clusters:
    def __init__(self, lab, til, C=CLUSTERS):
        lab = lab.long()
        H, S = lab.shape
        self.lab, self.til, self.C, self.H, self.S = lab, til, C, H, S
        self.k_idx = torch.sort(lab, dim=1, stable=True).indices.to(torch.int32).contiguous()
        cnt = torch.zeros(H, C, dtype=torch.long, device=lab.device).scatter_add_(1, lab, torch.ones_like(lab))
        self.k_off = torch.cat([cnt.new_zeros(H, 1), torch.cumsum(cnt, 1)], 1).to(torch.int32).contiguous()
        self.cnt = cnt
        tt = til.tok_tile.to(lab.device)
        self.pc = torch.zeros(H, til.nt * C, device=lab.device).scatter_add_(
            1, tt[None] * C + lab, torch.ones(H, S, device=lab.device)).view(H, til.nt, C).bfloat16()

    def subset(self, heads) -> Clusters:
        if len(heads) == self.H and all(int(h) == i for i, h in enumerate(heads)):
            return self
        return Clusters(self.lab[torch.as_tensor(heads, device=self.lab.device).long()], self.til, self.C)

    def counts(self, mask_hnn) -> torch.Tensor:
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return (self.cnt.float()[:, None, :] - torch.bmm(mask_hnn.to(self.pc.device).float(),
                                                             self.pc.float())).round()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

    def l2n(self, mask_hnn) -> torch.Tensor:
        n = self.counts(mask_hnn)
        return torch.where(n > 0.5, torch.log2(n.clamp(min=1)), torch.full_like(n, float("-inf"))).contiguous()


def tile_labels(til, H, C=CLUSTERS) -> torch.Tensor:
    t = til.tok_tile.long()
    n = int(t.max()) + 1
    return (t * C // n if n > C else t)[None].expand(H, -1).contiguous()


def build_clusters(k, til, C=CLUSTERS, iters=KMEANS_ITERS):
    lab, cen = kmeans_tf32(k, C, iters, None)
    return Clusters(lab, til, C), cen


def reassign_labels(k, centroids, til, C=CLUSTERS):
    lab, _ = kmeans_tf32(k, C, 0, centroids)
    return Clusters(lab, til, C)


class ClusterCache:
    def __init__(self, til, schedule="reassign", tied=False, C=CLUSTERS, iters=KMEANS_ITERS, build_step=BUILD_STEP):
        assert schedule in SCHEDULES, schedule
        self.til, self.schedule, self.tied, self.C, self.iters, self.b = til, schedule, tied, C, iters, build_step
        self.reset()

    def reset(self):
        self.lab, self.cen = {}, {}

    def update(self, k, step, layer, branch="cond") -> bool:
        key = (layer, branch)
        if self.tied:
            if step != self.b:
                return False
            self.lab[key] = Clusters(tile_labels(self.til, k.shape[2], self.C), self.til, self.C)
            return True
        if step == self.b or (self.schedule == "rebuild" and step > self.b and key in self.cen):
            self.lab[key], self.cen[key] = build_clusters(k, self.til, self.C, self.iters)
            return True
        if self.schedule == "reassign" and step > self.b and key in self.cen:
            self.lab[key] = reassign_labels(k, self.cen[key], self.til, self.C)
            return True
        return False

    def clusters(self, layer, branch="cond") -> Clusters | None:
        return self.lab.get((layer, branch))

    def get(self, k, step, layer, branch="cond") -> Clusters | None:
        self.update(k, step, layer, branch)
        return self.clusters(layer, branch)

    def drop(self, layer, branch="cond"):
        self.lab.pop((layer, branch), None)
