import torch

from decoupattn.clustering import ClusterCache, Clusters, build_clusters, kmeans_keep, reassign_labels, tile_labels
from decoupattn.layout import Tiling, own_mask

GRID, SHAPE, H, D, C = (2, 10, 20), (1, 8, 16), 2, 8, 16


def _keys(seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    S = GRID[0] * GRID[1] * GRID[2]
    return (torch.randint(-16, 17, (1, S, H, D), generator=g).float() / 8 * scale)


def _sqdist(k, lab, cen):
    kh = k[0].transpose(0, 1)
    return float(sum(((kh[h] - cen[h][lab[h]]) ** 2).sum() for h in range(H)))


def test_kmeans_labels_are_compact():
    k = _keys(0)
    lab, cen = kmeans_keep(k[0].transpose(0, 1).contiguous(), C, 10)
    assert lab.shape == (H, k.shape[1]) and cen.shape == (H, C, D)
    for h in range(H):
        u = torch.unique(lab[h])
        assert torch.equal(u, torch.arange(len(u)))


def test_reassignment_minimises_distance_to_cached_centroids():
    til = Tiling(*GRID, SHAPE)
    k1, k2 = _keys(1), _keys(2)
    _, cen = build_clusters(k1, til, C, 10)
    cen = (cen * 8).round() / 8
    re = reassign_labels(k2, cen, til, C)
    kh = k2[0].transpose(0, 1)
    d = ((kh[:, :, None, :] - cen[:, None, :, :]) ** 2).sum(-1)
    nearest = d.argmin(-1)
    present = torch.zeros(H, C, dtype=torch.long).scatter_(1, nearest, 1)
    assert torch.equal(re.lab, torch.gather(torch.cumsum(present, 1) - 1, 1, nearest))
    best = _sqdist(k2, nearest, cen)
    assert abs(best - float(d.min(-1).values.sum())) < 1e-6
    g = torch.Generator().manual_seed(3)
    for _ in range(5):
        assert best <= _sqdist(k2, torch.randint(0, C, (H, kh.shape[1]), generator=g), cen)


def test_counts_equal_direct_count():
    til = Tiling(*GRID, SHAPE)
    g = torch.Generator().manual_seed(4)
    lab = torch.randint(0, C, (H, til.S), generator=g)
    cl = Clusters(lab, til, C)
    m = (torch.rand(H, til.nt, til.nt, generator=g) < 0.4) | own_mask(til)[None]
    n = cl.counts(m)
    t = til.tok_tile
    for h in range(H):
        for gq in range(til.nt):
            uncovered = ~m[h, gq][t]
            direct = torch.bincount(lab[h][uncovered], minlength=C).float()
            assert torch.equal(n[h, gq], direct)
    l2n = cl.l2n(m)
    assert torch.equal(torch.isinf(l2n), n == 0)
    assert torch.equal(cl.subset([1, 1]).lab, lab[[1, 1]])


def test_schedules():
    til = Tiling(*GRID, SHAPE)
    k1, k2, k3 = _keys(5), _keys(6), _keys(7)
    caches = {s: ClusterCache(til, s, C=C) for s in ("frozen", "reassign", "rebuild")}
    for c in caches.values():
        assert not c.update(k1, 0, 1)
        assert c.update(k1, 1, 1)
    first = {s: c.clusters(1) for s, c in caches.items()}
    assert all(torch.equal(first["frozen"].lab, x.lab) for x in first.values())
    assert not caches["frozen"].update(k2, 2, 1) and caches["frozen"].clusters(1) is first["frozen"]
    assert caches["reassign"].update(k2, 2, 1)
    assert torch.equal(caches["reassign"].clusters(1).lab, reassign_labels(k2, caches["reassign"].cen[(1, "cond")],
                                                                           til, C).lab)
    assert caches["rebuild"].update(k3, 2, 1)
    assert torch.equal(caches["rebuild"].clusters(1).lab, build_clusters(k3, til, C)[0].lab)
    tied = ClusterCache(til, tied=True, C=C)
    assert tied.update(k1, 1, 1) and not tied.update(k2, 2, 1)
    assert torch.equal(tied.clusters(1).lab, tile_labels(til, H))
