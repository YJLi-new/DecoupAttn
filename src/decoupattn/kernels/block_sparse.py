from __future__ import annotations

import math

import torch

from ..layout import exec_density, pseudo_density, real_density
from ._triton import jit, tl

GS = 128
MERGE = {"RB": 64, "BN": 64, "warps": 4, "stages": 2, "hilo": False}


_FLOAT_WS: dict = {}
_BUFS: dict = {}
SCRATCH = {"hmax": None}
_SHAPES = {"qp": "HSD", "kp": "HSD", "vp": "HSD", "os": "HSD", "lse": "HS", "kbh": "HCD", "kbl": "HCD", "vb": "HCD"}
_BIG: dict = {}


def _float_ws(device):
    b = _FLOAT_WS.get(str(device))
    if b is None:
        b = _FLOAT_WS[str(device)] = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    return b


def set_scratch_heads(hmax: int | None) -> None:
    SCRATCH["hmax"] = hmax
    _BUFS.clear()
    _BIG.clear()


def _bufs(H, S, C, D, dtype, device):
    key = (H, S, C, D, dtype, str(device))
    b = _BUFS.get(key)
    if b is not None:
        return b
    hmax = SCRATCH["hmax"]
    if hmax is None:
        e = lambda *s, dt=dtype: torch.empty(*s, dtype=dt, device=device)
        b = {"qp": e(H, S, D), "kp": e(H, S, D), "vp": e(H, S, D), "os": e(H, S, D),
             "lse": e(H, S, dt=torch.float32), "kbh": e(H, C, D), "kbl": e(H, C, D), "vb": e(H, C, D)}
    else:
        bk = key[1:]
        big = _BIG.get(bk)
        if big is None or big[0] < H:
            for vk in [x for x in _BUFS if x[1:] == bk]:
                del _BUFS[vk]
            hb = max(H, hmax)
            dim = {"H": hb, "S": S, "C": C, "D": D}
            big = _BIG[bk] = (hb, {n: torch.empty(math.prod(dim[c] for c in sh),
                                                  dtype=torch.float32 if n == "lse" else dtype, device=device)
                                   for n, sh in _SHAPES.items()})
        dim = {"H": H, "S": S, "C": C, "D": D}
        b = {}
        for n, sh in _SHAPES.items():
            shp = [dim[c] for c in sh]
            b[n] = big[1][n][:math.prod(shp)].view(*shp)
    _BUFS[key] = b
    return b


@jit
def _gather_rows(X, XP, IDX, sx_s, sx_h, S_real, S, D: tl.constexpr, BM: tl.constexpr):
    i = tl.program_id(0)
    h = tl.program_id(1)
    rows = i * BM + tl.arange(0, BM)
    d = tl.arange(0, D)
    src = tl.load(IDX + h * S + rows).to(tl.int64)
    ok = (src < S_real)[:, None]
    x = tl.load(X + src[:, None] * sx_s + h * sx_h + d[None, :], mask=ok, other=0.0)
    tl.store(XP + (h * S + rows[:, None]).to(tl.int64) * D + d[None, :], x)


@jit
def _gather_kv_mean_np(K, V, KP, VP, KBH, KBL, VB, IDX, OFF, sx_s, sx_h, S_real, S, C,
                       D: tl.constexpr, BR: tl.constexpr):
    c = tl.program_id(0)
    h = tl.program_id(1)
    o0 = tl.load(OFF + h * (C + 1) + c)
    o1 = tl.load(OFF + h * (C + 1) + c + 1)
    d = tl.arange(0, D)
    ks = tl.zeros([D], tl.float32)
    vs = tl.zeros([D], tl.float32)
    for r0 in range(o0, o1, BR):
        rows = r0 + tl.arange(0, BR)
        m = rows < o1
        src = tl.load(IDX + h * S + rows, mask=m, other=0).to(tl.int64)
        ld = m & (src < S_real)
        soff = src[:, None] * sx_s + h * sx_h + d[None, :]
        k = tl.load(K + soff, mask=ld[:, None], other=0.0)
        v = tl.load(V + soff, mask=ld[:, None], other=0.0)
        doff = (h * S + rows[:, None]).to(tl.int64) * D + d[None, :]
        tl.store(KP + doff, k, mask=m[:, None])
        tl.store(VP + doff, v, mask=m[:, None])
        ks += tl.sum(k.to(tl.float32), 0)
        vs += tl.sum(v.to(tl.float32), 0)
    inv = 1.0 / tl.maximum((o1 - o0).to(tl.float32), 1.0)
    kb = ks * inv
    kh = kb.to(KBH.dtype.element_ty)
    o = (h * C + c) * D + d
    tl.store(KBH + o, kh)
    tl.store(KBL + o, (kb - kh.to(tl.float32)).to(KBL.dtype.element_ty))
    tl.store(VB + o, (vs * inv).to(VB.dtype.element_ty))


@jit
def _merge(QP, OS, LSE, KBH, KBL, VB, L2N, MASK, QIDX, OUT, S, C, Gq, sout_s, qk_scale_log2,
           D: tl.constexpr, RB: tl.constexpr, BN: tl.constexpr, G_SZ: tl.constexpr, COMP: tl.constexpr,
           HILO: tl.constexpr):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    rows = pid * RB + tl.arange(0, RB)
    d = tl.arange(0, D)
    qoff = (h * S + rows[:, None]).to(tl.int64) * D + d[None, :]
    o_s = tl.load(OS + qoff).to(tl.float32)
    dst = tl.load(QIDX + h * S + rows).to(tl.int64)
    if COMP:
        g = (pid * RB) // G_SZ
        q = tl.load(QP + qoff)
        m_i = tl.full([RB], float("-inf"), tl.float32)
        l_i = tl.zeros([RB], tl.float32)
        acc = tl.zeros([RB, D], tl.float32)
        for j0 in range(0, C, BN):
            cols = j0 + tl.arange(0, BN)
            cv = cols < C
            koff = (h * C + cols[:, None]) * D + d[None, :]
            kh = tl.load(KBH + koff, mask=cv[:, None], other=0.0)
            s = tl.dot(q, tl.trans(kh))
            if HILO:
                kl = tl.load(KBL + koff, mask=cv[:, None], other=0.0)
                s = tl.dot(q, tl.trans(kl), s)
            s = s * qk_scale_log2
            l2n = tl.load(L2N + h * C + cols, mask=cv, other=float("-inf"))
            sel = tl.load(MASK + (h * Gq + g) * C + cols, mask=cv, other=1)
            ok = (sel == 0) & (l2n > float("-inf"))
            s = tl.where(ok[None, :], s + l2n[None, :], float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp2(m_i - m_safe)
            p = tl.exp2(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            vb = tl.load(VB + koff, mask=cv[:, None], other=0.0)
            acc = tl.dot(p.to(VB.dtype.element_ty), vb, acc * alpha[:, None])
            m_i = m_new
        m_safe = tl.where(m_i == float("-inf"), 0.0, m_i)
        lse_c = tl.where(l_i > 0, m_safe + tl.log2(tl.maximum(l_i, 1e-30)), float("-inf"))
        lse_s = tl.load(LSE + h * S + rows)
        mx = tl.maximum(lse_s, lse_c)
        w_s = tl.exp2(lse_s - mx)
        w_c = tl.exp2(lse_c - mx)
        o = (o_s * w_s[:, None] + acc * (w_c / tl.maximum(l_i, 1e-30))[:, None]) / (w_s + w_c)[:, None]
    else:
        o = o_s
    tl.store(OUT + dst[:, None] * sout_s + h * D + d[None, :], o.to(OUT.dtype.element_ty))


@jit
def _kv_index_kernel(OUT, BASE, START, LEN, BL: tl.constexpr):
    t = tl.program_id(0)
    b = tl.load(BASE + t)
    s = tl.load(START + t).to(tl.int64)
    n = tl.load(LEN + t)
    r = tl.arange(0, BL)
    tl.store(OUT + s + r, b + r, mask=r < n)


_IDXBUF: dict = {}


def _idx_buf(n, device):
    b = _IDXBUF.get(str(device))
    if b is None or b.numel() < n:
        b = _IDXBUF[str(device)] = torch.empty(int(n * 1.05) + 1024, dtype=torch.int32, device=device)
    return b


def tiling_side(cq, ck, C):
    H = cq.shape[0]
    cql, ckl = cq.long(), ck.long()
    q_idx = torch.sort(cql, dim=1, stable=True).indices.to(torch.int32).contiguous()
    k_idx = torch.sort(ckl, dim=1, stable=True).indices.to(torch.int32).contiguous()
    cnt = torch.zeros(H, C, dtype=torch.long, device=cq.device).scatter_add_(1, ckl, torch.ones_like(ckl))
    k_off = torch.cat([cnt.new_zeros(H, 1), torch.cumsum(cnt, 1)], 1).to(torch.int32).contiguous()
    return q_idx, k_idx, k_off, cnt


class StaticPlan:
    def __init__(self, cq, ck, allowed, head_dim=128, backend="fa3", dtype=torch.bfloat16, pad_cluster=None,
                 int_ws_bytes=2 * 1024 * 1024, side=None):
        import flashinfer.sparse as fs
        from flashinfer.sparse import VariableBlockSparseAttentionWrapper

        H, S = cq.shape
        Gq = S // GS
        C = allowed.shape[-1]
        dev = cq.device
        self.H, self.S, self.Gq, self.C, self.D, self.backend = H, S, Gq, C, head_dim, backend
        self.q_idx, self.k_idx, self.k_off, cnt = side if side is not None else tiling_side(cq, ck, C)
        ne = cnt > 0
        al = allowed[:, :Gq].bool() & ne[:, None, :]
        assert bool(al.any(-1).all()), "every query tile must select >= 1 non-empty key block"
        self.mask_u8 = al.to(torch.uint8).contiguous()
        self.l2n = torch.where(ne, torch.log2(cnt.clamp(min=1).float()),
                               torch.full_like(cnt, float("-inf"), dtype=torch.float)).contiguous()
        if pad_cluster is not None:
            self.l2n[:, pad_cluster] = float("-inf")
            self.mask_u8[:, :, pad_cluster] = 1
        sel_keys = (al.long() * cnt[:, None, :]).sum(-1)
        unsel = ((~al) & ne[:, None, :]).sum(-1)
        self.pairs = int(sel_keys.sum()) * GS
        self.pseudo_pairs = int(unsel.sum()) * GS
        self.density_keys = self.pairs / (H * S * S)
        i32 = torch.int32
        cnt32 = cnt.to(i32)
        qo_indptr = torch.arange(0, H * Gq + 1, device=dev, dtype=i32) * GS
        row_len = (al.to(i32) * cnt32[:, None, :]).sum(-1, dtype=i32).flatten()
        self.exec_pairs = int((GS * ((row_len.long() + 95) // 96 * 96)).sum())
        kv_indptr = torch.zeros(H * Gq + 1, dtype=i32, device=dev)
        kv_indptr[1:] = torch.cumsum(row_len, 0, dtype=i32)
        col_off = torch.cumsum(cnt32, 1, dtype=i32) - cnt32 + (torch.arange(H, device=dev, dtype=i32) * S)[:, None]
        h_i, _, c_i = al.nonzero(as_tuple=True)
        lengths = cnt32[h_i, c_i]
        self.total = int(kv_indptr[-1])
        starts = torch.cumsum(lengths, 0, dtype=torch.int64) - lengths
        self.base = col_off[h_i, c_i].contiguous()
        self.start = starts.to(i32).contiguous() if self.total < 2**31 else starts.contiguous()
        self.len = lengths.contiguous()
        self.n_trip = int(self.base.numel())
        assert int(self.len.max()) <= 128
        qo_host, kv_host = qo_indptr.cpu(), kv_indptr.cpu()
        w = VariableBlockSparseAttentionWrapper(_float_ws(dev), backend="fa2")
        w._int_workspace_buffer = torch.empty(int_ws_bytes, dtype=torch.uint8, device=dev)
        w._pin_memory_int_workspace_buffer = torch.empty(int_ws_bytes, dtype=torch.uint8, pin_memory=True)
        w._qo_indptr = qo_indptr
        w._paged_kv_indptr_buf = kv_indptr
        w._paged_kv_indices_buf = None
        w._paged_kv_last_page_len = torch.ones(H * Gq, dtype=i32, device=dev)
        w._mask_mode = fs.MaskMode.NON_CAUSAL.value
        w._o_dtype = dtype
        w._backend = backend
        w._cached_module = fs.get_batch_prefill_module(backend, dtype, dtype, dtype, i32, head_dim, head_dim,
                                                       fs.PosEncodingMode["NONE"].value, False, False, False)
        kv_lens_host = kv_host[1:] - kv_host[:-1]
        w._kv_lens_buffer = kv_lens_host.to(dev)
        if backend == "fa3":
            w._vector_sparse_indptr_buffer = torch.cat([kv_indptr, kv_indptr.new_zeros(1)])
            w._vector_sparse_indices_buffer = torch.empty(1, dtype=i32, device=dev).expand(self.total + 2)
        w._plan_info = w._cached_module.plan(w._float_workspace_buffer, w._int_workspace_buffer,
                                             w._pin_memory_int_workspace_buffer, qo_host, kv_host, kv_lens_host,
                                             int(qo_host[-1]), H * Gq, 1, 1, 1, False, head_dim, head_dim, False)
        torch.cuda.synchronize()
        w._pos_encoding_mode = "NONE"
        w._use_fp16_qk_reduction = False
        w._logits_soft_cap = 0.0
        w._sm_scale = None
        w._rope_scale = None
        w._rope_theta = None
        w._num_kv_heads = H
        w._gqa_group_size = 1
        self.wrapper = w

    def get(self):
        buf = _idx_buf(self.total, self.base.device)
        _kv_index_kernel[(self.n_trip,)](buf, self.base, self.start, self.len, BL=128, num_warps=4)
        self.wrapper._paged_kv_indices_buf = buf[:self.total]
        return self


class SpatialPlan:
    def __init__(self, mask_hnn, til, D=128, backend="fa3"):
        H, NT = mask_hnn.shape[0], til.nt
        dev = til.cq.device
        self.til, self.D, self.backend, self.H = til, D, backend, H
        allowed = torch.zeros(H, mask_hnn.shape[1], NT + 1, dtype=torch.bool, device=dev)
        allowed[:, :, :NT] = mask_hnn.to(dev)
        hx = til.__dict__.setdefault("_hx", {})
        if H not in hx:
            cq, ck = til.cq[None].expand(H, -1).contiguous(), til.ck[None].expand(H, -1).contiguous()
            hx[H] = (cq, ck, tiling_side(cq, ck, NT + 1))
        cq, ck, side = hx[H]
        m = mask_hnn.to(dev)
        self.density = real_density(m, til)
        self.pseudo_density = pseudo_density(m, til)
        self.exec_density = exec_density(m, til)
        self._plan = StaticPlan(cq, ck, allowed, head_dim=D, backend=backend, pad_cluster=NT, side=side)

    def get(self):
        return self._plan.get()


def exact_part_np(q, k, v, plan, S_real):
    H, D, S = plan.H, plan.D, plan.S
    b = _bufs(H, S, plan.C, D, q.dtype, q.device)
    assert q.stride() == k.stride() == v.stride() and q.stride(-1) == 1
    _gather_rows[(S // 128, H)](q, b["qp"], plan.q_idx, q.stride(0), q.stride(1), S_real, S, D=D, BM=128, num_warps=8)
    _gather_rows[(S // 128, H)](k, b["kp"], plan.k_idx, k.stride(0), k.stride(1), S_real, S, D=D, BM=128, num_warps=8)
    _gather_rows[(S // 128, H)](v, b["vp"], plan.k_idx, v.stride(0), v.stride(1), S_real, S, D=D, BM=128, num_warps=8)
    plan.wrapper.run(b["qp"], b["kp"], b["vp"], out=b["os"].view(H * S, 1, D), lse=b["lse"].view(H * S, 1),
                     return_lse=True)
    return b


def _launch_merge(b, plan, o2, compensate, cfg=None):
    m = dict(MERGE, **(cfg or {}))
    H, S, D = plan.H, plan.S, plan.D
    _merge[(S // m["RB"], H)](b["qp"], b["os"], b["lse"], b["kbh"], b["kbl"], b["vb"], plan.l2n, plan.mask_u8,
                              plan.q_idx, o2, S, plan.C, plan.Gq, o2.stride(0), math.log2(math.e) / math.sqrt(D),
                              D=D, RB=m["RB"], BN=m["BN"], G_SZ=GS, COMP=bool(compensate), HILO=bool(m["hilo"]),
                              num_warps=m["warps"], num_stages=m["stages"])


def qb_attn_np(q, k, v, plan, S_real, compensate=True):
    H, D, S = plan.H, plan.D, plan.S
    b = _bufs(H, S, plan.C, D, q.dtype, q.device)
    assert q.stride() == k.stride() == v.stride() and q.stride(-1) == 1
    _gather_rows[(S // 128, H)](q, b["qp"], plan.q_idx, q.stride(0), q.stride(1), S_real, S, D=D, BM=128, num_warps=8)
    _gather_kv_mean_np[(plan.C, H)](k, v, b["kp"], b["vp"], b["kbh"], b["kbl"], b["vb"], plan.k_idx, plan.k_off,
                                    k.stride(0), k.stride(1), S_real, S, plan.C, D=D, BR=32, num_warps=4)
    plan.wrapper.run(b["qp"], b["kp"], b["vp"], out=b["os"].view(H * S, 1, D), lse=b["lse"].view(H * S, 1),
                     return_lse=True)
    out = torch.empty(S, 1, H * D, dtype=q.dtype, device=q.device)
    _launch_merge(b, plan, out[:, 0], compensate)
    return out[:, 0]
