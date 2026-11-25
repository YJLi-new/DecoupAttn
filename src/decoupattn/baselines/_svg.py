from __future__ import annotations

import sys
import types

import torch
import torch.nn.functional as F

from ._common import LN2, add_path, bhsd, upstream_path

UPSTREAM = "Sparse-VideoGen"


def import_svg() -> types.SimpleNamespace:
    add_path(upstream_path(UPSTREAM))
    try:
        import cuvs
    except ImportError:
        stub = types.ModuleType("cuvs")
        stub.cluster = types.ModuleType("cuvs.cluster")
        stub.cluster.kmeans = types.ModuleType("cuvs.cluster.kmeans")
        stub.cluster.kmeans.KMeansParams = stub.cluster.kmeans.fit = None
        sys.modules.update({"cuvs": stub, "cuvs.cluster": stub.cluster, "cuvs.cluster.kmeans": stub.cluster.kmeans})
    import flashinfer
    import svg
    from svg import kmeans_utils as ku
    from svg.flashinfer_patch import flashinfer_patch_enabled
    from svg.kernels.triton.permute import apply_inverse_permutation_triton, permute_tensor_by_labels_triton
    return types.SimpleNamespace(ku=ku, patch_ctx=flashinfer_patch_enabled, permute=permute_tensor_by_labels_triton,
                                 inv_permute=apply_inverse_permutation_triton,
                                 wrapper_cls=flashinfer.sparse.VariableBlockSparseAttentionWrapper)


class VarBlockSparse:
    FLOAT_WS = 128 * 1024 * 1024

    def __init__(self, svg):
        self.svg = svg
        self.float_ws = self.wrapper = self.idx_buf = None

    def _wrapper(self, dev):
        if self.wrapper is None:
            self.float_ws = torch.empty(self.FLOAT_WS, device=dev)
            self.wrapper = self.svg.wrapper_cls(self.float_ws, backend="auto")
        return self.wrapper

    def plan(self, block_map, qsz, ksz, head_dim, dtype):
        w = self._wrapper(block_map.device)
        B, H, qc, kc = block_map.shape
        need = int((block_map.reshape(B * H, qc, kc).long() * ksz.reshape(B * H, 1, kc).long()).sum())
        if self.idx_buf is None or self.idx_buf.numel() < need:
            w._vector_sparse_indices_buffer = self.idx_buf = None
            if block_map.is_cuda:
                torch.cuda.empty_cache()
            self.idx_buf = torch.empty(int(need * 1.25) + (1 << 20), dtype=torch.int32, device=block_map.device)
        with self.svg.patch_ctx():
            w.plan(block_mask_map=block_map.reshape(B * H, qc, kc), block_row_sz=qsz.reshape(B * H, qc),
                   block_col_sz=ksz.reshape(B * H, kc), num_qo_heads=B * H, num_kv_heads=B * H, head_dim=head_dim,
                   q_data_type=dtype, kv_data_type=dtype)
        return w

    def run_lse(self, q, k, v):
        B, H, S, D = q.shape
        w = self.wrapper
        w._vector_sparse_indices_buffer = self.idx_buf
        return w.run(q.reshape(B * H, S, D), k.reshape(B * H, S, D), v.reshape(B * H, S, D), return_lse=True)

    def run(self, q, k, v):
        B, H, S, D = q.shape
        w = self.wrapper
        w._vector_sparse_indices_buffer = self.idx_buf
        return w.run(q.reshape(B * H, S, D), k.reshape(B * H, S, D), v.reshape(B * H, S, D)).reshape(B, H, S, D)


class Clusters:
    def __init__(self, svg, qc: int, kc: int, iter_init: int, iter_step: int):
        self.svg, self.qc, self.kc, self.iter_init, self.iter_step = svg, qc, kc, iter_init, iter_step
        self.centroids: dict[int, tuple] = {}
        self.builds = 0

    def reset(self) -> None:
        self.centroids.clear()
        self.builds = 0

    def __call__(self, layer: int, q, k):
        ku = self.svg.ku
        _, H, S, D = q.shape
        prev = self.centroids.get(layer)
        iters = self.iter_init if prev is None else self.iter_step
        ql, qcen, qsz, _ = ku.batch_kmeans_Euclid(q.view(H, S, D), n_clusters=self.qc, max_iters=iters,
                                                  init_centroids=None if prev is None else prev[0])
        kl, kcen, ksz, _ = ku.batch_kmeans_Euclid(k.view(H, S, D), n_clusters=self.kc, max_iters=iters,
                                                  init_centroids=None if prev is None else prev[1])
        self.centroids[layer] = (qcen, kcen)
        self.builds += 1
        return ql, qcen, qsz.view(1, H, self.qc), kl, kcen, ksz.view(1, H, self.kc)


def permute_qkv(svg, q, k, v, ql, kl):
    q_p, q_idx = svg.permute(q, ql, dim=2)
    k_p, k_idx = svg.permute(k, kl, dim=2)
    v_p, _ = svg.permute(v, kl, dim=2, sorted_indices=k_idx)
    return q_p, k_p, v_p, q_idx


def joint_layout(q_p, k_p, v_p, q_idx, block_map, qsz, ksz, q, k, v, nv):
    nt = q.shape[1] - nv
    qj = torch.cat([q_p, bhsd(q[:, nv:])], 2)
    kj = torch.cat([k_p, bhsd(k[:, nv:])], 2)
    vj = torch.cat([v_p, bhsd(v[:, nv:])], 2)
    bm = F.pad(block_map, (0, 1, 0, 1), value=False)
    bm[..., :-1, -1] = True
    bm[..., -1, :] = True
    qs = F.pad(qsz, (0, 1), value=nt)
    ks = F.pad(ksz, (0, 1), value=nt)
    qi = F.pad(q_idx, (0, nt), value=0)
    qi[..., nv:] = torch.arange(nv, nv + nt, device=qi.device, dtype=qi.dtype)
    return qj, kj, vj, bm.contiguous(), qs.contiguous(), ks.contiguous(), qi.contiguous()


def fused_compensation(svg, sparse: VarBlockSparse, q_p, k_p, v_p, kcen, vcen, block_map, qsz, ksz):
    B, H, S, D = q_p.shape
    qc_num, kc_num = block_map.shape[-2:]
    o_flash, lse_flash = sparse.run_lse(q_p, k_p, v_p)
    qc_indptr = torch.zeros((B, H, qc_num + 1), device=q_p.device, dtype=torch.int32)
    qc_indptr[..., 1:] = torch.cumsum(qsz.to(q_p.device), dim=-1)
    q_flat = q_p.view(B * H, S, D)
    kc_flat = kcen.reshape(B * H, kc_num, D)
    vc_flat = vcen.reshape(B * H, kc_num, D)
    o_final = torch.empty((B * H, S, D), device=q_p.device, dtype=q_p.dtype)
    lse_final = torch.empty((B * H, S), device=q_p.device, dtype=torch.float32)
    svg.ku._fused_qc_kernel_opt[(qc_num, B * H)](
        Q=q_flat, K_centroids=kc_flat, V_centroids=vc_flat, block_mask_map=block_map, block_col_sz=ksz,
        O_flash=o_flash, LSE_flash=lse_flash, QC_INDPTR=qc_indptr, LSE_final=lse_final, Out=o_final,
        stride_qh=q_flat.stride(0), stride_qs=q_flat.stride(1), stride_qd=q_flat.stride(2),
        stride_ch=kc_flat.stride(0), stride_cn=kc_flat.stride(1), stride_cd=kc_flat.stride(2),
        stride_mb=block_map.stride(0), stride_mh=block_map.stride(1), stride_mq=block_map.stride(2),
        stride_mk=block_map.stride(3),
        stride_szb=ksz.stride(0), stride_szh=ksz.stride(1), stride_szn=ksz.stride(2),
        stride_ipb=qc_indptr.stride(0), stride_iph=qc_indptr.stride(1), stride_ipn=qc_indptr.stride(2),
        sm_scale=D ** -0.5, B=B, H=H, S=S, D=D, KC_NUM=kc_num, BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=2)
    return o_final.view(B, H, S, D), lse_final.view(B, H, S)


def flashinfer_lse_to_natural(lse: torch.Tensor) -> torch.Tensor:
    return lse * LN2
