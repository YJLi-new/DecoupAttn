from __future__ import annotations

import torch

CTA_Q, CTA_KV = 128, 96


def selected_pairs(block_map: torch.Tensor, qsz: torch.Tensor, ksz: torch.Tensor) -> torch.Tensor:
    nsel = (block_map.long() * ksz.long()[..., None, :]).sum(-1)
    return (nsel * qsz.long()).sum()


def vb_exec_pairs(block_map: torch.Tensor, qsz: torch.Tensor, ksz: torch.Tensor, cta_q: int = CTA_Q,
                  cta_kv: int = CTA_KV) -> torch.Tensor:
    nsel = (block_map.long() * ksz.long()[..., None, :]).sum(-1)
    q = qsz.long()
    return (((q + cta_q - 1) // cta_q * cta_q) * ((nsel + cta_kv - 1) // cta_kv * cta_kv) * (q > 0)).sum()


def pseudo_key_pairs(block_map: torch.Tensor, qsz: torch.Tensor, ksz: torch.Tensor) -> torch.Tensor:
    unsel = ((~block_map) & (ksz.long() > 0)[..., None, :]).long().sum(-1)
    return (unsel * qsz.long()).sum()


def block_topk(tokens: int, block: int, density: float) -> tuple[int, int]:
    nt = -(-tokens // block)
    return nt, max(1, int(density * nt))


def block_topk_exec_pairs(tokens: int, heads: int, block: int, density: float) -> int:
    nt, top_k = block_topk(tokens, block, density)
    return heads * nt * block * top_k * block


def threshold_exact_pairs(score: torch.Tensor, thr: torch.Tensor, block: int = 64) -> torch.Tensor:
    n = score.shape[-1]
    idx = torch.arange(n, device=score.device)
    exact = (score > thr[..., None]) | ((idx[:, None] - idx[None, :]).abs() <= 1)
    return exact.sum() * (block * block)
