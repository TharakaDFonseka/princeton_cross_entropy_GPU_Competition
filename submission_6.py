"""
submission_6 — Triton cross-entropy (v6): "wide uniform tile" variant of submission_5.

Popcorn rejected fused target-in-loop and num_warps=16 (500). This file keeps the *exact*
online-softmax kernels and launch pattern as submission_5 (runtime V, num_stages=2,
num_warps=8, separate target gather after LSE).

New idea vs submission_5: use a single large BLOCK_V=4096 for *all* vocab sizes (32k,
50k, 128k). submission_5 used 2048 for 32k/50k and 4096 only for 128k — more Python loop
iterations per row. Wider tiles cut trip count (e.g. 32k: 8 tiles vs 16) which can help
instruction/overhead-bound inner loops on A100, without changing Triton lowering shape
that already passed Popcorn for v5.

If this regresses locally (register spill), set _block_v back to submission_5's table.
torch + triton only; literals inside @triton.jit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ce_fwd_kernel(
    logits_ptr,
    tgt_ptr,
    loss_ptr,
    V,
    stride_l0,
    stride_l1,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * stride_l0
    neg_big = -1.0e30

    # Online max + sum(exp(x - m)) in one pass over the row (stable rescaling each tile).
    m = neg_big
    s = 0.0
    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V
        x = tl.load(
            logits_ptr + base + offs * stride_l1,
            mask=mask,
            other=neg_big,
        )
        x = x.to(tl.float32)
        tile_m = tl.max(tl.where(mask, x, neg_big))
        m_new = tl.maximum(m, tile_m)
        s = s * tl.exp(m - m_new) + tl.sum(tl.where(mask, tl.exp(x - m_new), 0.0))
        m = m_new

    lse = tl.log(s)
    tgt = tl.load(tgt_ptr + row).to(tl.int32)
    x_tgt = tl.load(logits_ptr + base + tgt * stride_l1).to(tl.float32)
    loss = -(x_tgt - m - lse)
    tl.store(loss_ptr + row, loss)


@triton.jit
def _ce_bwd_kernel(
    logits_ptr,
    tgt_ptr,
    gout_ptr,
    grad_ptr,
    V,
    stride_l0,
    stride_l1,
    stride_g0,
    stride_g1,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    base_l = row * stride_l0
    base_g = row * stride_g0
    g_row = tl.load(gout_ptr + row).to(tl.float32)
    neg_big = -1.0e30

    m = neg_big
    s = 0.0
    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V
        x = tl.load(
            logits_ptr + base_l + offs * stride_l1,
            mask=mask,
            other=neg_big,
        )
        x = x.to(tl.float32)
        tile_m = tl.max(tl.where(mask, x, neg_big))
        m_new = tl.maximum(m, tile_m)
        s = s * tl.exp(m - m_new) + tl.sum(tl.where(mask, tl.exp(x - m_new), 0.0))
        m = m_new

    lse = tl.log(s)
    tgt = tl.load(tgt_ptr + row).to(tl.int32)

    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V
        x = tl.load(
            logits_ptr + base_l + offs * stride_l1,
            mask=mask,
            other=0.0,
        )
        x = x.to(tl.float32)
        p = tl.exp(x - m - lse)
        grad = p * g_row
        grad = grad - tl.where((offs == tgt) & mask, g_row, 0.0)
        tl.store(
            grad_ptr + base_g + offs * stride_g1,
            grad.to(tl.bfloat16),
            mask=mask,
        )


def _block_v(_: int) -> int:
    # Uniform wide tile: fewer outer-loop iterations than v5 for V in {32k, 50k}.
    return 4096


def _launch_kw(block_v: int):
    nw = 8 if block_v >= 2048 else 4
    return {"BLOCK_V": block_v, "num_warps": nw, "num_stages": 2}


def cross_entropy_forward(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if not logits.is_cuda:
        raise RuntimeError("CUDA required")
    logits = logits.contiguous()
    targets = targets.contiguous()
    targets_i = targets if targets.dtype == torch.int32 else targets.to(torch.int32)

    B, V = logits.shape
    losses = torch.empty(B, device=logits.device, dtype=torch.float32)
    s0, s1 = int(logits.stride(0)), int(logits.stride(1))
    bv = _block_v(V)
    kw = _launch_kw(bv)

    _ce_fwd_kernel[(B,)](
        logits,
        targets_i,
        losses,
        V,
        s0,
        s1,
        **kw,
    )
    return losses


def cross_entropy_backward(
    logits: torch.Tensor,
    targets: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    if not logits.is_cuda:
        raise RuntimeError("CUDA required")
    logits = logits.contiguous()
    targets = targets.contiguous()
    grad_output = grad_output.contiguous()
    targets_i = targets if targets.dtype == torch.int32 else targets.to(torch.int32)

    B, V = logits.shape
    grad = torch.empty_like(logits)
    s0, s1 = int(logits.stride(0)), int(logits.stride(1))
    gs0, gs1 = int(grad.stride(0)), int(grad.stride(1))
    bv = _block_v(V)
    kw = _launch_kw(bv)

    _ce_bwd_kernel[(B,)](
        logits,
        targets_i,
        grad_output,
        grad,
        V,
        s0,
        s1,
        gs0,
        gs1,
        **kw,
    )
    return grad
