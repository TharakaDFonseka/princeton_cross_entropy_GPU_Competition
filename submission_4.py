"""
submission_4 — custom Triton forward + backward (stable softmax / log-sum-exp per row).
One kernel launch per direction per batch: row-parallel, tiled over V.

Compared to v2 (torch.compile + F.cross_entropy + compiled backward): aims for predictable
fused loops and fewer separate kernels from Inductor; must still pass atol/rtol on the grader.

API: cross_entropy_forward / cross_entropy_backward; torch + triton only.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Finite sentinel for masked max (avoids -inf edge cases in reductions)
_NEG = -1.0e30


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

    m = _NEG
    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V
        x = tl.load(
            logits_ptr + base + offs * stride_l1,
            mask=mask,
            other=_NEG,
        )
        x = x.to(tl.float32)
        m = tl.maximum(m, tl.max(tl.where(mask, x, _NEG)))

    acc = 0.0
    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V
        x = tl.load(
            logits_ptr + base + offs * stride_l1,
            mask=mask,
            other=0.0,
        )
        x = x.to(tl.float32)
        acc += tl.sum(tl.where(mask, tl.exp(x - m), 0.0))

    lse = tl.log(acc)

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

    m = _NEG
    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V
        x = tl.load(
            logits_ptr + base_l + offs * stride_l1,
            mask=mask,
            other=_NEG,
        )
        x = x.to(tl.float32)
        m = tl.maximum(m, tl.max(tl.where(mask, x, _NEG)))

    acc = 0.0
    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V
        x = tl.load(
            logits_ptr + base_l + offs * stride_l1,
            mask=mask,
            other=0.0,
        )
        x = x.to(tl.float32)
        acc += tl.sum(tl.where(mask, tl.exp(x - m), 0.0))

    lse = tl.log(acc)
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


def _block_v(v: int) -> int:
    # Larger tiles → fewer loop iterations on big vocab (tune on A100 if needed)
    if v >= 96_000:
        return 4096
    if v >= 48_000:
        return 2048
    return 2048


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
