from __future__ import annotations

import torch
import triton
import triton.language as tl


def _params_for_v(V: int):
    if V == 32000:
        return dict(BLOCK_V=1024, num_warps=4, num_stages=4, ROWS_PER_PROG=2)
    if V == 50264:
        return dict(BLOCK_V=2048, num_warps=8, num_stages=3, ROWS_PER_PROG=1)
    return dict(BLOCK_V=4096, num_warps=8, num_stages=2, ROWS_PER_PROG=1)


@triton.jit
def _ce_fwd_kernel_persistent(
    logits_ptr,
    targets_ptr,
    losses_ptr,
    B,
    V,
    stride_l0,
    stride_l1,
    stride_t0,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    neg_inf = float("-inf")

    for row_iter in range(ROWS_PER_PROG):
        row = pid + row_iter * n_programs
        row_mask = row < B
        base = row * stride_l0

        m = neg_inf
        s = 0.0

        for start in range(0, V, BLOCK_V):
            offs = start + tl.arange(0, BLOCK_V)
            mask = row_mask & (offs < V)

            x = tl.load(
                logits_ptr + base + offs * stride_l1,
                mask=mask,
                other=neg_inf,
            ).to(tl.float32)

            tile_m = tl.max(x, axis=0)
            m_new = tl.maximum(m, tile_m)
            s = s * tl.exp(m - m_new) + tl.sum(
                tl.where(mask, tl.exp(x - m_new), 0.0),
                axis=0,
            )
            m = m_new

        tgt = tl.load(targets_ptr + row * stride_t0, mask=row_mask, other=0).to(tl.int32)
        x_tgt = tl.load(
            logits_ptr + base + tgt * stride_l1,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)

        loss = (m + tl.log(s)) - x_tgt
        tl.store(losses_ptr + row, loss, mask=row_mask)


@triton.jit
def _ce_bwd_kernel_persistent(
    logits_ptr,
    targets_ptr,
    grad_out_ptr,
    grad_ptr,
    B,
    V,
    stride_l0,
    stride_l1,
    stride_t0,
    stride_go0,
    stride_g0,
    stride_g1,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    neg_inf = float("-inf")

    for row_iter in range(ROWS_PER_PROG):
        row = pid + row_iter * n_programs
        row_mask = row < B
        base_l = row * stride_l0
        base_g = row * stride_g0

        tgt = tl.load(targets_ptr + row * stride_t0, mask=row_mask, other=0).to(tl.int32)
        g = tl.load(grad_out_ptr + row * stride_go0, mask=row_mask, other=0.0).to(tl.float32)

        m = neg_inf
        s = 0.0

        for start in range(0, V, BLOCK_V):
            offs = start + tl.arange(0, BLOCK_V)
            mask = row_mask & (offs < V)

            x = tl.load(
                logits_ptr + base_l + offs * stride_l1,
                mask=mask,
                other=neg_inf,
            ).to(tl.float32)

            tile_m = tl.max(x, axis=0)
            m_new = tl.maximum(m, tile_m)
            s = s * tl.exp(m - m_new) + tl.sum(
                tl.where(mask, tl.exp(x - m_new), 0.0),
                axis=0,
            )
            m = m_new

        logden = m + tl.log(s)

        for start in range(0, V, BLOCK_V):
            offs = start + tl.arange(0, BLOCK_V)
            mask = row_mask & (offs < V)

            x = tl.load(
                logits_ptr + base_l + offs * stride_l1,
                mask=mask,
                other=neg_inf,
            ).to(tl.float32)

            p = tl.exp(x - logden)
            grad = p * g
            grad = grad - tl.where(offs == tgt, g, 0.0)

            tl.store(
                grad_ptr + base_g + offs * stride_g1,
                grad.to(tl.bfloat16),
                mask=mask,
            )


def cross_entropy_forward(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if not logits.is_cuda or not targets.is_cuda:
        raise RuntimeError("CUDA required")
    if logits.dtype != torch.bfloat16:
        raise RuntimeError("logits must be bfloat16")
    if targets.dtype != torch.int64:
        raise RuntimeError("targets must be int64")

    B, V = logits.shape
    losses = torch.empty((B,), device=logits.device, dtype=torch.float32)

    kw = _params_for_v(V)
    rows_per_prog = kw.pop("ROWS_PER_PROG")
    grid = (triton.cdiv(B, rows_per_prog),)

    _ce_fwd_kernel_persistent[grid](
        logits,
        targets,
        losses,
        B,
        V,
        logits.stride(0),
        logits.stride(1),
        targets.stride(0),
        ROWS_PER_PROG=rows_per_prog,
        **kw,
    )
    return losses


def cross_entropy_backward(
    logits: torch.Tensor,
    targets: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    if not logits.is_cuda or not targets.is_cuda or not grad_output.is_cuda:
        raise RuntimeError("CUDA required")
    if logits.dtype != torch.bfloat16:
        raise RuntimeError("logits must be bfloat16")
    if targets.dtype != torch.int64:
        raise RuntimeError("targets must be int64")
    if grad_output.dtype != torch.float32:
        raise RuntimeError("grad_output must be float32")

    B, V = logits.shape
    grad = torch.empty_like(logits)

    kw = _params_for_v(V)
    rows_per_prog = kw.pop("ROWS_PER_PROG")
    grid = (triton.cdiv(B, rows_per_prog),)

    _ce_bwd_kernel_persistent[grid](
        logits,
        targets,
        grad_output,
        grad,
        B,
        V,
        logits.stride(0),
        logits.stride(1),
        targets.stride(0),
        grad_output.stride(0),
        grad.stride(0),
        grad.stride(1),
        ROWS_PER_PROG=rows_per_prog,
        **kw,
    )
    return grad