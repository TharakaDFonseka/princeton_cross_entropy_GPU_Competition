"""v8: Triton @autotune over BLOCK_V/warps/stages + tl.constexpr V/strides + logden cache.

    Differs from submission_9.py, which uses a fixed _LAUNCH table instead of autotune.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# Reuse forward logden in backward when the same tensors are used (best for fwd+bwd timing).
_FWD_CACHE: dict = {}

B_FIXED = 4096


def _cache_key(logits: torch.Tensor, targets: torch.Tensor):
    return (
        int(logits.data_ptr()),
        int(targets.data_ptr()),
        logits.device.index,
        logits.shape,
        logits.stride(),
        targets.shape,
        targets.stride(),
    )


# Wider search space; warmup lets autotune pick per-V winners on A100-class GPUs.
_FWD_CONFIGS = [
    triton.Config({"BLOCK_V": 512}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_V": 1024}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_V": 2048}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_V": 2048}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=1),
    triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=2),
]

_BWD_CONFIGS = [
    triton.Config({"BLOCK_V": 512}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_V": 1024}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_V": 2048}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_V": 2048}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=1),
    triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=2),
]


@triton.autotune(configs=_FWD_CONFIGS, key=["V"])
@triton.jit
def _ce_fwd_kernel(
    logits_ptr,
    tgt_ptr,
    loss_ptr,
    logden_ptr,
    V: tl.constexpr,
    stride_l0: tl.constexpr,
    stride_l1: tl.constexpr,
    stride_t0: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * stride_l0
    neg_inf = float("-inf")

    m = neg_inf
    s = 0.0

    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V

        x = tl.load(
            logits_ptr + base + offs * stride_l1,
            mask=mask,
            other=neg_inf,
        ).to(tl.float32)

        tile_m = tl.max(x, axis=0)
        m_new = tl.maximum(m, tile_m)

        exp_x = tl.exp(x - m_new)
        exp_x = exp_x * mask

        s = s * tl.exp(m - m_new) + tl.sum(exp_x, axis=0)
        m = m_new

    logden = m + tl.log(s)

    tgt = tl.load(tgt_ptr + row * stride_t0).to(tl.int32)
    x_tgt = tl.load(logits_ptr + base + tgt * stride_l1).to(tl.float32)

    loss = logden - x_tgt

    tl.store(loss_ptr + row, loss)
    tl.store(logden_ptr + row, logden)


@triton.autotune(configs=_BWD_CONFIGS, key=["V"])
@triton.jit
def _ce_bwd_cached_kernel(
    logits_ptr,
    tgt_ptr,
    gout_ptr,
    logden_ptr,
    grad_ptr,
    V: tl.constexpr,
    stride_l0: tl.constexpr,
    stride_l1: tl.constexpr,
    stride_t0: tl.constexpr,
    stride_go0: tl.constexpr,
    stride_g0: tl.constexpr,
    stride_g1: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    base_l = row * stride_l0
    base_g = row * stride_g0
    neg_inf = float("-inf")

    g = tl.load(gout_ptr + row * stride_go0).to(tl.float32)
    logden = tl.load(logden_ptr + row).to(tl.float32)
    tgt = tl.load(tgt_ptr + row * stride_t0).to(tl.int32)

    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V

        x = tl.load(
            logits_ptr + base_l + offs * stride_l1,
            mask=mask,
            other=neg_inf,
        ).to(tl.float32)

        p = tl.exp(x - logden)
        grad = p * g
        grad = grad - g * ((offs == tgt) & mask)

        tl.store(
            grad_ptr + base_g + offs * stride_g1,
            grad.to(tl.bfloat16),
            mask=mask,
        )


@triton.autotune(configs=_BWD_CONFIGS, key=["V"])
@triton.jit
def _ce_bwd_recompute_kernel(
    logits_ptr,
    tgt_ptr,
    gout_ptr,
    grad_ptr,
    V: tl.constexpr,
    stride_l0: tl.constexpr,
    stride_l1: tl.constexpr,
    stride_t0: tl.constexpr,
    stride_go0: tl.constexpr,
    stride_g0: tl.constexpr,
    stride_g1: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    base_l = row * stride_l0
    base_g = row * stride_g0
    neg_inf = float("-inf")

    g = tl.load(gout_ptr + row * stride_go0).to(tl.float32)
    tgt = tl.load(tgt_ptr + row * stride_t0).to(tl.int32)

    m = neg_inf
    s = 0.0

    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V

        x = tl.load(
            logits_ptr + base_l + offs * stride_l1,
            mask=mask,
            other=neg_inf,
        ).to(tl.float32)

        tile_m = tl.max(x, axis=0)
        m_new = tl.maximum(m, tile_m)

        exp_x = tl.exp(x - m_new)
        exp_x = exp_x * mask

        s = s * tl.exp(m - m_new) + tl.sum(exp_x, axis=0)
        m = m_new

    logden = m + tl.log(s)

    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V

        x = tl.load(
            logits_ptr + base_l + offs * stride_l1,
            mask=mask,
            other=neg_inf,
        ).to(tl.float32)

        p = tl.exp(x - logden)
        grad = p * g
        grad = grad - g * ((offs == tgt) & mask)

        tl.store(
            grad_ptr + base_g + offs * stride_g1,
            grad.to(tl.bfloat16),
            mask=mask,
        )


def cross_entropy_forward(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if not logits.is_cuda or not targets.is_cuda:
        raise RuntimeError("CUDA tensors required")
    if logits.dtype != torch.bfloat16:
        raise RuntimeError("logits must be bfloat16")
    if targets.dtype != torch.int64:
        raise RuntimeError("targets must be int64")

    B, V = logits.shape
    if B != B_FIXED:
        raise RuntimeError(f"Expected batch size {B_FIXED}, got {B}")

    losses = torch.empty((B,), device=logits.device, dtype=torch.float32)
    logden = torch.empty((B,), device=logits.device, dtype=torch.float32)

    s0, s1 = int(logits.stride(0)), int(logits.stride(1))
    st = int(targets.stride(0))

    _ce_fwd_kernel[(B,)](
        logits,
        targets,
        losses,
        logden,
        V,
        s0,
        s1,
        st,
    )

    _FWD_CACHE[_cache_key(logits, targets)] = logden
    return losses


def cross_entropy_backward(
    logits: torch.Tensor,
    targets: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    if not logits.is_cuda or not targets.is_cuda or not grad_output.is_cuda:
        raise RuntimeError("CUDA tensors required")
    if logits.dtype != torch.bfloat16:
        raise RuntimeError("logits must be bfloat16")
    if targets.dtype != torch.int64:
        raise RuntimeError("targets must be int64")
    if grad_output.dtype != torch.float32:
        raise RuntimeError("grad_output must be float32")

    B, V = logits.shape
    if B != B_FIXED:
        raise RuntimeError(f"Expected batch size {B_FIXED}, got {B}")

    grad = torch.empty_like(logits)

    s0, s1 = int(logits.stride(0)), int(logits.stride(1))
    st = int(targets.stride(0))
    sgo = int(grad_output.stride(0))
    sg0, sg1 = int(grad.stride(0)), int(grad.stride(1))

    key = _cache_key(logits, targets)
    logden = _FWD_CACHE.pop(key, None)

    if logden is not None:
        _ce_bwd_cached_kernel[(B,)](
            logits,
            targets,
            grad_output,
            logden,
            grad,
            V,
            s0,
            s1,
            st,
            sgo,
            sg0,
            sg1,
        )
    else:
        _ce_bwd_recompute_kernel[(B,)](
            logits,
            targets,
            grad_output,
            grad,
            V,
            s0,
            s1,
            st,
            sgo,
            sg0,
            sg1,
        )

    return grad
