"""
COS 484 Assignment 4 — Cross-Entropy Competition
Correctness test and benchmarking script.

Usage:
    python test_cross_entropy.py <your_submission.py>

This will:
  1. Check that your cross_entropy_forward and cross_entropy_backward produce
     correct outputs (compared to PyTorch's reference implementation).
  2. Benchmark the combined forward + backward time and report achieved
     memory bandwidth.

Environment: Python 3.10+, PyTorch 2.x, Triton (optional).
GPU: NVIDIA A100 80GB (official scoring) — but you can test locally on any GPU.
"""

import argparse
import importlib.util
import math
import sys
import statistics

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Test / benchmark parameters (same as the assignment)
# ---------------------------------------------------------------------------
B = 4_096
VOCAB_SIZES = [32_000, 50_264, 128_256]
DTYPE = torch.bfloat16
DEVICE = "cuda"

# Correctness tolerances
ATOL = 1e-3
RTOL = 1e-2

# Benchmark parameters
WARMUP_ITERS = 20
BENCH_ITERS = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_submission(path: str):
    """Dynamically import a submission file and return its module."""
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_forward(logits, targets):
    """Reference forward: PyTorch cross-entropy, reduction='none', float32 output."""
    return F.cross_entropy(logits.float(), targets, reduction="none")


def reference_backward(logits, targets, grad_output):
    """Reference backward: softmax(logits) - one_hot(targets), scaled by grad_output."""
    probs = torch.softmax(logits.float(), dim=-1)
    grad = probs
    grad[torch.arange(logits.shape[0], device=logits.device), targets] -= 1.0
    grad = grad * grad_output.unsqueeze(1)
    return grad.to(logits.dtype)


def generate_inputs(B, V, seed=42):
    """Generate deterministic random inputs."""
    torch.manual_seed(seed)
    logits = torch.randn(B, V, dtype=DTYPE, device=DEVICE)
    targets = torch.randint(0, V, (B,), device=DEVICE)
    grad_output = torch.randn(B, dtype=torch.float32, device=DEVICE)
    return logits, targets, grad_output


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------

def check_correctness(mod, V):
    """Run correctness checks on the submission module for a given vocab size."""
    logits, targets, grad_output = generate_inputs(B, V)

    # --- Forward ---
    ref_loss = reference_forward(logits, targets)
    sub_loss = mod.cross_entropy_forward(logits, targets)

    assert sub_loss.shape == ref_loss.shape, (
        f"Forward shape mismatch: expected {ref_loss.shape}, got {sub_loss.shape}"
    )
    assert sub_loss.dtype == torch.float32, (
        f"Forward dtype mismatch: expected float32, got {sub_loss.dtype}"
    )

    fwd_close = torch.allclose(sub_loss, ref_loss, atol=ATOL, rtol=RTOL)
    max_fwd_err = (sub_loss - ref_loss).abs().max().item()

    # --- Backward ---
    ref_grad = reference_backward(logits, targets, grad_output)
    sub_grad = mod.cross_entropy_backward(logits, targets, grad_output)

    assert sub_grad.shape == ref_grad.shape, (
        f"Backward shape mismatch: expected {ref_grad.shape}, got {sub_grad.shape}"
    )
    assert sub_grad.dtype == DTYPE, (
        f"Backward dtype mismatch: expected {DTYPE}, got {sub_grad.dtype}"
    )

    bwd_close = torch.allclose(sub_grad, ref_grad, atol=ATOL, rtol=RTOL)
    max_bwd_err = (sub_grad.float() - ref_grad.float()).abs().max().item()

    return fwd_close, bwd_close, max_fwd_err, max_bwd_err


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_one(mod, V):
    """Benchmark forward + backward for a single vocab size. Returns timing dict."""
    logits, targets, grad_output = generate_inputs(B, V, seed=123)

    # Warmup
    for _ in range(WARMUP_ITERS):
        loss = mod.cross_entropy_forward(logits, targets)
        grad = mod.cross_entropy_backward(logits, targets, grad_output)
    torch.cuda.synchronize()

    # Forward timing
    fwd_times = []
    for _ in range(BENCH_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = mod.cross_entropy_forward(logits, targets)
        end.record()
        torch.cuda.synchronize()
        fwd_times.append(start.elapsed_time(end))

    # Backward timing
    bwd_times = []
    for _ in range(BENCH_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        grad = mod.cross_entropy_backward(logits, targets, grad_output)
        end.record()
        torch.cuda.synchronize()
        bwd_times.append(start.elapsed_time(end))

    # Combined timing (forward + backward together)
    combined_times = []
    for _ in range(BENCH_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = mod.cross_entropy_forward(logits, targets)
        grad = mod.cross_entropy_backward(logits, targets, grad_output)
        end.record()
        torch.cuda.synchronize()
        combined_times.append(start.elapsed_time(end))

    fwd_ms = statistics.median(fwd_times)
    bwd_ms = statistics.median(bwd_times)
    combined_ms = statistics.median(combined_times)

    # Memory I/O for bandwidth calculation (you should derive these in Theory Problem 1)
    fwd_bytes = 2 * B * V + 12 * B
    bwd_bytes = 4 * B * V + 12 * B
    total_bytes = fwd_bytes + bwd_bytes

    fwd_bw = fwd_bytes / (fwd_ms * 1e-3) / 1e9
    bwd_bw = bwd_bytes / (bwd_ms * 1e-3) / 1e9
    combined_bw = total_bytes / (combined_ms * 1e-3) / 1e9

    return dict(
        fwd_ms=fwd_ms, bwd_ms=bwd_ms, combined_ms=combined_ms,
        fwd_bytes=fwd_bytes, bwd_bytes=bwd_bytes, total_bytes=total_bytes,
        fwd_bw=fwd_bw, bwd_bw=bwd_bw, combined_bw=combined_bw,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="COS 484 A4 — Test and benchmark a cross-entropy submission."
    )
    parser.add_argument("submission", help="Path to your submission .py file")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available. This script requires a GPU.")
        sys.exit(1)

    mod = load_submission(args.submission)

    # Check required functions exist
    for fn_name in ("cross_entropy_forward", "cross_entropy_backward"):
        if not hasattr(mod, fn_name):
            print(f"ERROR: Submission is missing function '{fn_name}'.")
            sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"B = {B}")

    # ---- Correctness ----
    print("\n" + "=" * 70)
    print("Correctness check")
    print("=" * 70)

    all_passed = True
    for V in VOCAB_SIZES:
        fwd_ok, bwd_ok, fwd_err, bwd_err = check_correctness(mod, V)
        status = "PASS" if (fwd_ok and bwd_ok) else "FAIL"
        print(f"  V={V:>7}  fwd err={fwd_err:.6f} {'OK' if fwd_ok else 'FAIL'}  "
              f"bwd err={bwd_err:.6f} {'OK' if bwd_ok else 'FAIL'}  [{status}]")
        if not (fwd_ok and bwd_ok):
            all_passed = False

    if not all_passed:
        print("\nFAILED — submission is not numerically correct.")
        sys.exit(1)
    print("\nAll correctness checks PASSED.\n")

    # ---- Benchmark ----
    print("=" * 70)
    print("Benchmark")
    print("=" * 70)

    print(f"\n  {'V':>7} | {'Fwd ms':>8} {'Bwd ms':>8} {'Fwd+Bwd ms':>11} | "
          f"{'Fwd BW':>10} {'Bwd BW':>10} {'Fwd+Bwd BW':>11} | "
          f"{'Fwd GB':>7} {'Bwd GB':>7}")
    print("  " + "-" * 100)

    # Benchmark baseline (PyTorch eager)
    print("\n  Benchmarking baseline (PyTorch eager)...")
    baseline_mod = type(sys)("baseline")
    baseline_mod.cross_entropy_forward = lambda logits, targets: \
        F.cross_entropy(logits.float(), targets, reduction="none")
    def _baseline_bwd(logits, targets, grad_output):
        probs = torch.softmax(logits.float(), dim=-1)
        probs[torch.arange(logits.shape[0], device=logits.device), targets] -= 1.0
        return (probs * grad_output.unsqueeze(1)).to(logits.dtype)
    baseline_mod.cross_entropy_backward = _baseline_bwd

    baseline_times = {}
    for V in VOCAB_SIZES:
        r = benchmark_one(baseline_mod, V)
        baseline_times[V] = r['combined_ms']

    # Benchmark submission
    print(f"\n  {'V':>7} | {'Fwd ms':>8} {'Bwd ms':>8} {'Fwd+Bwd ms':>11} | "
          f"{'Fwd BW':>10} {'Bwd BW':>10} {'Fwd+Bwd BW':>11} | "
          f"{'Speedup':>8}")
    print("  " + "-" * 95)

    speedups = []
    for V in VOCAB_SIZES:
        r = benchmark_one(mod, V)
        speedup = baseline_times[V] / r['combined_ms']
        speedups.append(speedup)
        print(f"  {V:>7} | {r['fwd_ms']:>7.3f}  {r['bwd_ms']:>7.3f}  "
              f"{r['combined_ms']:>10.3f}  | "
              f"{r['fwd_bw']:>8.1f}  {r['bwd_bw']:>8.1f}  "
              f"{r['combined_bw']:>9.1f}  | "
              f"{speedup:>7.2f}x")

    geomean_speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))

    print(f"\n  Units: time in ms, bandwidth in GB/s")
    print(f"\n  ** Competition score (geomean speedup vs baseline): {geomean_speedup:.2f}x **\n")


if __name__ == "__main__":
    main()
