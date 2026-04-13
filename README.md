# Princeton COS484 — GPU Cross-Entropy Competition

This repository contains work for the **optional GPU bonus** in Princeton’s **COS484 (NLP)** course: implement **forward and backward cross-entropy** as fast as possible on GPU. **Lower wall-clock time wins**; rankings are based on speedup over a fixed baseline, not raw loss values.

## What you’re optimizing

You ship a **single Python file** that defines:

- **`cross_entropy_forward(logits, targets)`** — `logits` shape `(B, V)` in **bfloat16**, `targets` shape `(B,)` **int64**, returns per-example losses `(B,)` in **float32** (reduction=`none` semantics).
- **`cross_entropy_backward(logits, targets, grad_output)`** — same dtypes/shapes for inputs; returns **`grad_logits`** `(B, V)` in **bfloat16**.

Benchmark settings (fixed by the course): **batch size B = 4,096**, **bfloat16** logits, vocabulary sizes **V ∈ {32,000, 50,264, 128,256}**. Runs target an **NVIDIA A100 (80 GB)**-class setup.

## Correctness & allowed tools

- Outputs are checked against a **reference implementation** with tolerances **atol = 1e-3**, **rtol = 1e-2**.
- Dependencies are restricted to **`torch`** and **`triton`** (including submodules like `torch.nn.functional` and `triton.language`) — **no other third-party packages**.
- Typical evaluation stack: **PyTorch 2.11.0**, **Triton 3.6.0**, **CUDA 12.x**. You may use **custom Triton kernels**, **`torch.compile`**, or a mix.

## How you’re scored

For each vocabulary size, the grader measures **forward + backward** time with **`torch.cuda.Event`**, using the **median of 100 timed runs** after warmup. Your **competition score** is the **geometric mean** of **baseline_time / your_time** over the three **V** values, where the baseline is **PyTorch eager mode**. The test harness may still print forward-only and backward-only timings for debugging; **only the combined forward+backward metric affects ranking**.

## Local testing

Course materials include a checker/benchmark script and a reference-style baseline (e.g. `test_cross_entropy.py`, `baseline_submission.py`). Locally you can run:

```bash
python test_cross_entropy.py your_submission.py
```

That run checks numerical agreement and reports timing / bandwidth-style stats so you can iterate before any server upload.

*(Original handout assets were also distributed via the course Google Drive link posted on Ed.)*

## Submission & deadlines

**Exact upload steps and the benchmarking server URL are announced on Ed.** The competition window is **separate** from the main assignment lateness policy: you can keep tuning through the server’s open dates; **only your fastest correct submission counts**, and **late days do not apply** to the bonus server (it simply **closes at the announced deadline**, e.g. **11:59 PM Eastern** on the final day).

## Write-up (bonus credit)

For bonus credit, you also submit a short **optimization journal** (in the Colab notebook or a separate PDF) describing what you tried, what worked, and what you measured—enough that a reader can follow your engineering decisions.

---

*This README is a student-written summary of the competition rules; if anything disagrees with the official PDF or Ed post, the course staff version wins.*
