# Cross-Entropy Speed Competition — Optimization Journal: Tharaka Fonseka (tf8712)

- **Goal:** Fastest correct **forward + backward** cross-entropy in a **single Python file** with `cross_entropy_forward` and `cross_entropy_backward` only using **torch** and **triton** (and submodules).
- **Hardware / problem size:** NVIDIA **A100 80GB**, `B = 4,096`, **bfloat16** logits, `reduction='none'`, **V ∈ {32,000, 50,264, 128,256}**.
- **Correctness:** vs reference with **atol=1e-3**, **rtol=1e-2**.
- **Environment:** PyTorch **2.11.0**, Triton **3.6.0**, CUDA **12.x**.
- **Timing:** `torch.cuda.Event`, **median of 100** runs after warmup, **per V**. **Ranking uses geometric mean speedup** of **combined forward+backward** vs **PyTorch eager baseline** (the harness also prints separate fwd/bwd for analysis).


## Final submission summary

Implementation uses `torch.compile(fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")` for forward and backward so TorchInductor can emit fused kernels.

**Files (two versions + default upload name):**

| File | Role |
|------|------|
| `submission_1.py` | **v1** — backward uses `probs.clone()` before subtracting 1 at the target index, then multiplies by `grad_output` (earlier baseline). |
| `submission_2.py` | **v2** — backward uses `grad = probs * grad_output`, then `grad[rows, targets] -= grad_output` (same math, no extra full-matrix clone). |
| `submission.py` | **Same source as v2** — use this path for `test_cross_entropy.py` and for Popcorn upload unless you intentionally submit v1. |

### submission_1 vs submission_2 (what changed)

- **v1:** After softmax `probs`, clone the full `(B, V)` tensor, subtract `1.0` at each row’s target column, multiply by `grad_output[:, None]`, cast to bf16.
- **v2:** `grad = probs * grad_output[:, None]`, then subtract `grad_output` only at `(rows, targets)`. Algebra matches \((p - \mathrm{onehot}) \odot g\) without allocating a clone of `probs`.

**Why v2 might be faster:** One fewer large tensor allocation / memory touch on the backward path; Inductor may fuse differently. **Empirical:** compare geomean speedup from `test_cross_entropy.py` on the same GPU — compile caches differ, so always re-run both after a fresh runtime if you compare fairly.

## Environment

Target environment from the competition spec:
- PyTorch 2.11.0
- Triton 3.6.0
- CUDA 12.x
- A100 80GB

**Colab setup (reproducible snippet):** Load `submission.py`, fix `B, V = 4096, 128256`, build `logits` (bf16), `targets` (int64), `grad_output` (float32) on CUDA; check correctness vs `F.cross_entropy` / softmax–one-hot gradient. Benchmark: **10 warmup** iterations, **50** timed iterations, **`statistics.median`** over `torch.cuda.Event` elapsed ms (same event pattern as the assignment, but **not** the same counts as `test_cross_entropy.py`).

**Colab measurement below:** GPU = whatever `torch.cuda.get_device_name(0)` prints (not the official **A100 80GB**). PyTorch version = `torch.__version__` on that runtime. Treat these as **development measurements**; for the write-up and leaderboard, prefer `python test_cross_entropy.py submission.py` on **A100** (**20** warmup, **100** runs per the provided script) or Popcorn output.

### Version 1 — Google Colab

**Initial Version**.

**v1** (`submission_1.py`) was run on Colab (same runtime class as below: microbench / single-**V** focus at **V = 128,256**, **B = 4096**, median `torch.cuda.Event` ms):

| Phase | Median time |
|------|---:|
| Forward | **1.7285** ms |
| Backward | **6.3907** ms |
| Forward + backward (sum) | **~8.12** ms |

These are **not** the full three-**V** harness; they are useful for A/B vs v2 before running `test_cross_entropy.py` on all **V**.

### Version 2 — Google Colab (`test_cross_entropy.py` harness)

Recorded on **Google Colab**, **NVIDIA A100-SXM4-40GB**, **PyTorch 2.10.0+cu128**, using the course script (**20** warmup, **100** timed iterations, median ms). Correctness: **PASS** for all three **V** (max |fwd|/|bwd| error ≈ **1e−6** in the printed checks).

Printed summary (same structure as the harness table):

```
        V |   Fwd ms   Bwd ms  Fwd+Bwd ms |     Fwd BW     Bwd BW  Fwd+Bwd BW |  Speedup
  -----------------------------------------------------------------------------------------------
    32000 |   0.354    1.323       1.599  |    740.0     396.3      491.7  |    3.56x
    50264 |   0.603    2.231       2.756  |    682.8     369.1      448.3  |    3.31x
   128256 |   1.367    6.296       7.563  |    768.6     333.8      416.8  |    3.07x
```

Markdown copy of the same numbers:

| V | Fwd ms | Bwd ms | Fwd+Bwd ms | Fwd BW (GB/s) | Bwd BW (GB/s) | Fwd+Bwd BW (GB/s) | Speedup vs eager (combined) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32,000 | 0.354 | 1.323 | 1.599 | 740.0 | 396.3 | 491.7 | **3.56×** |
| 50,264 | 0.603 | 2.231 | 2.756 | 682.8 | 369.1 | 448.3 | **3.31×** |
| 128,256 | 1.367 | 6.296 | 7.563 | 768.6 | 333.8 | 416.8 | **3.07×** |

**Competition-style score from this run:** geometric mean speedup vs the script’s eager baseline ≈ **3.31×**.

**Run-to-run variance:** A second immediate harness run on the same machine reported combined **1.640 / 2.691 / 7.565** ms with the same **3.31×** geomean (baseline and submission medians both jitter slightly; **V = 128k** combined time was stable to **~0.03%**).



### Version 2 - Submission to Popcorn leaderboard 

After Colab looked good with V_2, the **first** official **leaderboard** upload used **`submission.py`** (aligned with **v2** / `submission_2.py`). Popcorn reported:

Note: From here onwards, every file submitted to leaderboard will be **`submission.py`** which will be identical to the version that is selected as the best version to be submitted. 

| Field | Value |
|------|--------|
| GitHub / display user | **TharakaDFonseka** |
| Reported combined time | **2716.164 μs** (~**2.716 ms**) |

That number is the grader’s own median over its **three vocabulary sizes** and environment (**A100**, PyTorch **2.11.0**, Triton **3.6.0**, CUDA **12.x**); it will not exactly match a single Colab row or the arithmetic sum of per-**V** “Fwd+Bwd ms” above. Use Popcorn for rank; use Colab + `test_cross_entropy.py` for iteration.

## Approaches I tried

### 1) PyTorch eager baseline
**Idea:** Start from the simplest correct implementation using `torch.nn.functional.cross_entropy` and a manual backward formula.

**Why try it:** Establish a correctness baseline and a timing reference.

**What happened:** Correct, but slower. The eager path launches multiple kernels and tends to materialize more intermediate tensors in HBM.

**Expected effect:** Lowest engineering effort, but not the fastest.

---

### 2) `torch.compile` around the forward only [Mention the assignment]
**Idea:** Compile only the forward `F.cross_entropy(..., reduction="none")`.

**Why try it:** The official `torch.compile` docs explain that it can lower Python/PyTorch code to optimized kernels, often with fusion. See the official docs:
- PyTorch `torch.compile` API: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- PyTorch tutorial: https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

**What happened:** Forward became much faster than eager. The compiled forward used TorchInductor/Triton-generated kernels.

**Why faster:** Fewer kernel launches and better fusion.

---

### 3) `torch.compile` for both forward and backward (final)
**Idea:** Keep the forward as compiled cross-entropy, and write the backward directly with the stable softmax-gradient formula:
\[
\nabla_x \ell = (\mathrm{softmax}(x) - \mathrm{onehot}(y)) \cdot g
\]
where `g = grad_output[:, None]`.

**Why try it:** This keeps the code compact and gives the compiler a clear computation graph to optimize.

**Implementation details:**
- convert logits to float32 inside the compiled region
- compute row-wise max for stability
- compute shifted logits, log-sum-exp, and probabilities
- subtract 1 at the target index
- multiply by `grad_output`
- cast final gradient back to bf16

**Why this is a good tradeoff:** It is simple, numerically stable, and much more likely to compile well than a more complicated custom autograd path.

---

### 4) Ideas I considered but did not fully pursue
- Custom Triton forward kernel
- Custom Triton backward kernel
- Fused forward+backward in one custom kernel
- Tuning Triton block sizes manually
- More aggressive memory-layout tricks

**Why I did not use them in the final version:** They may be faster, but they require significantly more engineering/debugging effort and correctness validation. For this submission, I prioritized a strong, correct, and relatively compact implementation.

## Why the final version should be faster than eager

Cross-entropy is mainly **memory-bound**, so reducing HBM traffic matters more than increasing raw FLOPs. That means fusion is important.

Relevant references:
- Triton fused softmax tutorial: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
- Triton fused softmax tutorial code: https://github.com/triton-lang/triton/blob/main/python/tutorials/02-fused-softmax.py

The Triton fused softmax tutorial explicitly explains that fusion helps bandwidth-bound operations by keeping more intermediate values on-chip instead of writing everything to global memory.

## Timing table

Byte counts and bandwidth below match `test_cross_entropy.py`:  
`fwd_bytes = 2*B*V + 12*B`, `bwd_bytes = 4*B*V + 12*B`, with `B = 4096`.

| Approach | Forward time (ms) | Backward time (ms) | Combined (ms) | Notes |
|---|---:|---:|---:|---|
| Eager baseline | — | — | — | From `test_cross_entropy.py` baseline block (reference for speedup). |
| `torch.compile` forward only | — | — | — | Optional intermediate experiment. |
| **submission_1** (clone in bwd) | **~1.7285** | **~6.3907** | **~8.12** | **Google Colab** microbench: **V = 128,256** only, **10** warmup / **50** iters (also printed as **1.728512** / **6.390784** ms in an earlier paste). |
| **submission_2** / **`submission.py`** | (see per-**V** table above) | (see per-**V** table above) | **1.599 / 2.756 / 7.563** | **A100 40GB** full **`test_cross_entropy.py`** run; **geomean speedup ≈ 3.31×** (see variance note above). |

For **competition-comparable** numbers, run **`python test_cross_entropy.py submission_1.py`** and **`submission_2.py`** (or `submission.py` for v2): **20** warmup, **100** runs, **three** **V** values, **geomean speedup** printed at the end.

## Achieved memory bandwidth

Effective bandwidth uses the same byte model as `test_cross_entropy.py`:  
`fwd_bytes = 2*B*V + 12*B`, `bwd_bytes = 4*B*V + 12*B`, **GB/s** = bytes / (time in s) / 10⁹.

The **submission_1** Colab timings below are only at **V = 128,256**, **B = 4096**. Recompute bandwidth for **submission_2** after you paste its fwd/bwd ms.

| Setting | Fwd GB/s | Bwd GB/s | Combined GB/s (sum of fwd + bwd medians) | Combined / 2039 peak |
|---:|---:|---:|---:|---:|
| **submission_1**, Colab, **V = 128,256**, fwd **1.7285** ms, bwd **6.3907** ms (harness also printed **1.728512** / **6.390784**) | **607.9** | **328.8** | **388.2** | **~19.0%** |
| **submission_2** (A100 40GB harness) | Use **Fwd+Bwd BW** from the main table: **492 / 448 / 417** GB/s at **V = 32k / 50k / 128k** | → combined / **2039** ≈ **24.1% / 22.0% / 20.4%** |

**Peak A100 bandwidth (per assignment):** 2039 GB/s.

- **submission_1** (older Colab, 128k only): see earlier row (~**19%** combined proxy on non-A100).
- **submission_2** on **A100**: combined effective bandwidth about **417–492 GB/s** across **V** (~**20–24%** of **2039**), in line with memory-bound softmax + CE + gradient traffic, kernel launches, and `max-autotune` multi-kernel backward.

**After Popcorn / grader PyTorch 2.11**, refresh this section if leaderboard numbers differ.

## Why I do not expect 100% of peak bandwidth

Even a very good implementation usually will not reach 100% of peak A100 bandwidth because of:
- kernel launch overhead
- imperfect fusion
- extra reads/writes for required outputs
- synchronization / reduction overhead
- non-ideal memory access patterns
- compiler/runtime overhead
- numerical-stability work (max, exp, log, reductions)

## How to run locally

```bash
python test_cross_entropy.py submission.py
python test_cross_entropy.py submission_1.py
python test_cross_entropy.py submission_2.py
```

Each run: correctness for all three **V**, timings, bandwidth, geomean speedup vs eager baseline.

## Google Colab setup — steps and commands

1. **Runtime → Change runtime type → GPU** (T4/L4/A100 depending on tier).

2. **Upload files** to `/content` (or use Drive): `submission.py`, `submission_1.py`, `submission_2.py`, `test_cross_entropy.py`.

   ```python
   from google.colab import files
   files.upload()   # select all four .py files
   ```

3. **Check GPU and PyTorch** (optional):

   ```python
   import torch
   print(torch.__version__, torch.cuda.get_device_name(0))
   ```

4. **Official harness (recommended)** — matches course script (**20** warmup, **100** iters). **Silence TorchInductor log spam** (otherwise you may see huge `torch/_inductor/graph.py` / `__output_code` dumps):

   ```python
   %cd /content
   %env TORCH_LOGS=
   !python test_cross_entropy.py submission_1.py
   !python test_cross_entropy.py submission_2.py
   !python test_cross_entropy.py submission.py
   ```

   If logs still appear, set the env **before** importing PyTorch in any earlier cell, e.g.:

   ```python
   import os
   os.environ["TORCH_LOGS"] = ""
   os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torchinductor_colab"  # optional: fixed cache dir
   ```

   Then restart the runtime or run the benchmark in a fresh process via `!python ...` only.


5. **Optional quick microbench** (single **V**, e.g. 128256 — good for A/B, not a substitute for step 4):

   ```python
   import importlib.util, sys, statistics, torch

   def load(path):
       spec = importlib.util.spec_from_file_location("sub", path)
       m = importlib.util.module_from_spec(spec)
       spec.loader.exec_module(m)
       return m

   def bench(mod, B=4096, V=128_256, warmup=10, iters=50):
       logits = torch.randn(B, V, device="cuda", dtype=torch.bfloat16)
       targets = torch.randint(0, V, (B,), device="cuda", dtype=torch.int64)
       go = torch.randn(B, device="cuda", dtype=torch.float32)
       def run_fwd():
           mod.cross_entropy_forward(logits, targets)
       def run_bwd():
           mod.cross_entropy_backward(logits, targets, go)
       for _ in range(warmup):
           run_fwd(); run_bwd()
       torch.cuda.synchronize()
       tf, tb = [], []
       for _ in range(iters):
           e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
           e0.record(); run_fwd(); e1.record(); torch.cuda.synchronize()
           tf.append(e0.elapsed_time(e1))
       for _ in range(iters):
           e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
           e0.record(); run_bwd(); e1.record(); torch.cuda.synchronize()
           tb.append(e0.elapsed_time(e1))
       return statistics.median(tf), statistics.median(tb)

   for name, path in [("v1", "/content/submission_1.py"), ("v2", "/content/submission_2.py")]:
       m = load(path)
       f, b = bench(m)
       print(name, "forward ms:", f, "backward ms:", b, "sum:", f + b)
   ```



I documented my optimization process in this journal. I compared **submission_1** (backward with `probs.clone()`) and **submission_2** (no clone; subtract `grad_output` on `grad = probs * g` at target indices). **`submission.py` matches submission_2** for the default benchmark and server upload. Full timings and geomean speedups come from `test_cross_entropy.py` on GPU (Colab or A100).
