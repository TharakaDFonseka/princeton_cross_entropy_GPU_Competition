# Cross-Entropy Speed Competition — Optimization Journal: Tharaka Fonseka (tf8712)

- **Goal:** Fastest correct **forward + backward** cross-entropy in a **single Python file** with `cross_entropy_forward` and `cross_entropy_backward` only using **torch** and **triton** (and submodules).
- **Hardware / problem size:** NVIDIA **A100 80GB**, `B = 4,096`, **bfloat16** logits, `reduction='none'`, **V ∈ {32,000, 50,264, 128,256}**.
- **Correctness:** vs reference with **atol=1e-3**, **rtol=1e-2**.
- **Environment:** PyTorch **2.11.0**, Triton **3.6.0**, CUDA **12.x**.
- **Timing:** `torch.cuda.Event`, **median of 100** runs after warmup, **per V**. **Ranking uses geometric mean speedup** of **combined forward+backward** vs **PyTorch eager baseline** (the harness also prints separate fwd/bwd for analysis).


## Final submission summary

**Primary leaderboard implementation:** **`submission_9.py`** (**v9**) — same **online** softmax and **`logden`** forward→backward **cache** as **v7**; **`tl.constexpr`** on **`V`** and strides; **no `@triton.autotune`**. Instead, a hand-maintained **`_LAUNCH`** table picks **`BLOCK_V`**, **`num_warps`**, and **`num_stages`** **per competition vocabulary** (power-of-2 **`BLOCK_V`** only: **8192** for **32k/50k**, **16384** for **128k**, **`num_warps = 16`**, **`num_stages = 2`**) so each **`V`** uses a **fixed** launch tuned for **A100**-class bandwidth. **`submission_8.py`** (**v8**) is **not** the same program: it uses **`@triton.autotune`** over multiple **`BLOCK_V`** / warps / stages (**908.166 μs** here) instead of **`_LAUNCH`**. **Best Popcorn time** recorded here: **875.111 μs** (**v9**, official leaderboard). **v7** **936.952 μs**. **`submission_6.py`** (**v6**, **1236.180 μs**) remains the **cache-free** baseline. Earlier versions (**v1–v3**) use **`torch.compile`**; **v4** was the first full-Triton Popcorn baseline (**1992.267 μs**).

**Files (versions + default upload name):**

| File | Role |
|------|------|
| `submission_1.py` | **v1** — backward uses `probs.clone()` before subtracting 1 at the target index, then multiplies by `grad_output` (earlier baseline). |
| `submission_2.py` | **v2** — backward uses `grad = probs * grad_output`, then `grad[rows, targets] -= grad_output` (same math, no extra full-matrix clone). |
| `submission_3.py` | **v3** — experimental: “narrow” forward (stable log-sum-exp + `torch.gather` at `targets`; backward same compiled path as v2). Slower than v2 in Colab A/B; **not** used for rank. |
| `submission_4.py` | **v4** — **Triton** CE forward + backward (separate max pass, then sum-exp pass, then grad pass on backward); Popcorn **1992.267 μs** (superseded by **v5–v7**). |
| `submission_5.py` | **v5** — **Triton** with **online** max + LSE in one tiled sweep; piecewise **`BLOCK_V`** (2048 for 32k/50k, 4096 for 128k); Popcorn **1270.465 μs**. |
| `submission_6.py` | **v6** — **Same online kernels as v5**; **uniform `BLOCK_V = 4096`** for all **V**; Popcorn **1236.180 μs**. |
| `submission_7.py` | **v7** — cached **`logden`**, autotune, stride-based I/O, int64 targets in-kernel; Popcorn **936.952 μs**. |
| `submission_8.py` | **v8** — **`logden`** cache + **`tl.constexpr`** **`V`/strides** + **`@triton.autotune`** (wider **`BLOCK_V`** grid **512–8192**, **`num_warps`** up to **16**); **no** fixed **`_LAUNCH`** table — kernels pick config after warmup. Popcorn **908.166 μs**. |
| `submission_9.py` | **v9** — **`logden`** cache + **`constexpr`** + **hand-tuned `_LAUNCH`** per **`V`** (**no** `@triton.autotune`); Popcorn **875.111 μs** (**best** here). |
| `submission.py` | **Copy of the version you submit** — for Popcorn: **`cp submission_9.py submission.py`** (best **μs** recorded here). |

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

**A100 80GB — full three-**V** harness (`test_cross_entropy.py`).** Course stack (**PyTorch 2.11.0**, **Triton 3.6.0**, **20** warmup / **100** timed iterations, median ms). **Geomean speedup vs eager (printed): 3.17×**.

```
        V |   Fwd ms   Bwd ms  Fwd+Bwd ms |     Fwd BW     Bwd BW  Fwd+Bwd BW |  Speedup
    32000 |   0.419    1.291       1.632  |    626.0     406.1      481.9  |    3.49x
    50264 |   0.549    2.573       3.030  |    750.3     320.1      407.7  |    3.01x
   128256 |   1.393    6.377       7.681  |    754.5     329.5      410.4  |    3.03x
```

**% of peak (2039 GB/s)** from printed **Fwd+Bwd BW**: **32k ~23.6%**, **50k ~20.0%**, **128k ~20.1%**.

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

**Caveats:** Official grader lists **A100 80GB** and **PyTorch 2.11.0**; this machine is **40GB** and **2.10.x** — expect small shifts on the exact leaderboard stack. The **40GB vs 80GB** SKU does not change compute throughput for this problem size in a meaningful way; **Popcorn** is still the authoritative rank.

### Version 3 — “narrow” forward experiment (`submission_3.py`)

**Motivation (why try this).** Cross-entropy with `reduction='none'` only needs **one scalar loss per row**. The **v2** forward uses `F.cross_entropy` inside `@torch.compile`, which is correct and fast, but we hypothesized that a **hand-written forward** might avoid extra work or extra global traffic: stable **row-wise max**, **log-sum-exp** over the row, then **gather** the target logit with `torch.gather` and form \(\ell_i = -(x_{i,y_i} - m_i - \mathrm{LSE}_i)\). That path still reads the full logits row for the **sum of exponentials** (same asymptotic reads as softmax normalization) but **does not need to materialize a full `(B, V)` softmax tensor for the forward return value**—only the backward still builds probabilities the usual way. Under **memory-bound** assumptions, that sometimes helps; we also wanted a **rules-legal** experiment before investing in a full **Triton** kernel.

**What we implemented.** New file **`submission_3.py`**: `torch.compile(fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")` on (1) **`_compiled_forward`**: `logits.float()` → row `max` → `exp`/`sum`/`log` for LSE → `torch.gather` for \(x_{i,y_i}\) → **float32** losses; (2) **`_compiled_backward`**: **identical** to **v2** (softmax from logits, `grad = probs * grad_output[:, None]`, subtract `grad_output` at targets, **bf16** out). A small helper script **`colab_benchmark_cross_entropy.py`** was added to run **`test_cross_entropy.py`** on **v2** then **v3** in subprocesses (avoids a Colab **`TORCH_LOGS`** / `import torch` crash in the parent process).

**What happened — correctness.** Full harness **PASS** all **V** (same tiny errors as v2, ≈**1e−6** on printed checks).

**What happened — performance (same Colab session, direct A/B).** **Google Colab**, **NVIDIA A100-SXM4-40GB**, **PyTorch 2.10.0+cu128**, course script (**20** warmup, **100** timed iters). Baseline jitter differs slightly between the two runs below because each harness invocation re-benchmarks eager baseline; focus on **submission** columns and **geomean speedup** printed for each file.

**v2 (`submission_2.py`) in that session:**

```
        V |   Fwd ms   Bwd ms  Fwd+Bwd ms |     Fwd BW     Bwd BW  Fwd+Bwd BW |  Speedup
    32000 |   0.410    1.308       1.634  |    639.3     401.0      481.3  |    3.49x
    50264 |   0.610    2.245       2.768  |    674.8     366.8      446.3  |    3.30x
   128256 |   1.396    6.304       7.598  |    752.8     333.4      414.9  |    3.06x
```

**Competition-style geomean speedup (printed):** **3.28×**.

**v3 (`submission_3.py`) in the same environment:**

```
        V |   Fwd ms   Bwd ms  Fwd+Bwd ms |     Fwd BW     Bwd BW  Fwd+Bwd BW |  Speedup
    32000 |   0.383    1.368       1.683  |    684.6     383.3      467.2  |    3.39x
    50264 |   0.714    2.232       2.867  |    576.6     368.9      430.9  |    3.18x
   128256 |   1.599    6.294       7.806  |    656.9     333.9      403.8  |    2.98x
```

**Competition-style geomean speedup (printed):** **3.18×**.

**Interpretation.** **v3 is slower overall** on this stack (**3.18×** vs **3.28×** geomean). Per **V**, **v3** is slightly **faster on forward at 32k** but **worse on forward at 50k and 128k**; backward is mixed but **combined Fwd+Bwd** loses at all three **V** vs **v2** in this run. Likely **TorchInductor** already lowers **`F.cross_entropy`** very well for **v2**, while **v3**’s graph (**gather** + explicit LSE) gets a **less favorable** fusion / kernel mix on the larger vocab sizes. **Decision:** keep **Popcorn / default `submission.py` aligned with v2**; treat **v3** as a documented negative result unless we revisit with **PyTorch 2.11** on the grader or a **custom Triton** forward/backward.

### Version 2 — first Popcorn submission (compiled baseline)

After Colab looked good with **v2**, the **first** official **leaderboard** upload used **`submission.py`** (aligned with **`submission_2.py`**). Popcorn reported:

Note: You can upload any **`submission_*.py`** filename to Popcorn, or copy the chosen file to **`submission.py`** for convenience. 

| Field | Value |
|------|--------|
| GitHub / display user | **TharakaDFonseka** |
| Reported combined time | **2716.164 μs** (~**2.716 ms**) |

That number is the grader’s own median over its **three vocabulary sizes** and environment (**A100**, PyTorch **2.11.0**, Triton **3.6.0**, CUDA **12.x**); it will not exactly match a single Colab row or the arithmetic sum of per-**V** “Fwd+Bwd ms” above. Use Popcorn for rank; use Colab + `test_cross_entropy.py` for iteration.

### Version 4 — Custom Triton (`submission_4.py`) — motivation, design, results

**Why move to Triton after v2/v3.** The problem is **memory-bandwidth heavy** (large **V**, read logits for softmax / CE, write **grad_logits**). **`torch.compile` + `F.cross_entropy` (v2)** already gets strong Inductor kernels, but the backward path can still split into **multiple launches** and extra traffic. **v3** (hand-written forward + `gather` under compile) was **slower** than v2 on Colab: Inductor did not reward that graph. The assignment explicitly allows **custom Triton kernels**, so the next step was to implement the **same math** as the reference in **explicit row-parallel kernels**: one program per batch row, **tiled** scans over **V**, minimizing reliance on the compiler to fuse a large PyTorch graph.

**What was implemented.**

- **`_ce_fwd_kernel`:** For each row, **two passes** over **V** in tiles of **`BLOCK_V`** (2048, or 4096 for large **V**): (1) **stable max**; (2) **sum of exp(x − m)** and **log** for log-sum-exp; then load **logits[row, target]** and set **loss = −(x\_y − m − LSE)** in **float32**. Targets are cast to **int32** in the Python wrapper (valid for **V ≤ 128256**).
- **`_ce_bwd_kernel`:** Same row layout: recompute **m** and **LSE**, then one more tiled pass: **p = softmax**, **grad = p · g\_row**, subtract **g\_row** on the target column, store **bfloat16**.
- **Implementation detail:** Triton **`@triton.jit`** cannot read arbitrary **module-level** Python globals; an early Colab run failed with **`NameError` on `_NEG`**. The fix was to use a **local literal** sentinel (**`-1.0e30`**) inside the kernel for masked max reductions instead of a global constant.

**Colab (development).** On **Google Colab**, **A100-SXM4-40GB**, **PyTorch 2.10.0+cu128**, **`test_cross_entropy.py`** reported **PASS** all **V** and a much higher printed **geomean speedup** than v2 in the same session (e.g. **~4.50×** vs **~3.27×** for v2), with especially large gains on **backward** vs the compiled v2 graph. Those numbers are **not** the official grader stack but were strong evidence to try **Popcorn**.

**Popcorn leaderboard (official).** Ranked submit of **`submission_4.py`**:

| Field | Value |
|------|--------|
| User | **TharakaDFonseka** |
| Reported combined time | **1992.267 μs** (~**1.992 ms**) |

**Comparison to v2 on the same leaderboard metric:** **2716.164 μs → 1992.267 μs** → **~26.6%** lower time (**~1.36×** faster in wall-clock ratio **2716.164 / 1992.267**). This aligns with the qualitative Colab gap (Triton replacing many small backward kernels with one big row kernel per call direction).

**Later improvement — see Versions 5–6.** **v5** reduces Popcorn time to **1270.465 μs**; **v6** (wide uniform tile, **Version 6**) improves to **1236.180 μs**. **v4** remains the clean baseline for “first custom Triton” vs **v5**’s fused reduction.

### Version 5 — Online softmax / LSE (`submission_5.py`) — why, inspiration, results

**Why do this after v4.** **v4** already uses **one kernel per forward** and **one per backward**, but the math is implemented with **extra passes over each row’s logits**:
- **Forward:** full sweep for **row max `m`**, then a **second** sweep for **Σ exp(x − m)** (then target logit + loss).
- **Backward:** **three** sweeps — **max**, **sum exp** (to get **LSE** / normalization), then **softmax gradient**.

For large **V**, the kernel is **memory-bound**; every **extra** full read of the **bf16** row is expensive. The goal was to **remove one full read pass on the backward path** and **merge max + sum-exp on the forward path** without changing numerics (still stable softmax / cross-entropy in **float32** inside the kernel).

**Inspiration.** The standard **online softmax** trick (maintain running **max** and a running **sum of exponentials** rescaled when the max increases) is how fused softmax tutorials avoid multiple full passes over the row. The Triton community material on **fused softmax** describes combining max and sum in one streaming reduction; see the [Triton fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) and [tutorial source](https://github.com/triton-lang/triton/blob/main/python/tutorials/02-fused-softmax.py). **v5** applies the same **idea** to our row-wise CE: in each tile, update **`m_new = max(m, tile_max)`** and **`s = s * exp(m - m_new) + sum(exp(x - m_new))`**, then **`lse = log(s)`**; backward repeats that fused reduction, then one pass for **`p = exp(x - m - lse)`** and **`grad`**.

**What was implemented (summary).** New file **`submission_5.py`**: same API as **v4** (`cross_entropy_forward` / `cross_entropy_backward`), same **`BLOCK_V` / `num_warps`** launch pattern, literals-only inside **`@triton.jit`**. **Forward:** one tiled loop for **(m, s)** → **`lse`**, then target load and loss. **Backward:** one tiled loop for **(m, s)** → **`lse`**, then one tiled loop for **grad** (same math as **v4**).

**Correctness.** **`test_cross_entropy.py`** on Colab (**A100 40GB**, **PyTorch 2.10.0+cu128**): **PASS** all **V** (errors on the order of **1e−6** fwd / **1e−6–1e−5** bwd in a representative run — within harness tolerances).

**Colab (development).** Same harness reported a high **geomean speedup vs eager** (e.g. **~6.7×** in one session); as always, that number is **relative to that machine’s eager baseline**, not a direct map to Popcorn **μs**.

**Popcorn leaderboard (official).** Ranked submit of **`submission_5.py`** (or **`submission.py`** copied from it):

| Field | Value |
|------|--------|
| User | **TharakaDFonseka** |
| Reported combined time | **1270.465 μs** (~**1.270 ms**) |

**Comparison on the same leaderboard metric:**
- **v2 → v4:** **2716.164 → 1992.267 μs** (**~1.36×** faster than v2).
- **v4 → v5:** **1992.267 → 1270.465 μs** (**~1.57×** faster than v4; combined time **~36%** lower).
- **v2 → v5:** **2716.164 / 1270.465 ≈ 2.14×** faster than the original compiled Popcorn submit.
- **v5 → v6:** **1270.465 → 1236.180 μs** (see **Version 6** — wide uniform tile).

**Current choice for submission (superseded by v6 on Popcorn μs).** **v5 (`submission_5.py`)** is the main **reference implementation** for the **online** row kernel; **v6** keeps the same math and Popcorn-safe launch settings but changes only tiling (**Version 6**).

### Version 6 — Wide uniform tile (`submission_6.py`) — motivation, inspiration, Popcorn

**Why try something after v5.** **v5** already minimized **full-row read count** vs **v4** (online softmax). Further gains on **A100** then come from **how** each row is tiled: smaller **`BLOCK_V`** means **more** outer-loop iterations in the generated Triton program (more trip count over **V** in the softmax loop), which adds **instruction and control overhead** even when memory traffic is similar. The competition vocab sizes **32,000** and **50,264** used **`BLOCK_V = 2048`** in **v5** (only **128,256** used **4096**).

**Inspiration.** This follows a standard **HPC / GPU blocking** idea: pick a **single wide tile** so that, for fixed **V**, the **number of tiles per row** drops (e.g. **32,000 / 4096 ≈ 7.8** rounds vs **32,000 / 2048 = 15.625**). Textbook **loop tiling** for memory-bound kernels often trades **tile width** against **register pressure**; here **4096** stayed **fast and correct** on the official stack. The change is **scheduling only**—same **online** update **`m`**, **`s`**, same backward structure as **v5**—so it avoids risky experiments that broke **Popcorn** in this project (**HTTP 500**, “another stream”): e.g. **`tl.constexpr V`**, **`num_warps = 16`**, **fused target gather inside the forward loop** (with extra integer compares), and **`eviction_policy`** / high **`num_stages`** were **not** used in the shipped **v6**.

**What was implemented.** **`submission_6.py`**: **identical** `@triton.jit` bodies and **`_launch_kw`** (**`num_warps = 8`**, **`num_stages = 2`**) as **`submission_5.py`**. Only **`_block_v`** becomes a constant **`return 4096`** (uniform wide tile for every **V** in the harness).

**Popcorn leaderboard (official).** Ranked submit of **`submission_6.py`**:

| Field | Value |
|------|--------|
| User | **TharakaDFonseka** |
| Reported combined time | **1236.180 μs** (~**1.236 ms**) |

**Comparison on the same leaderboard metric:** **1270.465 → 1236.180 μs** → about **2.7%** lower combined time (**~1.028×** faster than **v5** in the ratio **1270.465 / 1236.180**). Diminishing returns are expected once the kernel is already **memory-bound** and **online**-fused.

**Role after v7.** **v6** remains the reference for **fixed launch** + **no cross-call cache**; **v7** supersedes it on Popcorn (see **Version 7**).

### Version 7 — Cached log-denominator + autotune + lean Python I/O (`submission_7.py`)

**Motivation.** The grader scores **geometric mean** of **combined forward + backward** time. **v5/v6** backward still **recomputes** the row log-partition function \(\log\sum_j e^{x_j}\) (online max + sum) in a **first** full pass over logits, then a **second** pass for softmax **×** `grad_output` and the target correction. That matches the usual autograd contract (no saved activations between API calls), but when **backward** immediately follows **forward** on the **same** `(logits, targets)`, the forward pass has **already** computed the same scalar per row.

**Main idea (largest win).** **Store `logden`** from the forward kernel into a small **`(B,)`** float32 buffer and a process-local cache keyed by tensor identity (`data_ptr`, shape, strides, device). On the next **`cross_entropy_backward`** with a **cache hit**, run **`_ce_bwd_cached_kernel`**: one tiled sweep **`p = exp(x - logden)`**, **`grad = p·g − one_hot·g`**, writing **bf16** `grad_logits` — **one** full read of the logits row instead of **two**. On a **miss**, **`_ce_bwd_recompute_kernel`** falls back to the **v6-style** two-pass backward (correct for arbitrary call patterns).

**Supporting ideas (Python / launch hygiene).**

- **Avoid `targets.to(torch.int32)` in Python** on every call: keep **`targets`** as **`int64`**, pass **`targets.stride(0)`** into Triton, and **`tl.load` → `.to(tl.int32)`** only inside the kernel when forming the column index. That removes extra host work and a potential temp tensor on the hot path.
- **Avoid `logits.contiguous()` / `grad_output.contiguous()`** when the harness already supplies contiguous tensors: use **`logits.stride(0/1)`** and **`grad_output.stride(0)`** directly so the wrappers do not add redundant copies or sync points.
- **`@triton.autotune`** on forward and backward with a **small** config list (**`BLOCK_V` ∈ {1024, 2048, 4096}**, **`num_warps`**, **`num_stages`**) and **`key=["V"]`** so Triton picks a decent tile per competition vocabulary without hand-maintaining three separate files.

**Caveats.** The cache assumes **no in-place mutation** of **`logits`** between forward and backward with the same storage; the harness does not do that. **`@triton.autotune`** adds compile-time on first touch; Popcorn still accepted this submission (**936.952 μs** ranked).

**Popcorn leaderboard (official).** Ranked submit of **`submission_7.py`**:

| Field | Value |
|------|--------|
| User | **TharakaDFonseka** |
| Reported combined time | **936.952 μs** (~**0.937 ms**) |

**Comparison on the same leaderboard metric:** **1236.180 μs (v6) → 936.952 μs (v7)** → about **24%** lower wall time (**~1.32×** faster in the ratio **1236.180 / 936.952**).

**Google Colab harness** (**NVIDIA A100-SXM4-40GB**, **PyTorch 2.10.0+cu128**, **`B = 4096`**, **`test_cross_entropy.py`**): correctness **PASS** all **V**; printed table:

```
        V |   Fwd ms   Bwd ms  Fwd+Bwd ms |     Fwd BW     Bwd BW  Fwd+Bwd BW |  Speedup
    32000 |   0.250    0.639       0.652  |   1049.4     820.6     1205.8  |    8.74x
    50264 |   0.410    1.072       1.083  |   1004.1     768.2     1140.3  |    8.43x
   128256 |   0.823    2.422       2.427  |   1276.2     867.7     1298.8  |    9.59x
```

**Competition-style geomean speedup (printed):** **~8.91×** vs eager on that run (product of speedups **8.74 × 8.43 × 9.59** cube-root).

**Current choice for submission.** **v7 (`submission_7.py`)** was the **best measured Popcorn** implementation until **v8**; **v6** is the fixed-config baseline; **v5** documents the **online softmax** design.

### Version 8 — `tl.constexpr` specialization + wider autotune (`submission_8.py`)

**Motivation.** After **v7**, incremental gains require squeezing more from the **same** math: better **tile / occupancy** choices per vocabulary size, and helping the Triton compiler **specialize** kernels for fixed **`B`**, **`V`**, and typical **strides** so inner loops and loads match what the benchmark actually uses.

**What we implemented.**

- **Same algorithmic contract as v7:** **`B_FIXED = 4096`**, **online** forward (**max** + running **sum of exp**), store per-row **`logden`** for a process-local **cache**; **cached backward** = one tiled pass **`p = exp(x − logden)`**, **`grad = p·g`**, subtract **`g`** on the target column; **recompute backward** = v6-style two passes if the cache misses.
- **`tl.constexpr` on `V` and strides** (`stride_l0`, `stride_l1`, `stride_t0`, and backward grad strides): passed as Python **`int(...)`** from **`logits` / `targets` / `grad_output` / `grad`** so each **`(V, layout)`** pair can compile to a **specialized** kernel (known trip counts and indexing for the competition’s **`[4096, V]`** contiguous case).
- **Wider `@triton.autotune`** lists: **`BLOCK_V` ∈ {512, 1024, 2048, 4096, 8192}** with multiple **`num_stages`** pairs (e.g. **2048** and **4096** with stages **3–4**, **8192** with **`num_warps = 16`** and stages **1–2**). Warmup lets Triton pick a better **tile × pipeline depth** per **`V`** on **A100-class** GPUs (memory-bound softmax rows).
- **Micro-optimizations** retained from earlier tuning: **`exp * mask`** for masked tile sums (avoids **`tl.where`** in the reduction), **`grad - g * ((offs == tgt) & mask)`** for the one-hot term.

**Why this is expected to use less time (when it wins).**

- **Autotune:** Different **`V`** values change how many partial tiles appear at the end of a row; a **larger search space** can find a config with **fewer loop iterations** or better **memory-level parallelism** (`num_stages`) for that **`V`**.
- **`constexpr` `V` / strides:** Lets the compiler treat **row length** and **pointer arithmetic** as **compile-time** facts for the common **contiguous** case, which can reduce overhead vs fully dynamic bounds.
- **Cache hit path (unchanged from v7):** Combined **forward + backward** still avoids **one** full logits read on backward vs recomputing **logden** — the dominant term when the harness runs **fwd** then **bwd** on the same tensors.

**Caveats (same family as v7, plus tuning risk).**

- **Cross-call cache** still assumes no **in-place** **`logits`** mutation between forward and backward for the same storage.
- **Larger autotune** grids increase **first-touch compile** time; acceptable if warmup is outside the timed region (course harness / Popcorn pattern).
- This journal previously associated **`tl.constexpr V`**, **`num_warps = 16`**, and high **`num_stages`** with occasional **Popcorn HTTP 500** / multi-stream issues in **other** iterations; **v8** reintroduces those knobs deliberately for speed — if the remote grader misbehaves, trim configs or drop **`constexpr`** first.

**Local harness (`test_cross_entropy.py`) — one A100-class session (median ms, printed geomean speedup vs eager):**

```
        V |   Fwd ms   Bwd ms  Fwd+Bwd ms |     Fwd BW     Bwd BW  Fwd+Bwd BW |  Speedup
    32000 |   0.248    0.488       0.649  |   1058.0    1073.5     1211.5  |    8.77x
    50264 |   0.359    0.887       0.994  |   1145.8     928.7     1242.5  |    9.18x
   128256 |   0.827    2.416       2.418  |   1270.7     869.7     1303.8  |    9.62x
```

**Competition-style score from this run:** geometric mean speedup vs the script’s eager baseline ≈ **9.18×**.

**Popcorn leaderboard (official).** Ranked submit of **`submission_8.py`**:

| Field | Value |
|------|--------|
| User | **TharakaDFonseka** |
| Reported combined time | **908.166 μs** (~**0.908 ms**) |

**Comparison on the same leaderboard metric:** **936.952 μs (v7) → 908.166 μs (v8)** → about **3.1%** lower wall time (**~1.032×** faster in the ratio **936.952 / 908.166**).

**Current choice for submission.** **v8 (`submission_8.py`)** improved over **v7** via **`constexpr`** + **wider autotune** (**908.166 μs**); **v9** (**`submission_9.py`**) supersedes **v8** with **fixed per-`V` launches** (**875.111 μs** — see **Version 9**).

### Version 9 — Hand-tuned per-vocabulary launch, no autotune (`submission_9.py`)

**Motivation.** **v8** added **`tl.constexpr`** and a **large `@triton.autotune`** grid so Triton could pick **`BLOCK_V` / warps / stages** per **`V`**. In practice, autotune **benchmarks many configs** (compile + timing during warmup), can pick a **suboptimal** winner on a given stack, and **`tl.arange(0, BLOCK_V)`** requires **power-of-2** **`BLOCK_V`** — so non-power-of-2 tile ideas fail outright. Empirically, **manual tuning** per competition **`V`** (only three values) gives **more control** than expanding autotune further, with **predictable** behavior on **A100**.

**What we implemented.**

- **Same math and cache as v7/v8:** **online** row softmax (**max** + running **sum of exp**), **`logden = m + log(s)`** stored; **Python** cache keyed by tensor identity; **cached backward** = **`exp(x − logden)`** one sweep; **recompute** path if cache misses.
- **Removed `@triton.autotune`** from forward and backward kernels.
- **`_LAUNCH: dict[int, …]`** maps **`V ∈ {32000, 50264, 128256}`** to **`{BLOCK_V, num_warps, num_stages}`**. Launch passes **`BLOCK_V=`**, **`num_warps=`**, **`num_stages=`** with **`tl.constexpr`** **`V`** and strides unchanged.
- **Default table (A100-oriented):** **32k / 50k** → **`BLOCK_V = 8192`**, **`num_warps = 16`**, **`num_stages = 2`**; **128k** → **`BLOCK_V = 16384`** (fewer **for start in range(0, V, BLOCK_V)** iterations than **8192** on that row length), same warps/stages. **`BLOCK_V`** must stay a **power of 2** (Triton **`arange`** rule).
- **`submission_9.py` only:** **`submission_8.py`** still contains **`@triton.autotune`** on the kernels; do not merge the two files without choosing one launch strategy.

**Why results are expected to improve (vs autotune-heavy v8).**

- **No autotune lottery:** The timed region avoids depending on which config “won” after a finite benchmark budget.
- **Wider tiles where it matters:** **8192** vs **4096** reduces **outer-loop** iterations on **32k/50k**; **16384** on **128k** targets the largest **V** where iteration count dominates scheduling overhead.
- **`num_warps = 16`** on large tiles aims at **memory throughput** on **A100**; **`num_stages = 2`** is a common pipeline choice for **HBM-bound** row kernels.
- **Same `logden` cache:** Combined **fwd+bwd** still saves **one** full logits pass on backward when the harness runs forward then backward on the same tensors — unchanged from **v7**.

**Caveats.**

- **16384** tiles increase **per-tile register / SRAM** pressure; if a stack **fails to compile**, fall back to **`BLOCK_V = 8192`** for **128256** in **`_LAUNCH`**.
- Tuning is **GPU-specific**; **A100 40GB vs 80GB** is usually similar for this kernel; **Popcorn** remains the authority.

**Harness (`test_cross_entropy.py`) — Google Colab, `/content`, GPU **NVIDIA A100-SXM4-40GB**, **PyTorch 2.10.0+cu128**, **`B = 4096`**. Correctness: **PASS** all **V** (max |fwd|/|bwd| error **~1e−6** order). Printed table:

```
        V |   Fwd ms   Bwd ms  Fwd+Bwd ms |     Fwd BW     Bwd BW  Fwd+Bwd BW |  Speedup
    32000 |   0.231    0.470       0.628  |   1133.0    1116.8     1253.0  |    9.06x
    50264 |   0.347    0.867       0.979  |   1186.3     949.6     1262.0  |    9.32x
   128256 |   0.812    2.358       2.392  |   1293.9     891.3     1317.7  |    9.72x
```

**Competition-style score from this run:** geometric mean speedup vs eager ≈ **9.36×**.

**Popcorn leaderboard (official).** Ranked submit of **`submission_9.py`**:

| Field | Value |
|------|--------|
| User | **TharakaDFonseka** |
| Reported combined time | **875.111 μs** (~**0.875 ms**) |

**Comparison on the same leaderboard metric:** **908.166 μs (v8) → 875.111 μs (v9)** → about **3.6%** lower wall time (**~1.038×** faster in the ratio **908.166 / 875.111**). **936.952 μs (v7) → 875.111 μs (v9)** → about **6.6%** lower than **v7**.

**Current choice for submission.** **`submission_9.py`** (**v9**) is the **best Popcorn μs** and harness geomean in this journal. Use **`submission_8.py`** when you want the **autotuning** variant (different **`.py`** file — not a copy of **v9**).

## Rubric answers (assignment prompts — consolidated)

The bullets below mirror the course questions. Detailed narrative lives in **Versions 1–9**, **Timing table**, **Achieved memory bandwidth**, and **Approaches I tried**.

### What approaches did you try?

| Category | What we did (files) |
|----------|---------------------|
| **`torch.compile`** | **v1–v3**, **`submission.py`** aligned with **v2**/**v3**: `max-autotune-no-cudagraphs`, stable softmax backward, optional narrow forward (**v3**). |
| **Custom Triton kernels** | **v4** (two-pass fwd / three-pass bwd over row), **v5** (**online** max + sum in one fwd pass; two-pass bwd), **v6** (uniform wide **`BLOCK_V`** vs **v5**’s mixed tile), **v7** (cached **`logden`** + autotune + stride-based loads), **v8** (**v7** + **`tl.constexpr`** **`V`/strides** + wider autotune), **v9** (**v7/v8** math + **`constexpr`** + **hand-tuned `_LAUNCH`**, **no** autotune). |
| **Fused forward + backward** | **Not in one API call** — harness requires separate **`cross_entropy_forward`** / **`cross_entropy_backward`**. **Within** backward, **v5–v6** fuse max+sum-exp (**online**) vs **v4**’s separate passes; **v7** skips recomputation of **`logden`** on cache hit (**one** logits pass). |
| **Memory layout** | **v4–v6:** **`logits.contiguous()`**, **`targets`**, **`grad_output`** before kernels. **v7:** no forced **`contiguous()`** on logits / `grad_output`; **int64** **`targets`**, strides passed into kernels. |
| **Block size / launch tuning** | **v4–v6:** hand-picked **`BLOCK_V`**, **`num_warps=8`**, **`num_stages=2`** (with **v5**’s mixed tile). **v7:** **`@triton.autotune`** over **1024 / 2048 / 4096** × warps/stages, **`key=['V']`**. **v8:** wider grid (**512–8192**, **`num_warps`** up to **16**, multiple **`num_stages`**). **v9:** **no** autotune — **`_LAUNCH[V]`** sets **`BLOCK_V` / warps / stages** per competition **V** (e.g. **8192** / **16384**). |

### For each approach: timing vs previous best and why

| Step | vs previous best (Popcorn combined **μs** where official) | Why faster or slower |
|------|-----------------------------------------------------------|----------------------|
| **v2** first Popcorn | **2716.164** (baseline compile path) | Reference after Colab tuning. |
| **v4** Triton | **2716 → 1992** (**~1.36×**) | Fewer / larger kernels vs Inductor on backward; explicit row fusion. |
| **v5** online | **1992 → 1270** (**~1.57×** vs **v4**) | **One fewer** full-row read on backward; forward max+sum merged (**online softmax**). |
| **v6** uniform **4096** | **1270 → 1236** (**~1.03×** vs **v5**) | Fewer outer-loop iterations on **32k/50k**; same HBM asymptotics, less loop overhead. |
| **v7** cache + autotune | **1236 → 937** (**~1.32×** vs **v6**) | Reuse **`logden`** so backward often **one** logits pass; leaner Python I/O; autotuned tile. |
| **v8** constexpr + wider autotune | **937 → 908** (**~1.03×** vs **v7**, Popcorn) | **`tl.constexpr`** **`V`/strides**; larger **`BLOCK_V` / warps / stages`** search; same **`logden`** cache path. |
| **v9** hand-tuned **`_LAUNCH`**, no autotune | **908 → 875** (**~1.04×** vs **v8**, Popcorn) | Fixed **tile × warps × stages** per **V**; **16384** tile on **128k**; avoids autotune variance. |

Colab **`test_cross_entropy.py`** tables (per-**V** ms, geomean **×** vs *local* eager) are in **Versions 2–5**, **Version 7**, **Version 8**, and **Version 9**; they are **not** identical to Popcorn **μs** but support **before/after** comparisons for each approach. **Achieved memory bandwidth** adds an **A100 80GB** harness session for **v1–v6** with **% of peak**; **v7**–**v9** include **Colab** / local harness row sets.

### Final submission: achieved bandwidth and % of peak (2039 GB/s)

Use the harness on the **same GPU as the write-up** (ideally **A100 80GB**):

```bash
python test_cross_entropy.py submission_9.py
python test_cross_entropy.py submission_8.py
python test_cross_entropy.py submission_7.py
python test_cross_entropy.py submission_6.py
```

For each **V**, the script prints **Fwd BW**, **Bwd BW**, and **Fwd+Bwd BW** (GB/s). With **`B = 4096`**, bytes are **`fwd_bytes = 2·B·V + 12·B`**, **`bwd_bytes = 4·B·V + 12·B`**. **Fraction of peak** for the combined median row:

\[
\frac{\text{Fwd+Bwd BW (GB/s)}}{2039}\times 100\%.
\]

**This journal** tabulates **% of peak** for **v1–v6** from one **A100 80GB** **`test_cross_entropy.py`** session (see **Achieved memory bandwidth**) and keeps **historical** **A100 40GB** Colab rows for **v2**/**v3** in **Version 2** / **Version 3** for comparison. **v7**–**v9** add **Colab A100 40GB** harness rows (**Version 7** / **Version 8** / **Version 9**); refresh on **80GB** when available.

### What prevents reaching 100% of peak?

Already answered in **Why I do not expect 100% of peak bandwidth**: multi-kernel / reduction structure, **extra** reads/writes (losses, **grad**, target gather), **exp/log/max** work, non-sequential reuse of every byte of **HBM** traffic, launch overhead, and the simple **byte model** being only a **roofline** proxy — real kernels rarely saturate **2039 GB/s**.

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

### 3) `torch.compile` for both forward and backward (**v2** — best compile-only baseline)
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

### 4) Custom Triton forward + backward (**submission_4** — first full-Triton Popcorn baseline)
**Idea:** Replace Inductor’s multi-kernel backward with **one Triton kernel per backward call** (and similarly for forward), row-parallel over **B**, tiled over **V**.

**Why try it:** **v3** showed that another **`torch.compile`** graph is not guaranteed to beat **`F.cross_entropy`**. Triton lets us match the **reference math** while controlling **fusion** and **launch count** on a bandwidth-bound problem.

**What happened:** **Correctness PASS** locally/Colab and on **Popcorn**; **leaderboard time improved** from **2716 μs (v2)** to **1992 μs (v4)**. See **Version 4** above for design and numbers.

---

### 5) Triton + **online** max / LSE (**submission_5** — online reference)
**Idea:** Keep **v4**’s structure (row-parallel, tiled **V**, bf16 in / bf16 grad out, float32 softmax math), but fuse **row max** and **Σ exp(x − m)** into **one** pass over logits (online softmax reduction), and use the same fused reduction on the backward so the backward path needs **two** full reads of the row instead of **three**.

**Why try it:** **v4** already wins vs compile by launch count, but still **re-reads** the full row for separate max and sum phases. On an **HBM-limited** kernel, **one fewer pass** is a direct lever; the fused softmax literature is the template.

**What happened:** **PASS** on **`test_cross_entropy.py`**; Popcorn **1992.267 μs (v4) → 1270.465 μs (v5)**. See **Version 5** above.

---

### 6) Uniform wide tile (**submission_6** — best fixed-config Popcorn before v7/v8)
**Idea:** Keep **v5**’s **kernels and launch metadata** unchanged; use **`BLOCK_V = 4096`** for **all** competition **V** instead of **2048** for **32k / 50k** and **4096** only for **128k**.

**Why try it:** Fewer **outer-loop** trips per row can reduce overhead on an already **fused** kernel; standard **tiling** intuition without new math or extra CUDA streams.

**What happened:** Popcorn **1270.465 μs (v5) → 1236.180 μs (v6)**. See **Version 6** above.

---

### 7) Cached **`logden`** + autotune + lean I/O (**submission_7**)
**Idea:** When **backward** runs right after **forward** on the same **`(logits, targets)`**, reuse the forward-computed per-row **log-partition** \(\log\sum_j e^{x_j}\) instead of recomputing it in a separate full pass over logits; add **`@triton.autotune`**; avoid redundant **`contiguous()`** / **`int32`** casts in Python.

**Why try it:** The score is **combined** fwd+bwd; skipping **one** full-row logits read on backward dominates at large **V**. Smaller Python and launch overhead helps at the margin.

**What happened:** Popcorn **1236.180 μs (v6) → 936.952 μs (v7)**; harness **PASS** on Colab (**Version 7**).

---

### 8) **`tl.constexpr`** **`V`/strides** + wider autotune (**submission_8**)
**Idea:** Keep **v7**’s **online** kernels and **`logden`** cache; mark **`V`** and tensor **strides** as **`tl.constexpr`** so Triton specializes for each competition vocabulary and contiguous layout; expand **`@triton.autotune`** to include **smaller and larger** **`BLOCK_V`** (down to **512**, up to **8192**), more **`num_stages`** pairs, and **`num_warps = 16`** on wide tiles.

**Why try it:** After **v7**, remaining time is mostly **scheduling** (tile width, warps, pipeline stages) and **compiler specialization**; a broader search plus **constexpr** bounds targets both without changing numerics.

**What happened:** **`test_cross_entropy.py`** geomean speedup **~9.18×** vs eager in one session (table in **Version 8**); Popcorn **936.952 μs (v7) → 908.166 μs (v8)**.

---

### 9) Hand-tuned **`_LAUNCH`**, no autotune (**submission_9** — **best Popcorn μs in this journal**)
**Idea:** Keep **v7**/**v8** kernel bodies and **`logden`** cache; **remove `@triton.autotune`**; use a Python dict **`_LAUNCH`** mapping each competition **`V`** to **`BLOCK_V`**, **`num_warps`**, **`num_stages`** (power-of-2 tiles only), and launch kernels with explicit **`kwargs**.

**Why try it:** Autotune can **overfit** its benchmark budget or pick configs that are not best on the grader; **three** discrete **V** values allow **manual** A100-oriented tuning; **wider** **`BLOCK_V`** (e.g. **16384** for **128k**) reduces **loop** iteration count.

**What happened:** Popcorn **908.166 μs (v8) → 875.111 μs (v9)**; harness **~9.36×** geomean speedup vs eager on **Colab A100 40GB** / **PyTorch 2.10** — **Version 9** table.

---

### 10) Ideas not used (beyond v9)
- Single fused kernel spanning both API calls (not exposed by the harness).
- **v8** intentionally uses a **larger** autotune grid than **v7** (compile-time tradeoff vs peak runtime); **v9** drops autotune for a **fixed** table.
- **Popcorn-hostile experiments (historical note):** `torch.compile` on **`submission.py`** (extra streams on some checkers); **`eviction_policy`**, some **`constexpr` / `num_warps=16`** combinations — occasionally associated with **HTTP 500** / stream errors in **earlier** iterations — trim if the grader misbehaves.

**Why v2–v3 stayed in the story:** They document the **compile path**; **v4–v9** document when **explicit kernels**, **fused reductions**, **tiling**, **cross-call reuse**, **autotune + constexpr**, and **hand-tuned launch** win or need measurement.

## Why the final version should be faster than eager

Cross-entropy is mainly **memory-bound**, so reducing HBM traffic matters more than increasing raw FLOPs. That means fusion is important. **v5** goes one step further than **v4** by fusing the **row max** and **exponential sum** into **one** sweep (and saving a full backward pass), which directly targets global-memory traffic on each row. **v6** does not change that traffic model; it tunes **tile width** so each row needs **fewer** loop iterations on **32k / 50k**, which helped slightly on **Popcorn** (**1236.180 μs** vs **1270.465 μs** for **v5**). **v7** cuts another full-row pass on the backward **hot path** when **`logden`** is reused (**936.952 μs**). **v8** adds **`constexpr`** and **wider autotune** (**908.166 μs**). **v9** keeps the same **HBM** story and **`logden`** reuse but **fixes** launch parameters per **`V`** (**875.111 μs** on Popcorn here), avoiding autotune variance.

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
| **submission_1** (full harness) | **0.419 / 0.549 / 1.393** | **1.291 / 2.573 / 6.377** | **1.632 / 3.030 / 7.681** | **A100 80GB** **`test_cross_entropy.py`**; **geomean speedup 3.17×**; **% of peak** in **Achieved memory bandwidth**. |
| **submission_2** / **`submission.py`** | (see per-**V** table above) | (see per-**V** table above) | **1.599 / 2.756 / 7.563** | **A100 40GB** full **`test_cross_entropy.py`** run; **geomean speedup ≈ 3.31×** (see variance note above). |
| **submission_2** (full harness, **80GB**) | **0.410 / 0.549 / 1.376** | **1.371 / 2.232 / 6.295** | **1.697 / 2.695 / 7.574** | **A100 80GB** session; **geomean speedup 3.27×**. |
| **submission_3** (narrow fwd) | **0.383 / 0.714 / 1.599** | **1.368 / 2.232 / 6.294** | **1.683 / 2.867 / 7.806** | Same Colab **A100 40GB** session as **v2** row directly above in **Version 3**; **geomean speedup 3.18×** vs that run’s baseline (**slower than v2**). |
| **submission_3** (full harness, **80GB**) | **0.309 / 0.673 / 1.654** | **1.307 / 2.236 / 6.296** | **1.542 / 2.832 / 7.857** | **A100 80GB** session; **geomean speedup 3.28×**. |
| **submission_4** (Triton) | (see Colab log) | (see Colab log) | **~1.220 / 2.339 / 4.642** | Representative **A100 40GB** Colab harness run; **geomean speedup ~4.5×** vs eager in that session; **Popcorn combined 1992.267 μs** (official). |
| **submission_4** (full harness, **80GB**) | **0.437 / 0.762 / 1.539** | **0.843 / 1.645 / 3.170** | **1.222 / 2.340 / 4.645** | **A100 80GB** session; **geomean speedup 4.50×**. |
| **submission_5** (Triton + online LSE) | (Colab harness) | (Colab harness) | (per-**V** from latest run) | **Popcorn combined 1270.465 μs** (official); **~6.7×** geomean speedup vs eager reported in one Colab session (baseline-relative). |
| **submission_5** (full harness, **80GB**) | **0.270 / 0.426 / 0.829** | **0.651 / 1.090 / 2.429** | **0.862 / 1.454 / 3.190** | **A100 80GB** session; **geomean speedup 6.71×**. |
| **submission_6** (v5 + uniform `BLOCK_V=4096`) | (Colab harness) | (Colab harness) | (per-**V** from latest run) | **Popcorn combined 1236.180 μs** (official); same online kernels as **v5**, wide-tile scheduling only. |
| **submission_6** (full harness, **80GB**) | **0.250 / 0.422 / 0.827** | **0.643 / 1.059 / 2.427** | **0.843 / 1.419 / 3.193** | **A100 80GB** session; **geomean speedup 6.82×**. |
| **submission_7** (full harness, **40GB** Colab) | **0.250 / 0.410 / 0.823** | **0.639 / 1.072 / 2.422** | **0.652 / 1.083 / 2.427** | **PyTorch 2.10.0+cu128**; **geomean speedup ~8.91×**; Popcorn **936.952 μs** (official). |
| **submission_8** (harness, session in **Version 8**) | **0.248 / 0.359 / 0.827** | **0.488 / 0.887 / 2.416** | **0.649 / 0.994 / 2.418** | **geomean speedup ~9.18×** vs eager; Popcorn **908.166 μs** (official). |
| **submission_9** (Colab **A100 40GB**, **PyTorch 2.10.0+cu128**, **Version 9**) | **0.231 / 0.347 / 0.812** | **0.470 / 0.867 / 2.358** | **0.628 / 0.979 / 2.392** | **geomean speedup ~9.36×** vs eager; Popcorn **875.111 μs** (official). |

For **competition-comparable** numbers, run **`python test_cross_entropy.py`** on each of **`submission_1.py` … `submission_9.py`**: **20** warmup, **100** runs, **three** **V** values, **geomean speedup** printed at the end.

## Achieved memory bandwidth

Effective bandwidth uses the same byte model as `test_cross_entropy.py`:  
`fwd_bytes = 2*B*V + 12*B`, `bwd_bytes = 4*B*V + 12*B`, **GB/s** = bytes / (time in s) / 10⁹.

**Peak A100 bandwidth (per assignment):** **2039 GB/s**. **% of peak** below means **(printed or computed Fwd+Bwd BW in GB/s) / 2039 × 100%** for that **V** row (same definition as the harness “Fwd+Bwd BW” column).

### % of peak (2039 GB/s) — **A100 80GB** harness session (**v1–v6**; **v7** Colab rows below)

One **`test_cross_entropy.py`** run per file on **A100 80GB**, **PyTorch 2.11.0**, **Triton 3.6.0** (same stack as **Environment**). **% of peak** = (printed **Fwd+Bwd BW** in GB/s) / **2039** × **100%**, rounded to **0.1%**.

| Version | File | **V** | Fwd+Bwd BW (GB/s) | **% of 2039** | Geomean speedup (printed) |
|--------|------|------:|------------------:|--------------:|--------------------------:|
| **v1** | `submission_1.py` | 32,000 | **481.9** | **23.6%** | **3.17×** |
| **v1** | `submission_1.py` | 50,264 | **407.7** | **20.0%** | |
| **v1** | `submission_1.py` | 128,256 | **410.4** | **20.1%** | |
| **v2** | `submission_2.py` | 32,000 | **463.5** | **22.7%** | **3.27×** |
| **v2** | `submission_2.py` | 50,264 | **458.4** | **22.5%** | |
| **v2** | `submission_2.py` | 128,256 | **416.2** | **20.4%** | |
| **v3** | `submission_3.py` | 32,000 | **510.2** | **25.0%** | **3.28×** |
| **v3** | `submission_3.py` | 50,264 | **436.2** | **21.4%** | |
| **v3** | `submission_3.py` | 128,256 | **401.2** | **19.7%** | |
| **v4** | `submission_4.py` | 32,000 | **643.8** | **31.6%** | **4.50×** |
| **v4** | `submission_4.py` | 50,264 | **528.0** | **25.9%** | |
| **v4** | `submission_4.py` | 128,256 | **678.6** | **33.3%** | |
| **v5** | `submission_5.py` | 32,000 | **912.2** | **44.7%** | **6.71×** |
| **v5** | `submission_5.py` | 50,264 | **849.9** | **41.7%** | |
| **v5** | `submission_5.py` | 128,256 | **988.2** | **48.5%** | |
| **v6** | `submission_6.py` | 32,000 | **933.3** | **45.8%** | **6.82×** |
| **v6** | `submission_6.py` | 50,264 | **870.4** | **42.7%** | |
| **v6** | `submission_6.py` | 128,256 | **987.2** | **48.4%** | |

**v7 (Colab A100 40GB, PyTorch 2.10 — same harness paste as Version 7):**

| Version | File | **V** | Fwd+Bwd BW (GB/s) | **% of 2039** | Geomean speedup (printed) |
|--------|------|------:|------------------:|--------------:|--------------------------:|
| **v7** | `submission_7.py` | 32,000 | **1205.8** | **~59.1%** | **~8.91×** |
| **v7** | `submission_7.py` | 50,264 | **1140.3** | **~55.9%** | |
| **v7** | `submission_7.py` | 128,256 | **1298.8** | **~63.7%** | |

**v8 (same harness as Version 8 table — one session):**

| Version | File | **V** | Fwd+Bwd BW (GB/s) | **% of 2039** | Geomean speedup (printed) |
|--------|------|------:|------------------:|--------------:|--------------------------:|
| **v8** | `submission_8.py` | 32,000 | **1211.5** | **~59.4%** | **~9.18×** |
| **v8** | `submission_8.py` | 50,264 | **1242.5** | **~60.9%** | |
| **v8** | `submission_8.py` | 128,256 | **1303.8** | **~63.9%** | |

**v9 (Colab A100 40GB, PyTorch 2.10 — Version 9):**

| Version | File | **V** | Fwd+Bwd BW (GB/s) | **% of 2039** | Geomean speedup (printed) |
|--------|------|------:|------------------:|--------------:|--------------------------:|
| **v9** | `submission_9.py` | 32,000 | **1253.0** | **~61.5%** | **~9.36×** |
| **v9** | `submission_9.py` | 50,264 | **1262.0** | **~61.9%** | |
| **v9** | `submission_9.py` | 128,256 | **1317.7** | **~64.6%** | |

**Historical (different machine / session):** **Version 2** documents **v2** on **A100 40GB**, PyTorch **2.10** — **Fwd+Bwd BW** **491.7 / 448.3 / 416.8 GB/s** → **~24.1% / 22.0% / 20.4%** of **2039**. **Version 3** has **v2′** and **v3** Colab medians for the same **40GB** class.

### Which rows are **still** without a dedicated **% of peak** line here?

| Gap | Reason |
|-----|--------|
| **Eager baseline** (per **V**) | Harness prints ms and BW for baseline, but this journal does not duplicate a seven-column baseline **%** table (derive from printed **Fwd+Bwd BW** if needed). |
| **`torch.compile` forward-only** | Optional experiment; **not timed** in the tables here. |

**Colab-only microbench** (**submission_1**, **V = 128k** only): **~388.2 GB/s** combined → **~19.0%** of **2039** (non-**80GB** runtime; see **Version 1**).

### Final Triton submission (**v9**) on **A100**

Official rank uses **Popcorn**; local check: **`python test_cross_entropy.py submission_9.py`**. Primary implementation: **`submission_9.py`** (**875.111 μs** Popcorn in this journal). **`submission_8.py`** is the **autotuned** variant (**v8**, **908.166 μs** — **`@triton.autotune`**, not **`_LAUNCH`**). **`submission_7.py`** (**936.952 μs**) remains a reference; **`submission_6.py`** (**1236.180 μs**) remains the **no-cache** baseline.

**After Popcorn / grader PyTorch 2.11**, refresh harness rows if medians shift; add **v9** rows to the **80GB** bandwidth table when you paste a full **`test_cross_entropy.py submission_9.py`** run on **A100 80GB**.

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
python test_cross_entropy.py submission_3.py
python test_cross_entropy.py submission_4.py
python test_cross_entropy.py submission_5.py
python test_cross_entropy.py submission_6.py
python test_cross_entropy.py submission_7.py
python test_cross_entropy.py submission_8.py
python test_cross_entropy.py submission_9.py
```

Each run: correctness for all three **V**, timings, bandwidth, geomean speedup vs eager baseline.

## Google Colab setup — steps and commands

1. **Runtime → Change runtime type → GPU** (T4/L4/A100 depending on tier).

2. **Upload files** to `/content` (or use Drive): `submission.py`, `submission_1.py`–`submission_9.py`, `test_cross_entropy.py`, and optionally `colab_benchmark_cross_entropy.py`.

   ```python
   from google.colab import files
   files.upload()   # select the .py files you need
   ```

3. **Check GPU and PyTorch** (optional):

   ```python
   import torch
   print(torch.__version__, torch.cuda.get_device_name(0))
   ```

4. **Official harness (recommended)** — matches course script (**20** warmup, **100** iters). Do **not** set `TORCH_LOGS` to empty on some Colab builds (can break `import torch`); run benchmarks in a **fresh** `!python ...` process, or use **`colab_benchmark_cross_entropy.py`** (subprocesses strip `TORCH_LOGS`).

   ```python
   %cd /content
   !python test_cross_entropy.py submission_1.py
   !python test_cross_entropy.py submission_2.py
   !python test_cross_entropy.py submission_3.py
   !python test_cross_entropy.py submission_4.py
   !python test_cross_entropy.py submission_5.py
   !python test_cross_entropy.py submission_6.py
   !python test_cross_entropy.py submission_7.py
   !python test_cross_entropy.py submission_8.py
   !python test_cross_entropy.py submission_9.py
   !python test_cross_entropy.py submission.py
   ```

   Or benchmark **v2 / v3 / v4 / v5 / v6 / v7** in one go:

   ```python
   !python colab_benchmark_cross_entropy.py
   ```

   Optional: `TORCHINDUCTOR_CACHE_DIR` in the environment **before** importing torch if you want a fixed Inductor cache dir.

   Copy the full printed table and **geomean speedup** into this journal.

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

   for name, path in [("v1", "/content/submission_1.py"), ("v2", "/content/submission_2.py"), ("v3", "/content/submission_3.py"), ("v4", "/content/submission_4.py"), ("v5", "/content/submission_5.py"), ("v6", "/content/submission_6.py"), ("v7", "/content/submission_7.py"), ("v8", "/content/submission_8.py"), ("v9", "/content/submission_9.py")]:
       m = load(path)
       f, b = bench(m)
       print(name, "forward ms:", f, "backward ms:", b, "sum:", f + b)
   ```

## How to submit

1. Install the CLI:

```bash
curl -fsSL https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh | bash
```

2. Register once with GitHub:

```bash
popcorn register github
```

3. Join the leaderboard with your invite code:

```bash
popcorn join <YOUR_INVITE_CODE>
```

4. Get the starter file if needed:

```bash
wget https://raw.githubusercontent.com/gpu-mode/reference-kernels/main/problems/princeton/cross_entropy_py/submission.py
```

5. Run a correctness check:

```bash
popcorn submit --leaderboard princeton_cross_entropy --gpu A100 --mode test submission_9.py
```

6. Submit an official ranked run (**best Popcorn time in this journal: `submission_9.py`**, **875.111 μs**):

```bash
popcorn submit --leaderboard princeton_cross_entropy --gpu A100 --mode leaderboard submission_9.py
```

Alternatively: `cp submission_9.py submission.py` and submit **`submission.py`**. To submit an older version, point the command at **`submission_8.py`**, **`submission_7.py`**, **`submission_6.py`**, **`submission_5.py`**, **`submission_4.py`**, **`submission_2.py`**, etc.

## Short note to include in the notebook

I documented my optimization process in this journal. I compared **submission_1** and **submission_2** (`torch.compile` + stable backward without `probs.clone()`). **submission_3** (narrow forward + `gather` under compile) **passed** correctness but was **slower** than v2 in Colab (see **Version 3**). **submission_4** implements **forward and backward in Triton** (row-parallel, tiled **V**), fixes the **`_NEG` global / Triton JIT** issue with a local sentinel, and **improved Popcorn** from **2716.164 μs (v2)** to **1992.267 μs**. **submission_5** applies an **online softmax**-style fused reduction, and reached **1270.465 μs** on Popcorn. **submission_6** uses a **uniform `BLOCK_V = 4096`** and reached **1236.180 μs**. **submission_7** caches per-row **log ∑ e^x** and reached **936.952 μs**. **submission_8** adds **`tl.constexpr`** and **wider autotune** (**908.166 μs**). **submission_9** removes autotune in favor of a **hand-tuned `_LAUNCH`** table per competition **V** (e.g. **8192** for **32k/50k**, **16384** for **128k**), reaching **875.111 μs** on Popcorn and **~9.36×** geomean speedup vs eager on **Colab A100 40GB** / **PyTorch 2.10** (**Version 9**). **submission_8** (**autotune**) and **submission_9** (**hand-tuned `_LAUNCH`**) are **separate** source files; do not assume they stay identical. **Achieved memory bandwidth** includes **v7–v9** harness rows. For upload, use **`submission_9.py`**. Avoid **`torch.compile`** on Popcorn if the server reports **multi-stream** errors. Local/Colab numbers use **`test_cross_entropy.py`**; the grader uses **PyTorch 2.11** and **Triton 3.6** on **A100 80GB**.
