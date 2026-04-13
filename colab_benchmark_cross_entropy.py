#!/usr/bin/env python3
"""
Google Colab — compare cross-entropy submission timings (v2 vs v3, etc.)

This runner **does not import torch** in the parent process. On Colab, inherited
``TORCH_LOGS`` (often set to empty or other values in the notebook) can break
``torch._logging`` and crash on ``import torch``. All checks and benchmarks run
in **subprocesses** with ``TORCH_LOGS`` removed from their environment.

Setup
-----
1. Runtime → Change runtime type → **GPU** (A100 if available).
2. Upload to ``/content`` (or change ``WORKDIR`` below):
   - ``test_cross_entropy.py``
   - ``submission_2.py``
   - ``submission_3.py``

Run
---
::

    %cd /content
    !python colab_benchmark_cross_entropy.py

Minimal (no runner): upload the three files, then::

    %cd /content
    !python test_cross_entropy.py submission_2.py
    !python test_cross_entropy.py submission_3.py

If anything still fails, **Runtime → Restart session** (clears bad ``%env`` state),
then avoid ``%env TORCH_LOGS=``.
"""

from __future__ import annotations

import os
import subprocess
import sys

# --- config ---
WORKDIR = "/content"
SUBMISSIONS = [
    "submission_2.py",
    "submission_3.py",
]
TEST_SCRIPT = "test_cross_entropy.py"

# Subprocess env: drop TORCH_LOGS entirely (Colab/kernel values often break torch._logging).
_PROBE = r"""
import torch
print("torch:", torch.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA not available")
"""


def _child_env() -> dict[str, str]:
    e = dict(os.environ)
    e.pop("TORCH_LOGS", None)
    return e


def main() -> None:
    os.chdir(WORKDIR)
    env = _child_env()

    probe = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=WORKDIR,
        env=env,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        print("ERROR: could not import torch in a clean subprocess.")
        print(probe.stderr or probe.stdout)
        sys.exit(1)
    print(probe.stdout.strip())

    for name in [TEST_SCRIPT, *SUBMISSIONS]:
        path = os.path.join(WORKDIR, name)
        if not os.path.isfile(path):
            print(f"ERROR: missing file: {path}")
            print("Upload the course test script and submission .py files to", WORKDIR)
            sys.exit(1)

    for sub in SUBMISSIONS:
        print("\n" + "=" * 80)
        print(f" BENCHMARK: {sub}")
        print("=" * 80 + "\n")
        rc = subprocess.call(
            [sys.executable, TEST_SCRIPT, sub],
            cwd=WORKDIR,
            env=env,
        )
        if rc != 0:
            print(f"\n(test_cross_entropy.py exited with code {rc} for {sub})")


if __name__ == "__main__":
    main()
