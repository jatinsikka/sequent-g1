#!/usr/bin/env python
"""
SEQUENT — RUN ANY BODY SCRIPT ON A MACHINE WITHOUT AN NVIDIA GPU.

`amo_jit.pt` was traced on a CUDA box and the trace baked a literal
`device=cuda:0` into a torch.zeros call inside the graph. `map_location` moves
parameters but not constants, so every script in this repo that loads the AMO
controller dies on a Mac or a CPU-only Linux box with:

    NotImplementedError: Could not run 'aten::empty_strided' with arguments
    from the 'CUDA' backend

This wrapper installs a shim over `torch.jit.load` that forces CPU placement and
rewrites those device constants, then runs the target script unmodified, exactly
as if you had invoked it directly.

    python demo/run_on_cpu.py pick_table.py
    python demo/run_on_cpu.py play_amo.py
    python demo/run_on_cpu.py walk_and_grasp.py --some-flag

Nothing in the repo is edited. If you are on a real CUDA machine this wrapper is
a no-op — the constants are already correct there — so it is safe either way.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def install_cpu_shim() -> None:
    """Force every torch.jit.load in this process onto CPU, constants included."""
    import torch
    from demo.body_walk import patch_jit_to_cpu

    if torch.cuda.is_available():
        return

    original = torch.jit.load

    def patched(*args, **kwargs):
        kwargs["map_location"] = "cpu"
        module = original(*args, **kwargs)
        try:
            patch_jit_to_cpu(module)
        except Exception:
            pass                      # not every traced module has device constants
        return module

    torch.jit.load = patched

    # Scripts written for a CUDA box also call .to("cuda") / device="cuda"
    # directly. Make those resolve to CPU rather than raising.
    original_device = torch.device

    class _Device:
        def __new__(cls, *args, **kwargs):
            if args and isinstance(args[0], str) and args[0].startswith("cuda"):
                args = ("cpu",) + args[1:]
            return original_device(*args, **kwargs)

    torch.device = _Device


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    script = sys.argv[1]
    sys.argv = sys.argv[1:]           # the target script sees its own argv

    os.chdir(REPO)                    # repo scripts resolve g1.xml / *.pt / *.mjb from here
    sys.path.insert(0, str(REPO))

    install_cpu_shim()
    print(f"[cpu shim installed]  running {script}\n", flush=True)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
