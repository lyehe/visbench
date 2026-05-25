"""Panel-based standard benchmark: cross every (dataset x method) cell.

Two execution modes:
  1. **In-process sequential** (`run_standard`) — calls `run_batch` per method
     in a loop. Simplest; matcher state is reused across the panel.
  2. **Parallel via subprocess pool** (`run_standard_parallel`) — spawns one
     `python -m visbench batch` per method, distributes them across the given
     `devices` (CPU and/or CUDA), and enforces a per-method timeout. Use this
     for 50+ method panels where overlapping GPUs and bounded wall time matter.
"""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .batch import run_batch


def _read_panel(path: Path) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8")
    out = []
    for ln in raw.splitlines():
        s = ln.strip().lstrip("﻿")
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _per_dataset_max_pairs(datasets: list[str], pose: int | None,
                           homography: int | None) -> dict:
    from ..datasets import get as get_dataset_spec
    out: dict[str, int] = {}
    for ds in datasets:
        try:
            spec = get_dataset_spec(ds)
        except KeyError:
            continue
        if spec.task == "pose" and pose is not None:
            out[ds] = pose
        elif spec.task in ("homography", "correspondence", "fundamental") and homography is not None:
            out[ds] = homography
    return out


def run_standard(panel_path: Path, methods_path: Path, out_dir: Path,
                 data_root_default: Path, *,
                 max_pairs_pose: int | None = None,
                 max_pairs_homography: int | None = None,
                 ransac_runs: int = 5,
                 device: str = "cpu",
                 skip_existing: bool = True,
                 seed: int | None = 0) -> None:
    """Sequential execution: one method at a time, in-process."""
    datasets = _read_panel(Path(panel_path))
    methods = _read_panel(Path(methods_path))
    max_pairs = _per_dataset_max_pairs(datasets, max_pairs_pose, max_pairs_homography)

    for method in methods:
        run_batch(
            method=method,
            datasets=datasets,
            out_dir=out_dir,
            data_root_default=data_root_default,
            max_pairs=max_pairs,
            ransac_runs=ransac_runs,
            device=device,
            skip_existing=skip_existing,
            seed=seed,
        )


def run_standard_parallel(panel_path: Path, methods_path: Path, out_dir: Path,
                          data_root_default: Path, *,
                          devices: list[str] | tuple[str, ...] = ("cpu",),
                          max_pairs_pose: int | None = None,
                          max_pairs_homography: int | None = None,
                          ransac_runs: int = 5,
                          per_method_timeout: float | None = None,
                          skip_existing: bool = True,
                          seed: int | None = 0,
                          python_exe: str | None = None) -> None:
    """Parallel execution: one subprocess per method, distributed round-robin
    across `devices`. Each subprocess is `python -m visbench batch ...`."""
    datasets = _read_panel(Path(panel_path))
    methods = _read_panel(Path(methods_path))
    if not datasets or not methods:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    py = python_exe or sys.executable
    devices = list(devices) or ["cpu"]
    workers = len(devices)

    # Build per-method commands; round-robin device assignment.
    commands: list[tuple[str, str, list[str]]] = []  # (method, device, argv)
    for i, method in enumerate(methods):
        device = devices[i % workers]
        argv = [
            py, "-m", "visbench", "batch",
            "--method", method,
            "--datasets", ",".join(datasets),
            "--out-dir", str(out_dir),
            "--data-root-default", str(data_root_default),
            "--ransac-runs", str(ransac_runs),
            "--device", device,
        ]
        if max_pairs_pose is not None or max_pairs_homography is not None:
            # `visbench batch` accepts a single --max-pairs; use the smaller of
            # the two so neither task overruns its budget.
            mp = min(x for x in (max_pairs_pose, max_pairs_homography) if x is not None)
            argv += ["--max-pairs", str(mp)]
        if not skip_existing:
            argv += ["--no-skip"]
        if seed is not None:
            argv += ["--seed", str(seed)]
        commands.append((method, device, argv))

    print(f"[standard-parallel] {len(methods)} methods x {len(datasets)} datasets "
          f"on {workers} worker(s): {devices}", file=sys.stderr)

    def _run_one(m_d_argv):
        method, device, argv = m_d_argv
        # Create a new process group / job so we can take down any worker
        # processes the child spawns (DataLoader pools, NCCL helpers, etc.) on
        # timeout — `subprocess.run(timeout=...)` only kills the immediate
        # child, leaving orphans holding VRAM.
        popen_kw: dict = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if os.name == "nt":
            popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kw["start_new_session"] = True
        proc = subprocess.Popen(argv, **popen_kw)
        try:
            stdout, stderr = proc.communicate(timeout=per_method_timeout)
            return method, device, proc.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                proc.kill()
            else:
                import signal
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            tail = (stderr or "").strip().splitlines()[-3:]
            return method, device, -1, stdout, \
                f"TIMEOUT after {per_method_timeout}s\n" + "\n".join(tail)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_one, cmd): cmd[0] for cmd in commands}
        for fut in as_completed(futures):
            method, device, rc, stdout, stderr = fut.result()
            tag = "ok" if rc == 0 else f"fail rc={rc}"
            print(f"[{tag}] {method:<40} device={device}", file=sys.stderr)
            if rc != 0 and stderr:
                # Surface the last few lines of stderr for diagnosis.
                tail = "\n".join(stderr.strip().splitlines()[-6:])
                print(f"  stderr tail:\n    " + tail.replace("\n", "\n    "),
                      file=sys.stderr)
