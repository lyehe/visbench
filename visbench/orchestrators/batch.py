"""Run one method across multiple datasets in a single process.

Imports + matcher build happen ONCE — significant wall-time saving when running
a panel of N datasets per method (compared to N subprocesses).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ..core.harness import reset_caches
from ..datasets import get as get_dataset_spec
from .run import _evaluate, atomic_write_text, build_matcher, resolve_data_root


def run_batch(method: str, datasets: list[str], out_dir: Path,
              data_root_default: Path,
              max_pairs: dict | None = None, ransac_runs: int = 5,
              device: str = "cpu", skip_existing: bool = True) -> list[dict]:
    matcher = build_matcher(method, device=device)
    safe = method.replace(":", "-").replace("/", "-")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for ds in datasets:
        out_path = out_dir / f"{ds}__{safe}.json"
        if skip_existing and out_path.exists():
            continue
        spec = get_dataset_spec(ds)
        data_root = resolve_data_root(spec, None, data_root_default)
        if not data_root.exists():
            print(f"[skip] {ds}: data root missing ({data_root})", file=sys.stderr)
            continue
        mp = (max_pairs or {}).get(ds)
        pairs = spec.iter_pairs(data_root, max_pairs=mp)
        t0 = time.time()
        try:
            metrics = _evaluate(spec.task, matcher, pairs,
                                resize_long=spec.default_resize,
                                ransac_runs=ransac_runs, progress=False)
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {ds} on {method}: {type(e).__name__}: {e}", file=sys.stderr)
            reset_caches()
            continue
        metrics["total_time_s"] = time.time() - t0
        rec = {
            "dataset": ds, "method": method, "device": device,
            "data_root": str(data_root), "max_pairs": mp,
            "metrics": metrics,
        }
        atomic_write_text(out_path, json.dumps(rec, indent=2))
        results.append(rec)
        print(f"[ok] {ds:<22} {method:<40} {metrics['total_time_s']:>5.0f}s", file=sys.stderr)
        reset_caches()
    return results
