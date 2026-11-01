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
from .run import (
    RESULT_SCHEMA_VERSION,
    _evaluate,
    atomic_write_text,
    build_matcher,
    iter_dataset_pairs,
    resolve_data_root,
)

POSE_TASKS = ("pose",)
HOMOGRAPHY_TASKS = ("homography", "correspondence", "fundamental")


def per_dataset_max_pairs(datasets: list[str], pose: int | None, homography: int | None) -> dict:
    """Map each dataset to the pair cap for its task family."""
    out: dict[str, int] = {}
    for ds in datasets:
        try:
            spec = get_dataset_spec(ds)
        except KeyError:
            continue
        if spec.task in POSE_TASKS and pose is not None:
            out[ds] = pose
        elif spec.task in HOMOGRAPHY_TASKS and homography is not None:
            out[ds] = homography
    return out


def existing_result_is_current(path: Path, max_pairs: int | None, seed: int | None) -> bool:
    """True only for a complete record produced with the same cap, seed and schema."""
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(rec, dict) or rec.get("schema_version") != RESULT_SCHEMA_VERSION:
        return False
    if rec.get("max_pairs") != max_pairs or rec.get("seed") != seed:
        return False
    metrics = rec.get("metrics") or {}
    counts = [v for k, v in metrics.items() if k == "num_pairs" or k.endswith("_num_pairs")]
    return bool(counts) and max(counts) > 0


def run_batch(method: str, datasets: list[str], out_dir: Path,
              data_root_default: Path,
              max_pairs: dict | None = None, ransac_runs: int = 5,
              device: str = "cpu", skip_existing: bool = True,
              seed: int | None = 0) -> list[dict]:
    matcher = build_matcher(method, device=device)
    safe = method.replace(":", "-").replace("/", "-")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for ds in datasets:
        out_path = out_dir / f"{ds}__{safe}.json"
        mp = (max_pairs or {}).get(ds)
        if skip_existing and out_path.exists() and existing_result_is_current(out_path, mp, seed):
            continue
        spec = get_dataset_spec(ds)
        data_root = resolve_data_root(spec, None, data_root_default)
        if not data_root.exists():
            print(f"[skip] {ds}: data root missing ({data_root})", file=sys.stderr)
            continue
        pairs = iter_dataset_pairs(spec, data_root, mp, seed)
        t0 = time.time()
        try:
            metrics = _evaluate(spec.task, matcher, pairs,
                                resize_long=spec.default_resize,
                                ransac_runs=ransac_runs, progress=False,
                                seed=seed)
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {ds} on {method}: {type(e).__name__}: {e}", file=sys.stderr)
            reset_caches()
            continue
        metrics["total_time_s"] = time.time() - t0
        rec = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "dataset": ds, "method": method, "device": device,
            "data_root": str(data_root), "max_pairs": mp,
            "seed": seed,
            "metrics": metrics,
        }
        atomic_write_text(out_path, json.dumps(rec, indent=2))
        results.append(rec)
        print(f"[ok] {ds:<22} {method:<40} {metrics['total_time_s']:>5.0f}s", file=sys.stderr)
        reset_caches()
    return results
