"""Single-cell evaluation: one (dataset, method) -> one JSON record.

Library entry: `run_one(...)`. CLI dispatch lives in `visbench.__main__`.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

from ..core.harness import (
    eval_correspondence_pairs,
    eval_descriptor_pairs,
    eval_fundamental_pairs,
    eval_homography_pairs,
    eval_pose_pairs,
    reset_caches,
)
from ..datasets import get as get_dataset_spec
from ..methods import get as get_method_spec


def atomic_write_text(path: Path, text: str) -> None:
    """Atomic write so killed processes leave either old or new file, never half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


def resolve_data_root(spec, data_root: Path | None, default_root: Path) -> Path:
    if data_root is not None:
        return Path(data_root)
    return default_root / spec.default_root


def build_matcher(method: str, device: str):
    return get_method_spec(method).builder(device)


def _qualitative_sweep(matcher, pairs, resize_long=None):
    import numpy as np
    from tqdm import tqdm
    times, matches, inliers = [], [], []
    for pair in tqdm(pairs, desc="pairs"):
        img0 = matcher.load_image(pair["im_A_path"], resize=resize_long) if resize_long \
               else matcher.load_image(pair["im_A_path"])
        img1 = matcher.load_image(pair["im_B_path"], resize=resize_long) if resize_long \
               else matcher.load_image(pair["im_B_path"])
        t = time.time()
        r = matcher(img0, img1)
        times.append(time.time() - t)
        matches.append(len(r.get("matched_kpts0", [])))
        inliers.append(int(r.get("num_inliers", 0)))
    return {
        "mean_time_s": float(np.mean(times)) if times else 0.0,
        "mean_matches": float(np.mean(matches)) if matches else 0.0,
        "mean_inliers": float(np.mean(inliers)) if inliers else 0.0,
        "num_pairs": len(times),
        "resize_long": resize_long,
    }


def _evaluate(spec_task: str, matcher, pairs, *, resize_long, ransac_runs, progress):
    if spec_task == "pose":
        return eval_pose_pairs(matcher, pairs, resize_long=resize_long,
                               ransac_runs=ransac_runs, progress=progress)
    if spec_task == "homography":
        return eval_homography_pairs(matcher, pairs, resize_long=resize_long, progress=progress)
    if spec_task == "correspondence":
        return eval_correspondence_pairs(matcher, pairs, resize_long=resize_long, progress=progress)
    if spec_task == "fundamental":
        return eval_fundamental_pairs(matcher, pairs, resize_long=resize_long, progress=progress)
    if spec_task == "descriptor":
        return eval_descriptor_pairs(matcher, pairs, resize_long=resize_long, progress=progress)
    if spec_task == "qualitative":
        return _qualitative_sweep(matcher, pairs, resize_long=resize_long)
    # Unknown / future task — degrade gracefully to a qualitative sweep so
    # any new dataset still produces SOMETHING comparable across matchers.
    import warnings
    warnings.warn(f"Unknown task '{spec_task}' — falling back to qualitative sweep "
                  f"(speed + match-count only). Define the task in core/harness.py "
                  f"to get full metrics.", stacklevel=2)
    return _qualitative_sweep(matcher, pairs, resize_long=resize_long)


def run_one(dataset: str, method: str, *, data_root_override: Path | None = None,
            data_root_default: Path, max_pairs: int | None = None,
            resize: int | None = None, ransac_runs: int = 5,
            device: str = "cpu", out: Path | None = None,
            progress: bool = True) -> dict:
    spec = get_dataset_spec(dataset)
    data_root = resolve_data_root(spec, data_root_override, data_root_default)
    if not data_root.exists():
        raise SystemExit(
            f"Dataset '{dataset}' not found at {data_root}.\n"
            f"  -> Run `visbench download {dataset}` (if supported), or pass --data-root.\n"
            f"  -> See visbench/datasets/{dataset}.py for source URL."
        )

    matcher = build_matcher(method, device=device)
    pairs = spec.iter_pairs(data_root, max_pairs=max_pairs)
    resize_long = resize if resize is not None else spec.default_resize

    t0 = time.time()
    metrics = _evaluate(spec.task, matcher, pairs,
                        resize_long=resize_long, ransac_runs=ransac_runs, progress=progress)
    metrics["total_time_s"] = time.time() - t0

    record = {
        "dataset": dataset, "method": method, "device": device,
        "data_root": str(data_root), "max_pairs": max_pairs,
        "metrics": metrics,
    }
    if out is not None:
        atomic_write_text(Path(out), json.dumps(record, indent=2))
        print(f"wrote {out}", file=sys.stderr)
    reset_caches()
    return record
