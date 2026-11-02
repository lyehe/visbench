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
    eval_point_homography_pairs,
    eval_pose_pairs,
    reset_caches,
)
from ..core.sampling import sample_pairs
from ..datasets import get as get_dataset_spec
from ..methods import get as get_method_spec

# Bumped whenever a change makes earlier result files non-comparable (pair
# sampling, metric fixes). `batch --skip-existing` re-runs older records.
RESULT_SCHEMA_VERSION = 2


def iter_dataset_pairs(spec, data_root: Path, max_pairs: int | None, seed: int | None):
    """All of the dataset's pairs, then a seeded scene-balanced subset if capped."""
    pairs = spec.iter_pairs(data_root, max_pairs=None, **(spec.kwargs or {}))
    return sample_pairs(pairs, max_pairs, seed=seed)


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


def _qualitative_sweep(matcher, pairs, resize_long=None, progress: bool = True):
    """Cross-matcher speed/match-count sweep used as the fallback when a
    dataset's task isn't in `_TASK_DISPATCH`. Shares the same image cache,
    aspect-preserving long-side resize, progress and per-pair-failure
    warning conventions as the other eval_* paths (matters: vismatch's
    load_image treats a bare int as a SQUARE resize, which would distort
    non-square inputs).
    """
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm
    from ..core.harness import _prepare_pair_images, _warn_pair_failure, _warn_summary
    times, matches, inliers = [], [], []
    n_failures = 0
    for pair in tqdm(pairs, desc="pairs", disable=not progress):
        try:
            img0, img1, *_ = _prepare_pair_images(
                matcher, pair["im_A_path"], pair["im_B_path"],
                resize_long=resize_long,
            )
            t = time.time()
            r = matcher(img0, img1)
            times.append(time.time() - t)
            matches.append(len(r.get("matched_kpts0", [])))
            inliers.append(int(r.get("num_inliers", 0)))
        except Exception as e:  # noqa: BLE001
            n_failures += 1
            _warn_pair_failure("_qualitative_sweep",
                               Path(pair.get("im_A_path", "?")),
                               Path(pair.get("im_B_path", "?")),
                               e, n_failures)
    _warn_summary("_qualitative_sweep", len(times) + n_failures, n_failures,
                  "excluded from the qualitative aggregate.")
    return {
        "mean_time_s": float(np.mean(times)) if times else 0.0,
        "mean_matches": float(np.mean(matches)) if matches else 0.0,
        "mean_inliers": float(np.mean(inliers)) if inliers else 0.0,
        "num_pairs": len(times),
        "num_pair_failures": n_failures,
        "resize_long": resize_long,
    }


# Each entry is the bare eval function for the task; `_evaluate` filters the
# caller's kwargs against each function's signature so we don't have to repeat
# the same lambda wrapper just to drop `seed`/`ransac_runs` for tasks that
# don't use them (only `pose` consumes `seed` and `ransac_runs` right now).
_TASK_DISPATCH = {
    "pose": eval_pose_pairs,
    "homography": eval_homography_pairs,
    "homography_points": eval_point_homography_pairs,
    "correspondence": eval_correspondence_pairs,
    "fundamental": eval_fundamental_pairs,
    "descriptor": eval_descriptor_pairs,
    "qualitative": _qualitative_sweep,
}


def _accepted_kwargs(fn) -> set[str]:
    import inspect
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        # Function takes **kwargs — accept everything.
        return None  # type: ignore[return-value]
    return set(sig.parameters)


def known_tasks() -> tuple[str, ...]:
    """Tasks the dispatcher recognises. Sourced from `visbench._tasks.TASKS`
    so the orchestrator and the dataset registry can't drift; a drift check
    runs once at module import (see below)."""
    return tuple(_TASK_DISPATCH)


def _check_dispatch_matches_tasks() -> None:
    """One-time sanity check that _TASK_DISPATCH and visbench._tasks.TASKS
    agree. Runs at module import; raises immediately if a task was added to
    one without the other so the bug surfaces at startup, not at eval time.
    """
    from .._tasks import TASKS
    dispatch_keys = frozenset(_TASK_DISPATCH)
    if dispatch_keys != TASKS:
        only_dispatch = dispatch_keys - TASKS
        only_tasks = TASKS - dispatch_keys
        raise RuntimeError(
            f"_TASK_DISPATCH and visbench._tasks.TASKS have drifted. "
            f"Only in _TASK_DISPATCH: {sorted(only_dispatch)}. "
            f"Only in TASKS: {sorted(only_tasks)}. "
            f"Add the missing task to the other location."
        )


_check_dispatch_matches_tasks()


def _evaluate(spec_task: str, matcher, pairs, *, resize_long, ransac_runs,
              progress, seed: int | None = 0):
    fn = _TASK_DISPATCH.get(spec_task)
    if fn is None:
        # Unknown / future task — degrade gracefully to a qualitative sweep so
        # any new dataset still produces SOMETHING comparable across matchers.
        import warnings
        warnings.warn(f"Unknown task '{spec_task}' — falling back to qualitative sweep "
                      f"(speed + match-count only). Add a dispatch entry to "
                      f"orchestrators/run._TASK_DISPATCH to get full metrics.",
                      stacklevel=2)
        fn = _qualitative_sweep
    candidate_kwargs = {
        "resize_long": resize_long,
        "ransac_runs": ransac_runs,
        "progress": progress,
        "seed": seed,
    }
    accepted = _accepted_kwargs(fn)
    kwargs = (candidate_kwargs if accepted is None
              else {k: v for k, v in candidate_kwargs.items() if k in accepted})
    return fn(matcher, pairs, **kwargs)


def run_one(dataset: str, method: str, *, data_root_override: Path | None = None,
            data_root_default: Path, max_pairs: int | None = None,
            resize: int | None = None, ransac_runs: int = 5,
            device: str = "cpu", out: Path | None = None,
            progress: bool = True, seed: int | None = 0) -> dict:
    spec = get_dataset_spec(dataset)
    data_root = resolve_data_root(spec, data_root_override, data_root_default)
    if not data_root.exists():
        raise SystemExit(
            f"Dataset '{dataset}' not found at {data_root}.\n"
            f"  -> Run `visbench download {dataset}` (if supported), or pass --data-root.\n"
            f"  -> Source: {spec.source_url or 'see the dataset module docstring'}"
        )

    matcher = build_matcher(method, device=device)
    # Forward spec.kwargs (e.g. hardmatch vs hardmatch_rot90 differ only by
    # the kwargs they hand iter_pairs).
    # Production used to drop these, silently making `_wide`/`_hard` variants
    # identical to their base.
    pairs = iter_dataset_pairs(spec, data_root, max_pairs, seed)
    resize_long = resize if resize is not None else spec.default_resize

    t0 = time.time()
    metrics = _evaluate(spec.task, matcher, pairs,
                        resize_long=resize_long, ransac_runs=ransac_runs,
                        progress=progress, seed=seed)
    metrics["total_time_s"] = time.time() - t0

    record = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "dataset": dataset, "method": method, "device": device,
        "data_root": str(data_root), "max_pairs": max_pairs,
        "seed": seed,
        "metrics": metrics,
    }
    if out is not None:
        atomic_write_text(Path(out), json.dumps(record, indent=2))
        print(f"wrote {out}", file=sys.stderr)
    reset_caches()
    return record
