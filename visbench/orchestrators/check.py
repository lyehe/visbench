"""`visbench check <dataset>`: CPU sanity check of a loader's ground truth on real data.

- pose: SIFT matches on a scene-balanced sample of pairs must satisfy the epipolar
  constraint of the loader's GT pose + intrinsics (median Sampson error under 2 px)
  and do clearly better than under a 5-8 deg perturbed pose (see core/gt_check.py).
- correspondence / homography_points: images must load and every annotated point
  must lie inside its image.
- homography: images must load and the GT must be finite.

Returns True when the dataset passes. Nothing is written except rotated-image caches
that the loader itself creates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core import gt_check as rg
from ..core.sampling import sample_pairs
from ..datasets import get as get_dataset_spec

MAX_SIDE = 840
MAX_BAD_PAIR_FRACTION = 0.05


def _pose_summary(name: str, pairs: list[dict]) -> rg.Summary:
    rng = np.random.default_rng(1234)
    summary = rg.Summary(dataset=name)
    for i, p in enumerate(pairs):
        imgs, Ks = [], []
        for path, K in ((p["im_A_path"], p["K0"]), (p["im_B_path"], p["K1"])):
            img, K = rg.read_rgb(path), np.asarray(K, np.float64)
            h, w = img.shape[:2]
            s = MAX_SIDE / max(h, w)
            if s < 1.0:
                img, K = rg.resize_with_k(img, K, (round(h * s), round(w * s)))
            imgs.append(img)
            Ks.append(K)
        summary.results.append(rg.evaluate_pair(
            f"{p.get('scene', '?')}#{i} {Path(p['im_A_path']).name}->{Path(p['im_B_path']).name}",
            imgs[0], imgs[1], Ks[0], Ks[1], R=p["R_0to1"], t=p["t_0to1"], rng=rng))
    return summary


def _pair_label(p: dict) -> str:
    if p.get("pair_id"):
        return str(p["pair_id"])
    a = Path(p["im_A_path"])
    return f"{p.get('scene', a.parent.name)}/{a.name}"


def _points_problems(pairs: list[dict], task: str) -> dict[str, list[str]]:
    """Problems per pair label (empty dict when everything is in order)."""
    from PIL import Image

    problems: dict[str, list[str]] = {}
    for p in pairs:
        label = _pair_label(p)
        mine = problems.setdefault(label, [])
        sizes = []
        for key in ("im_A_path", "im_B_path"):
            try:
                with Image.open(p[key]) as im:
                    sizes.append(im.size)
            except OSError as exc:
                mine.append(f"{p[key]} unreadable ({exc})")
        if len(sizes) != 2:
            continue
        if task == "correspondence":
            c = np.asarray(p["corrs"], np.float64).reshape(-1, 4)
            pts = ((c[:, :2], sizes[0]), (c[:, 2:], sizes[1]))
        elif task == "homography_points":
            pts = ((np.asarray(p["pts_A"]).reshape(-1, 2), sizes[0]), (np.asarray(p["pts_B"]).reshape(-1, 2), sizes[1]))
        else:
            gt = p.get("H_gt")
            if gt is not None and not np.isfinite(np.asarray(gt, np.float64)).all():
                mine.append("non-finite H_gt")
            continue
        for arr, (w, h) in pts:
            out = (arr[:, 0] < 0) | (arr[:, 0] > w) | (arr[:, 1] < 0) | (arr[:, 1] > h)
            if out.any():
                mine.append(f"{int(out.sum())} annotation(s) outside a {w}x{h} image")
    return {k: v for k, v in problems.items() if v}


def check_dataset(name: str, data_root: Path, n_pairs: int = 8, seed: int = 0) -> bool:
    spec = get_dataset_spec(name)
    pairs = list(sample_pairs(spec.iter_pairs(data_root, max_pairs=None, **(spec.kwargs or {})),
                              n_pairs, seed=seed))
    if not pairs:
        print(f"{name}: FAIL no pairs under {data_root}")
        return False
    if spec.task == "pose":
        summary = _pose_summary(name, pairs)
        print(summary.report())
        return summary.passed
    problems = _points_problems(pairs, spec.task)
    # A few out-of-image annotations occur in released data (3 of 830 HardMatch
    # test pairs); reference evaluators keep them, so only a widespread problem fails.
    bad_pairs = len(problems)
    passed = bad_pairs <= MAX_BAD_PAIR_FRACTION * len(pairs)
    status = ("PASS" if not problems else "PASS with warnings") if passed else "FAIL"
    print(f"{name}: {status} ({len(pairs)} pairs checked, {bad_pairs} with problems, task={spec.task})")
    for label, lines in list(problems.items())[:20]:
        print(f"  {label}: {'; '.join(lines)}")
    return passed
