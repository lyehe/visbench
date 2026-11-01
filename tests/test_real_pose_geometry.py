"""Real-data GT-geometry regression check for visbench pose datasets.

For a handful of real pairs from each dataset's REAL ``iter_pairs`` we compute SIFT
matches and the median Sampson epipolar error under the yielded GT (R_0to1, t_0to1,
K0, K1), and under a 5-8 deg perturbed rotation.  Correct GT gives ~<2 px and is far
better than the perturbed pose; a convention bug (e.g. TartanAir NED poses used as
OpenCV, fixed 2026-10-01 in visbench/datasets/tartanair.py::_parse_poses) does not.

Data root: $VISBENCH_DATA_ROOT/<default_root>, else D:\\Yehe\\matcher_training\\<default_root>.
Tests skip cleanly when the data is absent.  Run:  pytest tests/test_real_pose_geometry.py -s
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _real_geom_check as rg  # noqa: E402


MATCHER_TRAINING = Path(os.environ.get("VISBENCH_REAL_DATA_ROOT", r"D:\Yehe\matcher_training"))
N_PAIRS = 8
MAX_SIDE = 840  # downscale big images (MegaDepth) to keep SIFT on CPU fast

# dataset -> (pairs to scan lazily, stride when picking pairs from the scan)
SAMPLING = {
    "megadepth1500": (1500, 187),
    "eth3d": (400, 50),
}


def _dataset_root(name: str) -> Path | None:
    from visbench.datasets import _registry  # noqa: F401
    import visbench.datasets  # noqa: F401  (registers all datasets)

    spec = _registry.get(name)
    candidates = []
    if os.environ.get("VISBENCH_DATA_ROOT"):
        candidates.append(Path(os.environ["VISBENCH_DATA_ROOT"]) / spec.default_root)
    candidates.append(MATCHER_TRAINING / spec.default_root)
    for c in candidates:
        if c.is_dir():
            return c
    return None


def _pick_pairs(name: str) -> list[dict]:
    from visbench.datasets import _registry

    root = _dataset_root(name)
    if root is None:
        pytest.skip(f"{name}: dataset root not found (VISBENCH_DATA_ROOT or {MATCHER_TRAINING})")
    spec = _registry.get(name)
    scan, stride = SAMPLING[name]
    try:
        pairs = list(spec.iter_pairs(root, max_pairs=scan, **(spec.kwargs or {})))
    except FileNotFoundError as exc:
        pytest.skip(f"{name}: {exc}")
    if not pairs:
        pytest.skip(f"{name}: iter_pairs yielded 0 pairs under {root}")
    return pairs[::stride][:N_PAIRS]


def _evaluate(name: str, pairs: list[dict]) -> rg.Summary:
    rng = np.random.default_rng(1234)
    summary = rg.Summary(dataset=f"visbench/{name}")
    for i, p in enumerate(pairs):
        img0, img1 = rg.read_rgb(p["im_A_path"]), rg.read_rgb(p["im_B_path"])
        K0, K1 = np.asarray(p["K0"], np.float64), np.asarray(p["K1"], np.float64)
        for k, (img, K) in enumerate(((img0, K0), (img1, K1))):
            h, w = img.shape[:2]
            s = MAX_SIDE / max(h, w)
            if s < 1.0:
                img, K = rg.resize_with_k(img, K, (round(h * s), round(w * s)))
            if k == 0:
                img0, K0 = img, K
            else:
                img1, K1 = img, K
        summary.results.append(
            rg.evaluate_pair(
                f"{p.get('scene', '?')}#{i} {Path(p['im_A_path']).name}->{Path(p['im_B_path']).name}",
                img0, img1, K0, K1, R=p["R_0to1"], t=p["t_0to1"], rng=rng,
            )
        )
    return summary


@pytest.mark.parametrize("name", ["megadepth1500", "eth3d"])
def test_real_pose_geometry(name):
    summary = _evaluate(name, _pick_pairs(name))
    print("\n" + summary.report())
    assert summary.usable, f"{name}: no pair had >= {rg.MIN_MATCHES} SIFT matches\n{summary.report()}"
    assert summary.passed, summary.report()
