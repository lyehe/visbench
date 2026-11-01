"""Advanced synthetic stressors — optical / sensor / scene simulations on HPatches.

Source:  derived from `hpatches`
License: inherits HPatches (CC BY-NC-SA 4.0).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .._registry import DatasetSpec, register
from ._synthetic_common import make_iter as _make_iter
from .synthetic import download as _download_hpatches


def download(data_root: Path | str) -> Path:
    return _download_hpatches(data_root)


def _motion_blur(k: int):
    ker = np.zeros((k, k), dtype=np.float32)
    ker[k // 2, :] = 1.0 / k
    def f(arr):
        return cv2.filter2D(arr, -1, ker)
    return f


def _low_light(factor: float = 0.25, sigma: float = 8):
    def f(arr):
        out = (arr.astype(np.float32) * factor)
        rng = np.random.default_rng(0)
        noise = rng.normal(0, sigma, out.shape).astype(np.float32)
        return np.clip(out + noise, 0, 255).astype(np.uint8)
    return f


def _fog(strength: float = 0.6):
    def f(arr):
        airlight = np.full_like(arr, 230)
        t = 1.0 - strength
        out = arr.astype(np.float32) * t + airlight.astype(np.float32) * (1 - t)
        return np.clip(out, 0, 255).astype(np.uint8)
    return f


_BASE_KW = dict(
    download=download,
    source_url="https://github.com/hpatches/hpatches-dataset",
    license="CC BY-NC-SA 4.0",
)


# Compact stress set (see synthetic.py).
register("synthetic_motion_blur_15", DatasetSpec(
    "homography", _make_iter("motion_blur_15", _motion_blur(15)), Path("hpatches"), **_BASE_KW))
register("synthetic_low_light", DatasetSpec(
    "homography", _make_iter("low_light", _low_light()), Path("hpatches"), **_BASE_KW))
register("synthetic_fog", DatasetSpec(
    "homography", _make_iter("fog", _fog()), Path("hpatches"), **_BASE_KW))
