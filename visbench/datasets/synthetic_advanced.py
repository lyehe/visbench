"""Advanced synthetic stressors — optical / sensor / scene simulations on HPatches.

Source:  derived from `hpatches`
License: inherits HPatches (CC BY-NC-SA 4.0).
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from ._registry import DatasetSpec, register
from .synthetic import _make_iter, download as _download_hpatches


def download(data_root: Path | str) -> Path:
    return _download_hpatches(data_root)


def _motion_blur(k: int):
    ker = np.zeros((k, k), dtype=np.float32)
    ker[k // 2, :] = 1.0 / k
    def f(arr):
        return cv2.filter2D(arr, -1, ker)
    return f


def _defocus(k: int):
    y, x = np.ogrid[-k:k + 1, -k:k + 1]
    disc = (x * x + y * y <= k * k).astype(np.float32)
    disc /= disc.sum()
    def f(arr):
        return cv2.filter2D(arr, -1, disc)
    return f


def _vignette(strength: float):
    def f(arr):
        h, w = arr.shape[:2]
        y, x = np.mgrid[0:h, 0:w]
        cx, cy = w / 2, h / 2
        d = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
        mask = np.clip(1.0 - strength * d, 0, 1).astype(np.float32)
        return (arr.astype(np.float32) * mask[..., None]).clip(0, 255).astype(np.uint8)
    return f


def _low_light(factor: float = 0.25, sigma: float = 8):
    def f(arr):
        out = (arr.astype(np.float32) * factor)
        rng = np.random.default_rng(0)
        noise = rng.normal(0, sigma, out.shape).astype(np.float32)
        return np.clip(out + noise, 0, 255).astype(np.uint8)
    return f


def _high_contrast(alpha: float = 2.0):
    def f(arr):
        return np.clip((arr.astype(np.float32) - 128) * alpha + 128, 0, 255).astype(np.uint8)
    return f


def _fog(strength: float = 0.6):
    def f(arr):
        airlight = np.full_like(arr, 230)
        t = 1.0 - strength
        out = arr.astype(np.float32) * t + airlight.astype(np.float32) * (1 - t)
        return np.clip(out, 0, 255).astype(np.uint8)
    return f


def _chromatic_aberration(shift: int = 3):
    def f(arr):
        out = arr.copy()
        out[..., 0] = np.roll(arr[..., 0], shift, axis=1)
        out[..., 2] = np.roll(arr[..., 2], -shift, axis=1)
        return out
    return f


def _grayscale():
    def f(arr):
        g = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return f


def _invert():
    def f(arr):
        return 255 - arr
    return f


def _bgr_swap():
    def f(arr):
        return arr[..., ::-1].copy()
    return f


def _clahe():
    c = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    def f(arr):
        lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
        lab[..., 0] = c.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return f


def _posterize(bits: int = 3):
    shift = 8 - bits
    def f(arr):
        return (arr >> shift) << shift
    return f


def _pixelize(block: int):
    def f(arr):
        h, w = arr.shape[:2]
        small = cv2.resize(arr, (max(1, w // block), max(1, h // block)),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return f


def _occlusion(frac: float = 0.25):
    def f(arr):
        h, w = arr.shape[:2]
        rng = np.random.default_rng(0)
        oh = int(math.sqrt(frac) * h)
        ow = int(math.sqrt(frac) * w)
        y0 = rng.integers(0, max(1, h - oh))
        x0 = rng.integers(0, max(1, w - ow))
        out = arr.copy()
        out[y0:y0 + oh, x0:x0 + ow] = 0
        return out
    return f


def _crop_center(ratio: float = 0.5):
    def f(arr):
        h, w = arr.shape[:2]
        ch, cw = int(h * ratio), int(w * ratio)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        crop = arr[y0:y0 + ch, x0:x0 + cw]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
    return f


_BASE_KW = dict(
    download=download,
    source_url="https://github.com/hpatches/hpatches-dataset",
    license="CC BY-NC-SA 4.0",
)


for k in (5, 15):
    register(f"synthetic_motion_blur_{k}", DatasetSpec(
        "homography", _make_iter(f"motion_blur_{k}", _motion_blur(k)), Path("hpatches"), **_BASE_KW))

for k in (5, 11):
    register(f"synthetic_defocus_{k}", DatasetSpec(
        "homography", _make_iter(f"defocus_{k}", _defocus(k)), Path("hpatches"), **_BASE_KW))

register("synthetic_vignette_strong", DatasetSpec(
    "homography", _make_iter("vignette_strong", _vignette(0.9)), Path("hpatches"), **_BASE_KW))
register("synthetic_low_light", DatasetSpec(
    "homography", _make_iter("low_light", _low_light()), Path("hpatches"), **_BASE_KW))
register("synthetic_high_contrast", DatasetSpec(
    "homography", _make_iter("high_contrast", _high_contrast()), Path("hpatches"), **_BASE_KW))
register("synthetic_fog", DatasetSpec(
    "homography", _make_iter("fog", _fog()), Path("hpatches"), **_BASE_KW))
register("synthetic_chromatic_aberration", DatasetSpec(
    "homography", _make_iter("chromatic_ab", _chromatic_aberration()), Path("hpatches"), **_BASE_KW))
register("synthetic_grayscale", DatasetSpec(
    "homography", _make_iter("grayscale", _grayscale()), Path("hpatches"), **_BASE_KW))
register("synthetic_invert", DatasetSpec(
    "homography", _make_iter("invert", _invert()), Path("hpatches"), **_BASE_KW))
register("synthetic_bgr_swap", DatasetSpec(
    "homography", _make_iter("bgr_swap", _bgr_swap()), Path("hpatches"), **_BASE_KW))
register("synthetic_clahe", DatasetSpec(
    "homography", _make_iter("clahe", _clahe()), Path("hpatches"), **_BASE_KW))
register("synthetic_posterize_3bit", DatasetSpec(
    "homography", _make_iter("posterize_3bit", _posterize(3)), Path("hpatches"), **_BASE_KW))

for b in (4, 8):
    register(f"synthetic_pixelize_{b}", DatasetSpec(
        "homography", _make_iter(f"pixelize_{b}", _pixelize(b)), Path("hpatches"), **_BASE_KW))

register("synthetic_occlusion_25", DatasetSpec(
    "homography", _make_iter("occlusion_25", _occlusion(0.25)), Path("hpatches"), **_BASE_KW))
register("synthetic_crop_center_50", DatasetSpec(
    "homography",
    _make_iter("crop_center_50", _crop_center(0.5)),
    Path("hpatches"), **_BASE_KW))
