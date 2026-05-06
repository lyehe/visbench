"""Geometric stressors that compose a transform with the GT H.

Source:  derived from `hpatches`
License: inherits HPatches (CC BY-NC-SA 4.0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ._registry import DatasetSpec, register
from .synthetic import _base_pairs, download as _download_hpatches


def download(data_root: Path | str) -> Path:
    return _download_hpatches(data_root)


def _compose_iter(name: str, make_H_and_warp: Callable[[int, int], tuple]):
    def it(data_root, max_pairs=None, **_):
        data_root = Path(data_root)
        cache = data_root / f"_synthetic_{name}"
        cache.mkdir(exist_ok=True)
        for pair in _base_pairs(data_root, max_pairs):
            im_B = Path(pair["im_B_path"])
            out_dir = cache / pair["seq"]
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"{pair['pair_idx']}.png"
            arr = cv2.imread(str(im_B), cv2.IMREAD_COLOR)
            h, w = arr.shape[:2]
            H_aug, (nw, nh) = make_H_and_warp(w, h)
            if not out_path.exists():
                warped = cv2.warpPerspective(arr, H_aug, (nw, nh), borderValue=(0, 0, 0))
                cv2.imwrite(str(out_path), warped)
            yield {
                **pair,
                "im_B_path": str(out_path),
                "H_gt": H_aug @ pair["H_gt"],
                "subset": f"{pair['subset']}_{name}",
            }
    return it


def _hflip():
    def make(w, h):
        H = np.array([[-1.0, 0.0, w], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        return H, (w, h)
    return make


def _vflip():
    def make(w, h):
        H = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, h], [0.0, 0.0, 1.0]])
        return H, (w, h)
    return make


def _shear(sx: float):
    def make(w, h):
        S = np.array([[1.0, sx, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
        wc = S @ corners
        xs, ys = wc[0], wc[1]
        tx, ty = -min(0, xs.min()), -min(0, ys.min())
        T = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
        H = T @ S
        nw = int(np.ceil(xs.max() - min(0, xs.min())))
        nh = int(np.ceil(ys.max() - min(0, ys.min())))
        return H, (nw, nh)
    return make


def _perspective(strength: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    def make(w, h):
        d = strength * min(w, h)
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst = src + rng.uniform(-d, d, src.shape).astype(np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        corners = np.concatenate([dst, np.ones((4, 1), dtype=np.float32)], axis=1).T
        xs = corners[0]; ys = corners[1]
        tx, ty = -min(0.0, float(xs.min())), -min(0.0, float(ys.min()))
        T = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
        H_full = T @ H
        nw = int(np.ceil(float(xs.max()) + tx))
        nh = int(np.ceil(float(ys.max()) + ty))
        return H_full, (nw, nh)
    return make


def _random_affine(seed: int = 0):
    rng = np.random.default_rng(seed)
    def make(w, h):
        theta = rng.uniform(-30, 30) * np.pi / 180.0
        s = rng.uniform(0.8, 1.3)
        sh = rng.uniform(-0.15, 0.15)
        R = np.array([[np.cos(theta) * s, -np.sin(theta) * s + sh, 0.0],
                      [np.sin(theta) * s, np.cos(theta) * s, 0.0],
                      [0.0, 0.0, 1.0]])
        cx, cy = w / 2.0, h / 2.0
        T1 = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
        wc = R @ T1 @ corners
        xs, ys = wc[0], wc[1]
        tx, ty = -min(0.0, float(xs.min())), -min(0.0, float(ys.min()))
        T2 = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
        H = T2 @ R @ T1
        nw = int(np.ceil(float(xs.max()) + tx))
        nh = int(np.ceil(float(ys.max()) + ty))
        return H, (nw, nh)
    return make


_BASE_KW = dict(
    download=download,
    source_url="https://github.com/hpatches/hpatches-dataset",
    license="CC BY-NC-SA 4.0",
)


register("synthetic_hflip", DatasetSpec("homography", _compose_iter("hflip", _hflip()), Path("hpatches"), **_BASE_KW))
register("synthetic_vflip", DatasetSpec("homography", _compose_iter("vflip", _vflip()), Path("hpatches"), **_BASE_KW))
register("synthetic_shear_0p1", DatasetSpec("homography", _compose_iter("shear_0p1", _shear(0.1)), Path("hpatches"), **_BASE_KW))
register("synthetic_shear_0p3", DatasetSpec("homography", _compose_iter("shear_0p3", _shear(0.3)), Path("hpatches"), **_BASE_KW))
register("synthetic_perspective_mild", DatasetSpec("homography", _compose_iter("persp_mild", _perspective(0.08)), Path("hpatches"), **_BASE_KW))
register("synthetic_perspective_strong", DatasetSpec("homography", _compose_iter("persp_strong", _perspective(0.20)), Path("hpatches"), **_BASE_KW))
register("synthetic_affine_random", DatasetSpec("homography", _compose_iter("affine_random", _random_affine()), Path("hpatches"), **_BASE_KW))
