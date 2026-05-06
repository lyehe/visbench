"""Synthetic perturbation pipelines over HPatches.

For each base illum+view pair, apply a controlled perturbation to image B and
compose with the GT homography where appropriate. Cached under
`<hpatches_root>/_synthetic_<name>/`.

Source:  derived from `hpatches`
License: inherits HPatches (CC BY-NC-SA 4.0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ._registry import DatasetSpec, register
from .hpatches import DEGENERATE, download as _download_hpatches


def download(data_root: Path | str) -> Path:
    return _download_hpatches(data_root)


def _base_pairs(data_root: Path, max_pairs: int | None):
    data_root = Path(data_root)
    seq_root = data_root / "hpatches-sequences-release"
    if not seq_root.exists():
        seq_root = data_root
    count = 0
    for seq in sorted(seq_root.iterdir()):
        if not seq.is_dir() or seq.name in DEGENERATE:
            continue
        if not (seq.name.startswith("i_") or seq.name.startswith("v_")):
            continue
        subset_type = "illum" if seq.name.startswith("i_") else "view"
        im1 = seq / "1.ppm"
        if not im1.exists():
            continue
        for k in range(2, 7):
            imk = seq / f"{k}.ppm"
            Hk = seq / f"H_1_{k}"
            if not (imk.exists() and Hk.exists()):
                continue
            yield {
                "im_A_path": str(im1),
                "im_B_path": str(imk),
                "H_gt": np.loadtxt(Hk),
                "subset": subset_type,
                "seq": seq.name,
                "pair_idx": k,
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


def _make_iter(pert_name: str, fn: Callable[[np.ndarray], np.ndarray],
               composes_H: bool = False,
               scale_xy: tuple[float, float] | None = None) -> Callable:
    def iter_pairs(data_root, max_pairs=None, **_):
        data_root = Path(data_root)
        cache_dir = data_root / f"_synthetic_{pert_name}"
        cache_dir.mkdir(exist_ok=True)
        for pair in _base_pairs(data_root, max_pairs):
            im_B = Path(pair["im_B_path"])
            out_dir = cache_dir / pair["seq"]
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"{pair['pair_idx']}.png"
            if not out_path.exists():
                arr = cv2.imread(str(im_B), cv2.IMREAD_COLOR)
                warped = fn(arr)
                cv2.imwrite(str(out_path), warped)
            H_gt = pair["H_gt"].copy()
            if scale_xy is not None:
                sx, sy = scale_xy
                S = np.diag([sx, sy, 1.0])
                H_gt = S @ H_gt
            yield {
                **pair,
                "im_B_path": str(out_path),
                "H_gt": H_gt,
                "subset": f"{pair['subset']}_{pert_name}",
            }
    return iter_pairs


def _jpeg(q: int):
    def f(arr):
        _, enc = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return f


def _gauss_noise(sigma: float):
    def f(arr):
        noise = np.random.default_rng(0).normal(0, sigma, arr.shape).astype(np.float32)
        return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return f


def _gauss_blur(ksize: int):
    def f(arr):
        return cv2.GaussianBlur(arr, (ksize, ksize), 0)
    return f


def _gamma(g: float):
    lut = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)], dtype=np.uint8)
    def f(arr):
        return cv2.LUT(arr, lut)
    return f


def _scale(s: float):
    def f(arr):
        h, w = arr.shape[:2]
        return cv2.resize(arr, (max(1, int(w * s)), max(1, int(h * s))),
                          interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return f


def _hsv_shift(dh: int):
    def f(arr):
        hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + dh) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return f


def _salt_pepper(p: float):
    def f(arr):
        rng = np.random.default_rng(0)
        mask = rng.uniform(0, 1, arr.shape[:2])
        out = arr.copy()
        out[mask < p / 2] = 0
        out[mask > 1 - p / 2] = 255
        return out
    return f


_BASE_KW = dict(
    download=download,
    source_url="https://github.com/hpatches/hpatches-dataset",
    license="CC BY-NC-SA 4.0",
)

for q in (10, 30, 50):
    register(f"synthetic_jpeg_q{q}", DatasetSpec(
        "homography", _make_iter(f"jpeg_q{q}", _jpeg(q)), Path("hpatches"), **_BASE_KW))

for sigma in (5, 15, 30):
    register(f"synthetic_noise_{sigma}", DatasetSpec(
        "homography", _make_iter(f"noise_{sigma}", _gauss_noise(float(sigma))),
        Path("hpatches"), **_BASE_KW))

for k in (3, 7, 15):
    register(f"synthetic_blur_{k}", DatasetSpec(
        "homography", _make_iter(f"blur_{k}", _gauss_blur(k)), Path("hpatches"), **_BASE_KW))

register("synthetic_gamma_0p5", DatasetSpec(
    "homography", _make_iter("gamma_0p5", _gamma(0.5)), Path("hpatches"), **_BASE_KW))
register("synthetic_gamma_2p0", DatasetSpec(
    "homography", _make_iter("gamma_2p0", _gamma(2.0)), Path("hpatches"), **_BASE_KW))

register("synthetic_scale_0p5", DatasetSpec(
    "homography",
    _make_iter("scale_0p5", _scale(0.5), scale_xy=(0.5, 0.5)),
    Path("hpatches"), **_BASE_KW))
register("synthetic_scale_2p0", DatasetSpec(
    "homography",
    _make_iter("scale_2p0", _scale(2.0), scale_xy=(2.0, 2.0)),
    Path("hpatches"), **_BASE_KW))

register("synthetic_hsv_shift_30", DatasetSpec(
    "homography", _make_iter("hsv_shift_30", _hsv_shift(30)), Path("hpatches"), **_BASE_KW))
register("synthetic_saltpepper_2pct", DatasetSpec(
    "homography", _make_iter("saltpepper_2pct", _salt_pepper(0.02)), Path("hpatches"), **_BASE_KW))
