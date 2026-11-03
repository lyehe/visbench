"""Shared primitives for synthetic-perturbation HPatches datasets.

Every synthetic module imports from here; nothing here imports from a sibling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import cv2

from ...core.imageio import imread
import numpy as np

from ..homography.hpatches import DEGENERATE


def base_pairs(data_root: Path, max_pairs: int | None) -> Iterator[dict]:
    """Yield each HPatches reference pair (1.ppm <-> k.ppm for k in 2..6)
    along with the published GT homography and subset (illum/view) tag.
    Skips DEGENERATE sequences and stops at `max_pairs` if set.
    """
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


def make_iter(pert_name: str, fn: Callable[[np.ndarray], np.ndarray],
              scale_xy: tuple[float, float] | None = None) -> Callable:
    """Build an iter_pairs function that runs `fn` on image B and caches the
    result under `<data_root>/_synthetic_<pert_name>/<seq>/<pair_idx>.png`.
    Optionally composes a diagonal scale matrix with H_gt when the
    perturbation rescales image B (e.g. `synthetic_scale_*`).
    """
    def iter_pairs(data_root, max_pairs=None, **_):
        data_root = Path(data_root)
        cache_dir = data_root / f"_synthetic_{pert_name}"
        cache_dir.mkdir(exist_ok=True)
        for pair in base_pairs(data_root, max_pairs):
            im_B = Path(pair["im_B_path"])
            out_dir = cache_dir / pair["seq"]
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"{pair['pair_idx']}.png"
            if not out_path.exists():
                arr = imread(str(im_B), cv2.IMREAD_COLOR)
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


# ---- Photometric / sensor perturbation primitives -------------------------
# Each returns a callable f(arr_uint8_bgr) -> arr_uint8_bgr; chained via
# `make_iter` above.

def jpeg(q: int):
    def f(arr):
        _, enc = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return f


def gauss_noise(sigma: float):
    def f(arr):
        noise = np.random.default_rng(0).normal(0, sigma, arr.shape).astype(np.float32)
        return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return f


def gauss_blur(ksize: int):
    def f(arr):
        return cv2.GaussianBlur(arr, (ksize, ksize), 0)
    return f
