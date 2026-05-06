"""HPatches Patches — descriptor-isolation benchmark.

Mimics Lenc-Mikolajczyk 2017 "HPatches Patches" but extracts patches on-the-fly
from `hpatches-sequences-release` using GT homographies (avoids the 12 GB
hpatches-release download).

Source:  derived from `hpatches`
License: inherits HPatches (CC BY-NC-SA 4.0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ._registry import DatasetSpec, register
from .hpatches import DEGENERATE, download as _download_hpatches


def download(data_root: Path | str) -> Path:
    return _download_hpatches(data_root)


def _detect_kpts(img1_gray, n_max: int = 300):
    sift = cv2.SIFT_create(nfeatures=n_max * 3)
    kp = sift.detect(img1_gray, None)
    if len(kp) == 0:
        return []
    return sorted(kp, key=lambda k: -k.response)[:n_max]


def _project(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    proj = pts_h @ H.T
    return proj[:, :2] / np.maximum(proj[:, 2:3], 1e-9)


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               skip_degenerate: bool = True, n_kpts: int = 300) -> Iterator[dict]:
    data_root = Path(data_root)
    seq_root = data_root / "hpatches-sequences-release"
    if not seq_root.exists():
        seq_root = data_root
    count = 0
    for seq in sorted(seq_root.iterdir()):
        if not seq.is_dir():
            continue
        if skip_degenerate and seq.name in DEGENERATE:
            continue
        if not (seq.name.startswith("i_") or seq.name.startswith("v_")):
            continue
        subset = "illum" if seq.name.startswith("i_") else "view"
        im1 = seq / "1.ppm"
        if not im1.exists():
            continue
        img1 = cv2.imread(str(im1), cv2.IMREAD_GRAYSCALE)
        if img1 is None:
            continue
        kp1 = _detect_kpts(img1, n_max=n_kpts)
        if not kp1:
            continue
        kpts_A_full = np.array([[k.pt[0], k.pt[1]] for k in kp1], dtype=np.float64)
        sizes_A = np.array([k.size for k in kp1], dtype=np.float64)
        h1, w1 = img1.shape
        for k_idx in range(2, 7):
            imk = seq / f"{k_idx}.ppm"
            Hk = seq / f"H_1_{k_idx}"
            if not (imk.exists() and Hk.exists()):
                continue
            H = np.loadtxt(Hk)
            imgk = cv2.imread(str(imk), cv2.IMREAD_GRAYSCALE)
            if imgk is None:
                continue
            hk, wk = imgk.shape
            kpts_B = _project(kpts_A_full, H)
            margin = 32
            valid = ((kpts_B[:, 0] >= margin) & (kpts_B[:, 0] < wk - margin) &
                     (kpts_B[:, 1] >= margin) & (kpts_B[:, 1] < hk - margin))
            valid &= ((kpts_A_full[:, 0] >= margin) & (kpts_A_full[:, 0] < w1 - margin) &
                      (kpts_A_full[:, 1] >= margin) & (kpts_A_full[:, 1] < h1 - margin))
            if valid.sum() < 8:
                continue
            yield {
                "im_A_path": str(im1),
                "im_B_path": str(imk),
                "kpts_A": kpts_A_full[valid].astype(np.float32),
                "kpts_B": kpts_B[valid].astype(np.float32),
                "kpt_sizes_A": sizes_A[valid].astype(np.float32),
                "H_gt": H,
                "subset": subset,
                "seq": seq.name,
                "pair_idx": k_idx,
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


register("hpatches_patches", DatasetSpec(
    task="descriptor",
    iter_pairs=iter_pairs,
    default_root=Path("hpatches"),
    default_resize=None,
    download=download,
    source_url="https://github.com/hpatches/hpatches-dataset",
    license="CC BY-NC-SA 4.0",
))
