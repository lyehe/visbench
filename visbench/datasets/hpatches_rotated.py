"""Rotated HPatches — stress-test rotation invariance.

Applies a fixed rotation angle to each target image and composes the rotation
with the GT homography. Most learned matchers (SuperPoint, LightGlue, RoMa)
collapse on rotation while SIFT/ORB stay flat.

Source:  derived from `hpatches`
License: inherits HPatches (CC BY-NC-SA 4.0).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image

from ._registry import DatasetSpec, register
from .hpatches import DEGENERATE, download as _download_hpatches


def download(data_root: Path | str) -> Path:
    return _download_hpatches(data_root)


def _rotation_homography(angle_deg: float, w: int, h: int) -> tuple[np.ndarray, tuple[int, int]]:
    a = math.radians(angle_deg)
    cos, sin = math.cos(a), math.sin(a)
    new_w = int(math.ceil(abs(w * cos) + abs(h * sin)))
    new_h = int(math.ceil(abs(w * sin) + abs(h * cos)))
    cx, cy = w / 2.0, h / 2.0
    ncx, ncy = new_w / 2.0, new_h / 2.0
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    R = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    T2 = np.array([[1, 0, ncx], [0, 1, ncy], [0, 0, 1]])
    return T2 @ R @ T1, (new_w, new_h)


def iter_pairs(data_root: Path | str, rotation_deg: float = 90.0,
               max_pairs: int | None = None, skip_degenerate: bool = True,
               cache_dir: Path | str | None = None) -> Iterator[dict]:
    data_root = Path(data_root)
    seq_root = data_root / "hpatches-sequences-release"
    if not seq_root.exists():
        seq_root = data_root

    cache_dir = Path(cache_dir) if cache_dir else data_root / f"_rotated_{rotation_deg:+.1f}"
    cache_dir.mkdir(exist_ok=True)

    count = 0
    for seq in sorted(seq_root.iterdir()):
        if not seq.is_dir():
            continue
        if skip_degenerate and seq.name in DEGENERATE:
            continue
        if not (seq.name.startswith("i_") or seq.name.startswith("v_")):
            continue
        subset = ("illum" if seq.name.startswith("i_") else "view") + f"_rot{int(rotation_deg)}"
        im1 = seq / "1.ppm"
        if not im1.exists():
            continue
        for k in range(2, 7):
            imk = seq / f"{k}.ppm"
            Hk = seq / f"H_1_{k}"
            if not (imk.exists() and Hk.exists()):
                continue
            H_1k = np.loadtxt(Hk)

            out_dir = cache_dir / seq.name
            out_dir.mkdir(exist_ok=True)
            rotated_path = out_dir / f"{k}.png"

            if not rotated_path.exists():
                im_bgr = cv2.imread(str(imk), cv2.IMREAD_COLOR)
                h, w = im_bgr.shape[:2]
                H_rot, (nw, nh) = _rotation_homography(rotation_deg, w, h)
                rotated = cv2.warpPerspective(im_bgr, H_rot, (nw, nh), borderValue=(0, 0, 0))
                cv2.imwrite(str(rotated_path), rotated)
            else:
                with Image.open(imk) as pim:
                    w, h = pim.size
                H_rot, (nw, nh) = _rotation_homography(rotation_deg, w, h)

            H_gt = H_rot @ H_1k
            yield {
                "im_A_path": str(im1),
                "im_B_path": str(rotated_path),
                "H_gt": H_gt,
                "subset": subset,
                "seq": seq.name,
                "pair_idx": k,
                "rotation_deg": rotation_deg,
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


def _make_rot(angle):
    def it(data_root, max_pairs=None, **_):
        return iter_pairs(data_root, rotation_deg=float(angle), max_pairs=max_pairs)
    return it


for _ang in (10, 20, 30, 45, 60, 90, 120, 135, 150, 180, 225, 270, 315):
    register(f"hpatches_rot{_ang}", DatasetSpec(
        task="homography",
        iter_pairs=_make_rot(_ang),
        default_root=Path("hpatches"),
        download=download,
        source_url="https://github.com/hpatches/hpatches-dataset",
        license="CC BY-NC-SA 4.0",
    ))
