"""Rotate image B of a pair by 90 degrees counter-clockwise and keep its GT consistent.

Recent matching papers report robustness on existing benchmarks with image B
rotated. Coordinates use the continuous convention (pixel corners at integers,
centres at +0.5), so a CCW rotation of a W-wide image maps (x, y) -> (y, W - x).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Rotation of camera coordinates that matches a CCW image rotation:
# K' Rz = A K with A the pixel map above, K' = [[fy,0,cy],[0,fx,W-cx],[0,0,1]].
_RZ = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def rotated_image(src: Path, cache_root: Path, rel: Path) -> tuple[Path, int, int]:
    """Cached CCW-rotated copy of `src`; returns (path, original W, original H)."""
    from PIL import Image

    out = (cache_root / rel).with_suffix(".rot90.png")
    with Image.open(src) as im:
        W, H = im.size
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            im.transpose(Image.Transpose.ROTATE_90).save(out)
    return out, W, H


def rotate_points(pts: np.ndarray, W: int) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 1]
    out[:, 1] = W - pts[:, 0]
    return out


def rotate_camera(K: np.ndarray, R_0to1: np.ndarray, t_0to1: np.ndarray, W: int):
    """New intrinsics of the rotated image and the relative pose to its camera."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    K_rot = np.array([[fy, 0.0, cy], [0.0, fx, W - cx], [0.0, 0.0, 1.0]])
    return K_rot, _RZ @ R_0to1, _RZ @ np.asarray(t_0to1, dtype=np.float64)
