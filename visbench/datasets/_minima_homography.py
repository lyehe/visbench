"""Shared helpers for MINIMA-style synthetic-homography cross-modal benchmarks.

Source:  https://github.com/LSXI7/MINIMA
Paper:   MINIMA, 2024.
License: see MINIMA repo (ports `src/utils/sample_h.py:sample_homography`).
"""

from __future__ import annotations

import hashlib
import random
from math import pi
from pathlib import Path

import cv2
import numpy as np
from numpy.random import uniform
from scipy import stats

MINIMA_DEFAULT_CONFIG = {
    "perspective": True,
    "scaling": True,
    "rotation": True,
    "translation": True,
    "n_scales": 5,
    "n_angles": 25,
    "scaling_amplitude": 0.05,
    "perspective_amplitude_x": 0.05,
    "perspective_amplitude_y": 0.05,
    "patch_ratio": 0.8,
    "max_angle": 10 * (pi / 180),
    "allow_artifacts": False,
    "translation_overflow": 0.0,
}


def _seed_from_name(image_name: str) -> int:
    h = hashlib.sha256(image_name.encode()).hexdigest()
    return int(h, 16) % (2 ** 32)


def sample_homography(shape, image_name: str, config: dict | None = None) -> np.ndarray:
    """Deterministic per-image-name homography matching MINIMA's sample_h."""
    seed = _seed_from_name(image_name)
    np_state = np.random.get_state()
    np.random.seed(seed)
    random.seed(seed)
    try:
        cfg = dict(MINIMA_DEFAULT_CONFIG)
        if config is not None:
            cfg.update(config)

        margin = (1 - cfg["patch_ratio"]) / 2
        pts1 = margin + np.array([
            [0, 0],
            [0, cfg["patch_ratio"]],
            [cfg["patch_ratio"], cfg["patch_ratio"]],
            [cfg["patch_ratio"], 0],
        ])
        pts2 = pts1.copy()

        if cfg["perspective"]:
            ax = min(cfg["perspective_amplitude_x"], margin)
            ay = min(cfg["perspective_amplitude_y"], margin)
            tnorm_y = stats.truncnorm(-2, 2, loc=0, scale=ay / 2)
            tnorm_x = stats.truncnorm(-2, 2, loc=0, scale=ax / 2)
            pd = tnorm_y.rvs(1)
            hl = tnorm_x.rvs(1)
            hr = tnorm_x.rvs(1)
            pts2 += np.array([[hl, pd], [hl, -pd], [hr, pd], [hr, -pd]]).squeeze()

        if cfg["scaling"]:
            mu, sigma = 1, cfg["scaling_amplitude"] / 2
            lower, upper = mu - 2 * sigma, mu + 2 * sigma
            tnorm_s = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma,
                                      loc=mu, scale=sigma)
            scales = tnorm_s.rvs(cfg["n_scales"])
            scales = np.concatenate(([1], scales))
            center = np.mean(pts2, axis=0, keepdims=True)
            scaled = (pts2 - center)[None] * scales[:, None, None] + center
            valid = (scaled >= 0.) * (scaled < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
            if valid.size > 0:
                idx = valid[np.random.randint(valid.size, size=1)].squeeze().astype(int)
                pts2 = scaled[idx]

        if cfg["translation"]:
            t_min = np.min(pts2, axis=0)
            t_max = np.min(1 - pts2, axis=0)
            pts2 += np.array([uniform(-t_min[0], t_max[0], 1),
                              uniform(-t_min[1], t_max[1], 1)]).T

        if cfg["rotation"]:
            angles = np.linspace(-cfg["max_angle"], cfg["max_angle"], num=cfg["n_angles"])
            angles = np.concatenate(([0.], angles))
            center = np.mean(pts2, axis=0, keepdims=True)
            rot_mat = np.stack([np.cos(angles), -np.sin(angles),
                                np.sin(angles),  np.cos(angles)], axis=1).reshape(-1, 2, 2)
            rotated = np.matmul((pts2 - center)[None], rot_mat) + center
            valid = (rotated >= 0.) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
            if valid.size > 0:
                idx = valid[np.random.randint(valid.size, size=1)].squeeze().astype(int)
                pts2 = rotated[idx]

        s = np.array([shape[1], shape[0]], dtype=np.float64)
        pts1_px = (pts1 * s[None]).astype(np.float32)
        pts2_px = (pts2 * s[None]).astype(np.float32)
        H = cv2.getPerspectiveTransform(pts1_px, pts2_px)
        H = np.linalg.inv(H)
        return H.astype(np.float64)
    finally:
        np.random.set_state(np_state)


def warp_and_cache(src_path: str | Path, H: np.ndarray, cache_path: Path,
                   target_shape: tuple[int, int] = (480, 640),
                   color_depth_npy: bool = False) -> Path:
    cache_path = Path(cache_path)
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if color_depth_npy:
        depth = np.load(src_path)
        finite = depth[np.isfinite(depth)]
        if finite.size == 0:
            depth_norm = np.zeros_like(depth, dtype=np.uint8)
        else:
            lo, hi = float(finite.min()), float(finite.max())
            denom = hi - lo if hi > lo else 1.0
            depth_norm = np.clip(((depth - lo) / denom) * 255.0, 0, 255).astype(np.uint8)
            depth_norm = depth_norm.squeeze()
        img = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    else:
        img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cv2 could not read {src_path}")

    img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    warped = cv2.warpPerspective(img, H, (target_shape[1], target_shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite(str(cache_path), warped)
    return cache_path


def load_resized_to_disk(src_path: str | Path, cache_path: Path,
                         target_shape: tuple[int, int] = (480, 640)) -> Path:
    cache_path = Path(cache_path)
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cv2 could not read {src_path}")
    img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(cache_path), img)
    return cache_path
