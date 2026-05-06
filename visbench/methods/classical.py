"""Reference handcrafted-feature plugins via OpenCV.

SIFT / RootSIFT / ORB / AKAZE / BRISK as drop-in `CustomFeatureMatcher`
subclasses — the baselines a learned method needs to beat.
"""

from __future__ import annotations

import cv2
import numpy as np

from ._registry import register
from .base import CustomFeatureMatcher


def _gray(img_u8: np.ndarray) -> np.ndarray:
    if img_u8.ndim == 3 and img_u8.shape[-1] == 3:
        return cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    return img_u8


class SIFTFeature(CustomFeatureMatcher):
    descriptor_metric = "l2"
    ratio_test = 0.8

    def __init__(self, max_kpts: int = 2048, contrast_threshold: float = 0.04,
                 edge_threshold: float = 10.0, device: str = "cpu", **kwargs):
        super().__init__(device=device, **kwargs)
        self._sift = cv2.SIFT_create(nfeatures=max_kpts,
                                     contrastThreshold=contrast_threshold,
                                     edgeThreshold=edge_threshold)

    def detect_and_describe(self, img_u8):
        g = _gray(img_u8)
        kp, desc = self._sift.detectAndCompute(g, None)
        if desc is None or len(kp) == 0:
            return np.empty((0, 2)), np.empty((0, 128), dtype=np.float32)
        pts = np.array([k.pt for k in kp], dtype=np.float32)
        return pts, desc.astype(np.float32)


class RootSIFTFeature(SIFTFeature):
    """SIFT with L1-normalize + sqrt (Arandjelovic & Zisserman 2012)."""

    def detect_and_describe(self, img_u8):
        pts, desc = super().detect_and_describe(img_u8)
        if len(desc):
            desc = desc / np.maximum(desc.sum(axis=1, keepdims=True), 1e-12)
            desc = np.sqrt(desc)
        return pts, desc


class ORBFeature(CustomFeatureMatcher):
    descriptor_metric = "hamming"
    ratio_test = None

    def __init__(self, max_kpts: int = 4096, scale_factor: float = 1.2,
                 n_levels: int = 8, device: str = "cpu", **kwargs):
        super().__init__(device=device, **kwargs)
        self._orb = cv2.ORB_create(nfeatures=max_kpts, scaleFactor=scale_factor, nlevels=n_levels)

    def detect_and_describe(self, img_u8):
        g = _gray(img_u8)
        kp, desc = self._orb.detectAndCompute(g, None)
        if desc is None or len(kp) == 0:
            return np.empty((0, 2)), np.empty((0, 32), dtype=np.uint8)
        pts = np.array([k.pt for k in kp], dtype=np.float32)
        return pts, desc


class AKAZEFeature(CustomFeatureMatcher):
    descriptor_metric = "hamming"
    ratio_test = None

    def __init__(self, threshold: float = 1e-3, device: str = "cpu", **kwargs):
        super().__init__(device=device, **kwargs)
        self._akaze = cv2.AKAZE_create(threshold=threshold)

    def detect_and_describe(self, img_u8):
        g = _gray(img_u8)
        kp, desc = self._akaze.detectAndCompute(g, None)
        if desc is None or len(kp) == 0:
            return np.empty((0, 2)), np.empty((0, 61), dtype=np.uint8)
        pts = np.array([k.pt for k in kp], dtype=np.float32)
        return pts, desc


class BRISKFeature(CustomFeatureMatcher):
    descriptor_metric = "hamming"
    ratio_test = None

    def __init__(self, threshold: int = 30, octaves: int = 3, device: str = "cpu", **kwargs):
        super().__init__(device=device, **kwargs)
        self._brisk = cv2.BRISK_create(thresh=threshold, octaves=octaves)

    def detect_and_describe(self, img_u8):
        g = _gray(img_u8)
        kp, desc = self._brisk.detectAndCompute(g, None)
        if desc is None or len(kp) == 0:
            return np.empty((0, 2)), np.empty((0, 64), dtype=np.uint8)
        pts = np.array([k.pt for k in kp], dtype=np.float32)
        return pts, desc


REGISTRY = {
    "sift": SIFTFeature,
    "rootsift": RootSIFTFeature,
    "orb": ORBFeature,
    "akaze": AKAZEFeature,
    "brisk": BRISKFeature,
}


def _make_builder(cls):
    def build(device: str = "cpu"):
        return cls(device=device)
    return build


for _name, _cls in REGISTRY.items():
    register(_name, _make_builder(_cls), kind="classical")
