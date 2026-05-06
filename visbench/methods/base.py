"""Plugin base class for handcrafted / custom feature experiments.

Subclass `CustomFeatureMatcher` and implement `detect_and_describe(img_u8)`.
Mutual-NN matching with optional Lowe ratio test (sparse handcrafted baseline).
"""

from __future__ import annotations

import numpy as np
import torch

from vismatch.base_matcher import BaseMatcher


def _to_numpy_uint8(img: torch.Tensor) -> np.ndarray:
    """(C,H,W) float tensor in [0,1] -> (H,W,3) uint8 RGB numpy."""
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def _l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    aa = (a * a).sum(1, keepdims=True)
    bb = (b * b).sum(1, keepdims=True).T
    d2 = aa + bb - 2.0 * (a @ b.T)
    return np.sqrt(np.clip(d2, 0, None))


def _hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.unpackbits(a.astype(np.uint8), axis=1)
    b = np.unpackbits(b.astype(np.uint8), axis=1)
    return (a[:, None, :] != b[None, :, :]).sum(axis=2).astype(np.float32)


def mutual_nearest_neighbor(desc0: np.ndarray, desc1: np.ndarray,
                            ratio: float | None = 0.8,
                            metric: str = "l2") -> np.ndarray:
    if len(desc0) == 0 or len(desc1) == 0:
        return np.empty((0, 2), dtype=np.int64)
    d01 = _hamming(desc0, desc1) if metric == "hamming" else _l2(desc0, desc1)
    nn01 = d01.argmin(axis=1)
    nn10 = d01.argmin(axis=0)
    idx0 = np.arange(len(desc0))
    mutual = nn10[nn01] == idx0
    if ratio is not None and d01.shape[1] >= 2:
        part = np.partition(d01, 2, axis=1)[:, :2]
        ratio_ok = part[:, 0] < ratio * np.maximum(part[:, 1], 1e-12)
        mutual = mutual & ratio_ok
    i0 = idx0[mutual]
    i1 = nn01[mutual]
    return np.stack([i0, i1], axis=1).astype(np.int64)


class CustomFeatureMatcher(BaseMatcher):
    """Subclass and override `detect_and_describe`.

    Returns (kpts: (N,2) float xy, desc: (N,D) float or (N,D8) uint8 packed).
    """

    descriptor_metric: str = "l2"
    ratio_test: float | None = 0.8

    def __init__(self, device: str = "cpu", **kwargs):
        super().__init__(device=device, **kwargs)

    def detect_and_describe(self, img_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _forward(self, img0: torch.Tensor, img1: torch.Tensor):
        u0 = _to_numpy_uint8(img0)
        u1 = _to_numpy_uint8(img1)
        k0, d0 = self.detect_and_describe(u0)
        k1, d1 = self.detect_and_describe(u1)
        if d0 is None or d1 is None or len(k0) == 0 or len(k1) == 0:
            empty = np.empty((0, 2))
            return empty, empty, k0 if len(k0) else empty, k1 if len(k1) else empty, \
                   np.empty((0, 2)), np.empty((0, 2))
        pairs = mutual_nearest_neighbor(d0, d1, ratio=self.ratio_test, metric=self.descriptor_metric)
        mk0 = k0[pairs[:, 0]] if len(pairs) else np.empty((0, 2))
        mk1 = k1[pairs[:, 1]] if len(pairs) else np.empty((0, 2))
        return mk0, mk1, k0, k1, d0, d1
