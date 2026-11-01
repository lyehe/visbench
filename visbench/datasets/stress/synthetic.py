"""Synthetic perturbation pipelines over HPatches.

For each base illum+view pair, apply a controlled perturbation to image B and
compose with the GT homography where appropriate. Cached under
`<hpatches_root>/_synthetic_<name>/`.

Source:  derived from `hpatches`
License: inherits HPatches (CC BY-NC-SA 4.0).
"""

from __future__ import annotations

from pathlib import Path

import cv2

from .._registry import DatasetSpec, register
from ._synthetic_common import (
    base_pairs,
    gauss_blur,
    gauss_noise,
    jpeg,
    make_iter,
)
from ..homography.hpatches import download as _download_hpatches

# Legacy underscore aliases — external code that imported from this module
# (e.g. synthetic_compound previously did `from .synthetic import _jpeg`)
# continues to work, but new code should import from `_synthetic_common`.
_base_pairs = base_pairs
_make_iter = make_iter
_jpeg = jpeg
_gauss_noise = gauss_noise
_gauss_blur = gauss_blur


def download(data_root: Path | str) -> Path:
    return _download_hpatches(data_root)


def _scale(s: float):
    def f(arr):
        h, w = arr.shape[:2]
        return cv2.resize(arr, (max(1, int(w * s)), max(1, int(h * s))),
                          interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return f


_BASE_KW = dict(
    download=download,
    source_url="https://github.com/hpatches/hpatches-dataset",
    license="CC BY-NC-SA 4.0",
)

# Compact stress set: one strong setting per degradation family. The wider
# sweeps (other JPEG/noise/blur levels, gamma, HSV, salt-and-pepper, 2x scale)
# were removed as low-signal: each variant costs a full HPatches pass.
register("synthetic_jpeg_q10", DatasetSpec(
    "homography", _make_iter("jpeg_q10", _jpeg(10)), Path("hpatches"), **_BASE_KW))
register("synthetic_noise_30", DatasetSpec(
    "homography", _make_iter("noise_30", _gauss_noise(30.0)), Path("hpatches"), **_BASE_KW))
register("synthetic_scale_0p5", DatasetSpec(
    "homography",
    _make_iter("scale_0p5", _scale(0.5), scale_xy=(0.5, 0.5)),
    Path("hpatches"), **_BASE_KW))
