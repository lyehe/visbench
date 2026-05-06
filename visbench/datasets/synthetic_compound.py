"""Compound stressors — multiple perturbations stacked.

Source:  derived from `hpatches`
License: inherits HPatches (CC BY-NC-SA 4.0).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ._registry import DatasetSpec, register
from .hpatches_rotated import _rotation_homography
from .synthetic import _base_pairs, _gauss_blur, _gauss_noise, _jpeg, download as _download_hpatches
from .synthetic_advanced import (
    _chromatic_aberration,
    _defocus,
    _fog,
    _low_light,
    _motion_blur,
)


def download(data_root: Path | str) -> Path:
    return _download_hpatches(data_root)


def _compound_iter(name: str, rotation_deg: float, photometric):
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
            if rotation_deg != 0:
                H_rot, (nw, nh) = _rotation_homography(rotation_deg, w, h)
                rotated = cv2.warpPerspective(arr, H_rot, (nw, nh), borderValue=(0, 0, 0))
            else:
                H_rot = np.eye(3)
                rotated = arr
            perturbed = photometric(rotated)
            if not out_path.exists():
                cv2.imwrite(str(out_path), perturbed)
            H_gt = H_rot @ pair["H_gt"]
            yield {
                **pair,
                "im_B_path": str(out_path),
                "H_gt": H_gt,
                "subset": f"{pair['subset']}_{name}",
            }
    return it


_combos = {
    "rot90_noise15":     (90.0, _gauss_noise(15)),
    "rot90_jpeg30":      (90.0, _jpeg(30)),
    "rot90_blur7":       (90.0, _gauss_blur(7)),
    "rot90_low_light":   (90.0, _low_light()),
    "rot90_fog":         (90.0, _fog()),
    "rot45_motion_blur": (45.0, _motion_blur(9)),
    "noise15_jpeg30":    (0.0,  lambda arr: _jpeg(30)(_gauss_noise(15)(arr))),
    "blur7_noise15":     (0.0,  lambda arr: _gauss_noise(15)(_gauss_blur(7)(arr))),
    "low_light_blur7":   (0.0,  lambda arr: _gauss_blur(7)(_low_light()(arr))),
    "fog_chromatic":     (0.0,  lambda arr: _chromatic_aberration()(_fog()(arr))),
    "defocus_low_light": (0.0,  lambda arr: _low_light()(_defocus(5)(arr))),
}

_BASE_KW = dict(
    download=download,
    source_url="https://github.com/hpatches/hpatches-dataset",
    license="CC BY-NC-SA 4.0",
)

for _name, (_angle, _fn) in _combos.items():
    register(f"synthetic_{_name}", DatasetSpec(
        "homography", _compound_iter(_name, _angle, _fn), Path("hpatches"), **_BASE_KW))
