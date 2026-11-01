"""LoMa through its own reference matching path. Use ``loma:<size>`` (b128, b, l, g, r).

``vismatch:loma-*`` feeds LoMa one pre-resized tensor for both stages. LoMa's
own evaluation (``eval.py`` in davnords/LoMa) instead passes image paths: the
DaD detector resizes to 1024 px and the DeDoDe descriptor to 784x784, the
resolutions it was trained at, and matches come back in original-image
pixels. This method calls that path API directly, at the paper's 4096
keypoints and LoMa's default filter threshold, so numbers are comparable to
the LoMa paper. The harness hands it file paths instead of resized images
(``consumes_paths``) and leaves coordinates in the original frame.

GPU only: the reference path runs in bfloat16 mixed precision, which fails on CPU.

Needs vismatch's LoMa submodule:
``git -C third_party/vismatch submodule update --init vismatch/third_party/LoMa``.
"""

from __future__ import annotations

import numpy as np

SIZES = {"b128": "LoMa-B128", "b": "LoMa-B", "l": "LoMa-L", "g": "LoMa-G", "r": "LoMa-R"}
NUM_KEYPOINTS = 4096


class LoMaOfficialMatcher:
    consumes_paths = True

    def __init__(self, size: str, device: str = "cuda", model=None):
        size = size.lower()
        if size not in SIZES:
            raise ValueError(f"loma:<size> takes one of {sorted(SIZES)}, got '{size}'")
        self.size = size
        if model is None:
            from vismatch.im_models.loma import LoMaMatcher

            model = LoMaMatcher(device, NUM_KEYPOINTS, arch=SIZES[size]).matcher
        self.model = model

    @staticmethod
    def load_image(path, resize=None):
        return str(path)

    def __call__(self, path0: str, path1: str) -> dict:
        k0, k1 = self.model.match(str(path0), str(path1), num_keypoints=NUM_KEYPOINTS)
        k0 = np.asarray(k0, dtype=np.float64).reshape(-1, 2)
        k1 = np.asarray(k1, dtype=np.float64).reshape(-1, 2)
        return {"matched_kpts0": k0, "matched_kpts1": k1,
                "all_kpts0": k0, "all_kpts1": k1, "num_inliers": len(k0)}


def build_loma_official(name: str, device: str = "cuda") -> LoMaOfficialMatcher:
    return LoMaOfficialMatcher(name.split(":", 1)[1], device=device)
