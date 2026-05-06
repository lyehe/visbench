"""Resolution-sweep benchmarks — same HPatches pairs at different long-side resolutions.

Source:  derived from `hpatches`
License: inherits HPatches (CC BY-NC-SA 4.0).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ._registry import DatasetSpec, register
from .synthetic import _base_pairs, download as _download_hpatches


def download(data_root: Path | str) -> Path:
    return _download_hpatches(data_root)


def _resize_iter(name: str, long_side: int):
    def it(data_root, max_pairs=None, **_):
        data_root = Path(data_root)
        cache = data_root / f"_synthetic_{name}"
        cache.mkdir(exist_ok=True)
        for pair in _base_pairs(data_root, max_pairs):
            im_A = Path(pair["im_A_path"])
            im_B = Path(pair["im_B_path"])
            out_dir = cache / pair["seq"]
            out_dir.mkdir(exist_ok=True)
            out_a = out_dir / f"A_{pair['pair_idx']}.png"
            out_b = out_dir / f"B_{pair['pair_idx']}.png"
            arr_a = cv2.imread(str(im_A), cv2.IMREAD_COLOR)
            arr_b = cv2.imread(str(im_B), cv2.IMREAD_COLOR)
            ha, wa = arr_a.shape[:2]
            hb, wb = arr_b.shape[:2]
            sa = long_side / max(ha, wa)
            sb = long_side / max(hb, wb)
            if not out_a.exists():
                cv2.imwrite(str(out_a), cv2.resize(arr_a, (int(wa * sa), int(ha * sa)),
                                                   interpolation=cv2.INTER_AREA))
            if not out_b.exists():
                cv2.imwrite(str(out_b), cv2.resize(arr_b, (int(wb * sb), int(hb * sb)),
                                                   interpolation=cv2.INTER_AREA))
            Sa = np.diag([sa, sa, 1.0])
            Sb = np.diag([sb, sb, 1.0])
            H_gt = Sb @ pair["H_gt"] @ np.linalg.inv(Sa)
            yield {
                **pair,
                "im_A_path": str(out_a),
                "im_B_path": str(out_b),
                "H_gt": H_gt,
                "subset": f"{pair['subset']}_{name}",
            }
    return it


_BASE_KW = dict(
    download=download,
    source_url="https://github.com/hpatches/hpatches-dataset",
    license="CC BY-NC-SA 4.0",
)

for _s in (240, 360, 480, 720, 960, 1440, 1920):
    register(f"resolution_{_s}", DatasetSpec(
        "homography", _resize_iter(f"res_{_s}", _s), Path("hpatches"), **_BASE_KW))
