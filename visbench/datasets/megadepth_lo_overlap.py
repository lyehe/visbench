"""MegaDepth-1500 low-overlap stressor — only the lowest-overlap bins.

Source:  derived from `megadepth1500`
License: inherits MegaDepth1500.
"""

from __future__ import annotations

from pathlib import Path

from ._registry import DatasetSpec, register
from .megadepth1500 import iter_pairs as _iter, download as _download


def iter_pairs(data_root, max_pairs=None, **_):
    return _iter(data_root, max_pairs=max_pairs,
                 scene_files=("0015_0.1_0.3.npz", "0022_0.1_0.3.npz"))


register("megadepth_lo_overlap", DatasetSpec(
    task="pose",
    iter_pairs=iter_pairs,
    default_root=Path("megadepth"),
    default_resize=1200,
    download=_download,
    source_url="https://github.com/zju3dv/LoFTR",
    license="Apache 2.0 (LoFTR); MegaDepth images per source",
))
