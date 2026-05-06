"""EVD — Extreme View Dataset (Mishkin), via HuggingFace vrg-prague/evd.

15 planar pairs with GT homographies. Tiny set — useful for homography
stress-testing under extreme viewpoint change.

Source:  https://huggingface.co/datasets/vrg-prague/evd
Paper:   Mishkin et al., 'MODS: Fast and Robust Method for Two-View Matching',
         CVIU 2015.
License: CC BY 4.0 (per HF dataset card).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ._registry import DatasetSpec, register
from ._download import download_hf

_HF_REPO = "vrg-prague/evd"
_CACHE_NAME = "_cache_images"


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    test_pq = data_root / "data" / "test-00000-of-00001.parquet"
    if test_pq.exists():
        return data_root
    return download_hf(_HF_REPO, data_root, repo_type="dataset",
                       allow_patterns=["data/*"])


def iter_pairs(data_root: Path | str, max_pairs: int | None = None) -> Iterator[dict]:
    import pyarrow.parquet as pq
    data_root = Path(data_root)
    test_pq = data_root / "data" / "test-00000-of-00001.parquet"
    if not test_pq.exists():
        return
    cache_dir = data_root / _CACHE_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(str(test_pq)).to_pylist()
    count = 0
    for i, row in enumerate(table):
        a = cache_dir / f"{i:04d}_A.png"
        b = cache_dir / f"{i:04d}_B.png"
        if not a.exists():
            a.write_bytes(row["img1"]["bytes"])
        if not b.exists():
            b.write_bytes(row["img2"]["bytes"])
        H = np.array(row["H"], dtype=np.float64)
        yield {
            "im_A_path": str(a),
            "im_B_path": str(b),
            "H_gt": H,
            "scene": row.get("name", f"evd_{i}"),
        }
        count += 1
        if max_pairs is not None and count >= max_pairs:
            return


register("evd", DatasetSpec(
    task="homography",
    iter_pairs=iter_pairs,
    default_root=Path("evd"),
    default_resize=1024,
    download=download,
    source_url=f"https://huggingface.co/datasets/{_HF_REPO}",
    license="CC BY 4.0",
))
