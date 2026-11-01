"""WxBS — Wide multi-Baseline Stereo (Mishkin et al.).

32 test pairs with simultaneous appearance, illumination, sensor and geometry
change, each with hand-annotated correspondences. Scored with the reference
protocol (F from matches via MAGSAC, mAA@10px of GT correspondences under the
fitted F), as reported by RoMa, RoMa v2 and LoMa.

Source:  https://huggingface.co/datasets/vrg-prague/wxbs (ungated; parquet)
Paper:   Mishkin et al., 'WxBS: Wide Baseline Stereo Generalizations', BMVC 2015.
License: Academic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .._download import download_hf
from .._registry import DatasetSpec, register
from .._rotation import rotate_points, rotated_image

_HF_REPO = "vrg-prague/wxbs"
_TEST_PARQUET = Path("data") / "test-00000-of-00001.parquet"
_CACHE_NAME = "_cache_images"


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if (data_root / _TEST_PARQUET).exists():
        return data_root
    return download_hf(_HF_REPO, data_root, repo_type="dataset", allow_patterns=["data/*"])


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               rotate_b: bool = False) -> Iterator[dict]:
    data_root = Path(data_root)
    test_pq = data_root / _TEST_PARQUET
    if not test_pq.exists():
        raise FileNotFoundError(f"WxBS parquet not found at {test_pq}; run `visbench download wxbs`.")
    import pyarrow.parquet as pq  # noqa: PLC0415  (lazy: pyarrow only needed when data exists)
    cache_dir = data_root / _CACHE_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(str(test_pq)).to_pylist()
    for count, (i, row) in enumerate(enumerate(rows)):
        if max_pairs is not None and count >= max_pairs:
            return
        a = cache_dir / f"{i:04d}_A.png"
        b = cache_dir / f"{i:04d}_B.png"
        if not a.exists():
            a.write_bytes(row["img1"]["bytes"])
        if not b.exists():
            b.write_bytes(row["img2"]["bytes"])
        raw = row["corrs"]
        corrs = (np.zeros((0, 4), dtype=np.float32) if not raw
                 else np.stack([np.asarray(r, dtype=np.float32) for r in raw], axis=0))
        if rotate_b and len(corrs):
            b, W, _ = rotated_image(b, data_root / "_cache_rot90", Path(b.name))
            corrs = np.c_[corrs[:, :2], rotate_points(corrs[:, 2:4], W)]
        yield {
            "im_A_path": str(a),
            "im_B_path": str(b),
            "corrs": corrs,
            "scene": row.get("name", f"wxbs_{i}"),
            "subset": row.get("set", "all"),
            "pair_name": row.get("pair", "") or "",
        }


register("wxbs", DatasetSpec(
    task="correspondence", iter_pairs=iter_pairs, default_root=Path("wxbs"),
    default_resize=1024, download=download,
    source_url=f"https://huggingface.co/datasets/{_HF_REPO}",
    license="Academic",
))
register("wxbs_rot90", DatasetSpec(
    task="correspondence", iter_pairs=iter_pairs, default_root=Path("wxbs"),
    default_resize=1024, kwargs={"rotate_b": True}, download=download,
    source_url=f"https://huggingface.co/datasets/{_HF_REPO}",
    license="Academic",
))
