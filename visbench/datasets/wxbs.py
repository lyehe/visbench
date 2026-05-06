"""WxBS — Wide multi-Baseline Stereo (Mishkin et al.).

32 brutal pairs with simultaneous occlusion/weather/sensor/geometry change.

Source:  https://cmp.felk.cvut.cz/wbs/ + https://github.com/ducha-aiki/wxbs-descriptors-benchmark
HF mirror (wxbs_hf): https://huggingface.co/datasets/vrg-prague/wxbs
Paper:   Mishkin et al., 'WxBS: Wide Baseline Stereo Generalizations', BMVC 2015.
License: Academic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ._download import download_hf
from ._registry import DatasetSpec, register


_HF_REPO = "vrg-prague/wxbs"


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if (data_root / "pairs").exists():
        return data_root
    raise NotImplementedError(
        "WxBS: clone https://github.com/ducha-aiki/wxbs-descriptors-benchmark and place\n"
        f"its `pairs/` tree under {data_root}/pairs/.\n"
        "Or use the `wxbs_hf` dataset which auto-downloads from HuggingFace."
    )


def iter_pairs(data_root: Path | str, max_pairs: int | None = None) -> Iterator[dict]:
    data_root = Path(data_root)
    pairs_root = data_root / "pairs"
    if not pairs_root.exists():
        raise FileNotFoundError(f"WxBS pairs not found at {pairs_root}.")

    count = 0
    for set_dir in sorted(pairs_root.iterdir()):
        if not set_dir.is_dir():
            continue
        imgs = sorted([p for p in set_dir.iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".ppm")])
        if len(imgs) < 2:
            continue
        F_path = None
        for name in ("F.txt", "fundamental.txt", "F_matrix.txt"):
            if (set_dir / name).exists():
                F_path = set_dir / name
                break
        if F_path is None:
            continue
        F = np.loadtxt(F_path)
        yield {
            "im_A_path": str(imgs[0]),
            "im_B_path": str(imgs[1]),
            "F_gt": F,
            "scene": set_dir.name,
            "subset": "wxbs",
        }
        count += 1
        if max_pairs is not None and count >= max_pairs:
            return


register("wxbs", DatasetSpec(
    task="fundamental", iter_pairs=iter_pairs, default_root=Path("wxbs"),
    download=download,
    source_url="https://cmp.felk.cvut.cz/wbs/",
    license="Academic",
))


# --- HuggingFace variant: vrg-prague/wxbs parquet format ---
_CACHE_NAME = "_cache_images"


def download_hf_wxbs(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    test_pq = data_root / "data" / "test-00000-of-00001.parquet"
    if test_pq.exists():
        return data_root
    return download_hf(_HF_REPO, data_root, repo_type="dataset", allow_patterns=["data/*"])


def iter_pairs_hf(data_root: Path | str, max_pairs: int | None = None) -> Iterator[dict]:
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
        raw = row["corrs"]
        corrs = (np.zeros((0, 4), dtype=np.float32) if not raw
                 else np.stack([np.asarray(r, dtype=np.float32) for r in raw], axis=0))
        errors = np.array(row["errors"], dtype=np.float32)
        subset = row.get("set", "all")
        yield {
            "im_A_path": str(a),
            "im_B_path": str(b),
            "corrs": corrs,
            "errors": errors,
            "scene": row.get("name", f"wxbs_{i}"),
            "subset": subset,
            "pair_name": row.get("pair", "") or "",
        }
        count += 1
        if max_pairs is not None and count >= max_pairs:
            return


register("wxbs_hf", DatasetSpec(
    task="correspondence", iter_pairs=iter_pairs_hf, default_root=Path("wxbs"),
    default_resize=1024, download=download_hf_wxbs,
    source_url=f"https://huggingface.co/datasets/{_HF_REPO}",
    license="Academic",
))
