"""HardMatch — hand-annotated hard matching pairs (released with LoMa, 2026).

1000 Wikimedia Commons pairs across 100 categories (aerial-ground, a century
apart, drawings, day-night, seasons, extreme viewpoints, ...). The test split
is 90 of the 100 categories (900 pairs); ``hardmatch_val`` is the other 10.

Protocol (reference evaluator, ``pip install hardmatch`` v1.0.0): matches on
the released ``A_lr.jpg`` / ``B_lr.jpg`` images; F via USAC_MAGSAC at 0.25 px;
PCK of the hand-annotated correspondences under the fitted F for 0..19 px;
headline mAA@10px. Pairs with fewer than 8 annotations or a ``dynamic``
category are excluded, leaving 830 scored test pairs. The paper (LoMa, Table 2)
computes errors at 640x640 and predates a licensing update of about 10 pairs,
so released-data numbers differ slightly from the paper.

``hardmatch_rot90`` rotates image B by 90 degrees (counter-clockwise) and maps
its annotations accordingly; it measures rotation robustness on the same pairs.

Source:  https://github.com/davnords/HardMatch
Paper:   Nordström, Edstedt et al., 'LoMa: Local Feature Matching Revisited', arXiv 2604.04931.
License: images from Wikimedia Commons under their individual licenses; annotations per repository.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import numpy as np

from ._download import download_url, extract_archive
from ._registry import DatasetSpec, register
from ._rotation import rotate_points, rotated_image

_ZIP_URL = "https://github.com/davnords/storage/releases/download/hardmatch/hardmatch.zip"
_SPLIT_SEED = 1337
_N_CATEGORIES = 100
_N_VAL_CATEGORIES = 10
MIN_ANNOTATIONS = 8


def _has_layout(root: Path) -> bool:
    return root.is_dir() and any(root.rglob("corresp_lr.txt"))


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if _has_layout(data_root):
        return data_root
    # The archive holds a top-level `hardmatch/` folder; extract next to data_root.
    data_root.parent.mkdir(parents=True, exist_ok=True)
    zip_path = data_root.parent / "hardmatch.zip"
    if not zip_path.exists():
        download_url(_ZIP_URL, zip_path)
    extract_archive(zip_path, data_root.parent)
    return data_root


def _pair_dirs(root: Path) -> list[Path]:
    """Pair folders exactly as the reference HardMatch dataset collects them."""
    out = []
    for corresp in root.rglob("corresp.txt"):
        d = corresp.parent
        if ((d / "A.jpg").exists() and (d / "B.jpg").exists()) or (
            (d / "A_lr.jpg").exists() and (d / "B_lr.jpg").exists()
        ):
            out.append(d)
    return sorted(out)


def split_pairs(root: Path, split: str) -> list[Path]:
    """Reference split: 100 categories shuffled with Random(1337); the first 10 are val."""
    pairs = _pair_dirs(root)
    categories = sorted({p.parent.parent for p in pairs})
    if len(categories) != _N_CATEGORIES:
        raise ValueError(f"HardMatch: expected {_N_CATEGORIES} categories under {root}, found {len(categories)}")
    shuffled = categories.copy()
    random.Random(_SPLIT_SEED).shuffle(shuffled)
    chosen = set(shuffled[:_N_VAL_CATEGORIES] if split == "val" else shuffled[_N_VAL_CATEGORIES:])
    return [p for p in pairs if p.parent.parent in chosen]


def _read_corresp(path: Path) -> np.ndarray:
    text = path.read_text().strip()
    if not text:
        return np.zeros((0, 4), dtype=np.float64)
    return np.asarray(text.split(), dtype=np.float64).reshape(-1, 4)


def _categories(pair_dir: Path) -> list[str]:
    f = pair_dir / "categories.txt"
    if not f.exists():
        return ["unknown"]
    return [ln.strip() for ln in f.read_text().splitlines() if ln.strip()] or ["unknown"]


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               split: str = "test", rotate_b: bool = False) -> Iterator[dict]:
    data_root = Path(data_root)
    if not _has_layout(data_root):
        raise FileNotFoundError(f"HardMatch not found at {data_root}; run `visbench download hardmatch`.")
    count = 0
    for pair_dir in split_pairs(data_root, split):
        corrs = _read_corresp(pair_dir / "corresp_lr.txt")
        cats = _categories(pair_dir)
        if len(corrs) < MIN_ANNOTATIONS or "dynamic" in cats:
            continue
        im_B = pair_dir / "B_lr.jpg"
        if rotate_b:
            rel = pair_dir.relative_to(data_root) / "B_lr.jpg"
            im_B, W, _ = rotated_image(im_B, data_root / "_cache_rot90", rel)
            corrs = np.c_[corrs[:, :2], rotate_points(corrs[:, 2:4], W)]
        yield {
            "im_A_path": str(pair_dir / "A_lr.jpg"),
            "im_B_path": str(im_B),
            "corrs": corrs,
            "scene": pair_dir.parent.parent.name,
            "subset": "all",
            "pair_id": pair_dir.relative_to(data_root).as_posix(),
        }
        count += 1
        if max_pairs is not None and count >= max_pairs:
            return


_KW = dict(
    download=download,
    source_url="https://github.com/davnords/HardMatch",
    license="Wikimedia Commons (per image); annotations per repository",
)
register("hardmatch", DatasetSpec("correspondence", iter_pairs, Path("hardmatch"), **_KW))
register("hardmatch_val", DatasetSpec("correspondence", iter_pairs, Path("hardmatch"),
                                      kwargs={"split": "val"}, **_KW))
register("hardmatch_rot90", DatasetSpec("correspondence", iter_pairs, Path("hardmatch"),
                                        kwargs={"rotate_b": True}, **_KW))
