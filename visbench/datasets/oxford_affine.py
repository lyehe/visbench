"""Oxford-Affine — Mikolajczyk-Schmid 2005 affine-invariant feature benchmark.

8 sequences x 6 images each (img1.ppm = anchor, img2..img6 = transformed)
with exact ground-truth homographies H1to{2..6}p. 40 pairs total. Each
sequence isolates ONE transformation type:

    bark      scale + rotation
    bikes     blur (out-of-focus)
    boat      scale + rotation
    graf      viewpoint (planar, large angle)
    leuven    illumination
    trees     blur (natural)
    ubc       JPEG compression
    wall      viewpoint (planar)

Source:  https://www.robots.ox.ac.uk/~vgg/research/affine/
Paper:   Mikolajczyk & Schmid, 'A performance evaluation of local descriptors',
         IEEE TPAMI 2005.
License: Academic / research use (per VGG release page).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ._registry import DatasetSpec, register
from ._download import download_url, extract_archive


_SEQUENCES = ("bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall")
_BASE_URL = "https://thor.robots.ox.ac.uk/affine/"


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    for seq in _SEQUENCES:
        seq_dir = data_root / seq
        if seq_dir.exists() and any(seq_dir.iterdir()):
            continue
        url = f"{_BASE_URL}{seq}.tar.gz"
        archive = data_root / f"{seq}.tar.gz"
        download_url(url, archive)
        seq_dir.mkdir(parents=True, exist_ok=True)
        extract_archive(archive, seq_dir)
        try:
            archive.unlink()
        except OSError:
            pass
    return data_root


def _find_image(seq_dir: Path, idx: int) -> Path | None:
    for ext in ("ppm", "pgm", "png", "jpg"):
        p = seq_dir / f"img{idx}.{ext}"
        if p.exists():
            return p
        for sub in seq_dir.iterdir():
            if sub.is_dir():
                p2 = sub / f"img{idx}.{ext}"
                if p2.exists():
                    return p2
    return None


def _find_homography(seq_dir: Path, idx: int) -> Path | None:
    for cand in (seq_dir / f"H1to{idx}p", seq_dir / f"H1to{idx}.txt"):
        if cand.exists():
            return cand
    for sub in seq_dir.iterdir():
        if sub.is_dir():
            for cand in (sub / f"H1to{idx}p", sub / f"H1to{idx}.txt"):
                if cand.exists():
                    return cand
    return None


def iter_pairs(data_root: Path | str, max_pairs: int | None = None) -> Iterator[dict]:
    data_root = Path(data_root)
    if not data_root.exists() or not any(data_root.iterdir()):
        return  # caller should run `visbench download oxford_affine` first

    count = 0
    for seq in _SEQUENCES:
        seq_dir = data_root / seq
        if not seq_dir.exists():
            continue
        anchor = _find_image(seq_dir, 1)
        if anchor is None:
            continue
        for idx in range(2, 7):
            target = _find_image(seq_dir, idx)
            H_path = _find_homography(seq_dir, idx)
            if target is None or H_path is None:
                continue
            try:
                H_gt = np.loadtxt(H_path)
            except Exception:
                continue
            if H_gt.shape != (3, 3):
                continue
            yield {
                "im_A_path": str(anchor),
                "im_B_path": str(target),
                "H_gt": H_gt,
                "scene": seq,
                "subset": seq,
                "pair_idx": idx,
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


register("oxford_affine", DatasetSpec(
    task="homography",
    iter_pairs=iter_pairs,
    default_root=Path("oxford_affine"),
    default_resize=None,
    download=download,
    source_url="https://www.robots.ox.ac.uk/~vgg/research/affine/",
    license="Academic / research use",
))
