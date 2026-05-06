"""Aachen Day-Night v1.1 — pairwise subset (qualitative, no GT pose).

Source:  https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/
Pair lists: https://github.com/cvg/Hierarchical-Localization (pairs/aachen_v1.1/)
Paper:   Sattler et al., 'Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions', CVPR 2018.
License: Academic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ._registry import DatasetSpec, register


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if any(data_root.glob("pairs-*.txt")):
        return data_root
    raise NotImplementedError(
        "Aachen v1.1: download images_upright/ from CIIRC and pair lists from\n"
        "cvg/Hierarchical-Localization; place under {data_root}."
    )


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               pair_list_name: str = "pairs-query-night-netvlad50.txt") -> Iterator[dict]:
    data_root = Path(data_root)
    pair_file = data_root / pair_list_name
    img_root = data_root / "images" / "images_upright"
    if not pair_file.exists():
        raise FileNotFoundError(f"Aachen pair list missing: {pair_file}")
    count = 0
    for line in pair_file.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        qa, qb = parts[0], parts[1]
        yield {
            "im_A_path": str(img_root / qa),
            "im_B_path": str(img_root / qb),
            "subset": "aachen_night" if "night" in pair_list_name else "aachen_day",
        }
        count += 1
        if max_pairs is not None and count >= max_pairs:
            return


def _iter_day(data_root, max_pairs=None, **_):
    return iter_pairs(data_root, max_pairs, pair_list_name="pairs-query-day-netvlad50.txt")


_KW = dict(
    download=download,
    source_url="https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/",
    license="Academic",
)

register("aachen_night", DatasetSpec("qualitative", iter_pairs, Path("aachen_v1.1"), **_KW))
register("aachen_day", DatasetSpec("qualitative", _iter_day, Path("aachen_v1.1"), **_KW))
