"""InLoc — pairwise subset (qualitative).

Source:  https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/InLoc/
Pair lists: https://github.com/cvg/Hierarchical-Localization (pairs/inloc/)
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
        "InLoc: download images from CIIRC and pair-list from cvg/Hierarchical-Localization;\n"
        f"place under {data_root}."
    )


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               pair_list_name: str = "pairs-query-netvlad40-temporal.txt") -> Iterator[dict]:
    data_root = Path(data_root)
    pair_file = data_root / pair_list_name
    if not pair_file.exists():
        raise FileNotFoundError(f"InLoc pair list missing: {pair_file}")
    count = 0
    for line in pair_file.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        yield {
            "im_A_path": str(data_root / parts[0]),
            "im_B_path": str(data_root / parts[1]),
            "subset": "inloc",
        }
        count += 1
        if max_pairs is not None and count >= max_pairs:
            return


register("inloc", DatasetSpec(
    task="qualitative", iter_pairs=iter_pairs, default_root=Path("inloc"),
    download=download,
    source_url="https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/InLoc/",
    license="Academic",
))
