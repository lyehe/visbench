"""SILDa — Specular/Illumination Long-term Dataset (day-night urban).

Source:  https://github.com/abmmusa/silda
License: Academic (per source repo).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ._registry import DatasetSpec, register


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if (data_root / "pairs.txt").exists():
        return data_root
    raise NotImplementedError(
        "SILDa: download from https://github.com/abmmusa/silda and place under "
        f"{data_root} (must contain images/ and pairs.txt)."
    )


def iter_pairs(data_root: Path | str, max_pairs: int | None = None) -> Iterator[dict]:
    data_root = Path(data_root)
    pair_file = data_root / "pairs.txt"
    if not pair_file.exists():
        raise FileNotFoundError(f"SILDa pair list missing: {pair_file}")
    count = 0
    for line in pair_file.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        yield {
            "im_A_path": str(data_root / parts[0]),
            "im_B_path": str(data_root / parts[1]),
            "subset": "silda",
        }
        count += 1
        if max_pairs is not None and count >= max_pairs:
            return


register("silda", DatasetSpec(
    task="qualitative", iter_pairs=iter_pairs, default_root=Path("silda"),
    download=download,
    source_url="https://github.com/abmmusa/silda",
    license="Academic",
))
