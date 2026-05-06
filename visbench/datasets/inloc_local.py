"""InLoc — query-only frame-skip pairs (qualitative).

Source:  https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/InLoc/
License: Academic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ._registry import DatasetSpec, register


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if (data_root / "queries" / "iphone7").exists():
        return data_root
    raise NotImplementedError(
        "InLoc queries: download iphone7.tar.gz from\n"
        "https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/InLoc/iphone7.tar.gz\n"
        f"and extract under {data_root}/queries/iphone7/."
    )


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               pair_skip: int = 5) -> Iterator[dict]:
    data_root = Path(data_root)
    q_dir = data_root / "queries" / "iphone7"
    if not q_dir.exists():
        return
    frames = sorted(q_dir.glob("*.JPG")) or sorted(q_dir.glob("*.jpg"))
    count = 0
    for i in range(0, len(frames) - pair_skip, max(1, pair_skip // 2)):
        j = i + pair_skip
        if j >= len(frames):
            break
        yield {
            "im_A_path": str(frames[i]),
            "im_B_path": str(frames[j]),
            "scene": "inloc_iphone7",
        }
        count += 1
        if max_pairs is not None and count >= max_pairs:
            return


_KW = dict(
    download=download,
    source_url="https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/InLoc/",
    license="Academic",
)

register("inloc_queries", DatasetSpec("qualitative", iter_pairs, Path("inloc"),
                                      default_resize=1024, kwargs={"pair_skip": 5}, **_KW))
register("inloc_queries_wide", DatasetSpec("qualitative", iter_pairs, Path("inloc"),
                                           default_resize=1024, kwargs={"pair_skip": 15}, **_KW))
