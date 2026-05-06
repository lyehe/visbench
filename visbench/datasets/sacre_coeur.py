"""Sacre Coeur — hloc demo dataset (10 tourism images, qualitative only).

Source:  https://github.com/cvg/Hierarchical-Localization (datasets/sacre_coeur/mapping)
License: Apache 2.0 (hloc); originally Yahoo Flickr Creative Commons 100M (YFCC).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ._download import download_url
from ._registry import DatasetSpec, register

_GH_API = "https://api.github.com/repos/cvg/Hierarchical-Localization/contents/datasets/sacre_coeur/mapping"


def download(data_root: Path | str) -> Path:
    """Fetch the 10 mapping images individually from the hloc repo."""
    import json
    import urllib.request

    data_root = Path(data_root)
    mapping = data_root / "mapping"
    if mapping.exists() and any(mapping.glob("*.jpg")):
        return data_root
    mapping.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(_GH_API, headers={"User-Agent": "visbench/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        listing = json.loads(r.read())

    for entry in listing:
        if entry.get("type") != "file":
            continue
        name = entry["name"]
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        download_url(entry["download_url"], mapping / name)
    return data_root


def iter_pairs(data_root: Path | str, max_pairs: int | None = None) -> Iterator[dict]:
    data_root = Path(data_root)
    img_dir = data_root / "mapping"
    frames = sorted(img_dir.glob("*.jpg"))
    count = 0
    for i, a in enumerate(frames):
        for j in range(i + 1, len(frames)):
            b = frames[j]
            yield {
                "im_A_path": str(a),
                "im_B_path": str(b),
                "scene": "sacre_coeur",
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


register("sacre_coeur", DatasetSpec(
    task="qualitative",
    iter_pairs=iter_pairs,
    default_root=Path("sacre_coeur"),
    default_resize=1024,
    download=download,
    source_url="https://github.com/cvg/Hierarchical-Localization/tree/master/datasets/sacre_coeur",
    license="Apache 2.0 (hloc)",
))
