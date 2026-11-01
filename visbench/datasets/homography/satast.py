"""SatAst — astronaut photo <-> satellite image matching (released with RoMa v2).

39 hand-annotated pairs of ISS astronaut photographs and satellite tiles
(from AIMS / EarthMatch), with large in-plane rotation and scale change.
Scored by the reference protocol: homography from the matches, pooled
reprojection error of the annotated points, AUC@10px headline
(RoMa v2 Table 9; LoMa supplement Table 9).

Layout (``data_root``)::

    annotations-v1/annotation_*.json   # from the satast repository
    data/<ISS id>/*.jpg                # images, fetched with download_images.sh

Download (public, no registration)::

    visbench download satast                     # annotations (GitHub archive)
    cd <data_root> && bash download_images.sh     # images, rsync://vandaldata.polito.it

Source:  https://github.com/georg-bn/satast
Paper:   Edstedt et al., 'RoMa v2: Harder Better Faster Denser Feature Matching', arXiv 2511.15706.
License: images per AIMS / EarthMatch; annotations per repository.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np

from .._download import download_url, extract_archive
from .._registry import DatasetSpec, register

_REPO = "https://github.com/georg-bn/satast"
_ARCHIVE = "https://github.com/georg-bn/satast/archive/refs/heads/main.zip"
_ANNOTATIONS = "annotations-v1"
_FILE_LIST = "included_images.txt"
_RSYNC_SOURCE = "rsync://vandaldata.polito.it/sf_xl/EarthMatch/data"


def _images_present(root: Path) -> bool:
    return (root / "data").is_dir() and any((root / "data").rglob("*.jpg"))


def download(data_root: Path | str) -> Path:
    """Fetch the annotations (GitHub archive); the images need rsync (download_images.sh)."""
    data_root = Path(data_root)
    if not (data_root / _ANNOTATIONS).is_dir():
        data_root.mkdir(parents=True, exist_ok=True)
        archive = data_root / "satast-main.zip"
        download_url(_ARCHIVE, archive)
        extract_archive(archive, data_root)
        inner = data_root / "satast-main"
        if inner.is_dir():
            for item in inner.iterdir():
                shutil.move(str(item), str(data_root / item.name))
            inner.rmdir()
        if archive.exists():
            archive.unlink()
    if not _images_present(data_root):
        rsync = shutil.which("rsync")
        if rsync is None or not (data_root / _FILE_LIST).exists():
            raise NotImplementedError(
                "SatAst: the images are served only over rsync (no registration).\n"
                "Install rsync and put it on PATH, then rerun `visbench download satast`."
            )
        # Same call as the repository's download_images.sh, run directly: on
        # Windows a bare `bash` can resolve to the WSL launcher.
        subprocess.run([rsync, "-rhz", "--info=progress2", "--ignore-existing",
                        f"--files-from={_FILE_LIST}", _RSYNC_SOURCE, "./data"],
                       cwd=str(data_root), check=True)
    return data_root


def iter_pairs(data_root: Path | str, max_pairs: int | None = None) -> Iterator[dict]:
    data_root = Path(data_root)
    ann_dir = data_root / _ANNOTATIONS
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"SatAst annotations not found at {ann_dir}; run `visbench download satast`.")
    count = 0
    for ann in sorted(ann_dir.glob("*.json")):
        d = json.loads(ann.read_text(encoding="utf-8"))
        last = d["correspondences"][-1]
        im_A = data_root / Path(d["query_path"].replace("\\", "/"))
        im_B = data_root / Path(d["pred_path"].replace("\\", "/"))
        if not (im_A.exists() and im_B.exists()):
            longest = max(len(str(im_A.resolve())), len(str(im_B.resolve())))
            hint = (f" The path is {longest} characters; Windows' 260-character limit hides it."
                    " Move the data to a shorter root or enable long paths."
                    if os.name == "nt" and longest >= 260 else "")
            raise FileNotFoundError(f"SatAst image missing for {ann.name} under {data_root}.{hint}")
        yield {
            "im_A_path": str(im_A),
            "im_B_path": str(im_B),
            "pts_A": np.asarray(last["pts_src"], dtype=np.float64).reshape(-1, 2),
            "pts_B": np.asarray(last["pts_dst"], dtype=np.float64).reshape(-1, 2),
            "scene": ann.stem.removeprefix("annotation_"),
        }
        count += 1
        if max_pairs is not None and count >= max_pairs:
            return


register("satast", DatasetSpec(
    task="homography_points", iter_pairs=iter_pairs, default_root=Path("satast"),
    download=download, source_url=_REPO,
    license="Images per AIMS / EarthMatch; annotations per repository",
))
