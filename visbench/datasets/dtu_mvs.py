"""DTU MVS — DTU Robot Image Data Sets, multi-view stereo benchmark.

Source:  https://roboimagedata.compute.dtu.dk/?page_id=36
Paper:   Aanaes, Jensen, Vogiatzis, Tola & Dahl, 'Large-Scale Data for Multiple-View Stereopsis', IJCV 2016.
License: Open data (per DTU Robot Image Data Sets release).

Used by MASt3R (Wang et al. 2024), DUSt3R, RoMa-V2 as a generalization test.
The canonical eval format is **MVSNet-preprocessed**: per-scan directories
with `images/<idx>.jpg`, `cams/<idx>_cam.txt` (extrinsic + intrinsic + depth
range), and `pair.txt` listing top-N covisibility neighbours.

Layout::

    datasets/dtu/
      scan1/
        images/{00000000..00000049}.jpg
        cams/{00000000..00000049}_cam.txt
        pair.txt
      scan4/  ...  scan114/  scan118/  (22 test scans)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.metrics.pose import compute_relative_pose
from ._download import download_hf
from ._registry import DatasetSpec, register

_HF_REPO = "jzhangbs/mvsdf_dtu"

# MVSNet's standard 22-scan test split (used by DUSt3R/MASt3R).
_TEST_SCANS = (1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118)


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if any((data_root / f"scan{n}" / "pair.txt").exists() for n in _TEST_SCANS):
        return data_root
    raise NotImplementedError(
        "DTU MVS: download the MVSNet-preprocessed test split. Either:\n"
        f"  (a) HF mirror: huggingface-cli download {_HF_REPO} --repo-type dataset --local-dir {data_root}\n"
        "      (preprocessed for MVSDF, ICCV 2021 — same MVSNet format).\n"
        "  (b) MVSNet repo: https://github.com/YoYo000/MVSNet (download dtu_test.zip per README).\n"
        "  (c) Original raw rectified set: https://roboimagedata.compute.dtu.dk/?page_id=36 \n"
        "      (Rectified.zip + camera params; you'll need to re-pack into MVSNet format).\n"
        f"Place 22 scan directories (scan1, scan4, ... scan118) under {data_root}."
    )


def _parse_cam_txt(p: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse MVSNet `<idx>_cam.txt`. Returns (K, R_cam_from_world, t_cam_from_world)."""
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    # Format:
    #   extrinsic
    #   e11 e12 e13 e14
    #   e21 e22 e23 e24
    #   e31 e32 e33 e34
    #   0 0 0 1
    #   intrinsic
    #   fx 0 cx
    #   0 fy cy
    #   0 0 1
    #   depth_min depth_interval
    i = lines.index("extrinsic") + 1
    ext = np.array([[float(x) for x in lines[i + r].split()] for r in range(4)])
    j = lines.index("intrinsic") + 1
    K = np.array([[float(x) for x in lines[j + r].split()] for r in range(3)])
    return K, ext[:3, :3], ext[:3, 3]


def _parse_pair_txt(p: Path) -> list[tuple[int, list[int]]]:
    """Returns [(view_idx, [neighbour_idx, ...]), ...] sorted by score."""
    txt = p.read_text().splitlines()
    n_views = int(txt[0])
    out = []
    for v in range(n_views):
        view_idx = int(txt[1 + 2 * v])
        nbrs_line = txt[2 + 2 * v].split()
        n_nbrs = int(nbrs_line[0])
        nbrs = [int(nbrs_line[1 + 2 * k]) for k in range(n_nbrs)]
        out.append((view_idx, nbrs))
    return out


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               top_k: int = 3) -> Iterator[dict]:
    """Yield (view, neighbour) pairs from MVSNet pair.txt for each test scan.

    `top_k` = how many neighbours per view to use (1..N from pair.txt).
    """
    data_root = Path(data_root)
    count = 0
    for n in _TEST_SCANS:
        scan = data_root / f"scan{n}"
        pair_file = scan / "pair.txt"
        if not pair_file.exists():
            continue
        cam_dir = scan / "cams"
        img_dir = scan / "images"
        for view_idx, nbrs in _parse_pair_txt(pair_file):
            view_cam = cam_dir / f"{view_idx:08d}_cam.txt"
            view_img = img_dir / f"{view_idx:08d}.jpg"
            if not (view_cam.exists() and view_img.exists()):
                continue
            K0, R0, t0 = _parse_cam_txt(view_cam)
            for nb in nbrs[:top_k]:
                nb_cam = cam_dir / f"{nb:08d}_cam.txt"
                nb_img = img_dir / f"{nb:08d}.jpg"
                if not (nb_cam.exists() and nb_img.exists()):
                    continue
                K1, R1, t1 = _parse_cam_txt(nb_cam)
                R, t = compute_relative_pose(R0, t0, R1, t1)
                yield {
                    "im_A_path": str(view_img),
                    "im_B_path": str(nb_img),
                    "K0": K0, "K1": K1,
                    "R_0to1": R, "t_0to1": t,
                    "scene": scan.name,
                }
                count += 1
                if max_pairs is not None and count >= max_pairs:
                    return


register("dtu_mvs", DatasetSpec(
    task="pose",
    iter_pairs=iter_pairs,
    default_root=Path("dtu"),
    default_resize=1200,
    download=download,
    source_url="https://roboimagedata.compute.dtu.dk/?page_id=36",
    license="Open (per DTU release)",
))
