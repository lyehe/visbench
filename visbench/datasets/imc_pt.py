"""IMC-PT (Phototourism) validation benchmark — wide-baseline tourism pairs.

Source:  https://www.cs.ubc.ca/~kmyi/imw2020/data.html
Paper:   Jin et al., 'Image Matching across Wide Baselines: From Paper to Practice', IJCV 2021.
License: Academic (per CVL UBC).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.metrics.pose import compute_relative_pose
from ._registry import DatasetSpec, register


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if any((data_root / s).exists() for s in ("sacre_coeur", "st_peters_square", "reichstag")):
        return data_root
    raise NotImplementedError(
        "IMC-PT: follow the image-matching-benchmark-baselines instructions at\n"
        "https://www.cs.ubc.ca/~kmyi/imw2020/data.html (~10 GB val bundle); place\n"
        f"<scene>/{{images,pairs,calibration}}/ subtrees under {data_root}."
    )


def _load_calib_h5(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import h5py
    with h5py.File(path, "r") as f:
        K = np.array(f["K"])
        R = np.array(f["R"])
        T = np.array(f["T"]).reshape(3)
    return K, R, T


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               scenes=("sacre_coeur", "st_peters_square", "reichstag")) -> Iterator[dict]:
    data_root = Path(data_root)
    count = 0
    for scene_name in scenes:
        scene = data_root / scene_name
        if not scene.exists():
            continue
        pairs_file = scene / "pairs" / "pairs.txt"
        if not pairs_file.exists():
            continue
        calib = scene / "calibration"
        imgs = scene / "images"
        for line in pairs_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            K0, R0, t0 = _load_calib_h5(calib / f"calibration_{a}.h5")
            K1, R1, t1 = _load_calib_h5(calib / f"calibration_{b}.h5")
            R, t = compute_relative_pose(R0, t0, R1, t1)
            yield {
                "im_A_path": str(imgs / f"{a}.jpg"),
                "im_B_path": str(imgs / f"{b}.jpg"),
                "K0": K0, "K1": K1,
                "R_0to1": R, "t_0to1": t,
                "scene": scene_name,
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


register("imc_pt", DatasetSpec(
    task="pose", iter_pairs=iter_pairs, default_root=Path("imc_pt"),
    default_resize=1200, download=download,
    source_url="https://www.cs.ubc.ca/~kmyi/imw2020/data.html",
    license="Academic",
))
