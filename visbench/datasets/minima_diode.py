"""DIODE-outdoor RGB-Depth synthetic-homography iterator (MINIMA protocol).

Source:  https://diode-dataset.org/ + MINIMA https://github.com/LSXI7/MINIMA
Paper:   DIODE: Vasiljevic et al., 2019; MINIMA, 2024.
License: MIT (DIODE).
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Iterator

from ._download import download_hf
from ._minima_homography import (
    load_resized_to_disk,
    sample_homography,
    warp_and_cache,
)
from ._registry import DatasetSpec, register

TARGET_SHAPE = (480, 640)
_HF_REPO = "diode-dataset/diode-outdoor-val"


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    val = data_root / "val" / "outdoor"
    if val.exists():
        return data_root
    raise NotImplementedError(
        "DIODE outdoor val requires a manual fetch.\n"
        "Download the val.tar from https://diode-dataset.org/ and extract under "
        f"{data_root}/val/outdoor/ ."
    )


def _enumerate_outdoor_rgbs(val_root: Path) -> list[Path]:
    out_root = val_root / "outdoor"
    if not out_root.exists():
        raise FileNotFoundError(f"DIODE val/outdoor not found at {out_root}")
    rgbs: list[Path] = []
    for png in sorted(out_root.rglob("*.png")):
        if "_depth" in png.name:
            continue
        rgbs.append(png)
    return rgbs


def iter_pairs(data_root: Path | str, max_pairs: int | None = None) -> Iterator[dict]:
    data_root = Path(data_root)
    val_root = data_root / "val"
    if not val_root.exists():
        raise FileNotFoundError(f"DIODE val/ not found at {val_root}")

    cache_root = data_root / "_warp_cache"
    rgb_paths = _enumerate_outdoor_rgbs(val_root)
    count = 0

    for rgb_path in rgb_paths:
        depth_npy = rgb_path.with_name(rgb_path.stem + "_depth.npy")
        if not depth_npy.exists():
            continue

        rel = rgb_path.relative_to(val_root)
        stem = rel.with_suffix("")

        H_rgb_to_depth = sample_homography(TARGET_SHAPE, str(rgb_path))
        H_depth_to_rgb = sample_homography(TARGET_SHAPE, str(depth_npy))

        rgb_resized = load_resized_to_disk(rgb_path, cache_root / "rgb" / rel, TARGET_SHAPE)
        depth_colorized = warp_and_cache(
            depth_npy, H=np.eye(3),
            cache_path=cache_root / "depth" / stem.with_suffix(".png"),
            target_shape=TARGET_SHAPE, color_depth_npy=True,
        )
        depth_warped = warp_and_cache(
            depth_npy, H=H_rgb_to_depth,
            cache_path=cache_root / "depth_warped" / stem.with_suffix(".png"),
            target_shape=TARGET_SHAPE, color_depth_npy=True,
        )
        rgb_warped = warp_and_cache(
            rgb_path, H=H_depth_to_rgb,
            cache_path=cache_root / "rgb_warped" / stem.with_suffix(".png"),
            target_shape=TARGET_SHAPE, color_depth_npy=False,
        )

        for direction, im_A, im_B, H_gt in [
            ("rgb_to_depth", rgb_resized, depth_warped, H_rgb_to_depth),
            ("depth_to_rgb", depth_colorized, rgb_warped, H_depth_to_rgb),
        ]:
            yield {
                "im_A_path": str(im_A),
                "im_B_path": str(im_B),
                "H_gt": H_gt,
                "subset": f"DIO-depth-{direction}",
                "seq": str(rel),
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


register("minima_diode", DatasetSpec(
    task="homography",
    iter_pairs=iter_pairs,
    default_root=Path("diode"),
    default_resize=None,
    download=download,
    source_url="https://diode-dataset.org/",
    license="MIT (DIODE)",
))
