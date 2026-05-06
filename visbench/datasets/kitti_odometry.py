"""KITTI visual odometry frame-skip pairs (gray stereo left).

Source:  https://huggingface.co/datasets/yujie2696/kitti_odometry_00 (sequences 00, 02, 03)
Paper:   Geiger et al., 'Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite', CVPR 2012.
License: CC BY-NC-SA 3.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ._download import download_hf
from ._registry import DatasetSpec, register


_HF_REPOS = (
    ("yujie2696/kitti_odometry_00", "00"),
    ("yujie2696/kitti_odometry_02", "02"),
    ("yujie2696/kitti_odometry_03", "03"),
)


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    for repo_id, seq in _HF_REPOS:
        target = data_root / seq
        if target.exists() and any(target.glob("**/*.png")):
            continue
        download_hf(repo_id, target, repo_type="dataset")
    return data_root


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               pair_skip: int = 10) -> Iterator[dict]:
    data_root = Path(data_root)
    count = 0
    for seq in sorted(data_root.iterdir()):
        if not seq.is_dir():
            continue
        img_dir = seq / "image_0"
        if not img_dir.exists():
            img_dir = seq
        frames = sorted(img_dir.glob("*.png"))
        for i in range(0, len(frames) - pair_skip, pair_skip // 2 or 1):
            j = i + pair_skip
            if j >= len(frames):
                break
            yield {
                "im_A_path": str(frames[i]),
                "im_B_path": str(frames[j]),
                "scene": seq.name,
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


_KW = dict(
    download=download,
    source_url="https://www.cvlibs.net/datasets/kitti/eval_odometry.php",
    license="CC BY-NC-SA 3.0",
)

register("kitti_odometry", DatasetSpec("qualitative", iter_pairs, Path("kitti_odometry"),
                                       default_resize=1024, kwargs={"pair_skip": 10}, **_KW))
register("kitti_odometry_wide", DatasetSpec("qualitative", iter_pairs, Path("kitti_odometry"),
                                            default_resize=1024, kwargs={"pair_skip": 30}, **_KW))
