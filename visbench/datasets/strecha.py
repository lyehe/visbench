"""Strecha multi-view — classical small SfM benchmark (Strecha et al., CVPR 2008).

Source:  https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/
         (original EPFL host; mirrored by many matching repos)
Paper:   Strecha, von Hansen, Van Gool, Fua & Thoennessen,
         'On Benchmarking Camera Calibration and Multi-View Stereo for High Resolution Imagery', CVPR 2008.
License: Academic / research use (per EPFL release).

Five scenes, ~80 images total: fountain-P11, herzjesu-P8, castle-P19, castle-P30, entry-P10.
Still cited as a low-pair sanity check (LightGlue paper Fig. 4, RoMa supplementary).

Layout::

    datasets/strecha/
      fountain-P11/
        urd/0000.png ... urd/0010.png        # rectified images
        urd/0000.png.camera ... 0010.png.camera
      herzjesu-P8/  castle-P19/  castle-P30/  entry-P10/

`.camera` file format (per Strecha 2008 release)::

    fx 0  cx
    0  fy cy
    0  0  1
    0  0  0          (radial distortion, usually all zero)
    r11 r12 r13
    r21 r22 r23
    r31 r32 r33      (R: world-from-camera rotation)
    cx cy cz         (camera center C in world coords)
    width height
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.metrics.pose import compute_relative_pose
from ._registry import DatasetSpec, register

_SCENES = ("fountain-P11", "herzjesu-P8", "castle-P19", "castle-P30", "entry-P10")


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if any((data_root / s).exists() for s in _SCENES):
        return data_root
    raise NotImplementedError(
        "Strecha: the original EPFL host (cvlabwww.epfl.ch / documents.epfl.ch) is\n"
        "intermittent. Get the per-scene tarballs from one of:\n"
        "  - The MASt3R / DUSt3R data prep scripts (they bundle Strecha as part of\n"
        "    their evaluation recipes):\n"
        "      https://github.com/naver/mast3r\n"
        "      https://github.com/naver/dust3r\n"
        "  - hloc's data scripts: https://github.com/cvg/Hierarchical-Localization\n"
        "  - LightGlue's reference repo (uses Strecha as a low-pair sanity check):\n"
        "      https://github.com/cvg/LightGlue\n"
        f"Place 5 scene directories ({', '.join(_SCENES)}) under {data_root}.\n"
        "Each scene needs <name>/urd/*.png and matching *.png.camera files."
    )


def _parse_camera(p: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse Strecha .camera. Returns (K, R_cam_from_world, t_cam_from_world)."""
    nums = []
    for ln in p.read_text().splitlines():
        for tok in ln.split():
            try:
                nums.append(float(tok))
            except ValueError:
                pass
    K = np.array(nums[0:9], dtype=np.float64).reshape(3, 3)
    # nums[9:12] = radial distortion (skip)
    R_wc = np.array(nums[12:21], dtype=np.float64).reshape(3, 3)  # world-from-camera
    C = np.array(nums[21:24], dtype=np.float64)  # camera centre in world
    # camera-from-world: R_cw = R_wc.T, t_cw = -R_cw @ C
    R_cw = R_wc.T
    t_cw = -R_cw @ C
    return K, R_cw, t_cw


def _scene_views(scene_dir: Path) -> list[tuple[Path, Path]]:
    """Find (image, camera) pairs for a scene. Tolerates `urd/`, `images/`, or flat layout."""
    candidates = (scene_dir / "urd", scene_dir / "images", scene_dir)
    for d in candidates:
        if not d.is_dir():
            continue
        imgs = sorted(d.glob("*.png")) or sorted(d.glob("*.jpg"))
        pairs: list[tuple[Path, Path]] = []
        for img in imgs:
            cam = img.with_suffix(img.suffix + ".camera")
            if cam.exists():
                pairs.append((img, cam))
        if pairs:
            return pairs
    return []


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               pair_skip: int = 1) -> Iterator[dict]:
    """Yield (view_i, view_{i+pair_skip}) pairs across the 5 scenes."""
    data_root = Path(data_root)
    count = 0
    for scene_name in _SCENES:
        scene = data_root / scene_name
        if not scene.is_dir():
            continue
        views = _scene_views(scene)
        if len(views) < 2:
            continue
        for i in range(len(views) - pair_skip):
            j = i + pair_skip
            img_a, cam_a = views[i]
            img_b, cam_b = views[j]
            try:
                K0, R0, t0 = _parse_camera(cam_a)
                K1, R1, t1 = _parse_camera(cam_b)
            except Exception:
                continue
            R, t = compute_relative_pose(R0, t0, R1, t1)
            yield {
                "im_A_path": str(img_a),
                "im_B_path": str(img_b),
                "K0": K0, "K1": K1,
                "R_0to1": R, "t_0to1": t,
                "scene": scene_name,
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


_KW = dict(
    download=download,
    source_url="https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/",
    license="Academic",
)

register("strecha", DatasetSpec(
    task="pose", iter_pairs=iter_pairs, default_root=Path("strecha"),
    default_resize=1200, kwargs={"pair_skip": 1}, **_KW,
))
register("strecha_wide", DatasetSpec(
    task="pose", iter_pairs=iter_pairs, default_root=Path("strecha"),
    default_resize=1200, kwargs={"pair_skip": 3}, **_KW,
))
