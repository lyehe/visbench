"""ETH3D high-res multi-view pose pairs.

Source:  https://www.eth3d.net/datasets
Paper:   Schöps et al., 'A Multi-View Stereo Benchmark with High-Resolution Images', CVPR 2017.
License: CC BY-NC-SA 4.0.

Two on-disk layouts are accepted per scene:

- native ETH3D:  <scene>/dslr_calibration_undistorted/{cameras,images}.txt and
  <scene>/images/dslr_images_undistorted/*.JPG (images.txt names carry that prefix);
- flattened:     <scene>/{cameras,images,pairs}.txt and <scene>/images/*.JPG.

Pairs come from <scene>/pairs.txt when present, else from SfM co-visibility (shared
3D points in images.txt).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ...core.metrics.pose import compute_relative_pose
from .._registry import DatasetSpec, register

# Co-visibility pairing (no pairs.txt): keep the TOP_K most-covisible partners per
# image that share at least MIN_COVIS 3D points.
MIN_COVIS = 150
TOP_K_PER_IMAGE = 5


def _scene_files(scene: Path):
    """(cameras.txt, images.txt) for the native or flattened layout, or None."""
    for base in (scene / "dslr_calibration_undistorted", scene):
        if (base / "cameras.txt").exists() and (base / "images.txt").exists():
            return base / "cameras.txt", base / "images.txt"
    return None


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if data_root.exists() and any(p.is_dir() and _scene_files(p) for p in data_root.iterdir()):
        return data_root
    raise NotImplementedError(
        "ETH3D: download the high-res multi-view (undistorted) splits from\n"
        f"https://www.eth3d.net/datasets and extract <scene>/{{dslr_calibration_undistorted,images}}/ under {data_root}."
    )


def _parse_colmap_cameras(path: Path) -> dict:
    cams = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        tok = line.split()
        cam_id = int(tok[0])
        params = list(map(float, tok[4:]))
        if tok[1] in ("PINHOLE", "OPENCV") and len(params) >= 4:
            fx, fy, cx, cy = params[:4]
        else:  # SIMPLE_PINHOLE / SIMPLE_RADIAL: single shared focal
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        cams[cam_id] = K
    return cams


def _parse_colmap_images(path: Path) -> dict:
    """name -> (R, t, cam_id, observed point3D ids)."""
    out = {}
    # Keep blank lines: COLMAP writes an empty points line for an image with no
    # observations, and dropping it would shift every later image by a record.
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        tok = lines[i].split()
        qw, qx, qy, qz = map(float, tok[1:5])
        tx, ty, tz = map(float, tok[5:8])
        cam_id = int(tok[8])
        name = tok[9]
        q = np.array([qw, qx, qy, qz])
        q /= np.linalg.norm(q) + 1e-12
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        t = np.array([tx, ty, tz])
        pts = lines[i + 1].split() if i + 1 < len(lines) else []
        ids = {int(pts[j + 2]) for j in range(0, len(pts) - 2, 3) if pts[j + 2] != "-1"}
        out[name] = (R, t, cam_id, ids)
        i += 2
    return out


def _covisible_pairs(imgs: dict) -> Iterator[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    for a, (_, _, _, ids_a) in imgs.items():
        if not ids_a:
            continue
        scored = sorted(((len(ids_a & v[3]), b) for b, v in imgs.items() if b != a), reverse=True)
        for score, b in scored[:TOP_K_PER_IMAGE]:
            key = (a, b) if a < b else (b, a)
            if score >= MIN_COVIS and key not in seen:
                seen.add(key)
                yield key


def _image_path(scene: Path, name: str) -> Path:
    """images.txt names carry dslr_images_undistorted/; flattened copies drop it."""
    p = scene / "images" / name
    return p if p.exists() else scene / "images" / Path(name).name


def iter_pairs(data_root: Path | str, max_pairs: int | None = None) -> Iterator[dict]:
    data_root = Path(data_root)
    count = 0
    for scene in sorted(data_root.iterdir()):
        if not scene.is_dir():
            continue
        files = _scene_files(scene)
        if files is None:
            continue
        cams = _parse_colmap_cameras(files[0])
        imgs = _parse_colmap_images(files[1])
        pair_file = scene / "pairs.txt"
        if pair_file.exists():
            pairs = (tuple(ln.split()[:2]) for ln in pair_file.read_text().splitlines() if len(ln.split()) >= 2)
        else:
            pairs = _covisible_pairs(imgs)
        for a, b in pairs:
            if a not in imgs or b not in imgs:
                continue
            R0, t0, cam0, _ = imgs[a]
            R1, t1, cam1, _ = imgs[b]
            R, t = compute_relative_pose(R0, t0, R1, t1)
            yield {
                "im_A_path": str(_image_path(scene, a)),
                "im_B_path": str(_image_path(scene, b)),
                "K0": cams[cam0], "K1": cams[cam1],
                "R_0to1": R, "t_0to1": t,
                "scene": scene.name,
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


_KW = dict(
    download=download,
    source_url="https://www.eth3d.net/datasets",
    license="CC BY-NC-SA 4.0",
)
register("eth3d", DatasetSpec("pose", iter_pairs, Path("eth3d"), default_resize=1200, **_KW))
