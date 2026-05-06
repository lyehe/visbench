"""Aachen Day-Night v1.1 — pose pairs from COLMAP SfM model.

Source:  https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/
License: Academic.

Requires `pycolmap` (install separately) to read the 3D-models/aachen_v_1_1 reconstruction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.metrics.pose import compute_relative_pose
from ._registry import DatasetSpec, register


_RECONSTRUCTION = {}


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if (data_root / "3D-models" / "aachen_v_1_1").exists():
        return data_root
    raise NotImplementedError(
        "Aachen v1.1: download aachen_v1_1.zip + database_and_query_images.zip from CIIRC\n"
        f"and extract under {data_root}."
    )


def _get_model(model_path: Path):
    key = str(model_path)
    if key not in _RECONSTRUCTION:
        import pycolmap
        _RECONSTRUCTION[key] = pycolmap.Reconstruction(str(model_path))
    return _RECONSTRUCTION[key]


def _qvec_to_R(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               min_shared: int = 100) -> Iterator[dict]:
    data_root = Path(data_root)
    model_path = data_root / "3D-models" / "aachen_v_1_1"
    if not model_path.exists():
        return
    images_root = data_root / "images_upright"
    r = _get_model(model_path)

    db_imgs = [(iid, img) for iid, img in r.images.items() if img.name.startswith("db/")]
    db_imgs.sort(key=lambda x: x[0])

    id_to_pts: dict[int, set] = {}
    for iid, img in db_imgs:
        pts = set()
        for p2d in img.points2D:
            if p2d.has_point3D():
                pts.add(p2d.point3D_id)
        id_to_pts[iid] = pts

    count = 0
    for i, (iid_a, img_a) in enumerate(db_imgs):
        pts_a = id_to_pts[iid_a]
        if len(pts_a) < min_shared // 2:
            continue
        for j in range(i + 1, min(i + 9, len(db_imgs))):
            iid_b, img_b = db_imgs[j]
            shared = len(pts_a & id_to_pts[iid_b])
            if shared < min_shared:
                continue
            rigid_a = img_a.cam_from_world()
            rigid_b = img_b.cam_from_world()
            qa_xyzw = np.asarray(rigid_a.rotation.quat)
            qb_xyzw = np.asarray(rigid_b.rotation.quat)
            ta = np.asarray(rigid_a.translation)
            tb = np.asarray(rigid_b.translation)
            Ra = _qvec_to_R(np.array([qa_xyzw[3], qa_xyzw[0], qa_xyzw[1], qa_xyzw[2]]))
            Rb = _qvec_to_R(np.array([qb_xyzw[3], qb_xyzw[0], qb_xyzw[1], qb_xyzw[2]]))
            R, t = compute_relative_pose(Ra, ta, Rb, tb)
            cam_a = r.cameras[img_a.camera_id]
            cam_b = r.cameras[img_b.camera_id]
            fx, fy, cx, cy = cam_a.params[0], cam_a.params[1], cam_a.params[2], cam_a.params[3]
            K0 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float64)
            fx, fy, cx, cy = cam_b.params[0], cam_b.params[1], cam_b.params[2], cam_b.params[3]
            K1 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float64)
            yield {
                "im_A_path": str(images_root / img_a.name),
                "im_B_path": str(images_root / img_b.name),
                "K0": K0, "K1": K1,
                "R_0to1": R, "t_0to1": t,
                "scene": "aachen_db",
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


_KW = dict(
    download=download,
    source_url="https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/",
    license="Academic",
)

register("aachen_v1_1", DatasetSpec("pose", iter_pairs, Path("aachen_v1"),
                                    default_resize=1024, kwargs={"min_shared": 100}, **_KW))
register("aachen_v1_1_hard", DatasetSpec("pose", iter_pairs, Path("aachen_v1"),
                                         default_resize=1024, kwargs={"min_shared": 30}, **_KW))
