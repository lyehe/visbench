"""TartanAir — multi-environment synthetic SLAM/odometry dataset (Wang et al., IROS 2020).

Source:  https://theairlab.org/tartanair-dataset/
HF mirror: https://huggingface.co/datasets/theairlabcmu/tartanair (BSD-3, official)
Paper:   Wang et al., 'TartanAir: A Dataset to Push the Limits of Visual SLAM', IROS 2020.
License: BSD-3-Clause (per CMU AirLab release).

Used as a cross-domain generalization test by recent matching papers (MASt3R,
MicKey predecessors). The full set is multi-TB; visbench iterates frame-skip
pairs from whatever environments/trajectories you've placed under `data_root`.

Layout (per CMU AirLab release):

    datasets/tartanair/
      <env>/
        Easy/  (or Hard/)
          P000/
            image_left/000000_left.png
            pose_left.txt          # 8 cols: tx ty tz qx qy qz qw (NED frame)
            ...

Intrinsics are fixed: fx=fy=320, cx=320, cy=240 for 640x480 RGB images.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.metrics.pose import compute_relative_pose
from ._download import download_hf
from ._registry import DatasetSpec, register

_HF_REPO = "theairlabcmu/tartanair"
_DEFAULT_ENVS = ("abandonedfactory",)
_DEFAULT_DIFFICULTY = "Easy"
_DEFAULT_TRAJ = "P000"

# Per the TartanAir docs (https://github.com/castacks/tartanair_tools).
K_TARTANAIR = np.array([
    [320.0, 0.0, 320.0],
    [0.0, 320.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)


def download(data_root: Path | str) -> Path:
    """Fetch a TartanAir subset from the official HF mirror.

    The full release is multi-TB; the default fetches one trajectory
    (`abandonedfactory/Easy/P000`). Override via env vars:

      VISBENCH_TARTANAIR_ENVS         comma-separated env list (default: 'abandonedfactory')
      VISBENCH_TARTANAIR_DIFFICULTY   'Easy' or 'Hard' (default: 'Easy')
      VISBENCH_TARTANAIR_TRAJ         comma-separated trajectories (default: 'P000')
    """
    data_root = Path(data_root)
    if any(data_root.glob("*/Easy/P*/image_left/*.png")) or \
       any(data_root.glob("*/Hard/P*/image_left/*.png")):
        return data_root
    envs = tuple(s.strip() for s in os.environ.get(
        "VISBENCH_TARTANAIR_ENVS", ",".join(_DEFAULT_ENVS)).split(",") if s.strip())
    difficulty = os.environ.get("VISBENCH_TARTANAIR_DIFFICULTY", _DEFAULT_DIFFICULTY)
    trajs = tuple(s.strip() for s in os.environ.get(
        "VISBENCH_TARTANAIR_TRAJ", _DEFAULT_TRAJ).split(",") if s.strip())
    allow_patterns: list[str] = []
    for env in envs:
        for traj in trajs:
            allow_patterns.append(f"{env}/{difficulty}/{traj}/image_left/*.png")
            allow_patterns.append(f"{env}/{difficulty}/{traj}/pose_left.txt")
    download_hf(_HF_REPO, data_root, repo_type="dataset",
                allow_patterns=allow_patterns)
    return data_root


def _q_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def _parse_poses(p: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """pose_left.txt: each line `tx ty tz qx qy qz qw` — camera-to-world (NED)."""
    out = []
    for ln in p.read_text().splitlines():
        toks = ln.strip().split()
        if len(toks) < 7:
            continue
        tx, ty, tz, qx, qy, qz, qw = (float(x) for x in toks[:7])
        R_wc = _q_to_R(qx, qy, qz, qw)
        t_wc = np.array([tx, ty, tz], dtype=np.float64)
        # camera-from-world (what compute_relative_pose expects).
        R_cw, t_cw = R_wc.T, -R_wc.T @ t_wc
        out.append((R_cw, t_cw))
    return out


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               pair_skip: int = 10) -> Iterator[dict]:
    data_root = Path(data_root)
    count = 0
    for env in sorted(data_root.iterdir()):
        if not env.is_dir():
            continue
        for diff in sorted(env.iterdir()):
            if not diff.is_dir() or diff.name not in ("Easy", "Hard"):
                continue
            for traj in sorted(diff.iterdir()):
                if not traj.is_dir() or not traj.name.startswith("P"):
                    continue
                pose_file = traj / "pose_left.txt"
                img_dir = traj / "image_left"
                if not (pose_file.exists() and img_dir.exists()):
                    continue
                poses = _parse_poses(pose_file)
                frames = sorted(img_dir.glob("*.png"))
                if len(frames) < 2 or len(poses) != len(frames):
                    continue
                for i in range(0, len(frames) - pair_skip, max(1, pair_skip // 2)):
                    j = i + pair_skip
                    if j >= len(frames):
                        break
                    R0, t0 = poses[i]
                    R1, t1 = poses[j]
                    R, t = compute_relative_pose(R0, t0, R1, t1)
                    yield {
                        "im_A_path": str(frames[i]),
                        "im_B_path": str(frames[j]),
                        "K0": K_TARTANAIR, "K1": K_TARTANAIR,
                        "R_0to1": R, "t_0to1": t,
                        "scene": f"{env.name}/{diff.name}/{traj.name}",
                    }
                    count += 1
                    if max_pairs is not None and count >= max_pairs:
                        return


_KW = dict(
    download=download,
    source_url="https://huggingface.co/datasets/theairlabcmu/tartanair",
    license="BSD-3-Clause",
)

register("tartanair", DatasetSpec(
    task="pose", iter_pairs=iter_pairs, default_root=Path("tartanair"),
    default_resize=640, kwargs={"pair_skip": 10}, **_KW,
))
register("tartanair_wide", DatasetSpec(
    task="pose", iter_pairs=iter_pairs, default_root=Path("tartanair"),
    default_resize=640, kwargs={"pair_skip": 30}, **_KW,
))
