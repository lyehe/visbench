"""Real-data GT-geometry regression check helpers (shared, dependency-light).

Canonical copy: featurexx/scripts/real_geom_check.py.  Identical vendored copies live in
  featurexx/third_party/mvroma/tests/_real_geom_check.py
  featurexx/third_party/gim/tools/_real_geom_check.py
  visbench/tests/_real_geom_check.py
  depth-anything-3/scripts/_real_geom_check.py
Keep them in sync (only numpy + cv2 are required).

Why: on 2026-10-01 several loaders fed wrong GT poses (OpenGL c2w used as OpenCV,
TartanAir NED used as OpenCV).  Rigidity checks did not catch it.  This check does:
SIFT matches on a real image pair must satisfy the epipolar constraint implied by the
loader's GT relative pose + intrinsics (median Sampson error ~ < 2 px), and must do so
clearly better than under a deliberately perturbed pose (5-8 deg rotation).

Conventions: ``T_0to1`` is a 4x4 (or R,t) that maps camera-0 coordinates to camera-1
coordinates (OpenCV cameras: x right, y down, z forward), i.e. X1 = R @ X0 + t.
Points are pixel coordinates in the images the intrinsics refer to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

GT_MEDIAN_MAX_PX = 2.0
GT_TO_PERTURBED_MAX_RATIO = 0.25
MIN_MATCHES = 30
MIN_PASS_FRACTION = 0.7


def to_gray_u8(image) -> np.ndarray:
    """Accept HxW / HxWx3 uint8 or float (0..1 or 0..255), CHW torch-ish arrays too."""
    a = np.asarray(image)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        if a.size and float(np.nanmax(a)) <= 1.5:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.shape[2] == 3 else a[..., 0]
    return np.ascontiguousarray(a)


def sift_matches(img0, img1, n_features: int = 4000, ratio: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    g0, g1 = to_gray_u8(img0), to_gray_u8(img1)
    sift = cv2.SIFT_create(n_features)
    k0, d0 = sift.detectAndCompute(g0, None)
    k1, d1 = sift.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(k0) < 2 or len(k1) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2))
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d0, d1, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    p0 = np.array([k0[m.queryIdx].pt for m in good], dtype=np.float64).reshape(-1, 2)
    p1 = np.array([k1[m.trainIdx].pt for m in good], dtype=np.float64).reshape(-1, 2)
    return p0, p1


def ransac_inliers(p0, p1, threshold_px: float = 1.5) -> np.ndarray:
    """GT-agnostic outlier rejection: keep matches consistent with *some* fundamental
    matrix (MAGSAC).  Removes SIFT mismatches in low-texture/repetitive scenes without
    using the GT pose, so a wrong GT pose still fails on the surviving inliers."""
    if len(p0) < 15:
        return np.ones(len(p0), dtype=bool)
    method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
    try:
        _, mask = cv2.findFundamentalMat(p0, p1, method, threshold_px, 0.999, 10000)
    except cv2.error:  # MAGSAC raises (-215 !model.empty()) on degenerate match sets
        mask = None
    if mask is None:
        return np.ones(len(p0), dtype=bool)
    return mask.ravel().astype(bool)


def split_pose(T_0to1=None, R=None, t=None) -> tuple[np.ndarray, np.ndarray]:
    if T_0to1 is not None:
        T = np.asarray(T_0to1, dtype=np.float64).reshape(-1, 4)[:3]
        return T[:, :3], T[:, 3]
    return np.asarray(R, dtype=np.float64).reshape(3, 3), np.asarray(t, dtype=np.float64).reshape(3)


def fundamental(K0, K1, R, t) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(3)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], dtype=np.float64)
    E = tx @ np.asarray(R, dtype=np.float64)
    K0 = np.asarray(K0, dtype=np.float64).reshape(3, 3)
    K1 = np.asarray(K1, dtype=np.float64).reshape(3, 3)
    return np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)


def sampson_px(p0, p1, K0, K1, R, t) -> np.ndarray:
    """Per-match sqrt(Sampson) epipolar error in pixels (scale-invariant in |t|)."""
    F = fundamental(K0, K1, R, t)
    x0 = np.concatenate([np.asarray(p0, np.float64), np.ones((len(p0), 1))], 1)
    x1 = np.concatenate([np.asarray(p1, np.float64), np.ones((len(p1), 1))], 1)
    Fx0 = x0 @ F.T
    Ftx1 = x1 @ F
    num = np.sum(x1 * Fx0, axis=1) ** 2
    den = Fx0[:, 0] ** 2 + Fx0[:, 1] ** 2 + Ftx1[:, 0] ** 2 + Ftx1[:, 1] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(num / den)


def rotation_about(axis, deg: float) -> np.ndarray:
    axis = np.asarray(axis, np.float64)
    axis = axis / np.linalg.norm(axis)
    rvec = axis * np.deg2rad(deg)
    return cv2.Rodrigues(rvec.reshape(3, 1))[0]


def perturb_pose(R, t, rng: np.random.Generator, deg_range=(5.0, 8.0)) -> tuple[np.ndarray, np.ndarray]:
    deg = float(rng.uniform(*deg_range))
    Rp = rotation_about(rng.normal(size=3), deg)
    return Rp @ np.asarray(R, np.float64), np.asarray(t, np.float64)


@dataclass
class PairResult:
    name: str
    n_matches: int
    gt_median: float = float("nan")
    perturbed_median: float = float("nan")

    @property
    def usable(self) -> bool:
        return self.n_matches >= MIN_MATCHES and np.isfinite(self.gt_median)

    @property
    def ok(self) -> bool:
        return (
            self.usable
            and self.gt_median < GT_MEDIAN_MAX_PX
            and np.isfinite(self.perturbed_median)
            and self.gt_median < GT_TO_PERTURBED_MAX_RATIO * self.perturbed_median
        )

    def line(self) -> str:
        tag = "ok " if self.ok else ("skip" if not self.usable else "BAD")
        return (
            f"[{tag}] {self.name}: n={self.n_matches} gt_med={self.gt_median:.2f}px "
            f"pert_med={self.perturbed_median:.2f}px"
        )


def evaluate_pair(
    name: str,
    img0,
    img1,
    K0,
    K1,
    T_0to1=None,
    *,
    R=None,
    t=None,
    rng: np.random.Generator | None = None,
    n_features: int = 4000,
    ratio: float = 0.7,
    matches: tuple[np.ndarray, np.ndarray] | None = None,
    prefilter: bool = False,
) -> PairResult:
    """``prefilter=True`` drops matches that fit no fundamental matrix (see ransac_inliers)
    before scoring; recommended for low-texture data, off by default for compatibility."""
    rng = rng if rng is not None else np.random.default_rng(0)
    R, t = split_pose(T_0to1, R, t)
    p0, p1 = matches if matches is not None else sift_matches(img0, img1, n_features, ratio)
    if prefilter and len(p0):
        keep = ransac_inliers(p0, p1)
        p0, p1 = p0[keep], p1[keep]
    res = PairResult(name=name, n_matches=int(len(p0)))
    if len(p0) == 0 or np.linalg.norm(t) < 1e-9:
        return res
    gt = sampson_px(p0, p1, K0, K1, R, t)
    Rp, tp = perturb_pose(R, t, rng)
    pe = sampson_px(p0, p1, K0, K1, Rp, tp)
    res.gt_median = float(np.nanmedian(gt))
    res.perturbed_median = float(np.nanmedian(pe))
    return res


@dataclass
class Summary:
    dataset: str
    results: list[PairResult] = field(default_factory=list)

    @property
    def usable(self) -> list[PairResult]:
        return [r for r in self.results if r.usable]

    @property
    def passed(self) -> bool:
        u = self.usable
        return bool(u) and sum(r.ok for r in u) >= MIN_PASS_FRACTION * len(u)

    def medians(self) -> tuple[float, float]:
        u = self.usable
        if not u:
            return float("nan"), float("nan")
        return float(np.median([r.gt_median for r in u])), float(np.median([r.perturbed_median for r in u]))

    def report(self) -> str:
        gt, pe = self.medians()
        u = self.usable
        head = (
            f"{self.dataset}: {'PASS' if self.passed else 'FAIL'} "
            f"ok={sum(r.ok for r in u)}/{len(u)} usable (of {len(self.results)}) "
            f"median-of-medians gt={gt:.2f}px perturbed={pe:.2f}px"
        )
        return "\n".join([head] + ["  " + r.line() for r in self.results])


def resize_with_k(image: np.ndarray, K, size_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Resize an image to (H, W) and rescale K accordingly (pixel-corner convention)."""
    h0, w0 = image.shape[:2]
    h, w = size_hw
    K = np.asarray(K, np.float64).copy()
    K[0] *= w / w0
    K[1] *= h / h0
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA), K


def read_rgb(path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
