"""Unified eval loop — consumes a vismatch-compatible matcher and a pair iterator, returns metrics."""

import os
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .metrics.pose import (
    compute_pose_error,
    estimate_pose_essential,
    pose_auc,
)
from .metrics.homography import (
    estimate_homography,
    homography_corner_error,
    hpatches_auc,
)
from .metrics.classical import (
    matching_score,
    mma_homography,
    repeatability,
    symmetric_epipolar_distance,
)


# ---- Tunables -------------------------------------------------------------
POSE_RANSAC_PX = float(os.environ.get("VISBENCH_POSE_RANSAC_PX", 0.5))
HOMOG_RANSAC_PX_BASE = float(os.environ.get("VISBENCH_HOMOG_RANSAC_PX_BASE", 3.0))
HOMOG_REF_MIN_SIDE = 480.0
FUND_RANSAC_PX = float(os.environ.get("VISBENCH_FUND_RANSAC_PX", 0.5))
FUND_RANSAC_CONF = 0.999
FUND_RANSAC_MAX_ITERS = 100_000
POSE_FAIL_DEG = 90.0


def _scale_K(K: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    K = K.copy()
    K[0] *= scale_x
    K[1] *= scale_y
    return K


_SIZE_CACHE: dict = {}
_IMAGE_CACHE: dict = {}
_IMAGE_CACHE_MAX = int(os.environ.get("VISBENCH_IMG_CACHE_MAX", 256))


def _pil_size(path):
    key = str(path)
    if key in _SIZE_CACHE:
        return _SIZE_CACHE[key]
    with Image.open(path) as im:
        wh = im.size
    if len(_SIZE_CACHE) < 4096:
        _SIZE_CACHE[key] = wh
    return wh


def _cached_load(matcher, path, resize=None):
    key = (str(path), resize if not isinstance(resize, tuple) else tuple(resize))
    cached = _IMAGE_CACHE.get(key)
    if cached is not None:
        return cached
    img = matcher.load_image(path, resize=resize) if resize is not None else matcher.load_image(path)
    if len(_IMAGE_CACHE) >= _IMAGE_CACHE_MAX:
        _IMAGE_CACHE.pop(next(iter(_IMAGE_CACHE)))
    _IMAGE_CACHE[key] = img
    return img


def reset_caches():
    """Call between matcher runs to free memory."""
    _IMAGE_CACHE.clear()
    _SIZE_CACHE.clear()


def _prepare_pair_images(matcher, im_A_path, im_B_path,
                         resize_long: int | None = None,
                         resize_hw: tuple[int, int] | None = None):
    w0, h0 = _pil_size(im_A_path)
    w1, h1 = _pil_size(im_B_path)

    if resize_hw is not None:
        nh0, nw0 = int(resize_hw[0]), int(resize_hw[1])
        nh1, nw1 = nh0, nw0
        sx0, sy0 = nw0 / w0, nh0 / h0
        sx1, sy1 = nw1 / w1, nh1 / h1
        img0 = _cached_load(matcher, im_A_path, resize=(nh0, nw0))
        img1 = _cached_load(matcher, im_B_path, resize=(nh1, nw1))
    elif resize_long is not None:
        s0 = resize_long / max(w0, h0)
        s1 = resize_long / max(w1, h1)
        nh0, nw0 = int(round(h0 * s0)), int(round(w0 * s0))
        nh1, nw1 = int(round(h1 * s1)), int(round(w1 * s1))
        sx0, sy0 = nw0 / w0, nh0 / h0
        sx1, sy1 = nw1 / w1, nh1 / h1
        img0 = _cached_load(matcher, im_A_path, resize=(nh0, nw0))
        img1 = _cached_load(matcher, im_B_path, resize=(nh1, nw1))
    else:
        nh0, nw0 = h0, w0
        nh1, nw1 = h1, w1
        sx0 = sy0 = sx1 = sy1 = 1.0
        img0 = _cached_load(matcher, im_A_path)
        img1 = _cached_load(matcher, im_B_path)
    return (img0, img1,
            (w0, h0), (w1, h1),
            (nh0, nw0), (nh1, nw1),
            (sx0, sy0), (sx1, sy1))


def eval_pose_pairs(matcher, pairs, resize_long: int = 1200, ransac_runs: int = 5,
                    thresholds=(5, 10, 20), progress: bool = True):
    """`pairs` yields {im_A_path, im_B_path, K0, K1, R_0to1, t_0to1}.

    RoMa/LoFTR MegaDepth convention: long-side resize, scale K, essential-matrix
    RANSAC averaged over `ransac_runs` shuffles, AUC at thresholds in degrees.
    """
    tot_err_pose, tot_err_t, tot_err_R = [], [], []
    timings, inlier_counts, match_counts = [], [], []
    epipolar_precision, n_kpts0, n_kpts1 = [], [], []
    n_pair_failures = 0

    for pair in tqdm(pairs, desc="pairs", disable=not progress):
        im_A_path = Path(pair["im_A_path"])
        im_B_path = Path(pair["im_B_path"])
        K0 = np.asarray(pair["K0"], dtype=np.float64)
        K1 = np.asarray(pair["K1"], dtype=np.float64)
        R_gt = np.asarray(pair["R_0to1"], dtype=np.float64)
        t_gt = np.asarray(pair["t_0to1"], dtype=np.float64)

        resize_hw = pair.get("resize_hw")
        K_already_scaled = bool(pair.get("K_already_scaled", False))
        try:
            (img0, img1, _, _, _, _,
             (sx0, sy0), (sx1, sy1)) = _prepare_pair_images(
                matcher, im_A_path, im_B_path,
                resize_long=resize_long, resize_hw=resize_hw,
            )
            if K_already_scaled:
                K0s, K1s = K0.copy(), K1.copy()
            else:
                K0s = _scale_K(K0, sx0, sy0)
                K1s = _scale_K(K1, sx1, sy1)

            t0 = time.time()
            result = matcher(img0, img1)
            timings.append(time.time() - t0)
        except Exception as e:  # noqa: BLE001
            n_pair_failures += 1
            if n_pair_failures <= 3:
                warnings.warn(
                    f"eval_pose_pairs: pair failed ({type(e).__name__}: {e}); "
                    f"skipping {im_A_path.name} <-> {im_B_path.name}",
                    stacklevel=2,
                )
            for _ in range(ransac_runs):
                tot_err_pose.append(POSE_FAIL_DEG)
                tot_err_t.append(POSE_FAIL_DEG)
                tot_err_R.append(POSE_FAIL_DEG)
            match_counts.append(0)
            inlier_counts.append(0)
            n_kpts0.append(0)
            n_kpts1.append(0)
            epipolar_precision.append(0.0)
            continue

        kpts0 = result["matched_kpts0"]
        kpts1 = result["matched_kpts1"]
        match_counts.append(len(kpts0))
        inlier_counts.append(int(result.get("num_inliers", 0)))
        n_kpts0.append(len(result.get("all_kpts0", np.empty((0, 2)))))
        n_kpts1.append(len(result.get("all_kpts1", np.empty((0, 2)))))

        if len(kpts0) > 0:
            epi = symmetric_epipolar_distance(kpts0, kpts1, K0s, K1s, R_gt, t_gt)
            epipolar_precision.append(float((epi < 2.0).mean()))
        else:
            epipolar_precision.append(0.0)

        for _ in range(ransac_runs):
            if len(kpts0) < 5:
                tot_err_pose.append(POSE_FAIL_DEG)
                tot_err_t.append(POSE_FAIL_DEG)
                tot_err_R.append(POSE_FAIL_DEG)
                continue
            perm = np.random.permutation(len(kpts0))
            k0p = kpts0[perm]
            k1p = kpts1[perm]
            norm_thr = POSE_RANSAC_PX / (
                np.mean(np.abs(K0s[:2, :2])) + np.mean(np.abs(K1s[:2, :2]))
            )
            out = estimate_pose_essential(k0p, k1p, K0s, K1s, norm_thr)
            if out is None:
                tot_err_pose.append(POSE_FAIL_DEG)
                tot_err_t.append(POSE_FAIL_DEG)
                tot_err_R.append(POSE_FAIL_DEG)
                continue
            R_est, t_est, _ = out
            e_t, e_R = compute_pose_error(R_est, t_est, R_gt, t_gt)
            tot_err_t.append(e_t)
            tot_err_R.append(e_R)
            tot_err_pose.append(max(e_t, e_R))

    if not match_counts:
        warnings.warn("eval_pose_pairs: zero pairs consumed.", stacklevel=2)
    if n_pair_failures > 0:
        warnings.warn(
            f"eval_pose_pairs: {n_pair_failures} pair(s) failed and were counted as POSE_FAIL_DEG.",
            stacklevel=2,
        )
    errs = np.array(tot_err_pose) if tot_err_pose else np.array([POSE_FAIL_DEG])
    aucs = pose_auc(errs, list(thresholds))
    acc = {f"acc_{int(t)}": float((errs < t).mean()) for t in thresholds}
    return {
        **{f"auc_{int(t)}": a for t, a in zip(thresholds, aucs)},
        **acc,
        "num_pair_failures": n_pair_failures,
        "num_pairs": len(match_counts),
        "num_ransac_runs": ransac_runs,
        "mean_matches": float(np.mean(match_counts)) if match_counts else 0.0,
        "mean_inliers": float(np.mean(inlier_counts)) if inlier_counts else 0.0,
        "mean_epi_precision_2px": float(np.mean(epipolar_precision)) if epipolar_precision else 0.0,
        "mean_kpts0": float(np.mean(n_kpts0)) if n_kpts0 else 0.0,
        "mean_kpts1": float(np.mean(n_kpts1)) if n_kpts1 else 0.0,
        "mean_time_s": float(np.mean(timings)) if timings else 0.0,
        "resize_long": resize_long,
    }


def eval_homography_pairs(matcher, pairs, resize_long: int | None = None,
                          thresholds=(1, 3, 5, 10),
                          ransac_thresh_base: float = HOMOG_RANSAC_PX_BASE,
                          progress: bool = True):
    """`pairs` yields {im_A_path, im_B_path, H_gt, subset}. HPatches protocol."""
    errors_by_subset: dict[str, list] = {}
    mma_by_subset: dict[str, list] = {}
    repeat_by_subset: dict[str, list] = {}
    mscore_by_subset: dict[str, list] = {}
    timings, match_counts, n_kpts0, n_kpts1 = [], [], [], []
    mma_thresholds = (1, 2, 3, 5, 10)
    n_pair_failures = 0

    for pair in tqdm(pairs, desc="pairs", disable=not progress):
        im_A_path = Path(pair["im_A_path"])
        im_B_path = Path(pair["im_B_path"])
        H_gt = np.asarray(pair["H_gt"], dtype=np.float64)
        subset = pair.get("subset", "all")

        try:
            (img0, img1,
             (w0, h0), (w1, h1),
             _, _,
             (sx0, sy0), (sx1, sy1)) = _prepare_pair_images(
                matcher, im_A_path, im_B_path, resize_long=resize_long,
            )
            t0 = time.time()
            result = matcher(img0, img1)
            timings.append(time.time() - t0)
        except Exception as e:  # noqa: BLE001
            n_pair_failures += 1
            if n_pair_failures <= 3:
                warnings.warn(
                    f"eval_homography_pairs: pair failed ({type(e).__name__}: {e}); "
                    f"skipping {im_A_path.name} <-> {im_B_path.name}",
                    stacklevel=2,
                )
            w0, h0 = _pil_size(im_A_path)
            err = float(max(w0, h0))
            errors_by_subset.setdefault(subset, []).append(err)
            errors_by_subset.setdefault("all", []).append(err)
            match_counts.append(0)
            n_kpts0.append(0)
            n_kpts1.append(0)
            continue

        kpts0 = result["matched_kpts0"]
        kpts1 = result["matched_kpts1"]
        all_k0 = result.get("all_kpts0", np.empty((0, 2)))
        all_k1 = result.get("all_kpts1", np.empty((0, 2)))
        match_counts.append(len(kpts0))
        n_kpts0.append(len(all_k0))
        n_kpts1.append(len(all_k1))

        k0 = kpts0.copy().astype(np.float64) if len(kpts0) else kpts0
        k1 = kpts1.copy().astype(np.float64) if len(kpts1) else kpts1
        ak0 = all_k0.copy().astype(np.float64) if len(all_k0) else all_k0
        ak1 = all_k1.copy().astype(np.float64) if len(all_k1) else all_k1
        if resize_long is not None and len(k0):
            k0[:, 0] /= sx0; k0[:, 1] /= sy0
            k1[:, 0] /= sx1; k1[:, 1] /= sy1
            if len(ak0):
                ak0[:, 0] /= sx0; ak0[:, 1] /= sy0
            if len(ak1):
                ak1[:, 0] /= sx1; ak1[:, 1] /= sy1

        if len(k0) >= 4:
            ransac_thr = ransac_thresh_base * min(w1, h1) / HOMOG_REF_MIN_SIDE
            H_est = estimate_homography(k0, k1, ransac_thr)
        else:
            H_est = None
        err = float(max(w0, h0)) if H_est is None else homography_corner_error(H_est, H_gt, w0, h0)
        errors_by_subset.setdefault(subset, []).append(err)
        errors_by_subset.setdefault("all", []).append(err)

        mma = mma_homography(k0, k1, H_gt, thresholds=mma_thresholds)
        mma_by_subset.setdefault(subset, []).append(mma)
        mma_by_subset.setdefault("all", []).append(mma)

        rep = repeatability(ak0, ak1, H_gt, image_shape1=(h1, w1), px_thresh=3.0)
        repeat_by_subset.setdefault(subset, []).append(rep["repeatability"])
        repeat_by_subset.setdefault("all", []).append(rep["repeatability"])

        ms = matching_score(k0, k1, ak0, H_gt, image_shape1=(h1, w1), px_thresh=3.0)
        mscore_by_subset.setdefault(subset, []).append(ms)
        mscore_by_subset.setdefault("all", []).append(ms)

    if not match_counts:
        warnings.warn("eval_homography_pairs: zero pairs consumed.", stacklevel=2)
    if n_pair_failures > 0:
        warnings.warn(
            f"eval_homography_pairs: {n_pair_failures} pair(s) failed; counted as worst-case AUC.",
            stacklevel=2,
        )
    out = {"num_pair_failures": n_pair_failures}
    for subset, errs in errors_by_subset.items():
        aucs = hpatches_auc(np.array(errs), list(thresholds))
        for t, a in zip(thresholds, aucs):
            out[f"{subset}_auc_{int(t)}"] = a
        out[f"{subset}_num_pairs"] = len(errs)
    for subset, recs in mma_by_subset.items():
        for t in mma_thresholds:
            out[f"{subset}_mma_{int(t)}"] = float(np.mean([r[f"mma_{int(t)}"] for r in recs]))
    for subset, vals in repeat_by_subset.items():
        out[f"{subset}_repeatability"] = float(np.mean(vals))
    for subset, vals in mscore_by_subset.items():
        out[f"{subset}_matching_score"] = float(np.mean(vals))
    out["mean_time_s"] = float(np.mean(timings)) if timings else 0.0
    out["mean_matches"] = float(np.mean(match_counts)) if match_counts else 0.0
    out["mean_kpts0"] = float(np.mean(n_kpts0)) if n_kpts0 else 0.0
    out["mean_kpts1"] = float(np.mean(n_kpts1)) if n_kpts1 else 0.0
    out["resize_long"] = resize_long
    return out


def _epi_distance(F, kpts0, kpts1):
    if F is None or len(kpts0) == 0:
        return np.full((len(kpts0),), np.inf, dtype=np.float64)
    h0 = np.concatenate([kpts0, np.ones((len(kpts0), 1))], axis=1)
    h1 = np.concatenate([kpts1, np.ones((len(kpts1), 1))], axis=1)
    Fx0 = h0 @ F.T
    Ftx1 = h1 @ F
    num = np.abs((h1 * Fx0).sum(axis=1))
    n1 = np.sqrt(Fx0[:, 0] ** 2 + Fx0[:, 1] ** 2)
    n0 = np.sqrt(Ftx1[:, 0] ** 2 + Ftx1[:, 1] ** 2)
    return num / np.maximum(n1, 1e-12) + num / np.maximum(n0, 1e-12)


def eval_correspondence_pairs(matcher, pairs, resize_long: int | None = None,
                              thresholds=(1, 2, 5, 10), progress: bool = True,
                              ransac_thresh: float = FUND_RANSAC_PX,
                              ths=tuple(range(20))):
    """`pairs` yields {im_A_path, im_B_path, corrs[Nx4 GT correspondences]}.
    LoMa-canonical WxBS protocol: F via USAC_MAGSAC, per-pair PCK, mAA = mean cumulative.
    """
    pck_by_subset: dict[str, list] = {}
    ths_arr = np.asarray(ths, dtype=np.float64)
    DEGENERATE_F = np.array([[0.0, 0.0, 0.0],
                             [0.0, 0.0, -1.0],
                             [0.0, 1.0, 0.0]], dtype=np.float64)
    timings, match_counts, inlier_counts, n_kpts0, n_kpts1 = [], [], [], [], []
    n_pair_failures = 0

    for pair in tqdm(pairs, desc="pairs", disable=not progress):
        im_A_path = Path(pair["im_A_path"])
        im_B_path = Path(pair["im_B_path"])
        gt = np.asarray(pair["corrs"], dtype=np.float64).reshape(-1, 4)
        subset = str(pair.get("subset") or pair.get("set") or "all")

        try:
            (img0, img1, _, _, _, _,
             (sx0, sy0), (sx1, sy1)) = _prepare_pair_images(
                matcher, im_A_path, im_B_path, resize_long=resize_long,
            )
            t0 = time.time()
            result = matcher(img0, img1)
            timings.append(time.time() - t0)
            kpts0 = np.asarray(result["matched_kpts0"], dtype=np.float64).copy()
            kpts1 = np.asarray(result["matched_kpts1"], dtype=np.float64).copy()
        except Exception as e:  # noqa: BLE001
            n_pair_failures += 1
            if n_pair_failures <= 3:
                warnings.warn(
                    f"eval_correspondence_pairs: pair failed ({type(e).__name__}: {e}); "
                    f"skipping {im_A_path.name} <-> {im_B_path.name}",
                    stacklevel=2,
                )
            zero_pck = np.zeros(len(ths_arr), dtype=np.float64)
            pck_by_subset.setdefault(subset, []).append(zero_pck)
            pck_by_subset.setdefault("all", []).append(zero_pck)
            match_counts.append(0)
            inlier_counts.append(0)
            n_kpts0.append(0)
            n_kpts1.append(0)
            continue

        match_counts.append(len(kpts0))
        inlier_counts.append(int(result.get("num_inliers", 0)))
        n_kpts0.append(len(result.get("all_kpts0", np.empty((0, 2)))))
        n_kpts1.append(len(result.get("all_kpts1", np.empty((0, 2)))))

        gt_resized = gt.copy()
        gt_resized[:, 0] *= sx0; gt_resized[:, 1] *= sy0
        gt_resized[:, 2] *= sx1; gt_resized[:, 3] *= sy1

        F_pred = None
        if len(kpts0) >= 8:
            try:
                F_pred, _ = cv2.findFundamentalMat(
                    kpts0.astype(np.float32),
                    kpts1.astype(np.float32),
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=ransac_thresh,
                    confidence=FUND_RANSAC_CONF,
                    maxIters=FUND_RANSAC_MAX_ITERS,
                )
            except cv2.error:
                F_pred = None
        if F_pred is None or F_pred.shape != (3, 3):
            F_pred = DEGENERATE_F

        errs = _epi_distance(F_pred, gt_resized[:, :2], gt_resized[:, 2:4])
        pck = (errs[:, None] <= ths_arr[None, :]).mean(axis=0)
        pck_by_subset.setdefault(subset, []).append(pck)
        pck_by_subset.setdefault("all", []).append(pck)

    if not match_counts:
        warnings.warn("eval_correspondence_pairs: zero pairs consumed.", stacklevel=2)
    if n_pair_failures > 0:
        warnings.warn(
            f"eval_correspondence_pairs: {n_pair_failures} pair(s) failed; PCK=0 for those.",
            stacklevel=2,
        )
    out: dict = {"num_pair_failures": n_pair_failures}
    for subset, vecs in pck_by_subset.items():
        if not vecs:
            continue
        avg = np.stack(vecs, axis=0).mean(axis=0)
        for T in (1, 2, 5, 10):
            up_to = min(T + 1, len(avg))
            out[f"{subset}_mAA_{T}"] = float(avg[:up_to].mean())
        if len(avg) > 5:
            out[f"{subset}_pck_5"] = float(avg[5])
        if len(avg) > 10:
            out[f"{subset}_pck_10"] = float(avg[10])
        out[f"{subset}_num_pairs"] = len(vecs)
    out["mean_time_s"] = float(np.mean(timings)) if timings else 0.0
    out["mean_matches"] = float(np.mean(match_counts)) if match_counts else 0.0
    out["mean_inliers"] = float(np.mean(inlier_counts)) if inlier_counts else 0.0
    out["mean_kpts0"] = float(np.mean(n_kpts0)) if n_kpts0 else 0.0
    out["mean_kpts1"] = float(np.mean(n_kpts1)) if n_kpts1 else 0.0
    out["resize_long"] = resize_long
    return out


def eval_fundamental_pairs(matcher, pairs, resize_long: int | None = None,
                           thresholds=(1, 3, 5, 10), progress: bool = True):
    """`pairs` yields {im_A_path, im_B_path, F_gt}. WxBS-style fundamental eval.

    Uses the GT fundamental matrix F_gt directly to score the matcher's
    predicted matches: for each match (x0, x1), compute symmetric epipolar
    distance to F_gt; report per-pair PCK at thresholds + mAA aggregation.
    """
    pck_by_subset: dict[str, list] = {}
    timings, match_counts, n_kpts0, n_kpts1 = [], [], [], []
    n_pair_failures = 0

    for pair in tqdm(pairs, desc="pairs", disable=not progress):
        im_A_path = Path(pair["im_A_path"])
        im_B_path = Path(pair["im_B_path"])
        F_gt = np.asarray(pair["F_gt"], dtype=np.float64)
        subset = str(pair.get("subset") or pair.get("scene") or "all")

        try:
            (img0, img1, _, _, _, _,
             (sx0, sy0), (sx1, sy1)) = _prepare_pair_images(
                matcher, im_A_path, im_B_path, resize_long=resize_long,
            )
            t0 = time.time()
            result = matcher(img0, img1)
            timings.append(time.time() - t0)
        except Exception as e:  # noqa: BLE001
            n_pair_failures += 1
            if n_pair_failures <= 3:
                warnings.warn(
                    f"eval_fundamental_pairs: pair failed ({type(e).__name__}: {e}); "
                    f"skipping {im_A_path.name} <-> {im_B_path.name}",
                    stacklevel=2,
                )
            zero_pck = np.zeros(len(thresholds), dtype=np.float64)
            pck_by_subset.setdefault(subset, []).append(zero_pck)
            pck_by_subset.setdefault("all", []).append(zero_pck)
            match_counts.append(0)
            n_kpts0.append(0)
            n_kpts1.append(0)
            continue

        kpts0 = np.asarray(result["matched_kpts0"], dtype=np.float64).copy()
        kpts1 = np.asarray(result["matched_kpts1"], dtype=np.float64).copy()
        match_counts.append(len(kpts0))
        n_kpts0.append(len(result.get("all_kpts0", np.empty((0, 2)))))
        n_kpts1.append(len(result.get("all_kpts1", np.empty((0, 2)))))

        # Bring matches back to ORIGINAL image frame so they're comparable to F_gt
        # (which is defined in the original image coordinates).
        if resize_long is not None and len(kpts0):
            kpts0[:, 0] /= sx0; kpts0[:, 1] /= sy0
            kpts1[:, 0] /= sx1; kpts1[:, 1] /= sy1

        ths_arr = np.asarray(thresholds, dtype=np.float64)
        if len(kpts0) == 0:
            pck = np.zeros(len(thresholds), dtype=np.float64)
        else:
            errs = _epi_distance(F_gt, kpts0, kpts1)
            pck = (errs[:, None] <= ths_arr[None, :]).mean(axis=0)
        pck_by_subset.setdefault(subset, []).append(pck)
        pck_by_subset.setdefault("all", []).append(pck)

    if not match_counts:
        warnings.warn("eval_fundamental_pairs: zero pairs consumed.", stacklevel=2)
    if n_pair_failures > 0:
        warnings.warn(
            f"eval_fundamental_pairs: {n_pair_failures} pair(s) failed; PCK=0 for those.",
            stacklevel=2,
        )

    out: dict = {"num_pair_failures": n_pair_failures}
    for subset, vecs in pck_by_subset.items():
        if not vecs:
            continue
        avg = np.stack(vecs, axis=0).mean(axis=0)
        for t, val in zip(thresholds, avg):
            out[f"{subset}_pck_{int(t)}"] = float(val)
        # mAA = trapezoidal area under PCK(threshold) curve up to max threshold.
        out[f"{subset}_mAA"] = float(np.trapezoid(avg, x=np.asarray(thresholds, dtype=np.float64))
                                      / float(thresholds[-1]))
        out[f"{subset}_num_pairs"] = len(vecs)
    out["mean_time_s"] = float(np.mean(timings)) if timings else 0.0
    out["mean_matches"] = float(np.mean(match_counts)) if match_counts else 0.0
    out["mean_kpts0"] = float(np.mean(n_kpts0)) if n_kpts0 else 0.0
    out["mean_kpts1"] = float(np.mean(n_kpts1)) if n_kpts1 else 0.0
    out["resize_long"] = resize_long
    return out


def _matcher_describe(matcher, raw_u8: np.ndarray, kpts_xy: np.ndarray,
                      kpt_sizes: np.ndarray | None = None):
    """Best-effort extract descriptors at fixed kpts via cv2 feature objects.
    Supports SIFT/RootSIFT/ORB/AKAZE/BRISK exposed as ._sift / ._orb / etc. on the matcher.
    """
    sizes = kpt_sizes if kpt_sizes is not None else \
            np.full(len(kpts_xy), 8.0, dtype=np.float32)

    feat_obj = None
    for attr in ("_sift", "_rootsift", "_orb", "_akaze", "_brisk", "_kaze"):
        f = getattr(matcher, attr, None)
        if f is not None and hasattr(f, "compute"):
            feat_obj = f
            break
    if feat_obj is None:
        return None
    gray = raw_u8 if raw_u8.ndim == 2 else cv2.cvtColor(raw_u8, cv2.COLOR_RGB2GRAY)
    cv_kpts = [cv2.KeyPoint(float(x), float(y), float(s))
               for (x, y), s in zip(kpts_xy, sizes)]
    try:
        _kp_out, desc = feat_obj.compute(gray, cv_kpts)
        if desc is None or len(desc) == 0:
            return None
        if "Root" in type(matcher).__name__ and desc.dtype != np.uint8:
            d32 = desc.astype(np.float32)
            d32 = d32 / np.maximum(d32.sum(axis=1, keepdims=True), 1e-12)
            desc = np.sqrt(d32)
        metric = "hamming" if desc.dtype == np.uint8 else "l2"
        return (desc, metric)
    except Exception:
        return None


def _compute_map_topk(d_a: np.ndarray, d_b: np.ndarray, metric: str = "l2") -> dict:
    if len(d_a) == 0 or len(d_b) == 0 or len(d_a) != len(d_b):
        return {"top1_acc": 0.0, "top5_acc": 0.0, "mAP": 0.0, "n": 0}
    if metric == "hamming":
        a = d_a.astype(np.uint8)
        b = d_b.astype(np.uint8)
        bits = np.unpackbits(a[:, None, :] ^ b[None, :, :], axis=-1).sum(axis=-1)
        D = bits
    else:
        a2 = (d_a ** 2).sum(axis=1, keepdims=True)
        b2 = (d_b ** 2).sum(axis=1, keepdims=True).T
        D = a2 + b2 - 2.0 * d_a @ d_b.T
    order = np.argsort(D, axis=1)
    n = len(d_a)
    ranks = np.empty(n, dtype=np.int64)
    for i in range(n):
        ranks[i] = int(np.where(order[i] == i)[0][0])
    return {
        "top1_acc": float((ranks == 0).mean()),
        "top5_acc": float((ranks < 5).mean()),
        "mAP": float((1.0 / (ranks + 1)).mean()),
        "n": n,
    }


def eval_descriptor_pairs(matcher, pairs, resize_long: int | None = None,
                          progress: bool = True):
    """`pairs` yields {im_A_path, im_B_path, kpts_A, kpts_B, kpt_sizes_A, subset}.
    Pure descriptor isolation: extract at fixed kpts, rank by distance.
    Only supports cv2-feature CustomFeatureMatcher subclasses; other matchers report `skipped`.
    """
    timings = []
    metrics_by_subset: dict[str, list] = {}
    skipped = False

    for pair in tqdm(pairs, desc="pairs", disable=not progress):
        im_A_path = Path(pair["im_A_path"])
        im_B_path = Path(pair["im_B_path"])
        kpts_A = np.asarray(pair["kpts_A"], dtype=np.float32)
        kpts_B = np.asarray(pair["kpts_B"], dtype=np.float32)
        sizes_A = np.asarray(pair.get("kpt_sizes_A",
                                       np.full(len(kpts_A), 8.0, dtype=np.float32)),
                              dtype=np.float32)
        subset = str(pair.get("subset") or "all")

        rawA = cv2.imread(str(im_A_path), cv2.IMREAD_COLOR)
        rawB = cv2.imread(str(im_B_path), cv2.IMREAD_COLOR)
        if rawA is None or rawB is None or len(kpts_A) == 0:
            continue
        rawA = cv2.cvtColor(rawA, cv2.COLOR_BGR2RGB)
        rawB = cv2.cvtColor(rawB, cv2.COLOR_BGR2RGB)

        t0 = time.time()
        rA = _matcher_describe(matcher, rawA, kpts_A, sizes_A)
        rB = _matcher_describe(matcher, rawB, kpts_B, sizes_A)
        timings.append(time.time() - t0)
        if rA is None or rB is None:
            skipped = True
            break
        d_a, metric = rA
        d_b, _ = rB
        n = min(len(d_a), len(d_b))
        if n < 8:
            continue
        m = _compute_map_topk(d_a[:n], d_b[:n], metric=metric)
        metrics_by_subset.setdefault(subset, []).append(m)
        metrics_by_subset.setdefault("all", []).append(m)

    out: dict = {"skipped": skipped}
    for subset, results in metrics_by_subset.items():
        if not results:
            continue
        out[f"{subset}_top1"] = float(np.mean([r["top1_acc"] for r in results]))
        out[f"{subset}_top5"] = float(np.mean([r["top5_acc"] for r in results]))
        out[f"{subset}_mAP"] = float(np.mean([r["mAP"] for r in results]))
        out[f"{subset}_n_pairs"] = len(results)
        out[f"{subset}_n_kpts"] = int(np.sum([r["n"] for r in results]))
    out["mean_time_s"] = float(np.mean(timings)) if timings else 0.0
    out["resize_long"] = resize_long
    return out
