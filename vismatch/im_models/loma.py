import torch

from typing import Literal

from vismatch import THIRD_PARTY_DIR, BaseMatcher  # noqa: F401
from vismatch.utils import add_to_path, resize_to_divisible, set_device_globals

add_to_path(THIRD_PARTY_DIR.joinpath("LoMa/src"))

from loma.loma import (
    LoMaB,
    LoMaB128,
    LoMaL,
    LoMaG,
    LoMaR,
    LoMa,
    filter_matches,
    to_pixel_coords,
)


class LoMaMatcher(BaseMatcher):
    divisible_size = 14  # for DINOv2 in the descriptor of LoMa-{B, L, G, R}. LoMa-B128 can handle arbitrary resolutions and is more lightweight.

    def __init__(
        self,
        device="cpu",
        max_num_keypoints=2048,
        arch: Literal["LoMa-B", "LoMa-L", "LoMa-G", "LoMa-B128", "LoMa-R"] = "LoMa-B",
        **kwargs,
    ):
        super().__init__(device, **kwargs)
        self.max_num_keypoints = max_num_keypoints

        arch = arch.lower()

        if arch == "loma-b":
            cfg = LoMaB()
        elif arch == "loma-l":
            cfg = LoMaL()
        elif arch == "loma-g":
            cfg = LoMaG()
        elif arch == "loma-b128":
            cfg = LoMaB128()
        elif arch == "loma-r":
            cfg = LoMaR()
        else:
            raise ValueError(
                f"Unsupported architecture '{arch}' for LoMa. Supported: 'LoMa-B', 'LoMa-L', 'LoMa-G', 'LoMa-B128', 'LoMa-R'."
            )

        # LoMa moves inputs to a module-level `device` global, resolved at import time to cuda
        # whenever a GPU is visible; rebind it so inference tensors follow the requested device.
        # On cpu also drop the bfloat16 amp global: cpus lack native bf16 kernels and run it far
        # slower than fp32 (measured ~5x on x86, 100x+ on Apple Silicon).
        set_device_globals("loma", self.device, amp_dtype=torch.float32 if self.device == "cpu" else None)

        # This automatically loads weights using torch.hub.load_state_dict_from_url
        self.matcher = LoMa(cfg).to(self.device)

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        img = resize_to_divisible(img, self.divisible_size)
        img = img.unsqueeze(0).to(self.device)
        return img, orig_shape

    # ------------------------------------------------------------------
    # Cached sliding-window path (video pseudo-labeling).
    # ------------------------------------------------------------------
    _DD_CACHE_CAP = 6

    def _detect_describe_cached(self, img, key):
        cache = getattr(self, "_dd_cache", None)
        if cache is None:
            from collections import OrderedDict

            cache = self._dd_cache = OrderedDict()
        hit = cache.get(key)
        if hit is not None:
            return hit
        img_p, orig_shape = self.preprocess(img)
        H, W = img_p.shape[-2:]
        kpts, desc, _, _ = self.matcher.detect_and_describe(img_p, self.max_num_keypoints)
        hit = (kpts, desc, orig_shape, H, W)
        cache[key] = hit
        while len(cache) > self._DD_CACHE_CAP:
            cache.popitem(last=False)
        return hit

    @torch.inference_mode()
    def match_cached(self, img0, key0, img1, key1):
        """Sliding-window matching: detect+describe results are cached per
        frame `key`, so a video frame appearing in two consecutive pairs is
        only detected/described once (the dominant cost for LoMa-G). Skips
        BaseMatcher.forward()'s homography RANSAC — callers of this path run
        their own robust fitting. Returns {'matched_kpts0','matched_kpts1'}
        as tensors in original-image pixel coords (same convention as
        _forward, including the COLMAP -0.5 offset)."""
        kpts0, desc0, shape0, H0, W0 = self._detect_describe_cached(img0, key0)
        kpts1, desc1, shape1, H1, W1 = self._detect_describe_cached(img1, key1)

        scores = self.matcher(kpts0, kpts1, desc0, desc1)["scores"]
        m0, _, _, _ = filter_matches(scores, self.matcher.cfg.filter_threshold)

        valid = m0[0] > -1
        matched_kpts0 = to_pixel_coords(kpts0[0][torch.where(valid)[0]], H0, W0)
        matched_kpts1 = to_pixel_coords(kpts1[0][m0[0][valid]], H1, W1)

        matched_kpts0 = self.rescale_coords(matched_kpts0, *shape0, H0, W0)
        matched_kpts1 = self.rescale_coords(matched_kpts1, *shape1, H1, W1)

        offset = 0.5  # COLMAP convention, same as _forward
        return {
            "matched_kpts0": matched_kpts0 - offset,
            "matched_kpts1": matched_kpts1 - offset,
        }

    @torch.inference_mode()
    def match_cached_conf(self, img0, key0, img1, key1):
        """Like match_cached but also returns per-correspondence confidence
        (mutual-NN match score from filter_matches' mscores0), for distilling
        LoMa correspondences into a student with teacher-confidence weighting.
        Returns {'matched_kpts0','matched_kpts1','confidence'} (conf in [0,1])."""
        kpts0, desc0, shape0, H0, W0 = self._detect_describe_cached(img0, key0)
        kpts1, desc1, shape1, H1, W1 = self._detect_describe_cached(img1, key1)

        scores = self.matcher(kpts0, kpts1, desc0, desc1)["scores"]
        m0, _, mscores0, _ = filter_matches(scores, self.matcher.cfg.filter_threshold)

        valid = m0[0] > -1
        matched_kpts0 = to_pixel_coords(kpts0[0][torch.where(valid)[0]], H0, W0)
        matched_kpts1 = to_pixel_coords(kpts1[0][m0[0][valid]], H1, W1)

        matched_kpts0 = self.rescale_coords(matched_kpts0, *shape0, H0, W0)
        matched_kpts1 = self.rescale_coords(matched_kpts1, *shape1, H1, W1)

        offset = 0.5  # COLMAP convention, same as _forward / match_cached
        return {
            "matched_kpts0": matched_kpts0 - offset,
            "matched_kpts1": matched_kpts1 - offset,
            "confidence": mscores0[0][valid],
        }

    @torch.inference_mode()
    def match_cached_distill(self, img0, key0, img1, key1):
        """match_cached_conf + the LoMa (DINOv2) descriptors at each matched
        keypoint, for true distillation (descriptor + soft-assignment) into a
        student. Returns matched_kpts0/1 (orig px), confidence, and desc0/desc1
        ([N, 256] teacher descriptors aligned to the matches)."""
        kpts0, desc0, shape0, H0, W0 = self._detect_describe_cached(img0, key0)
        kpts1, desc1, shape1, H1, W1 = self._detect_describe_cached(img1, key1)

        scores = self.matcher(kpts0, kpts1, desc0, desc1)["scores"]
        m0, _, mscores0, _ = filter_matches(scores, self.matcher.cfg.filter_threshold)

        valid = m0[0] > -1
        idxA = torch.where(valid)[0]
        idxB = m0[0][valid]
        matched_kpts0 = to_pixel_coords(kpts0[0][idxA], H0, W0)
        matched_kpts1 = to_pixel_coords(kpts1[0][idxB], H1, W1)
        matched_kpts0 = self.rescale_coords(matched_kpts0, *shape0, H0, W0)
        matched_kpts1 = self.rescale_coords(matched_kpts1, *shape1, H1, W1)

        offset = 0.5
        return {
            "matched_kpts0": matched_kpts0 - offset,
            "matched_kpts1": matched_kpts1 - offset,
            "confidence": mscores0[0][valid],
            "desc0": desc0[0][idxA],
            "desc1": desc1[0][idxB],
        }

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        H0, W0 = img0.shape[-2:]
        H1, W1 = img1.shape[-2:]

        kpts0, desc0, _, _ = self.matcher.detect_and_describe(img0, self.max_num_keypoints)
        kpts1, desc1, _, _ = self.matcher.detect_and_describe(img1, self.max_num_keypoints)

        scores = self.matcher(kpts0, kpts1, desc0, desc1)["scores"]
        m0, _, mscores0, _ = filter_matches(scores, self.matcher.cfg.filter_threshold)

        # LoMa returns bfloat16 confidences; cast to float so numpy can convert them.
        valid = m0[0] > -1
        matched_conf = mscores0[0][valid].float()
        matched_kpts0 = to_pixel_coords(kpts0[0][torch.where(valid)[0]], H0, W0)
        matched_kpts1 = to_pixel_coords(kpts1[0][m0[0][valid]], H1, W1)

        all_kpts0 = to_pixel_coords(kpts0[0], H0, W0)
        all_kpts1 = to_pixel_coords(kpts1[0], H1, W1)

        matched_kpts0 = self.rescale_coords(matched_kpts0, *img0_orig_shape, H0, W0)
        matched_kpts1 = self.rescale_coords(matched_kpts1, *img1_orig_shape, H1, W1)
        all_kpts0 = self.rescale_coords(all_kpts0, *img0_orig_shape, H0, W0)
        all_kpts1 = self.rescale_coords(all_kpts1, *img1_orig_shape, H1, W1)

        # LoMa uses COLMAP convention for pixel coords (see https://github.com/gmberton/vismatch/pull/63) so we subtact 0.5 for repo compatability
        offset = 0.5
        matched_kpts0 -= offset
        matched_kpts1 -= offset
        all_kpts0 -= offset
        all_kpts1 -= offset

        return matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, desc0[0], desc1[0], matched_conf
