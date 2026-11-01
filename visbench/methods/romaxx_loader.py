"""Optional adapter for RoMaXX matchers (TensorRT and PyTorch). Use ``romaxx:<kind>[:<arg>]``.

RoMaXX is not a visbench dependency; install it into the same environment
(for example ``uv pip install -e <path-to-romaxx>``) to use these methods.

=====================  =========================================================
``romaxx:flexglue:core``      FlexGlue PyTorch, 512 core (``refined`` for two-scale refinement)
``romaxx:flexglue-trt:<dir>`` FlexGlueTRT from a built or released engine bundle directory
``romaxx:romatrt:<setting>``  RoMaTRT (turbo / fast / base / precise)
``romaxx:minimaromatrt:<s>``  MinimaRoMaTRT (A->B only)
``romaxx:lomatrt:<preset>``   LoMaTRT (loma_B128 / loma_B / loma_L / loma_G / loma_R); images are
                              resized to the engine size (VISBENCH_LOMATRT_SIZE, default 512)
=====================  =========================================================

Dense matchers are sampled with the model's own ``sample()`` (5000
correspondences, the RoMa pose-benchmark convention; set
``VISBENCH_ROMAXX_NUM_CORRESP`` to change it) and lifted to input pixels with
``to_pixel_coordinates``. LoMaTRT returns its mutual matches directly.
"""

from __future__ import annotations

import os

import numpy as np

_DENSE_KINDS = ("flexglue", "flexglue-trt", "romatrt", "minimaromatrt")
KINDS = _DENSE_KINDS + ("lomatrt",)


def _load_image(path, resize=None):
    """CHW float tensor in [0, 1]; `resize` is (H, W) or a square int, as in vismatch."""
    import torch
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        if resize is not None:
            h, w = (resize, resize) if isinstance(resize, int) else resize
            im = im.resize((int(w), int(h)), Image.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _build_model(kind: str, arg: str | None, device: str):
    import romaxx

    if kind == "flexglue":
        mode = (arg or "core").lower()
        if mode not in ("core", "refined"):
            raise ValueError("romaxx:flexglue takes 'core' or 'refined'")
        return romaxx.FlexGlue(romaxx.FlexGlue.Cfg(device=device, refine=mode == "refined"))
    if kind == "flexglue-trt":
        if not arg:
            raise ValueError("romaxx:flexglue-trt needs an engine bundle directory: romaxx:flexglue-trt:<dir>")
        return romaxx.FlexGlueTRT(arg)
    if kind == "romatrt":
        return romaxx.RoMaTRT(setting=arg or "fast")
    if kind == "minimaromatrt":
        return romaxx.MinimaRoMaTRT(setting=arg or "fast")
    if kind == "lomatrt":
        return romaxx.LoMaTRT(preset=arg or "loma_B128")
    raise ValueError(f"Unknown romaxx method kind '{kind}'. Known: {', '.join(KINDS)}")


def _resize(img, size: int):
    """(C,H,W) -> (1,C,size,size), antialiased bilinear."""
    import torch.nn.functional as F

    return F.interpolate(img[None].float(), size=(size, size), mode="bilinear",
                         antialias=True, align_corners=False)


class RomaxxMatcher:
    """Harness-facing wrapper: ``load_image`` + ``__call__(img0, img1) -> match dict``."""

    def __init__(self, kind: str, arg: str | None, device: str = "cuda", model=None):
        if kind not in KINDS:
            raise ValueError(f"Unknown romaxx method kind '{kind}'. Known: {', '.join(KINDS)}")
        self.kind = kind
        self.device = device
        self.num_corresp = int(os.environ.get("VISBENCH_ROMAXX_NUM_CORRESP", 5000))
        self.lomatrt_size = int(os.environ.get("VISBENCH_LOMATRT_SIZE", 512))
        self.model = model if model is not None else _build_model(kind, arg, device)

    @staticmethod
    def load_image(path, resize=None):
        return _load_image(path, resize=resize)

    def __call__(self, img0, img1) -> dict:
        import torch

        H_A, W_A = int(img0.shape[-2]), int(img0.shape[-1])
        H_B, W_B = int(img1.shape[-2]), int(img1.shape[-1])
        with torch.inference_mode():
            if self.kind == "lomatrt":
                # LoMaTRT engines have one fixed input size and no auto-build:
                # resize to it and map the matches back to the input frame.
                S = self.lomatrt_size
                a = _resize(img0, S).to("cuda")
                b = _resize(img1, S).to("cuda")
                out = self.model.match_points(a, b)
                pts_A, pts_B = out[0] if isinstance(out, list) else out
                pts_A = np.asarray(pts_A, dtype=np.float64) * [W_A / S, H_A / S]
                pts_B = np.asarray(pts_B, dtype=np.float64) * [W_B / S, H_B / S]
            else:
                preds = self.model.match(img0, img1)
                matches, _ = self.model.sample(preds, self.num_corresp)
                pts_A, pts_B = self.model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
        pts_A = pts_A.detach().cpu().numpy() if isinstance(pts_A, torch.Tensor) else np.asarray(pts_A)
        pts_B = pts_B.detach().cpu().numpy() if isinstance(pts_B, torch.Tensor) else np.asarray(pts_B)
        pts_A = pts_A.reshape(-1, 2).astype(np.float64)
        pts_B = pts_B.reshape(-1, 2).astype(np.float64)
        return {"matched_kpts0": pts_A, "matched_kpts1": pts_B,
                "all_kpts0": pts_A, "all_kpts1": pts_B, "num_inliers": len(pts_A)}


def parse(name: str) -> tuple[str, str | None]:
    """'romaxx:kind[:arg]' -> (kind, arg). The arg may itself contain ':' (Windows paths)."""
    rest = name.split(":", 1)[1]
    kind, _, arg = rest.partition(":")
    return kind.lower(), (arg or None)


def build_romaxx(name: str, device: str = "cuda") -> RomaxxMatcher:
    kind, arg = parse(name)
    return RomaxxMatcher(kind, arg, device=device)
