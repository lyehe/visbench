"""Pass-through to `vismatch.get_matcher`. Use `vismatch:<name>` from the CLI."""

from __future__ import annotations


# LoMa sizes that vismatch exposes only through the `arch` kwarg of "loma".
# They run at LoMa's paper setting of 4096 keypoints (arXiv 2604.04931, B).
_ALIASES: dict[str, tuple[str, dict]] = {
    "loma-b": ("loma", {"arch": "LoMa-B", "max_num_keypoints": 4096}),
    "loma-b128": ("loma", {"arch": "LoMa-B128", "max_num_keypoints": 4096}),
    "loma-l": ("loma", {"arch": "LoMa-L", "max_num_keypoints": 4096}),
    "loma-g": ("loma", {"arch": "LoMa-G", "max_num_keypoints": 4096}),
}


def build_vismatch(name: str, device: str = "cpu", max_num_keypoints: int = 2048):
    """Resolve a vismatch matcher by name. See `vismatch.available_models` for the list."""
    import vismatch
    base, extra = _ALIASES.get(name, (name, {}))
    kwargs = {"max_num_keypoints": max_num_keypoints, **extra}
    return vismatch.get_matcher(base, device=device, **kwargs)


def list_vismatch_matchers() -> list[str]:
    """Return the list of vismatch matcher names known at import time."""
    try:
        import vismatch
        names = getattr(vismatch, "available_models", None)
        if names is None:
            names = getattr(vismatch, "MATCHER_NAMES", [])
        return sorted(set(names) | set(_ALIASES))
    except Exception:
        return []
