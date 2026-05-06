"""Pass-through to `vismatch.get_matcher`. Use `vismatch:<name>` from the CLI."""

from __future__ import annotations


def build_vismatch(name: str, device: str = "cpu", max_num_keypoints: int = 2048):
    """Resolve a vismatch matcher by name. See `vismatch.available_models` for the list."""
    import vismatch
    return vismatch.get_matcher(matcher_name=name, device=device,
                                max_num_keypoints=max_num_keypoints)


def list_vismatch_matchers() -> list[str]:
    """Return the list of vismatch matcher names known at import time."""
    try:
        import vismatch
        names = getattr(vismatch, "available_models", None)
        if names is None:
            names = getattr(vismatch, "MATCHER_NAMES", [])
        return list(names)
    except Exception:
        return []
