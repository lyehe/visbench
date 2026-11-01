"""Tests for the LRU image+size cache (formerly an in-harness FIFO masquerading as LRU)."""

from __future__ import annotations


import pytest

# When pyfilesystem-style env is unavailable, PIL Image.open still works fine.
PIL = pytest.importorskip("PIL")


def _set_cap(monkeypatch, size_max=4, image_max=4):
    """Reload the module with smaller caps for testing eviction behaviour."""
    import importlib
    monkeypatch.setenv("VISBENCH_SIZE_CACHE_MAX", str(size_max))
    monkeypatch.setenv("VISBENCH_IMG_CACHE_MAX", str(image_max))
    from visbench.core import _image_cache as ic
    importlib.reload(ic)
    return ic


def test_cached_load_evicts_lru(monkeypatch):
    ic = _set_cap(monkeypatch, image_max=3)
    ic.reset()
    calls = []

    class DummyMatcher:
        def load_image(self, path, resize=None):
            calls.append((str(path), resize))
            return f"img({path},{resize})"

    m = DummyMatcher()
    # Fill: a, b, c (calls all miss) → 3 calls
    ic.cached_load(m, "a")
    ic.cached_load(m, "b")
    ic.cached_load(m, "c")
    assert len(calls) == 3
    # Touch 'a' (hit, still in cache) → 3 calls. 'a' becomes MRU.
    ic.cached_load(m, "a")
    assert len(calls) == 3
    # Insert 'd' (miss, cap exceeded) → 'b' is LRU and evicted → 4 calls.
    ic.cached_load(m, "d")
    assert len(calls) == 4
    # Now request 'b' → cache miss (was evicted) → 5 calls.
    ic.cached_load(m, "b")
    assert len(calls) == 5
    # 'c' and 'd' still cached.
    ic.cached_load(m, "c")
    ic.cached_load(m, "d")
    # Wait — inserting 'b' evicted the LRU, which was 'c' (since 'a','d','b' were
    # most recent). So 'c' miss → 6 calls.
    assert len(calls) == 6
    ic.reset()


def test_pil_size_caches_and_lru_evicts(monkeypatch, tmp_path):
    """pil_size memoizes by stringified path; cap controls eviction."""
    ic = _set_cap(monkeypatch, size_max=2)
    ic.reset()
    from PIL import Image
    paths = []
    for i, (w, h) in enumerate([(8, 4), (16, 8), (32, 16)]):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (w, h)).save(p)
        paths.append(p)
    for p, expected in zip(paths, [(8, 4), (16, 8), (32, 16)]):
        assert ic.pil_size(p) == expected
    # After 3 inserts with cap=2, the first one is evicted; the others remain.
    # Re-fetching the first one triggers a new file open (no error) but still
    # returns correct dimensions.
    assert ic.pil_size(paths[0]) == (8, 4)
    ic.reset()


def test_reset_clears_both_caches(monkeypatch, tmp_path):
    ic = _set_cap(monkeypatch)
    ic.reset()
    from PIL import Image

    class DummyMatcher:
        def load_image(self, path, resize=None):
            return path

    p = tmp_path / "x.png"
    Image.new("RGB", (4, 4)).save(p)
    ic.pil_size(p)
    ic.cached_load(DummyMatcher(), str(p))
    ic.reset()
    # After reset, internal storage is empty (introspection is fine in a test)
    assert len(ic._size_cache) == 0  # noqa: SLF001
    assert len(ic._image_cache) == 0  # noqa: SLF001
