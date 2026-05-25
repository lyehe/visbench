"""Bounded LRU caches for image sizes and loaded image tensors.

Replaces the FIFO-pretending-to-be-LRU dict eviction in the harness with a real
recency-ordered cache (uses `collections.OrderedDict.move_to_end`).

The harness calls `reset_caches()` between matcher runs to free GPU/CPU memory;
the same call also resets these caches.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any

from PIL import Image

_SIZE_CACHE_MAX = int(os.environ.get("VISBENCH_SIZE_CACHE_MAX", 4096))
_IMAGE_CACHE_MAX = int(os.environ.get("VISBENCH_IMG_CACHE_MAX", 256))

_size_cache: OrderedDict[str, tuple[int, int]] = OrderedDict()
_image_cache: OrderedDict[Any, Any] = OrderedDict()


def pil_size(path) -> tuple[int, int]:
    """(w, h) of an image — cached with LRU eviction."""
    key = str(path)
    if key in _size_cache:
        _size_cache.move_to_end(key)
        return _size_cache[key]
    with Image.open(path) as im:
        wh = im.size
    _size_cache[key] = wh
    if len(_size_cache) > _SIZE_CACHE_MAX:
        _size_cache.popitem(last=False)
    return wh


def cached_load(matcher, path, resize=None):
    """matcher.load_image(...) memoized by (path, resize) with LRU eviction.

    `resize` may be None, an int (long-side), or an (h, w) tuple.
    """
    key = (str(path), resize if not isinstance(resize, tuple) else tuple(resize))
    if key in _image_cache:
        _image_cache.move_to_end(key)
        return _image_cache[key]
    img = matcher.load_image(path, resize=resize) if resize is not None else matcher.load_image(path)
    _image_cache[key] = img
    if len(_image_cache) > _IMAGE_CACHE_MAX:
        _image_cache.popitem(last=False)
    return img


def reset() -> None:
    """Clear all caches — call between matcher runs to free memory."""
    _image_cache.clear()
    _size_cache.clear()
