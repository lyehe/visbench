"""Unicode-safe OpenCV image reading.

``cv2.imread`` cannot open non-ASCII paths on Windows (HardMatch has folders
such as ``Ilyushin_Il-14_at_Cämmerswalde``) and silently returns None.
Reading the bytes with NumPy and decoding them avoids that.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR):
    """Drop-in for ``cv2.imread``: returns None when the file is missing or undecodable."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)
