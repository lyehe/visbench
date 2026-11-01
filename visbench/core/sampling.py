"""Pair-cap sampling that is representative of the whole dataset.

Loaders yield pairs in a fixed order (usually scene by scene). Truncating that
stream at ``max_pairs`` evaluates only the first few scenes. ``sample_pairs``
instead draws a seeded, scene-balanced subset: pairs are grouped by their
``scene`` (else ``subset``) key, shuffled within each group, and taken
round-robin across groups until the cap is reached. With no cap, the stream
is returned unchanged.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Iterator

import numpy as np


class EmptyPairsError(RuntimeError):
    """A dataset produced zero evaluable pairs (missing data or a bad layout)."""


def _group_key(pair: dict) -> str:
    for key in ("scene", "subset", "set"):
        value = pair.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def sample_pairs(pairs: Iterable[dict], max_pairs: int | None, seed: int | None = 0) -> Iterator[dict]:
    if max_pairs is None:
        yield from pairs
        return
    if max_pairs <= 0:
        return
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for pair in pairs:
        groups.setdefault(_group_key(pair), []).append(pair)
    total = sum(len(g) for g in groups.values())
    if total <= max_pairs:
        for group in groups.values():
            yield from group
        return
    rng = np.random.default_rng(seed)
    queues = []
    for group in groups.values():
        order = rng.permutation(len(group))
        queues.append([group[i] for i in order])
    taken = 0
    depth = 0
    while taken < max_pairs:
        for queue in queues:
            if depth < len(queue):
                yield queue[depth]
                taken += 1
                if taken >= max_pairs:
                    return
        depth += 1
