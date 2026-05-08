"""Per-benchmark smoke tests.

Every registered dataset gets four parametrized tests:

  1. test_spec_valid[<name>]
     - task is one the dispatcher knows about
     - iter_pairs is callable
     - download is either None or callable
     - source_url is set (not empty)

  2. test_iter_pairs_empty_root[<name>]
     - Calling iter_pairs on a non-existent root either yields nothing OR
       raises FileNotFoundError. Anything else (TypeError, ImportError on a
       lazy import, etc.) is a bug.

  3. test_download_behaviour[<name>]
     - With _download primitives monkey-patched to no-op, download() either
       returns a Path OR raises NotImplementedError. Anything else (random
       URL exception, KeyError, unbound name) means the download() body has
       a bug independent of network conditions.

  4. test_iter_pairs_yields_well_formed_pairs[<name>]  (skipped unless data exists)
     - If real data is on disk under VISBENCH_DATA_ROOT, run iter_pairs with
       max_pairs=2 and check every yielded dict has the keys required by its
       task. Skipped by default so CI stays offline.

This gives 132 datasets x 3 always-on tests = 396 per-benchmark assertions,
plus opt-in smoke when data is present.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from visbench.datasets import available, get
from visbench.datasets._registry import VALID_TASKS

# --- Fixtures ---------------------------------------------------------------

ALL_DATASETS = available()


@pytest.fixture(scope="module")
def patched_download(monkeypatch_module):
    """Module-scoped monkey-patch of `_download` primitives so download()
    can be invoked without touching the network."""
    from visbench.datasets import _download as dl
    import sys

    def fake_url(url, dest, sha256=None, **_):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"")
        return Path(dest)

    def fake_hf(repo_id, dest, repo_type="dataset", **_):
        Path(dest).mkdir(parents=True, exist_ok=True)
        return Path(dest)

    def fake_gdown(file_id, dest, **_):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"")
        return Path(dest)

    def fake_extract(path, dest=None, strip_components=0):
        return Path(dest or Path(path).parent)

    monkeypatch_module.setattr(dl, "download_url", fake_url)
    monkeypatch_module.setattr(dl, "download_hf", fake_hf)
    monkeypatch_module.setattr(dl, "download_gdown", fake_gdown)
    monkeypatch_module.setattr(dl, "download_gdown_bypass", fake_gdown)
    monkeypatch_module.setattr(dl, "extract_archive", fake_extract)

    # Also patch the names re-bound inside dataset modules at import time.
    for modname, mod in list(sys.modules.items()):
        if not modname.startswith("visbench.datasets."):
            continue
        for n in ("download_url", "download_hf", "download_gdown",
                  "download_gdown_bypass", "extract_archive"):
            if hasattr(mod, n):
                monkeypatch_module.setattr(mod, n, getattr(dl, n))
    yield


@pytest.fixture(scope="module")
def monkeypatch_module():
    """pytest's `monkeypatch` is function-scoped; we want module-scoped."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


# --- Tests ------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_DATASETS)
def test_dataset_has_reference_entry(name):
    """Every dataset must be covered by reference_scores.json — either via a
    direct entry, an alias to another dataset, or an explicit
    `_no_published_reference` marker. Catches new datasets that ship without
    a corresponding row."""
    from visbench.orchestrators.reporting import _load_references
    refs, aliases, no_pub = _load_references()
    if name in refs or name in aliases or name in no_pub:
        return
    pytest.fail(
        f"{name}: not present in visbench/reference_scores.json. Add a direct "
        f"entry with paper-cited numbers, or alias it via `_aliases`, or list "
        f"it under `_no_published_reference` with a one-line explanation."
    )


@pytest.mark.parametrize("name", ALL_DATASETS)
def test_spec_valid(name):
    spec = get(name)
    assert spec.task in VALID_TASKS, f"{name}: invalid task '{spec.task}'"
    assert callable(spec.iter_pairs), f"{name}: iter_pairs not callable"
    assert spec.download is None or callable(spec.download), \
        f"{name}: download must be None or callable"
    assert spec.source_url, f"{name}: source_url is empty"
    # default_root must be a Path; default_resize must be None or positive int.
    assert isinstance(spec.default_root, Path), f"{name}: default_root not Path"
    if spec.default_resize is not None:
        assert isinstance(spec.default_resize, int) and spec.default_resize > 0, \
            f"{name}: default_resize must be None or positive int"


@pytest.mark.parametrize("name", ALL_DATASETS)
def test_iter_pairs_empty_root(name):
    """Calling iter_pairs on a non-existent root must either yield nothing OR
    raise FileNotFoundError. Any other exception is a per-dataset bug."""
    spec = get(name)
    fake = Path(tempfile.mkdtemp()) / "definitely_does_not_exist"
    kwargs = dict(spec.kwargs or {})
    try:
        result = spec.iter_pairs(fake, max_pairs=0, **kwargs)
        # iter_pairs may be a generator or a callable returning one.
        pairs = list(result)
        assert pairs == [], f"{name}: yielded pairs from a non-existent root: {pairs[:1]}"
    except FileNotFoundError:
        pass  # acceptable
    except NotImplementedError:
        pass  # rare but acceptable
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"{name}: iter_pairs(non_existent_root) raised "
                    f"{type(e).__name__}: {e} (expected FileNotFoundError or empty yield)")


@pytest.mark.parametrize("name", ALL_DATASETS)
def test_download_behaviour(name, patched_download):
    """download() must return a Path or raise NotImplementedError; nothing else."""
    spec = get(name)
    if spec.download is None:
        pytest.skip(f"{name}: no download() registered")
    fake_root = Path(tempfile.mkdtemp()) / "fake_visbench_target"
    try:
        result = spec.download(fake_root)
        assert isinstance(result, (Path, str)), \
            f"{name}: download() returned {type(result).__name__}, expected Path"
    except NotImplementedError:
        pass  # documented: scripted-manual datasets emit instructions this way
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"{name}: download() raised {type(e).__name__}: {e} "
                    f"(expected Path return or NotImplementedError)")


# --- Opt-in real-data smoke test -------------------------------------------

_PAIR_KEYS_BY_TASK = {
    "pose":          ("im_A_path", "im_B_path", "K0", "K1", "R_0to1", "t_0to1"),
    "homography":    ("im_A_path", "im_B_path", "H_gt"),
    "correspondence":("im_A_path", "im_B_path", "corrs"),
    "fundamental":   ("im_A_path", "im_B_path", "F_gt"),
    "descriptor":    ("im_A_path", "im_B_path", "kpts_A", "kpts_B"),
    "qualitative":   ("im_A_path", "im_B_path"),
}


@pytest.mark.skipif(not os.environ.get("VISBENCH_SMOKE"),
                    reason="Set VISBENCH_SMOKE=1 (and VISBENCH_DATA_ROOT) for real-data smoke tests.")
@pytest.mark.parametrize("name", ALL_DATASETS)
def test_iter_pairs_yields_well_formed_pairs(name):
    """If real data is on disk, the first 2 pairs should have all keys their task requires."""
    spec = get(name)
    data_root_default = Path(os.environ.get("VISBENCH_DATA_ROOT", Path.cwd() / "datasets"))
    data_root = data_root_default / spec.default_root
    if not data_root.exists():
        pytest.skip(f"{name}: data not on disk at {data_root}")

    kwargs = dict(spec.kwargs or {})
    required = _PAIR_KEYS_BY_TASK[spec.task]
    yielded = 0
    for pair in spec.iter_pairs(data_root, max_pairs=2, **kwargs):
        for key in required:
            assert key in pair, f"{name}: yielded pair missing required key '{key}'"
        # Path strings should point at files that exist.
        for path_key in ("im_A_path", "im_B_path"):
            assert Path(pair[path_key]).exists(), \
                f"{name}: yielded {path_key}={pair[path_key]} does not exist"
        yielded += 1
        if yielded >= 2:
            break
    if yielded == 0:
        pytest.skip(f"{name}: data root exists but no pairs yielded (incomplete data?)")
