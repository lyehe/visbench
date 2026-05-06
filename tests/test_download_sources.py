"""Per-benchmark download-source validation.

For each dataset, monkey-patch the `_download` primitives so they RECORD intent
(URL / HF repo / gdown id) instead of fetching. Then HEAD-check every recorded
URL and `HfApi.repo_info`-check every HF repo. No bytes are saved.

Default: skipped (offline). Enable with VISBENCH_NETWORK=1.

This is the runtime cousin of `tests/test_per_benchmark.py::test_download_behaviour`,
which only checks that download() returns/raises cleanly with all primitives
mocked. This file goes further and validates that the recorded sources are
actually reachable on the public internet.
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

import pytest

from visbench.datasets import available, get

ALL_DATASETS = available()
NETWORK_ENABLED = bool(os.environ.get("VISBENCH_NETWORK"))


# --- Per-session caches so a URL/repo is HEAD-checked at most once ---------

_URL_CACHE: dict[str, str] = {}
_HF_CACHE: dict[str, str] = {}


def _check_url(url: str, timeout: int = 15) -> str:
    """HEAD-check a URL. Returns 'ok' or a short failure reason."""
    if url in _URL_CACHE:
        return _URL_CACHE[url]
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "visbench-test/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            code = r.getcode()
            status = "ok" if 200 <= code < 400 else f"HTTP {code}"
    except Exception as e:  # noqa: BLE001
        status = f"{type(e).__name__}: {e}"
    _URL_CACHE[url] = status
    return status


def _check_hf(repo_id: str) -> str:
    """Verify an HF repo exists. Tries `dataset` first, falls back to `model`."""
    if repo_id in _HF_CACHE:
        return _HF_CACHE[repo_id]
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        for kind in ("dataset", "model"):
            try:
                api.repo_info(repo_id, repo_type=kind)
                _HF_CACHE[repo_id] = "ok"
                return "ok"
            except Exception:
                continue
        status = "not found in dataset or model namespaces"
    except Exception as e:  # noqa: BLE001
        status = f"{type(e).__name__}: {e}"
    _HF_CACHE[repo_id] = status
    return status


def _record_intents_for(name: str) -> tuple[list[dict], str | None]:
    """Run `<name>.download(fake_root)` with primitives patched to no-op record.

    Returns (recorded_intents, manual_message_or_None).
    """
    spec = get(name)
    if spec.download is None:
        return [], None

    from visbench.datasets import _download as dl
    import sys

    intents: list[dict] = []

    def fake_url(url, dest, sha256=None, **_):
        intents.append({"kind": "url", "url": url, "dest": str(dest)})
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"")
        return Path(dest)

    def fake_hf(repo_id, dest, repo_type="dataset", **_):
        intents.append({"kind": "hf", "repo_id": repo_id,
                        "repo_type": repo_type, "dest": str(dest)})
        Path(dest).mkdir(parents=True, exist_ok=True)
        return Path(dest)

    def fake_gdown(file_id, dest, **_):
        intents.append({"kind": "gdown", "file_id": file_id, "dest": str(dest)})
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"")
        return Path(dest)

    def fake_extract(path, dest=None, strip_components=0):
        return Path(dest or Path(path).parent)

    saved = {}
    for n, fn in (("download_url", fake_url), ("download_hf", fake_hf),
                  ("download_gdown", fake_gdown),
                  ("download_gdown_bypass", fake_gdown),
                  ("extract_archive", fake_extract)):
        saved[n] = getattr(dl, n)
        setattr(dl, n, fn)

    # Re-bind into dataset modules that did `from ._download import …` at import.
    re_bound: list[tuple[object, str, object]] = []
    for modname, mod in list(sys.modules.items()):
        if not modname.startswith("visbench.datasets."):
            continue
        for n, fn in (("download_url", fake_url), ("download_hf", fake_hf),
                      ("download_gdown", fake_gdown),
                      ("download_gdown_bypass", fake_gdown),
                      ("extract_archive", fake_extract)):
            if hasattr(mod, n):
                re_bound.append((mod, n, getattr(mod, n)))
                setattr(mod, n, fn)

    fake_root = Path(tempfile.mkdtemp()) / "visbench_dl_test_root"
    manual_msg: str | None = None
    try:
        spec.download(fake_root)
    except NotImplementedError as e:
        manual_msg = str(e)
    finally:
        for n, fn in saved.items():
            setattr(dl, n, fn)
        for mod, n, fn in re_bound:
            setattr(mod, n, fn)
    return intents, manual_msg


# --- Test ------------------------------------------------------------------

@pytest.mark.skipif(not NETWORK_ENABLED,
                    reason="Set VISBENCH_NETWORK=1 to HEAD-check download sources.")
@pytest.mark.parametrize("name", ALL_DATASETS)
def test_download_sources_reachable(name):
    spec = get(name)
    if spec.download is None:
        pytest.skip(f"{name}: no download() registered")

    intents, manual_msg = _record_intents_for(name)

    # Manual datasets: NotImplementedError raised. The message should embed a
    # URL — sanity-check the source_url field for those.
    if manual_msg is not None:
        url = spec.source_url
        if not url:
            pytest.fail(f"{name}: manual download() raised but spec.source_url is empty")
        # Source URL itself need not 200 (some are landing pages with auth or
        # are forms). We accept any 2xx/3xx OR "401 Unauthorized" (CIIRC etc.
        # serve 401 for HEAD on directory listings even when GET works).
        status = _check_url(url)
        # Only fail on hard-broken statuses (DNS error, connection refused, 404).
        if any(s in status for s in ("getaddrinfo failed", "404", "Connection refused",
                                      "Name or service not known", "No address")):
            pytest.fail(f"{name}: source_url unreachable -> {url}: {status}")
        return

    # Auto-download datasets: at least one intent must be recorded, and every
    # URL / HF repo recorded must be reachable.
    if not intents:
        pytest.fail(f"{name}: download() returned without recording any intent "
                    "(early-exit because the fake target already 'existed'?)")

    failures = []
    for i in intents:
        if i["kind"] == "url":
            status = _check_url(i["url"])
            if status != "ok":
                failures.append(f"URL {i['url']} -> {status}")
        elif i["kind"] == "hf":
            status = _check_hf(i["repo_id"])
            if status != "ok":
                failures.append(f"HF {i['repo_id']} -> {status}")
        elif i["kind"] == "gdown":
            # Google Drive doesn't allow HEAD without an auth dance; only
            # validate the file_id is a non-empty string.
            if not i["file_id"] or not isinstance(i["file_id"], str):
                failures.append(f"gdown file_id is empty/invalid: {i['file_id']!r}")

    if failures:
        pytest.fail(f"{name}: " + "; ".join(failures))


# --- Aggregate sanity check (always-on) ------------------------------------

def test_every_auto_dataset_records_at_least_one_intent():
    """Module-level safety net: every dataset whose download() succeeds (i.e.,
    doesn't raise NotImplementedError) must record at least one intent.

    This catches `download()` bodies that have an early-exit hitting our fake
    root in a way that skips the actual fetch logic — a class of silent bug
    where the dataset would never actually download anything.
    """
    auto_no_intent: list[str] = []
    for name in ALL_DATASETS:
        spec = get(name)
        if spec.download is None:
            continue
        intents, manual_msg = _record_intents_for(name)
        if manual_msg is not None:
            continue
        if not intents:
            auto_no_intent.append(name)
    assert not auto_no_intent, \
        f"datasets whose download() succeeded silently (no recorded intent): {auto_no_intent}"
