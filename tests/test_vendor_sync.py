"""Enforce fork -> subtree flow without fetching or modifying Git state."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sync_vismatch", ROOT / "scripts" / "sync_vismatch.py")
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


def test_vendored_vismatch_matches_last_sync():
    if not (ROOT / ".git").exists():
        pytest.skip("subtree history is unavailable in a source archive")
    assert sync.check() == 0


@pytest.mark.parametrize("drift", ["none", "committed", "staged", "working", "untracked", "missing-history"])
def test_guard_detects_drift(monkeypatch, drift):
    def git(*args):
        if args[0] == "log":
            return "" if drift == "missing-history" else "squash\n"
        if args[0] == "rev-parse":
            return "other-tree\n" if drift == "committed" and args[1].startswith("HEAD:") else "tree\n"
        if args[0] == "diff":
            kind = "staged" if "--cached" in args else "working"
            return "third_party/vismatch/file.py\n" if drift == kind else ""
        if args[0] == "ls-files":
            return "third_party/vismatch/new.py\n" if drift == "untracked" else ""
        raise AssertionError(args)

    monkeypatch.setattr(sync, "git", git)
    assert sync.check() == (0 if drift == "none" else 1)
