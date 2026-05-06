"""CLI argparse + subcommand routing — no data, no network."""

from __future__ import annotations

import pytest

from visbench.__main__ import _build_parser


def test_parser_builds():
    p = _build_parser()
    assert p is not None


def test_run_requires_dataset_and_method():
    p = _build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["run"])


def test_list_datasets_parses():
    p = _build_parser()
    args = p.parse_args(["list", "datasets"])
    assert args.kind == "datasets"


def test_download_parses():
    p = _build_parser()
    args = p.parse_args(["download", "hpatches"])
    assert args.name == "hpatches"
    assert args.all_auto is False
