"""Milestone 1 smoke test: the package installs and imports cleanly."""

import subprocess
import sys

import naive_bayes


def test_import_and_version() -> None:
    assert naive_bayes.__version__ == "0.1.0"


def test_cli_entry_point_reports_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "naive_bayes.cli", "version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "0.1.0"
