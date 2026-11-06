"""Tests for CLI subcommands using Typer CliRunner."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from naive_bayes.cli import app

runner = CliRunner()


def test_cli_version() -> None:
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert "0.1.0" in res.output


def test_cli_verify_math() -> None:
    res = runner.invoke(app, ["verify-math"])
    assert res.exit_code == 0
    assert "All mathematical verification tests passed" in res.output


def test_cli_train_tabular() -> None:
    for ds in ["iris", "wine", "breast_cancer"]:
        res = runner.invoke(app, ["train", "--dataset", ds])
        assert res.exit_code == 0
        assert f"Trained GaussianNB on {ds}" in res.output


def test_cli_train_text() -> None:
    res = runner.invoke(app, ["train", "--dataset", "sms_spam", "--alpha", "1.0"])
    assert res.exit_code == 0
    assert "Trained MultinomialNB on sms_spam" in res.output


def test_cli_train_invalid_dataset() -> None:
    res = runner.invoke(app, ["train", "--dataset", "invalid_ds"])
    assert res.exit_code == 1


def test_cli_compare() -> None:
    res = runner.invoke(app, ["compare"])
    assert res.exit_code == 0
    assert "Running continuous feature comparisons" in res.output
    assert "Running text feature comparisons" in res.output


def test_cli_report(tmp_path: Path) -> None:
    out_file = tmp_path / "report.md"
    res = runner.invoke(app, ["report", "-o", str(out_file)])
    assert res.exit_code == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Mathematical Verification" in content
