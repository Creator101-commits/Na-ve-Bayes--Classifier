"""Kaggle entrypoint for the Naive Bayes benchmark.

Installs ``naive_bayes`` from the pinned source dataset uploaded by
``scripts/kaggle_run.sh``, downloads the real datasets (SMS Spam Collection,
20 Newsgroups), runs the full benchmark, and writes artifacts under
``/kaggle/working/``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
DATA = WORK / "data"
BENCH = WORK / "benchmark"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _find_project_root() -> Path | None:
    for candidate in sorted(KAGGLE_INPUT.rglob("pyproject.toml")):
        if (candidate.parent / "src" / "naive_bayes").is_dir():
            return candidate.parent
    return None


def _install_package() -> None:
    proj_root = _find_project_root()
    print(f"project root: {proj_root}", flush=True)
    if proj_root is None:
        raise FileNotFoundError(
            "No installable project found under /kaggle/input "
            "(expected pyproject.toml + src/naive_bayes)."
        )
    proj = WORK / "project"
    shutil.copytree(proj_root, proj)
    _run([sys.executable, "-m", "pip", "install", "--quiet", str(proj)])


def _git_sha() -> str:
    proj_root = _find_project_root()
    info = proj_root / "run_info.json" if proj_root is not None else None
    if info is not None and info.exists():
        return str(json.loads(info.read_text()).get("git_sha", "unknown"))
    return "unknown"


def _dataset_sources() -> list[str]:
    proj_root = _find_project_root()
    meta = proj_root / "kaggle" / "kernel-metadata.json" if proj_root is not None else None
    if meta is not None and meta.exists():
        try:
            return list(json.loads(meta.read_text()).get("dataset_sources", []))
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _package_versions() -> dict[str, str]:
    names = [
        "naive-bayes",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "typer",
    ]
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def main() -> None:
    t0 = time.time()
    _install_package()

    git_sha = _git_sha()
    print(f"git_sha={git_sha}", flush=True)
    os.environ["GIT_SHA"] = git_sha

    _run([sys.executable, "-m", "naive_bayes.cli", "prepare-data", "--out", str(DATA)])
    _run(
        [
            sys.executable,
            "-m",
            "naive_bayes.cli",
            "benchmark",
            "--data",
            str(DATA),
            "--out",
            str(BENCH),
        ]
    )

    manifest = json.loads((BENCH / "manifest.json").read_text())
    if manifest.get("git_sha") != git_sha:
        raise AssertionError(
            f"manifest git_sha {manifest.get('git_sha')} != kernel git_sha {git_sha}"
        )
    if not manifest.get("ordering_ok"):
        raise AssertionError("selection was not written before the first test read")

    env = {
        "git_sha": git_sha,
        "python_version": sys.version.split()[0],
        "package_versions": _package_versions(),
        "kaggle_dataset_sources": _dataset_sources(),
        "wall_seconds_total": time.time() - t0,
    }
    (WORK / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    print("\nBenchmark complete; artifacts under", BENCH, flush=True)


if __name__ == "__main__":
    main()
