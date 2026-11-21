"""Reproduce image.png from curated results/ benchmark outputs.

Reads ``results/metrics.json`` (copied unedited from the Kaggle run) and
``results/manifest.json`` for provenance, then renders the four panels
described in the README: frozen-test macro-F1 with bootstrap intervals,
the ECE comparison, validation training-size curves, and the synthetic
correlated-feature ECE stress test. Writes ``figure-data.json`` alongside
the plotted values so every number in the figure traces to results/.

Requires the project dependencies and an environment with Matplotlib,
e.g. ``uv run --with matplotlib python scripts/render_figure.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)
COLORS = ["#4169e1", "#10a68b", "#d95f45", "#8659cf"]
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DATASETS = [("sms_spam", "SMS Spam"), ("20newsgroups", "20 Newsgroups")]
MODELS = [("multinomial", "MNB"), ("bernoulli", "BNB")]
# One color per (dataset, model), shared by every panel.
ENTRY_COLORS = {
    ("sms_spam", "multinomial"): "#4169e1",
    ("sms_spam", "bernoulli"): "#10a68b",
    ("20newsgroups", "multinomial"): "#d95f45",
    ("20newsgroups", "bernoulli"): "#8659cf",
}


def _entry(metrics: dict[str, Any], dataset: str, model: str) -> dict[str, Any]:
    return metrics["text_test"][f"{dataset}_{model}"]


def _entries() -> list[tuple[str, str, str, str]]:
    return [
        (ds, md, ds_label, md_label)
        for ds, ds_label in DATASETS
        for md, md_label in MODELS
    ]


def main() -> None:
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), layout="constrained")
    entries = _entries()
    labels = [f"{ds_label}\n{md_label}" for _, _, ds_label, md_label in entries]
    xticks = np.arange(len(labels))

    def _dot_panel(ax: Any, key: str, ci_key: str, ylabel: str, title: str) -> None:
        est, lo, hi = [], [], []
        for ds, md, _, _ in entries:
            e = _entry(metrics, ds, md)
            ci = e[ci_key]
            est.append(float(e[key]))
            lo.append(float(ci["ci_lower"]))
            hi.append(float(ci["ci_upper"]))
        est_a, lo_a, hi_a = np.array(est), np.array(lo), np.array(hi)
        ax.errorbar(
            xticks, est_a,
            yerr=[est_a - lo_a, hi_a - est_a],
            fmt="o", ms=7, lw=1.8, capsize=5, ecolor="#64748b",
            color="#64748b",
        )
        for i, (ds, md, _, _) in enumerate(entries):
            ax.plot(i, est_a[i], "o", ms=7, color=ENTRY_COLORS[(ds, md)])
            ax.annotate(
                f"{est_a[i]:.3f}", (i, hi_a[i]), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=8,
            )
        ax.set_xticks(xticks, labels, fontsize=9)
        pad = 0.04 * (est_a.max() - est_a.min() + 0.05)
        ax.set_ylim(max(0.0, est_a.min() - pad), est_a.max() + pad + 0.02)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)

    # Panel 1: frozen-test macro-F1 with 95% bootstrap intervals.
    _dot_panel(
        axes[0, 0], "macro_f1", "macro_f1_ci", "Macro-F1",
        "Frozen test macro-F1 (95% bootstrap CI)",
    )

    # Panel 2: ECE comparison on the same frozen test predictions.
    _dot_panel(
        axes[0, 1], "ece", "ece_ci", "ECE (15 equal-width bins)",
        "Frozen test expected calibration error",
    )

    # Panel 3: validation macro-F1 vs training-set fraction.
    ax = axes[1, 0]
    ts = metrics["controlled"]["training_size"]
    for dataset, ds_label in DATASETS:
        for model, m_label in MODELS:
            rows = ts[dataset][model]
            frac = np.array([r["fraction"] for r in rows], dtype=float)
            mean = np.array([r["macro_f1_mean"] for r in rows], dtype=float)
            std = np.array([r["macro_f1_std"] for r in rows], dtype=float)
            color = ENTRY_COLORS[(dataset, model)]
            ax.fill_between(frac, mean - std, mean + std, color=color, alpha=0.15)
            ax.plot(frac, mean, marker="o", ms=4, lw=1.8, color=color,
                    label=f"{ds_label} {m_label}")
    ax.set_xscale("log")
    ax.set_xlabel("Training fraction of train partition")
    ax.set_ylabel("Validation macro-F1 (mean +/- std, 5 seeds)")
    ax.set_title("Training-size study (validation only)")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.2)

    # Panel 4: synthetic correlated-feature stress test, ECE vs k.
    ax = axes[1, 1]
    cf = metrics["controlled"]["correlated_feature"]["rows"]
    k = np.array([r["k"] for r in cf], dtype=float)
    gnb_ece = np.array([r["gnb_ece"] for r in cf], dtype=float)
    lr_ece = np.array([r["lr_ece"] for r in cf], dtype=float)
    ax.plot(k, gnb_ece, marker="o", ms=5, lw=2, color=COLORS[0],
            label="GaussianNB (custom)")
    ax.plot(k, lr_ece, marker="s", ms=5, lw=2, ls="--", color=COLORS[1],
            label="LogisticRegression (reference)")
    ax.set_xlabel("k exact copies of feature 1 (synthetic)")
    ax.set_ylabel("ECE")
    ax.set_title("Correlated-feature stress test (synthetic, n=5000)")
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.grid(alpha=0.2)

    fig.suptitle(
        "Naive Bayes benchmark | frozen test metrics and controlled studies",
        fontsize=15,
    )
    fig.supxlabel(
        f"Source: results/metrics.json, git_sha {manifest.get('git_sha', 'unknown')}. "
        "Text splits are duplicate-aware; the correlated-feature panel is synthetic.",
        fontsize=9,
    )

    payload: dict[str, Any] = {
        "source": "results/metrics.json copied unedited from the Kaggle benchmark run",
        "git_sha": manifest.get("git_sha"),
        "protocol": manifest.get("protocol"),
        "training_size": ts,
        "correlated_feature": metrics["controlled"]["correlated_feature"],
    }
    payload["macro_f1_test"] = {}
    payload["ece_test"] = {}
    for dataset, _ in DATASETS:
        for model, _ in MODELS:
            e = _entry(metrics, dataset, model)
            payload["macro_f1_test"][f"{dataset}_{model}"] = {
                "estimate": float(e["macro_f1"]),
                "ci_lower": float(e["macro_f1_ci"]["ci_lower"]),
                "ci_upper": float(e["macro_f1_ci"]["ci_upper"]),
            }
            payload["ece_test"][f"{dataset}_{model}"] = {
                "estimate": float(e["ece"]),
                "ci_lower": float(e["ece_ci"]["ci_lower"]),
                "ci_upper": float(e["ece_ci"]["ci_upper"]),
            }

    source_hash = hashlib.sha256()
    for name in ("metrics.json", "manifest.json"):
        path = RESULTS / name
        if path.exists():
            source_hash.update(name.encode())
            source_hash.update(path.read_bytes())
    payload["results_sha256"] = source_hash.hexdigest()

    fig.savefig(ROOT / "image.png", dpi=180, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    (ROOT / "scripts" / "figure-data.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {ROOT / 'image.png'} and scripts/figure-data.json")


if __name__ == "__main__":
    main()
