"""Exploratory analysis of the ten quality filters.

Computes pairwise Spearman correlations between the filter scores over all documents and
within topics (correlation of within-topic percentile ranks), and plots raw scores against
their within-topic percentile.

Writes:
  results/filter_correlations.json          global and within-topic Spearman matrices, max |delta rho|
  figures/filter_correlations.png           global Spearman heatmap
  figures/within_topic_percentile.png       raw DCLM / FineWeb-Edu score vs. within-topic percentile
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pretrain_quality import paths
from pretrain_quality.features.columns import TOPIC_NAMES

SCORES = {
    "dclm_score": "DCLM",
    "fwedu_rounded": "FineWeb-Edu",
    "pc_arc_easy": "PC-ARC",
    "pc_piqa": "PC-PIQA",
    "pc_sciq": "PC-SciQ",
    "pc_lambada": "PC-LAMBADA",
    "qr_writing_style": "QR-Writing",
    "qr_required_expertise": "QR-Expertise",
    "qr_facts_and_trivia": "QR-Facts",
    "qr_educational_value": "QR-EduVal",
}


def heatmap(mat: pd.DataFrame, title: str, path) -> None:
    labels = [SCORES[c] for c in mat.columns]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = mat.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else "black")
    ax.set_title(title, fontweight="bold")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def percentile_plot(df: pd.DataFrame, path, n: int = 30_000, seed: int = 123) -> None:
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    cmap = plt.get_cmap("tab20", 24)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, col in zip(axes, ["dclm_score", "fwedu_rounded"]):
        pct = sample.groupby("topic")[col].rank(pct=True)
        for t in range(24):
            m = (sample["topic"] == t).to_numpy()
            ax.scatter(sample[col].to_numpy()[m], pct.to_numpy()[m], s=3, alpha=0.3,
                       color=cmap(t), label=TOPIC_NAMES[t], rasterized=True)
        ax.set_xlabel(f"Raw {SCORES[col]} score")
        ax.set_title(f"{SCORES[col]}: raw score vs. within-topic percentile")
    axes[0].set_ylabel("Within-topic percentile")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8,
               markerscale=4, frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", default=str(paths.WORK_DIR / "master.parquet"))
    args = ap.parse_args()

    cols = list(SCORES)
    df = pd.read_parquet(args.master, columns=cols + ["topic"]).dropna()
    print(f"{len(df):,} documents with all ten scores")

    ranks = df[cols].rank()
    rho_global = ranks.corr()
    within = df.groupby("topic")[cols].rank(pct=True)
    rho_within = within.corr()
    delta = (rho_global - rho_within).abs().to_numpy().copy()
    np.fill_diagonal(delta, 0)

    out = paths.REPO_ROOT / "results" / "filter_correlations.json"
    out.write_text(json.dumps({
        "n_documents": int(len(df)),
        "spearman_global": rho_global.round(4).to_dict(),
        "spearman_within_topic": rho_within.round(4).to_dict(),
        "max_abs_delta_global_vs_within": float(delta.max()),
    }, indent=2) + "\n")
    figs = paths.REPO_ROOT / "figures"
    heatmap(rho_global, "Spearman correlations between quality filters",
            figs / "filter_correlations.png")
    percentile_plot(df, figs / "within_topic_percentile.png")
    print(f"max |global - within-topic| Spearman: {delta.max():.3f}")
    print(rho_global.round(2).to_string())


if __name__ == "__main__":
    main()
