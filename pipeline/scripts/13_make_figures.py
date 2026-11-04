"""Paper figures for the swarm target and the EBM.

Writes to figures/:
  swarm_val_loss.png    validation loss across the 593 diagnostic runs
  target_overview.png   distribution of y, run coverage, and y against three filters
  calibration.png       EBM-no-PC calibration on the held-out test split
  ebm_shapes.png        EBM-no-PC shape functions for the six most important features

Requires the full pipeline outputs in $PQ_WORK_DIR (master.parquet, the target,
the diagnostic runs and masks, and the trained EBM).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import DATA_DIR, MASKS_DIR, MASTER_PARQUET, MODELS_DIR, RESULTS_DIR  # noqa: E402
from pretrain_quality import paths  # noqa: E402

OUT = paths.REPO_ROOT / "figures"

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "grid.color": "#bbbbbb",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

PRETTY_FEATURE = {
    "qr_educational_value": "QuRater educational value",
    "qr_required_expertise": "QuRater required expertise",
    "qr_facts_and_trivia": "QuRater facts & trivia",
    "qr_writing_style": "QuRater writing style",
    "textbook_quality": "Textbook-quality score",
    "ai_generated": "P(AI-generated)",
    "proper_noun_density": "Proper-noun density",
    "entity_density": "Entity density",
    "first_person_fraction": "First-person sentence fraction",
    "mean_word_length": "Mean word length",
    "char_count": "Document length (characters)",
    "frac_alpha": "Fraction alphabetic",
    "flesch_kincaid_grade": "Flesch-Kincaid grade level",
    "flesch_reading_ease": "Flesch reading ease",
    "mean_line_length": "Mean line length",
}

FAMILY_COLOR = {
    "QuRater": "#1d3156",
    "Textbook": "#fed6ce",
    "Entity": "#a37ab4",
    "Structural": "#d486b8",
    "Length": "#abbcd6",
}

RUN_TYPE = {  # selection family -> (colour, legend label)
    "single_threshold": ("#f4a8a8", "single-feature threshold"),
    "multi_intersection": ("#f6c79a", "multi-feature intersection"),
    "linear_top": ("#f3e29a", "random linear combination"),
    "linear_top_simple": ("#c7e6b2", "linear combination (simple)"),
    "targeted_top": ("#a8d8c9", "targeted filter baselines"),
    "within_topic_top": ("#a8c8ec", "within-topic top-k"),
    "dclm_top_then_restrict": ("#c9b3e3", "DCLM-restricted"),
    "random": ("#f1b6da", "random subset"),
}


def feature_family(name: str) -> str:
    if name.startswith("qr_"):
        return "QuRater"
    if name in ("textbook_quality", "ai_generated"):
        return "Textbook"
    if name in ("entity_density", "entity_diversity", "entity_per_sentence",
                "proper_noun_density", "numeric_density"):
        return "Entity"
    if name.startswith(("discourse_", "explanation_", "question_", "pronoun_",
                        "first_person_", "hedge_", "hapax_", "header_", "code_",
                        "list_", "avg_paragraph", "flesch")):
        return "Structural"
    return "Length"


def load_runs() -> pd.DataFrame:
    rows = []
    for f in sorted(RESULTS_DIR.glob("run_*.json")):
        r = json.loads(f.read_text())
        if r.get("val_loss") is not None:
            rows.append({"run_id": r["run_id"], "type": r["cfg"]["type"],
                         "val_loss": float(r["val_loss"])})
    return pd.DataFrame(rows)


def train_pool() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Training-pool rows of master.parquet, the target y, and the row mask."""
    master = pd.read_parquet(MASTER_PARQUET)
    in_pool = master["__in_train_pool"].to_numpy()
    df = master[in_pool].reset_index(drop=True)
    y = np.load(DATA_DIR / "document_quality_target.npy")
    assert len(df) == len(y)
    return df, y, in_pool


def test_split(df: pd.DataFrame) -> np.ndarray:
    """The EBM's held-out split (80/20, stratified by topic, seed 42)."""
    _, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42,
                             stratify=df["topic"].to_numpy())
    return te


def fig_swarm_val_loss(runs: pd.DataFrame) -> None:
    vals, types = runs["val_loss"].to_numpy(), runs["type"].tolist()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.6),
                                     gridspec_kw={"width_ratios": [2.2, 1]})
    order = np.argsort(vals)
    ax_l.scatter(np.arange(len(vals)), vals[order], s=14, edgecolor="none",
                 c=[RUN_TYPE.get(types[i], ("#cccccc", ""))[0] for i in order])
    q25, med, q75 = np.percentile(vals, [25, 50, 75])
    ax_l.axhspan(q25, q75, color="#000", alpha=0.05, zorder=0)
    ax_l.axhline(med, color="k", ls=":", lw=0.8, label=f"median = {med:.3f}")
    ax_l.annotate(f"best: {vals[order[0]]:.3f}", xy=(0, vals[order[0]]), xytext=(15, 18),
                  textcoords="offset points", fontsize=9,
                  arrowprops=dict(arrowstyle="->", lw=0.7))
    ax_l.annotate(f"worst: {vals[order[-1]]:.3f}", xy=(len(vals) - 1, vals[order[-1]]),
                  xytext=(-90, -18), textcoords="offset points", fontsize=9,
                  arrowprops=dict(arrowstyle="->", lw=0.7))
    ax_l.set_xlabel("Run rank (sorted by validation loss)")
    ax_l.set_ylabel("CommonCrawl validation loss")
    ax_l.set_title(f"Validation loss across {len(vals)} swarm runs "
                   "(15M parameters, 150M unique tokens each)")
    ax_l.legend(loc="upper left", fontsize=9)

    present = [t for t in RUN_TYPE if t in set(types)]
    bins = np.linspace(vals.min(), vals.max(), 35)
    ax_r.hist([vals[np.array(types) == t] for t in present], bins=bins, stacked=True,
              color=[RUN_TYPE[t][0] for t in present],
              label=[RUN_TYPE[t][1] for t in present], edgecolor="white", lw=0.3)
    ax_r.set_xlabel("Validation loss")
    ax_r.set_ylabel("Runs")
    ax_r.set_title("By subset family")
    ax_r.legend(loc="upper right", fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "swarm_val_loss.png", dpi=160)
    plt.close()


def fig_target_overview(df: pd.DataFrame, y: np.ndarray, runs: pd.DataFrame) -> None:
    coverage = np.zeros(len(y), dtype=np.int32)
    for rid in runs["run_id"]:
        coverage += np.load(MASKS_DIR / f"run_{rid:04d}.npy").astype(np.int32)
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ax = axes[0, 0]
    ax.hist(y, bins=80, color="#4c78a8", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Document contribution $y$")
    ax.set_ylabel("Documents")
    ax.set_title(f"Distribution of $y$ (n = {len(y):,})")

    ax = axes[0, 1]
    ax.hist(coverage, bins=50, color="#f58518", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Swarm runs that include the document")
    ax.set_ylabel("Documents")
    ax.set_title(f"Run coverage (mean {coverage.mean():.1f}, median {np.median(coverage):.0f})")

    ax = axes[0, 2]
    ax.hist(runs["val_loss"], bins=40, color="#54a24b", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Validation loss")
    ax.set_ylabel("Runs")
    ax.set_title(f"Validation loss per run (n = {len(runs)})")

    for ax, col, label, cmap in [(axes[1, 0], "dclm_score", "DCLM-fastText score", "viridis"),
                                 (axes[1, 2], "qr_educational_value",
                                  "QuRater educational value", "magma")]:
        idx = np.where(df[col].notna().to_numpy())[0]
        take = rng.choice(idx, size=min(80_000, idx.size), replace=False)
        x = df[col].to_numpy()[take]
        ax.hexbin(x, y[take], gridsize=60, cmap=cmap, mincnt=2)
        rho, _ = spearmanr(x, y[take])
        ax.set_xlabel(label)
        ax.set_ylabel("$y$")
        ax.set_title(f"$y$ vs. {label.split(' score')[0]} (Spearman ρ = {rho:.2f})")

    ax = axes[1, 1]
    ok = df["fwedu_rounded"].notna().to_numpy()
    fwe = df["fwedu_rounded"].to_numpy()[ok].astype(int)
    ax.boxplot([y[ok][fwe == k] for k in range(6)], positions=range(6), widths=0.6,
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#e45756", alpha=0.5))
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("FineWeb-Edu score (rounded, 0-5)")
    ax.set_ylabel("$y$")
    ax.set_title("$y$ vs. FineWeb-Edu")

    plt.tight_layout()
    plt.savefig(OUT / "target_overview.png", dpi=130)
    plt.close()


def fig_calibration(df: pd.DataFrame, y: np.ndarray, in_pool: np.ndarray) -> None:
    pred = np.load(DATA_DIR / "ebm_interp_predicted_scores.npy")[in_pool]
    te = test_split(df)
    y_te, p_te = y[te], pred[te]
    r, _ = pearsonr(p_te, y_te)
    r2 = 1 - ((y_te - p_te) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum()

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    hb = ax.hexbin(p_te, y_te, gridsize=70, cmap="BuPu", mincnt=3,
                   extent=(-0.13, 0.11, -0.18, 0.14))
    ax.plot((-0.18, 0.14), (-0.18, 0.14), color="#d486b8", lw=1.4, ls="--", label="y = ŷ")
    bins = np.linspace(-0.10, 0.09, 22)
    which = np.digitize(p_te, bins) - 1
    mids = 0.5 * (bins[:-1] + bins[1:])
    means = np.array([y_te[which == i].mean() if (which == i).sum() > 50 else np.nan
                      for i in range(len(mids))])
    ax.plot(mids, means, color="#1d3156", lw=2.4, marker="o", ms=4.5,
            markerfacecolor="white", markeredgecolor="#1d3156", label="binned mean of y")
    ax.set_xlim(-0.13, 0.11)
    ax.set_ylim(-0.18, 0.14)
    ax.set_xlabel(r"EBM (no perplexity-correlation) prediction $\hat{y}$")
    ax.set_ylabel(r"Document contribution $y$")
    ax.set_title(f"Held-out calibration (n = {len(te) / 1e3:.0f}K, r = {r:.3f}, "
                 f"$R^2$ = {r2:.2f})")
    ax.legend(loc="upper left", fontsize=9)
    plt.colorbar(hb, ax=ax).set_label("documents per hexagon")
    plt.tight_layout()
    plt.savefig(OUT / "calibration.png", dpi=200)
    plt.close()


def fig_ebm_shapes() -> None:
    ebm = joblib.load(MODELS_DIR / "ebm_interp.joblib")
    glob = ebm.explain_global()
    gd = glob.data()
    names = list(ebm.feature_names_in_)
    terms = sorted(((n, s) for n, s in zip(gd["names"], gd["scores"])
                    if " & " not in n and n not in ("topic", "format")), key=lambda kv: -kv[1])

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.6), sharey=True)
    for ax, (feat, _) in zip(axes.flat, terms[:6]):
        sd = glob.data(names.index(feat))
        edges = np.asarray(sd["names"], dtype=float)
        scores = np.asarray(sd["scores"], dtype=float)
        mids = 0.5 * (edges[:-1] + edges[1:])
        lo, hi = np.quantile(edges, [0.01, 0.99])  # trim long tails for legibility
        keep = (mids >= lo) & (mids <= hi)
        x, s = mids[keep], scores[keep]
        c = FAMILY_COLOR[feature_family(feat)]
        if sd.get("upper_bounds") is not None:
            ax.fill_between(x, np.asarray(sd["lower_bounds"], float)[keep],
                            np.asarray(sd["upper_bounds"], float)[keep],
                            color=c, alpha=0.18, linewidth=0)
        ax.plot(x, s, color=c, lw=2.0)
        ax.axhline(0, color="#444", ls=":", lw=0.7, zorder=0)
        ax.scatter([x[np.argmax(s)]], [s.max()], s=22, color=c, zorder=5,
                   edgecolor="white", linewidths=0.6)
        ax.scatter([x[np.argmin(s)]], [s.min()], s=22, color="#cc3333", zorder=5,
                   edgecolor="white", linewidths=0.6)
        label = PRETTY_FEATURE.get(feat, feat)
        ax.set_xlabel(label, fontsize=10)
        ax.set_title(label.split(" (")[0], fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Contribution to $y$ ($\Delta y$)")
    fig.suptitle("Shape functions of the top six EBM features", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "ebm_shapes.png", dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*",
                    choices=["swarm", "target", "calibration", "shapes"],
                    help="subset of figures to make (default: all)")
    args = ap.parse_args()
    wanted = set(args.only or ["swarm", "target", "calibration", "shapes"])
    OUT.mkdir(parents=True, exist_ok=True)

    runs = load_runs()
    if "swarm" in wanted:
        fig_swarm_val_loss(runs)
    if wanted & {"target", "calibration"}:
        df, y, in_pool = train_pool()
        if "target" in wanted:
            fig_target_overview(df, y, runs)
        if "calibration" in wanted:
            fig_calibration(df, y, in_pool)
    if "shapes" in wanted:
        fig_ebm_shapes()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
