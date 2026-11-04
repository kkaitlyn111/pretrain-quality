"""Figures for the README and the paper, all drawn with the shared palette.

Writes to figures/:
  filter_correlations.png   Spearman correlations between the ten filters
  factor_loadings.png       two-factor loadings
  quality_diversity.png     quality vs. topic entropy for the top-20% two-factor selection
  r2.png                    test R² of four predictors of y
  calibration.png           EBM prediction vs. measured y on held-out documents
  ebm_shapes.png            EBM shape functions for the six most important features
  feature_importance.png    top-12 EBM features by importance
  selection.png             CommonCrawl val loss of 60M LMs by selection policy
  swarm_val_loss.png, target_overview.png   (diagnostics, not in the README)

Figures from results/ need only the repository; the others need the pipeline outputs in
$PQ_WORK_DIR (and $PQ_WORK_DIR/factor_model for the two factor-model figures).
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import DATA_DIR, MASKS_DIR, MASTER_PARQUET, MODELS_DIR, RESULTS_DIR  # noqa: E402
from pretrain_quality import paths  # noqa: E402
from pretrain_quality.palette import DARK_BLUE, GRAY, LIGHT_BLUE, PINK, PURPLE  # noqa: E402

OUT = paths.REPO_ROOT / "figures"
RESULTS = paths.REPO_ROOT / "results"
BLUES = LinearSegmentedColormap.from_list("pq_blues", ["#ffffff", LIGHT_BLUE, DARK_BLUE])

# Test R² as reported in report/paper.pdf.
R2 = [("Topic only", 0.076), ("Lasso", 0.408), ("EBM, no PC", 0.656), ("EBM, all", 0.668)]

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
    "savefig.facecolor": "white",
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
    "explanation_density": "Explanation marker density",
}


RUN_TYPE = {  # selection family -> (colour, legend label)
    "single_threshold": (DARK_BLUE, "single-feature threshold"),
    "multi_intersection": (PURPLE, "multi-feature intersection"),
    "linear_top": (PINK, "random linear combination"),
    "linear_top_simple": (LIGHT_BLUE, "linear combination (simple)"),
    "targeted_top": ("#4f6aa3", "targeted filter baselines"),
    "within_topic_top": ("#b3a3e0", "within-topic top-k"),
    "dclm_top_then_restrict": ("#f3bfd7", "DCLM-restricted"),
    "random": (GRAY, "random subset"),
}


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
    ax.hist(y, bins=80, color=DARK_BLUE, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Document contribution $y$")
    ax.set_ylabel("Documents")
    ax.set_title(f"Distribution of $y$ (n = {len(y):,})")

    ax = axes[0, 1]
    ax.hist(coverage, bins=50, color=PURPLE, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Swarm runs that include the document")
    ax.set_ylabel("Documents")
    ax.set_title(f"Run coverage (mean {coverage.mean():.1f}, median {np.median(coverage):.0f})")

    ax = axes[0, 2]
    ax.hist(runs["val_loss"], bins=40, color=PINK, edgecolor="white", linewidth=0.3)
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
               boxprops=dict(facecolor=PINK, alpha=0.5))
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
    hb = ax.hexbin(p_te, y_te, gridsize=70, cmap="Purples", mincnt=3,
                   extent=(-0.13, 0.11, -0.18, 0.14))
    ax.plot((-0.18, 0.14), (-0.18, 0.14), color=PINK, lw=1.4, ls="--", label="y = ŷ")
    bins = np.linspace(-0.10, 0.09, 22)
    which = np.digitize(p_te, bins) - 1
    mids = 0.5 * (bins[:-1] + bins[1:])
    means = np.array([y_te[which == i].mean() if (which == i).sum() > 50 else np.nan
                      for i in range(len(mids))])
    ax.plot(mids, means, color=DARK_BLUE, lw=2.4, marker="o", ms=4.5,
            markerfacecolor="white", markeredgecolor=DARK_BLUE, label="binned mean of y")
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
        if sd.get("upper_bounds") is not None:
            ax.fill_between(x, np.asarray(sd["lower_bounds"], float)[keep],
                            np.asarray(sd["upper_bounds"], float)[keep],
                            color=LIGHT_BLUE, alpha=0.45, linewidth=0)
        ax.plot(x, s, color=DARK_BLUE, lw=2.0)
        ax.axhline(0, color=GRAY, ls=":", lw=0.8, zorder=0)
        ax.scatter([x[np.argmax(s)]], [s.max()], s=28, color=PINK, zorder=5,
                   edgecolor="white", linewidths=0.8)
        ax.scatter([x[np.argmin(s)]], [s.min()], s=28, color=PURPLE, zorder=5,
                   edgecolor="white", linewidths=0.8)
        ax.set_title(PRETTY_FEATURE.get(feat, feat), fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Contribution to $y$ ($\Delta y$)")
    fig.suptitle("Shape functions of the top six EBM features", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "ebm_shapes.png", dpi=200)
    plt.close()


def _save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


def fig_filter_correlations() -> None:
    d = json.loads((RESULTS / "filter_correlations.json").read_text())
    rho = pd.DataFrame(d["spearman_global"])
    labels = {"dclm_score": "DCLM", "fwedu_rounded": "FineWeb-Edu", "pc_arc_easy": "PC-ARC",
              "pc_piqa": "PC-PIQA", "pc_sciq": "PC-SciQ", "pc_lambada": "PC-LAMBADA",
              "qr_writing_style": "QR-Writing", "qr_required_expertise": "QR-Expertise",
              "qr_facts_and_trivia": "QR-Facts", "qr_educational_value": "QR-EduVal"}
    cols = [c for c in labels if c in rho.columns]
    m = rho.loc[cols, cols].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(m, cmap=BLUES, vmin=-0.2, vmax=1)
    ax.set_xticks(range(len(cols)), [labels[c] for c in cols], rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), [labels[c] for c in cols])
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if m[i, j] > 0.6 else "#1f2937")
    ax.grid(False)
    ax.set_title("Spearman correlation between filters")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    _save(fig, "filter_correlations.png")


def fig_factor_loadings(factor_dir: Path) -> None:
    data = np.load(factor_dir / "two_factor_em_outputs.npz", allow_pickle=True)
    loading = data["loading"].astype(float)
    names = {"dclm_score": "DCLM", "fwedu_rounded": "FineWeb-Edu", "pc_arc_easy": "ARC-Easy",
             "pc_piqa": "PIQA", "pc_sciq": "SciQ", "pc_lambada": "LAMBADA"}
    labels = [names.get(str(n), str(n)) for n in data["score_names"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(x - 0.19, loading[:, 0], 0.38, color=DARK_BLUE, label="Factor 1 (benchmark)")
    ax.bar(x + 0.19, loading[:, 1], 0.38, color=PINK, label="Factor 2 (generative quality)")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Loading")
    ax.set_title("Two-factor loadings")
    ax.set_ylim(top=1.12)
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=9)
    ax.grid(axis="x", visible=False)
    plt.tight_layout()
    _save(fig, "factor_loadings.png")


def fig_quality_diversity(factor_dir: Path) -> None:
    d = json.loads((factor_dir / "quality_diversity_frontier.json").read_text())
    pts = d["two_factor_frontier"]
    x = np.array([p["topic_entropy"] for p in pts])
    y = np.array([p["mean_quality"] for p in pts])
    floor = np.array([p["topic_floor_fraction"] for p in pts])
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(x, y, color=DARK_BLUE, lw=1.4, zorder=2)
    sc = ax.scatter(x, y, c=floor, cmap=BLUES, vmin=-0.05, vmax=floor.max(), s=46,
                    edgecolor=DARK_BLUE, linewidths=0.6, zorder=3)
    ref = d["references"]
    ax.scatter(ref["global_dclm"]["topic_entropy"], ref["global_dclm"]["mean_quality"],
               marker="s", s=60, color=PINK, zorder=3)
    ax.scatter(ref["random"]["topic_entropy"], ref["random"]["mean_quality"],
               marker="o", s=60, color=GRAY, zorder=3)
    handles = [Line2D([], [], color=DARK_BLUE, marker="o", markerfacecolor=LIGHT_BLUE,
                      label="Two-factor, top 20%"),
               Line2D([], [], color=PINK, marker="s", ls="", label="Global DCLM"),
               Line2D([], [], color=GRAY, marker="o", ls="", label="Random")]
    ax.legend(handles=handles, frameon=False, loc="center left")
    ax.set_xlabel("Topic entropy")
    ax.set_ylabel("Mean held-out quality")
    ax.set_title("Quality vs. topic diversity")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Within-topic floor")
    cbar.ax.set_ylim(0, floor.max())
    plt.tight_layout()
    _save(fig, "quality_diversity.png")


def fig_r2() -> None:
    names, vals = zip(*R2)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bars = ax.bar(names, vals, color=[LIGHT_BLUE, PINK, PURPLE, DARK_BLUE], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center",
                va="bottom", fontsize=10)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("Test R²")
    ax.set_title("Predicting y on held-out documents")
    ax.grid(axis="x", visible=False)
    plt.tight_layout()
    _save(fig, "r2.png")


def _shape_kind(scores: np.ndarray, edges: np.ndarray) -> str:
    """Classify a shape function as up, down or peak (interior maximum)."""
    mids = 0.5 * (edges[:-1] + edges[1:])
    lo, hi = np.quantile(edges, [0.02, 0.98])
    s = scores[(mids >= lo) & (mids <= hi)]
    n = len(s)
    if n < 5:
        return "up"
    k = max(1, n // 10)
    left, right = s[:k].mean(), s[-k:].mean()
    span = max(s.max() - s.min(), 1e-9)
    if int(0.2 * n) <= int(np.argmax(s)) < n - int(0.2 * n) and s.max() - max(left, right) > 0.35 * span:
        return "peak"
    return "up" if right > left else "down"


def fig_feature_importance(top: int = 12) -> None:
    ebm = joblib.load(MODELS_DIR / "ebm_interp.joblib")
    glob = ebm.explain_global()
    gd = glob.data()
    names = list(ebm.feature_names_in_)
    terms = sorted(((n, float(s)) for n, s in zip(gd["names"], gd["scores"])
                    if " & " not in n and n not in ("topic", "format")), key=lambda kv: -kv[1])[:top]
    kinds = [_shape_kind(np.asarray(glob.data(names.index(n))["scores"], float),
                         np.asarray(glob.data(names.index(n))["names"], float)) for n, _ in terms]
    glyph = {"up": "\u2191", "down": "\u2193", "peak": "\u2229"}
    color = {"up": DARK_BLUE, "down": PINK, "peak": PURPLE}
    terms, kinds = terms[::-1], kinds[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    vals = [v for _, v in terms]
    ax.barh(range(len(terms)), vals, color=[color[k] for k in kinds], height=0.7)
    ax.set_yticks(range(len(terms)),
                  [f"{glyph[k]}  {PRETTY_FEATURE.get(n, n.replace('_', ' '))}" for (n, _), k in zip(terms, kinds)])
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"{v:.4f}", va="center", fontsize=8.5)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel(r"Mean $|\Delta y|$")
    ax.set_title("Top EBM features by importance")
    ax.grid(axis="y", visible=False)
    handles = [Line2D([], [], color=color[k], lw=6, label=lab) for k, lab in
               [("up", "\u2191 increasing"), ("down", "\u2193 decreasing"), ("peak", "\u2229 peaked")]]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    plt.tight_layout()
    _save(fig, "feature_importance.png")


def fig_selection() -> None:
    df = pd.read_csv(RESULTS / "validation_60m_2ep.csv")
    pretty = {"ebm_within_topic_eq150M": "EBM within topics", "random_eq150M_seed42": "Random (seed 42)",
              "random_eq150M_seed43": "Random (seed 43)", "random_eq150M_seed44": "Random (seed 44)",
              "ebm_eq150M": "EBM, all features", "ebm_eq150M_seed43": "EBM, all features (seed 43)",
              "ebm_interp_eq150M": "EBM, no PC", "lasso_eq150M": "Lasso", "fwedu_eq150M": "FineWeb-Edu",
              "dclm_eq150M": "DCLM"}
    df = df.sort_values("val_loss", ascending=False).reset_index(drop=True)
    rand = float(df.loc[df["name"] == "random_eq150M_seed42", "val_loss"].iloc[0])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for i, r in df.iterrows():
        n = r["name"]
        c = PINK if n.startswith("ebm_within") else (GRAY if n.startswith("random") else DARK_BLUE)
        ax.plot([rand, r["val_loss"]], [i, i], color=LIGHT_BLUE, lw=2, zorder=1)
        ax.scatter(r["val_loss"], i, s=50, color=c, zorder=2)
    ax.axvline(rand, color=GRAY, ls="--", lw=1)
    ax.set_yticks(range(len(df)), [pretty.get(n, n) for n in df["name"]])
    ax.set_xlabel("CommonCrawl val loss (lower is better)")
    ax.set_title("Held-out selection, 60M LMs")
    ax.grid(axis="y", visible=False)
    plt.tight_layout()
    _save(fig, "selection.png")


def main() -> None:
    readme = ["correlations", "loadings", "diversity", "r2", "calibration", "shapes",
              "importance", "selection"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=readme + ["swarm", "target"],
                    help="subset of figures to make (default: the README figures)")
    ap.add_argument("--factor-dir", type=Path, default=DATA_DIR / "factor_model")
    args = ap.parse_args()
    wanted = set(args.only or readme)
    OUT.mkdir(parents=True, exist_ok=True)

    if "correlations" in wanted:
        fig_filter_correlations()
    if "loadings" in wanted:
        fig_factor_loadings(args.factor_dir)
    if "diversity" in wanted:
        fig_quality_diversity(args.factor_dir)
    if "r2" in wanted:
        fig_r2()
    if "selection" in wanted:
        fig_selection()
    if "importance" in wanted:
        fig_feature_importance()
    if "shapes" in wanted:
        fig_ebm_shapes()
    if wanted & {"swarm", "target"}:
        runs = load_runs()
        if "swarm" in wanted:
            fig_swarm_val_loss(runs)
    if wanted & {"target", "calibration"}:
        df, y, in_pool = train_pool()
        if "target" in wanted:
            fig_target_overview(df, y, load_runs())
        if "calibration" in wanted:
            fig_calibration(df, y, in_pool)
    print(f"wrote figures to {OUT}")

if __name__ == "__main__":
    main()
