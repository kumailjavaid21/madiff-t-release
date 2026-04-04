from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


DATASET_ORDER = ["breast_cancer", "adult_income", "pima", "heart", "credit"]
DATASET_DISPLAY = {
    "breast_cancer": "Breast Cancer",
    "adult_income": "Adult Income",
    "pima": "PIMA",
    "heart": "Heart",
    "credit": "Credit",
}

METHOD_ORDER = ["GC", "TVAE", "CTGAN", "TabDDPM", "TabSyn", "STaSy", "MADiffT"]
METHOD_DISPLAY = {
    "GC": "Gaussian Copula",
    "TVAE": "TVAE",
    "CTGAN": "CTGAN",
    "TabDDPM": "TabDDPM",
    "TabSyn": "TabSyn",
    "STaSy": "STaSy",
    "MADiffT": "MADiff-T",
}
METHOD_COLORS = {
    "GC": "#6E6E6E",
    "TVAE": "#4C78A8",
    "CTGAN": "#F58518",
    "TabDDPM": "#54A24B",
    "TabSyn": "#B279A2",
    "STaSy": "#ECA3D2",
    "MADiffT": "#E45756",
}


def set_elsevier_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "TeX Gyre Termes",
                "STIXGeneral",
            ],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "lines.linewidth": 1.4,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )


def _load_latest_madifft_metrics(root: Path) -> pd.DataFrame:
    candidates = list((root / "outputs").rglob("madifft_generation_metrics.csv"))
    if not candidates:
        return pd.DataFrame()
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pd.read_csv(candidates[0])


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="both", alpha=0.2, linewidth=0.55)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")


def _bold_legend(legend_obj) -> None:
    if legend_obj is None:
        return
    for text in legend_obj.get_texts():
        text.set_fontweight("bold")
    title = legend_obj.get_title()
    if title is not None:
        title.set_fontweight("bold")


def build_comparative_utility(fig_dir: Path) -> None:
    path = fig_dir / "comparative_utility_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["dataset"].isin(DATASET_ORDER) & df["method"].isin(METHOD_ORDER)]
    datasets = [d for d in DATASET_ORDER if d in set(df["dataset"])]
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    if not datasets or not methods:
        return

    x = np.arange(len(datasets))
    width = 0.82 / len(methods)
    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)

    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("dataset")
        means = [sub.loc[d, "mean_acc_pct"] if d in sub.index else np.nan for d in datasets]
        stds = [sub.loc[d, "std_acc_pct"] if d in sub.index else 0.0 for d in datasets]
        pos = x - 0.41 + width / 2 + i * width
        ax.bar(
            pos,
            means,
            width=width,
            yerr=stds,
            capsize=2.0,
            color=METHOD_COLORS.get(method, "#999999"),
            edgecolor="black",
            linewidth=0.45,
            alpha=0.95,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
            label=METHOD_DISPLAY.get(method, method),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_DISPLAY[d] for d in datasets], rotation=16, ha="right")
    ax.set_ylabel("TSTR Accuracy (%)")
    _style_axes(ax)
    leg = ax.legend(
        ncol=min(4, len(methods)),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        frameon=False,
        columnspacing=1.1,
    )
    _bold_legend(leg)
    fig.savefig(fig_dir / "comparative_utility_IS.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_privacy_audits(fig_dir: Path) -> None:
    path = fig_dir / "privacy_summary_for_figures.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["dataset"].isin(DATASET_ORDER) & df["method"].isin(METHOD_ORDER)]
    datasets = [d for d in DATASET_ORDER if d in set(df["dataset"])]
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    if not datasets or not methods:
        return

    x = np.arange(len(datasets))
    width = 0.82 / len(methods)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), constrained_layout=True)

    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("dataset")
        dcr_mean = [sub.loc[d, "dcr_mean"] if d in sub.index else np.nan for d in datasets]
        dcr_std = [sub.loc[d, "dcr_std"] if d in sub.index else 0.0 for d in datasets]
        mia_mean = [sub.loc[d, "mia_mean"] if d in sub.index else np.nan for d in datasets]
        mia_std = [sub.loc[d, "mia_std"] if d in sub.index else 0.0 for d in datasets]
        pos = x - 0.41 + width / 2 + i * width
        kwargs = dict(
            width=width,
            capsize=1.8,
            color=METHOD_COLORS.get(method, "#999999"),
            edgecolor="black",
            linewidth=0.45,
            alpha=0.95,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
            label=METHOD_DISPLAY.get(method, method),
        )
        axes[0].bar(pos, dcr_mean, yerr=dcr_std, **kwargs)
        axes[1].bar(pos, mia_mean, yerr=mia_std, **kwargs)

    axes[0].set_title("DCR q=0.01 (higher safer)")
    axes[1].set_title("Distance-MIA AUC (lower safer)")
    axes[0].set_ylabel("DCR")
    axes[1].set_ylabel("MIA AUC")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_DISPLAY[d] for d in datasets], rotation=14, ha="right")
        _style_axes(ax)

    axes[1].axhline(0.5, linestyle="--", color="black", linewidth=0.9, alpha=0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        ncol=min(4, len(methods)),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
        columnspacing=1.1,
    )
    _bold_legend(leg)
    fig.savefig(fig_dir / "privacy_audits_IS.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_tradeoff_scatter(fig_dir: Path) -> None:
    path = fig_dir / "tradeoff_scatter_data.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["dataset"].isin(DATASET_ORDER) & df["method"].isin(METHOD_ORDER)]
    if df.empty:
        return

    dataset_markers = {
        "breast_cancer": "o",
        "adult_income": "s",
        "pima": "^",
        "heart": "D",
        "credit": "P",
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)

    for _, row in df.iterrows():
        method = row["method"]
        ds = row["dataset"]
        ax.scatter(
            row["mia_mean"],
            row["mean_acc_pct"],
            s=56,
            marker=dataset_markers.get(ds, "o"),
            color=METHOD_COLORS.get(method, "#777777"),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.95,
        )

    ax.set_xlabel("Distance-MIA AUC (lower safer)")
    ax.set_ylabel("TSTR Accuracy (%)")
    _style_axes(ax)

    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=METHOD_COLORS[m],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=6.6,
            label=METHOD_DISPLAY[m],
        )
        for m in METHOD_ORDER
        if m in set(df["method"])
    ]
    leg = ax.legend(handles=method_handles, loc="upper right", frameon=True, framealpha=0.9, edgecolor="#bbbbbb")
    _bold_legend(leg)
    fig.savefig(fig_dir / "tradeoff_scatter_IS.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_mia_roc_curves(fig_dir: Path) -> None:
    path = fig_dir / "mia_roc_curve_data.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["dataset"].isin(DATASET_ORDER) & df["method"].isin(METHOD_ORDER)]
    if df.empty:
        return

    datasets = [d for d in DATASET_ORDER if d in set(df["dataset"])]
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    ncols = 3
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.8, 7.0), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = df[df["dataset"] == dataset]
        for method in methods:
            msub = subset[subset["method"] == method].sort_values("fpr")
            if msub.empty:
                continue
            color = METHOD_COLORS.get(method, "#777777")
            ax.plot(msub["fpr"], msub["tpr_mean"], color=color, linewidth=1.4, label=METHOD_DISPLAY[method])
            if int(msub["n_seeds"].iloc[0]) > 1:
                low = np.clip(msub["tpr_mean"] - msub["tpr_std"], 0.0, 1.0)
                high = np.clip(msub["tpr_mean"] + msub["tpr_std"], 0.0, 1.0)
                ax.fill_between(msub["fpr"], low, high, color=color, alpha=0.08, linewidth=0.0)
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.8, alpha=0.85)
        ax.set_title(DATASET_DISPLAY[dataset], pad=2)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_aspect("equal", adjustable="box")
        _style_axes(ax)
        row = idx // ncols
        col = idx % ncols
        ax.set_ylabel("TPR" if col == 0 else "")
        ax.set_xlabel("FPR" if row == nrows - 1 else "")

    for ax in axes[len(datasets):]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], linewidth=1.5, label=METHOD_DISPLAY[m])
        for m in methods
    ]
    legend_ax = axes[-1]
    legend_ax.axis("off")
    leg = legend_ax.legend(handles=handles, loc="center", frameon=False, ncol=1, title="Methods")
    _bold_legend(leg)
    fig.savefig(fig_dir / "mia_roc_curves_IS.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_fidelity_analysis(fig_dir: Path, root: Path) -> None:
    df = _load_latest_madifft_metrics(root)
    if df.empty or not {"dataset", "generation", "fidelity"}.issubset(df.columns):
        return
    df = df.copy()
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce")
    df["fidelity"] = pd.to_numeric(df["fidelity"], errors="coerce")
    df = df.dropna(subset=["generation", "fidelity"])
    if df.empty:
        return

    datasets = [d for d in DATASET_ORDER if d in set(df["dataset"])]
    ncols = 3
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.8, 7.0), constrained_layout=True, sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        sub = df[df["dataset"] == dataset]
        summary = sub.groupby("generation")["fidelity"].agg(["mean", "std"]).reset_index().sort_values("generation")
        x = summary["generation"].to_numpy(dtype=float)
        y = summary["mean"].to_numpy(dtype=float)
        s = summary["std"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, color="#4C78A8", linewidth=1.5, label="Mean fidelity")
        ax.fill_between(x, y - s, y + s, color="#4C78A8", alpha=0.14, label="Mean +/- std")
        if "selected" in sub.columns:
            sel = sub[sub["selected"].astype(str).str.lower().isin(["true", "1", "yes"])]
            if not sel.empty:
                sel_s = sel.groupby("generation")["fidelity"].mean().reset_index().sort_values("generation")
                ax.scatter(sel_s["generation"], sel_s["fidelity"], marker="*", s=40, color="#E45756", zorder=4, label="Selected")
        ax.set_title(DATASET_DISPLAY.get(dataset, dataset), pad=2)
        _style_axes(ax)
        row, col = divmod(idx, ncols)
        if col == 0:
            ax.set_ylabel("Fidelity proxy")
        if row == nrows - 1:
            ax.set_xlabel("Generation")

    for ax in axes[len(datasets):]:
        ax.axis("off")

    legend_handles = [
        Line2D([0], [0], color="#4C78A8", linewidth=1.5, label="Mean fidelity"),
        Line2D([0], [0], color="#4C78A8", linewidth=6, alpha=0.14, label="Mean +/- std"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#E45756", markersize=8, label="Selected"),
    ]
    legend_ax = axes[-1]
    legend_ax.axis("off")
    leg = legend_ax.legend(handles=legend_handles, loc="center", frameon=False, ncol=1, title="Curve elements")
    _bold_legend(leg)
    fig.savefig(fig_dir / "fidelity_analysis_IS.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_ablation(fig_dir: Path, root: Path) -> None:
    df = _load_latest_madifft_metrics(root)
    required = {"dataset", "seed", "utility", "mia_auc"}
    if df.empty or not required.issubset(df.columns):
        return
    df = df.copy()
    df["utility"] = pd.to_numeric(df["utility"], errors="coerce")
    df["mia_auc"] = pd.to_numeric(df["mia_auc"], errors="coerce")
    if "fitness" in df.columns:
        df["fitness"] = pd.to_numeric(df["fitness"], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df = df.dropna(subset=["dataset", "seed", "utility", "mia_auc"])
    if df.empty:
        return
    if df["utility"].max() <= 1.5:
        df["utility"] *= 100.0

    rows = []
    for (dataset, seed), grp in df.groupby(["dataset", "seed"]):
        g = grp.copy()
        if "selected" in g.columns:
            sel = g[g["selected"].astype(str).str.lower().isin(["true", "1", "yes"])]
            if sel.empty and "fitness" in g.columns:
                sel = g.sort_values("fitness", ascending=False).head(1)
        elif "fitness" in g.columns:
            sel = g.sort_values("fitness", ascending=False).head(1)
        else:
            sel = g.sort_values("utility", ascending=False).head(1)
        util_pick = g.sort_values("utility", ascending=False).head(1)
        priv_pick = g.sort_values("mia_auc", ascending=True).head(1)
        for label, frame in [
            ("Selected (privacy-in-loop)", sel),
            ("Utility-only pick", util_pick),
            ("Privacy-only pick", priv_pick),
        ]:
            if frame.empty:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "utility": float(frame["utility"].iloc[0]),
                    "mia_auc": float(frame["mia_auc"].iloc[0]),
                }
            )
    proxy = pd.DataFrame(rows)
    if proxy.empty:
        return

    labels = ["Selected (privacy-in-loop)", "Utility-only pick", "Privacy-only pick"]
    color_map = {
        "Selected (privacy-in-loop)": "#E45756",
        "Utility-only pick": "#4C78A8",
        "Privacy-only pick": "#54A24B",
    }
    marker_map = {
        "Selected (privacy-in-loop)": "X",
        "Utility-only pick": "o",
        "Privacy-only pick": "^",
    }
    datasets = [d for d in DATASET_ORDER if d in set(proxy["dataset"])]
    ncols = 3
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.8, 7.0), constrained_layout=True, sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        sub = proxy[proxy["dataset"] == dataset]
        for label in labels:
            pts = sub[sub["label"] == label]
            if pts.empty:
                continue
            ax.scatter(
                pts["mia_auc"],
                pts["utility"],
                s=40,
                marker=marker_map[label],
                color=color_map[label],
                edgecolor="black",
                linewidth=0.35,
                alpha=0.95,
            )
        ax.set_title(DATASET_DISPLAY.get(dataset, dataset), pad=2)
        _style_axes(ax)
        row, col = divmod(idx, ncols)
        if col == 0:
            ax.set_ylabel("Utility (%)")
        if row == nrows - 1:
            ax.set_xlabel("MIA AUC (lower safer)")

    for ax in axes[len(datasets):]:
        ax.axis("off")

    handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map[label],
            linestyle="",
            color=color_map[label],
            markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=6.5,
            label=label,
        )
        for label in labels
    ]
    legend_ax = axes[-1]
    legend_ax.axis("off")
    leg = legend_ax.legend(handles=handles, loc="center", frameon=False, ncol=1)
    _bold_legend(leg)
    fig.savefig(fig_dir / "ablation_study_IS.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_evolution_curves(fig_dir: Path, root: Path) -> None:
    df = _load_latest_madifft_metrics(root)
    if df.empty or not {"dataset", "generation", "fitness"}.issubset(df.columns):
        return
    df = df.copy()
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce")
    df["fitness"] = pd.to_numeric(df["fitness"], errors="coerce")
    df = df.dropna(subset=["generation", "fitness"])
    if df.empty:
        return

    datasets = [d for d in DATASET_ORDER if d in set(df["dataset"])]
    ncols = 3
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.8, 7.0), constrained_layout=True, sharex=True)
    axes = np.array(axes).reshape(-1)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        sub = df[df["dataset"] == dataset]
        summary = (
            sub.groupby("generation")["fitness"]
            .agg(["mean", "std", "max", "median"])
            .reset_index()
            .sort_values("generation")
        )
        x = summary["generation"].to_numpy(dtype=float)
        y_mean = summary["mean"].to_numpy(dtype=float)
        y_std = summary["std"].fillna(0.0).to_numpy(dtype=float)
        y_best = summary["max"].to_numpy(dtype=float)
        y_median = summary["median"].to_numpy(dtype=float)

        ax.plot(x, y_mean, color="#4C78A8", linewidth=1.5, label="Mean")
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, color="#4C78A8", alpha=0.14, label="Mean +/- std")
        ax.plot(x, y_best, color="#E45756", linewidth=1.4, linestyle="--", label="Best")
        ax.plot(x, y_median, color="#54A24B", linewidth=1.2, linestyle=":", label="Median")

        if "selected" in sub.columns:
            sel = sub[sub["selected"].astype(str).str.lower().isin(["true", "1", "yes"])]
            if not sel.empty:
                sel_summary = sel.groupby("generation")["fitness"].mean().reset_index().sort_values("generation")
                ax.scatter(sel_summary["generation"], sel_summary["fitness"], marker="*", s=40, color="#B279A2", zorder=4, label="Selected")

        ax.set_title(DATASET_DISPLAY.get(dataset, dataset), pad=2)
        _style_axes(ax)
        row, col = divmod(idx, ncols)
        if col == 0:
            ax.set_ylabel("Fitness")
        if row == nrows - 1:
            ax.set_xlabel("Generation")

    for ax in axes[len(datasets):]:
        ax.axis("off")

    legend_ax = axes[-1]
    legend_ax.axis("off")
    legend_handles = [
        Line2D([0], [0], color="#4C78A8", linewidth=1.5, label="Mean"),
        Line2D([0], [0], color="#4C78A8", linewidth=6, alpha=0.14, label="Mean +/- std"),
        Line2D([0], [0], color="#E45756", linewidth=1.4, linestyle="--", label="Best"),
        Line2D([0], [0], color="#54A24B", linewidth=1.2, linestyle=":", label="Median"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#B279A2", markersize=8, label="Selected"),
    ]
    leg = legend_ax.legend(handles=legend_handles, loc="center", frameon=False, ncol=1, title="Curve elements")
    _bold_legend(leg)
    fig.savefig(fig_dir / "evolution_curves_IS.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="D:\\MAS_genAI")
    parser.add_argument("--fig_dir", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    fig_dir = (
        Path(args.fig_dir).resolve()
        if args.fig_dir
        else root / "outputs" / "final_bundle" / "journal_figures"
    )
    fig_dir.mkdir(parents=True, exist_ok=True)

    set_elsevier_style()
    build_comparative_utility(fig_dir)
    build_privacy_audits(fig_dir)
    build_tradeoff_scatter(fig_dir)
    build_mia_roc_curves(fig_dir)
    build_fidelity_analysis(fig_dir, root)
    build_ablation(fig_dir, root)
    build_evolution_curves(fig_dir, root)

    print(f"Saved journal-style styled PDFs in: {fig_dir}")
    print(
        "Files: ablation_study_IS.pdf, comparative_utility_IS.pdf, fidelity_analysis_IS.pdf, "
        "evolution_curves_IS.pdf, mia_roc_curves_IS.pdf, privacy_audits_IS.pdf, tradeoff_scatter_IS.pdf"
    )


if __name__ == "__main__":
    main()
