"""
heterogeneity_modelling.py
──────────────────────────────────────────────────────────────────────────────
Subject-heterogeneity analysis pipeline.

Blocks
------
1. Responder rate summary
2. Mutual Information (MI) analysis
3. MI summary with relative tercile ranking
4. Combined subject clustering (Ward / agglomerative)
5. Cluster → moderator scatter plots
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage, set_link_color_palette

from src.pipeline.signal_features import compute_feature_mi_importance
from src.pipeline.visualizations import plot_scatter
import src.utils.file_management as filemgmt


# ══════════════════════════════════════════════════════════════════════════════
#  Types / constants
# ══════════════════════════════════════════════════════════════════════════════

PlotKey = Literal["cooks_d", "dfbeta", "contrast"]

_METRIC_ORDER: dict[str, int] = {"DFBETA": 0, "CooksD": 1, "Contrast": 2}
_CLUSTER_PALETTE: list[str] = ["#e377c2", "#17becf", "#2ca02c"]
_METRIC_COLORS: dict[str, str] = {
    "DFBETA": "#d62728",
    "CooksD": "red",
    "Contrast": "#1f77b4",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Block 1 — Responder Rate Summary
# ══════════════════════════════════════════════════════════════════════════════

def compute_responder_summary(
    subject_contrast_frame: pd.DataFrame,
    dep_vars: list[str],
    conditions_to_evaluate: dict[str, tuple[str, list[str]]],
    subj_col: str = "Subject_ID",
    dep_var_col: str = "Dependent_Variable",
) -> pd.DataFrame:
    """Return a tidy responder-rate table across DVs, levels, and conditions."""
    rows: list[dict] = []
    for dep_var in dep_vars:
        contrast_sub = subject_contrast_frame.loc[subject_contrast_frame[dep_var_col] == dep_var]
        for level_key, (cond_var, conditions) in conditions_to_evaluate.items():
            level_sub = contrast_sub.loc[contrast_sub["Condition_Variable"] == cond_var]
            for condition in conditions:
                cond_rows = level_sub.loc[level_sub["Condition"] == condition]
                n_subj = cond_rows[subj_col].nunique()
                n_resp = cond_rows.loc[cond_rows["Responder_Flag"], subj_col].nunique()
                rows.append({
                    dep_var_col: dep_var,
                    "Level": level_key,
                    "Condition_Variable": cond_var,
                    "Condition": condition,
                    "N_Subjects": n_subj,
                    "N_Responders": n_resp,
                    "Responder_Rate": round(n_resp / n_subj, 3) if n_subj > 0 else np.nan,
                })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  Block 2 — Mutual Information Analysis
# ══════════════════════════════════════════════════════════════════════════════

def _run_mi_single(
    feature_df: pd.DataFrame,
    target_col: str,
    target_type: str,
    dep_var: str,
    level: str,
    cond_var: str,
    attr_cols: list[str],
    condition: str | None = None,
    plot_key: PlotKey | None = None,
    output_dir: Path | None = None,
) -> list[dict]:
    """Run MI for one (target, DV, condition) combination. Returns result rows."""
    valid = feature_df.dropna(subset=[target_col])
    if len(valid) < 4 or valid[target_col].nunique() < 2:
        return []

    target_arr = (
        valid[target_col].astype(int).values
        if target_type == "discrete"
        else valid[target_col].astype(float).values
    )
    mi_out = compute_feature_mi_importance(
        feature_array=valid[attr_cols].values,
        target_array=target_arr,
        feature_labels=attr_cols,
        target_label=f"{target_col}{f'[{condition}]' if condition else ''} │ {dep_var}",
        target_type=target_type,
        include_barplot=plot_key is not None,
        plot_save_dir=output_dir if plot_key is not None else None,
    )
    scores: dict = mi_out[2] if isinstance(mi_out, tuple) else mi_out
    return [
        {
            "Dependent_Variable": dep_var,
            "Level": level,
            "Condition_Variable": cond_var,
            "Condition": condition,
            "Target": target_col,
            "Feature": feat,
            "MI_Score": score,
        }
        for feat, score in scores.items()
    ]


def compute_mi_results(
    dep_vars: list[str],
    influence_frame: pd.DataFrame,
    contrast_frame: pd.DataFrame,
    coefficient_frame: pd.DataFrame,
    personal_df: pd.DataFrame,
    attr_cols: list[str],
    conditions_to_evaluate: dict[str, tuple[str, list[str]]],
    plot_mi_categories: list[PlotKey],
    alpha_omnibus: float = 0.05,
    analyse_dfbetas: bool = True,
    output_dir: Path | None = None,
    subj_col: str = "Subject_ID",
    dep_var_col: str = "Dependent_Variable",
) -> pd.DataFrame:
    """Run MI analysis across all DVs, levels, and conditions. Returns raw MI frame."""
    all_rows: list[dict] = []

    for dep_var in dep_vars:
        print(f"\n{'=' * 72}\n  MI Analysis │ {dep_var}\n{'=' * 72}")

        influence_sub = influence_frame.loc[influence_frame[dep_var_col] == dep_var].copy()
        contrast_sub = contrast_frame.loc[contrast_frame[dep_var_col] == dep_var].copy()

        if influence_sub.empty:
            warnings.warn(f"  [skip] No influence data for '{dep_var}'.")
            continue

        # ── Cook's D vs. personal attributes ─────────────────────────────────
        cooks_per_subj = (
            influence_sub.groupby(subj_col, as_index=False)["CooksD"].mean()
            .merge(personal_df, on=subj_col, how="left")
            .dropna(subset=attr_cols + ["CooksD"])
        )
        if len(cooks_per_subj) >= 4:
            print(f"\n  [Cook's D] n={len(cooks_per_subj)}")
            all_rows.extend(_run_mi_single(
                cooks_per_subj, "CooksD", "continuous", dep_var, "influence", "—",
                attr_cols, plot_key="cooks_d" if "cooks_d" in plot_mi_categories else None,
                output_dir=output_dir,
            ))

        # ── DFBeta vs. personal attributes (one run per significant parameter) ─
        if analyse_dfbetas:
            sig_params = coefficient_frame.loc[
                (coefficient_frame[dep_var_col] == dep_var)
                & (coefficient_frame["Model_Type"] == "LME")
                & (coefficient_frame["p_value_adjusted"] < alpha_omnibus),
                "Parameter",
            ].unique()
            if not len(sig_params):
                print(f"  [DFBeta] No significant parameters for '{dep_var}'.")
            for param in sig_params:
                param_rows = (
                    influence_sub.loc[influence_sub["Parameter"] == param]
                    .merge(personal_df, on=subj_col, how="left")
                    .dropna(subset=attr_cols + ["DFBETA"])
                )
                if len(param_rows) < 4:
                    continue
                print(f"\n  [DFBeta] '{param}'  n={len(param_rows)}")
                all_rows.extend(_run_mi_single(
                    param_rows, "DFBETA", "continuous", dep_var, "influence", "—",
                    attr_cols, condition=param,
                    plot_key="dfbeta" if "dfbeta" in plot_mi_categories else None,
                    output_dir=output_dir,
                ))

        # ── Per-condition: Responder_Flag + Normalised_Contrast ───────────────
        for level_key, (cond_var, conditions) in conditions_to_evaluate.items():
            level_sub = contrast_sub.loc[contrast_sub["Condition_Variable"] == cond_var]
            for condition in conditions:
                cond_rows = (
                    level_sub.loc[level_sub["Condition"] == condition]
                    .merge(personal_df, on=subj_col, how="left")
                    .dropna(subset=attr_cols)
                )
                if len(cond_rows) < 4:
                    warnings.warn(f"  [{level_key}|{condition}] Too few subjects.")
                    continue
                print(f"\n  [{level_key}] '{condition}'  n={len(cond_rows)}")
                all_rows.extend(_run_mi_single(
                    cond_rows, "Responder_Flag", "discrete", dep_var, level_key, cond_var,
                    attr_cols, condition=condition,
                ))
                all_rows.extend(_run_mi_single(
                    cond_rows, "Normalised_Contrast", "continuous", dep_var, level_key, cond_var,
                    attr_cols, condition=condition,
                    plot_key="contrast" if "contrast" in plot_mi_categories else None,
                    output_dir=output_dir,
                ))

    return pd.DataFrame(all_rows)


# ══════════════════════════════════════════════════════════════════════════════
#  Block 3 — MI Summary with Tercile Ranking
# ══════════════════════════════════════════════════════════════════════════════

def _assign_tercile_band(grp: pd.DataFrame) -> pd.Series:
    scores = grp["MI_Score"]
    t33, t67 = scores.quantile([1 / 3, 2 / 3])
    if t33 == t67:
        return pd.Series(["Medium"] * len(scores), index=scores.index)
    return scores.apply(lambda s: "High" if s >= t67 else ("Medium" if s >= t33 else "Low"))


def build_mi_summary(mi_df: pd.DataFrame, dep_var_col: str = "Dependent_Variable") -> pd.DataFrame:
    """Add tercile band + moderator candidate flags; return aggregated summary."""
    mi_df = mi_df.copy()
    mi_df["MI_Band"] = (
        mi_df
        .groupby([dep_var_col, "Level", "Condition", "Target"], group_keys=False)
        .apply(_assign_tercile_band)
    )
    high_features = set(mi_df.loc[mi_df["MI_Band"] == "High", "Feature"].unique())
    mi_df["Moderator_Candidate"] = mi_df["Feature"].isin(high_features)

    summary = (
        mi_df
        .groupby(["Feature", "Level", "Condition", "Target"])
        .apply(lambda g: pd.Series({
            "Max_MI_Score": g["MI_Score"].max(),
            "Band_at_Max": g.loc[g["MI_Score"].idxmax(), "MI_Band"],
            "N_High_across_DVs": int((g["MI_Band"] == "High").sum()),
            "Moderator_Candidate": g["Moderator_Candidate"].any(),
        }))
        .reset_index()
        .sort_values(["Level", "Condition", "Max_MI_Score"], ascending=[True, True, False])
    )
    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  Block 4 — Combined Subject Clustering
# ══════════════════════════════════════════════════════════════════════════════

def _scaled_pivot(
    long_df: pd.DataFrame,
    index_col: str,
    col_col: str,
    val_col: str,
) -> pd.DataFrame:
    """Pivot → drop columns with any NaN → z-score standardise."""
    piv = long_df.pivot_table(
        index=index_col, columns=col_col, values=val_col, aggfunc="mean"
    ).dropna(axis=1, how="any")
    return pd.DataFrame(
        StandardScaler().fit_transform(piv.values),
        index=piv.index, columns=piv.columns,
    )


def _sort_pivot_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sort columns: metric block → condition/parameter → DV."""
    return df.reindex(
        sorted(
            df.columns,
            key=lambda c: (
                _METRIC_ORDER.get(c.split("│")[0], 99),
                c.split("│")[-1],
                c.split("│")[1],
            ),
        ),
        axis=1,
    )


def build_combined_pivot(
    influence_frame: pd.DataFrame,
    contrast_frame: pd.DataFrame,
    dep_vars: list[str],
    sig_pairs: pd.DataFrame,
    conditions_to_evaluate: dict[str, tuple[str, list[str]]],
    clustering_measures: list[str],
    subj_col: str = "Subject_ID",
    dep_var_col: str = "Dependent_Variable",
) -> pd.DataFrame:
    """Assemble and standardise the subject × feature matrix for clustering."""
    blocks: list[pd.DataFrame] = []

    if "dfbeta" in clustering_measures:
        df_long = influence_frame.merge(sig_pairs, on=[dep_var_col, "Parameter"], how="inner").copy()
        df_long["col_key"] = (
            "DFBETA│"
            + df_long[dep_var_col].str.replace("CMC_", "", regex=False)
            + "│" + df_long["Parameter"]
        )
        blocks.append(_scaled_pivot(df_long, subj_col, "col_key", "DFBETA"))

    if "cooks_d" in clustering_measures:
        ck_long = influence_frame.loc[influence_frame[dep_var_col].isin(dep_vars)].copy()
        ck_long["col_key"] = "CooksD│" + ck_long[dep_var_col].str.replace("CMC_", "", regex=False)
        blocks.append(_scaled_pivot(ck_long, subj_col, "col_key", "CooksD"))

    if "contrast" in clustering_measures:
        ct_rows = pd.concat([
            contrast_frame.loc[
                contrast_frame[dep_var_col].isin(dep_vars)
                & (contrast_frame["Condition_Variable"] == cond_var)
                & (contrast_frame["Condition"].isin(conditions))
            ]
            for _, (cond_var, conditions) in conditions_to_evaluate.items()
        ], ignore_index=True)
        ct_rows["col_key"] = (
            "Contrast│"
            + ct_rows[dep_var_col].str.replace("CMC_", "", regex=False)
            + "│" + ct_rows["Condition"].astype(str)
        )
        blocks.append(_scaled_pivot(ct_rows, subj_col, "col_key", "Normalised_Contrast"))

    combined = blocks[0].copy()
    for blk in blocks[1:]:
        combined = combined.join(blk, how="inner")

    return _sort_pivot_columns(combined)


def select_best_k(
    X: np.ndarray,
    k_range: range,
    min_cluster_size: int,
) -> tuple[int, dict[int, float]]:
    """Return best k by silhouette score, enforcing minimum cluster size."""
    valid_scores: dict[int, float] = {}
    for k in k_range:
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        if np.all(np.bincount(labels) >= min_cluster_size):
            valid_scores[k] = silhouette_score(X, labels)
        else:
            print(f"  [Clustering] k={k} skipped — has cluster < {min_cluster_size} subjects")

    if not valid_scores:
        warnings.warn(f"  [Clustering] No valid k in {list(k_range)} with min_size={min_cluster_size}. Falling back to k=2.")
        return 2, valid_scores

    best_k = max(valid_scores, key=valid_scores.get)
    print(f"  [Clustering] Silhouette scores: { {k: f'{v:.3f}' for k, v in valid_scores.items()} }")
    return best_k, valid_scores


def plot_clustering(
    combined_pivot: pd.DataFrame,
    cluster_labels: np.ndarray,
    linkage_matrix: np.ndarray,
    best_k: int,
    clustering_measures: list[str],
    dep_vars: list[str],
    output_dir: Path,
) -> None:
    """Render and save the Ward dendrogram + heatmap figure."""
    row_order = leaves_list(linkage_matrix)[::-1]
    ordered_data = combined_pivot.iloc[row_order]

    col_colors = [
        _METRIC_COLORS.get(c.split("│")[0], "#888888")
        for c in combined_pivot.columns
    ]

    n_rows, n_cols = combined_pivot.shape
    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(14, n_cols * 0.7), 9),
        gridspec_kw={"width_ratios": [1, 2], "wspace": 0.14},
    )

    # ── Dendrogram ────────────────────────────────────────────────────────────
    set_link_color_palette(_CLUSTER_PALETTE)
    dendrogram(
        linkage_matrix,
        labels=[f"Sub_{i:02d}" for i in combined_pivot.index],
        orientation="left",
        ax=axes[0],
        color_threshold=linkage_matrix[-(best_k - 1), 2],
    )
    set_link_color_palette(None)

    axes[0].legend(
        handles=[Patch(color=_CLUSTER_PALETTE[k], label=f"Cluster {k + 1}") for k in range(best_k)],
        loc="lower left", framealpha=0.7, title="Clusters",
    )
    axes[0].set_title("Ward Dendrogram")
    axes[0].set_xlabel("Distance")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    vlim = np.nanpercentile(np.abs(ordered_data.values), 97)
    im = axes[1].imshow(ordered_data.values, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    axes[1].set_xticks(range(n_cols))
    axes[1].set_xticklabels(combined_pivot.columns.tolist(), rotation=45, ha="right")
    for tick, col in zip(axes[1].get_xticklabels(), col_colors):
        tick.set_color(col)
    axes[1].set_yticks(range(len(ordered_data)))
    axes[1].set_yticklabels([""] * len(ordered_data))  # labels come from dendrogram
    axes[1].set_title("Combined heterogeneity features")
    plt.colorbar(im, ax=axes[1], label="Standardised value", shrink=0.7)
    _MEASURE_LABEL_MAP = {
        "dfbeta": ("DFBETA", "DFBeta"),
        "cooks_d": ("CooksD", "Cook's D"),
        "contrast": ("Contrast", "Contrast"),
    }
    axes[1].legend(
        handles=[
            Patch(color=_METRIC_COLORS[_MEASURE_LABEL_MAP[m][0]], label=_MEASURE_LABEL_MAP[m][1])
            for m in clustering_measures
            if m in _MEASURE_LABEL_MAP
        ],
        loc="upper right", framealpha=0.7,
    )

    fig.suptitle(
        f"Subject Heterogeneity Clustering │ {' + '.join(clustering_measures)}\n"
        f"DVs: {', '.join(dep_vars)}  │  k={best_k} clusters  │  Ward linkage",
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.28, left=0.03, right=1.0)
    fig.savefig(output_dir / filemgmt.file_title("Heterogeneity Combined Clustering", ".png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def run_clustering(
    combined_pivot: pd.DataFrame,
    personal_df: pd.DataFrame,
    clustering_measures: list[str],
    dep_vars: list[str],
    min_cluster_size: int,
    output_dir: Path,
    subj_col: str = "Subject_ID",
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Fit clustering, save outputs, return (cluster_assign_df, silhouette_scores)."""
    X = combined_pivot.values
    Z = linkage(X, method="ward", metric="euclidean")
    k_range = range(2, min(6, combined_pivot.shape[0]))

    best_k, sil_scores = select_best_k(X, k_range, min_cluster_size)
    print(f"  [Clustering] Selected k = {best_k}")
    cluster_labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)

    plot_clustering(combined_pivot, cluster_labels, Z, best_k, clustering_measures, dep_vars, output_dir)

    cluster_df = (
        pd.DataFrame({subj_col: combined_pivot.index, "Cluster": cluster_labels})
        .sort_values("Cluster")
        .merge(personal_df, on=subj_col, how="left")
    )
    cluster_df.to_csv(output_dir / filemgmt.file_title("Heterogeneity Subject Clusters", ".csv"), index=False)
    pd.DataFrame([{"k": k, "Silhouette": v} for k, v in sil_scores.items()]).to_csv(
        output_dir / filemgmt.file_title("Heterogeneity Silhouette Scores", ".csv"), index=False
    )
    return cluster_df, sil_scores


# ══════════════════════════════════════════════════════════════════════════════
#  Block 5 — Cluster → Moderator Scatter Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_moderator_scatters(
    cluster_df: pd.DataFrame,
    contrast_frame: pd.DataFrame,
    personal_df: pd.DataFrame,
    mi_summary: pd.DataFrame,
    dep_vars: list[str],
    conditions_to_evaluate: dict[str, tuple[str, list[str]]],
    top_n: int,
    output_dir: Path,
    subj_col: str = "Subject_ID",
    dep_var_col: str = "Dependent_Variable",
) -> None:
    """Scatter top-MI moderators against mean normalised contrast, coloured by cluster."""
    top_moderators = (
        mi_summary.loc[mi_summary["Moderator_Candidate"]]
        .sort_values("Max_MI_Score", ascending=False)
        ["Feature"].unique()[:top_n]
    )
    print(f"\n  [Scatter] Top moderators: {list(top_moderators)}")

    lvl1_cond_var, lvl1_conditions = conditions_to_evaluate["lvl_1"]
    mean_contrast = (
        contrast_frame.loc[
            contrast_frame[dep_var_col].isin(dep_vars)
            & (contrast_frame["Condition_Variable"] == lvl1_cond_var)
            & (contrast_frame["Condition"].isin(lvl1_conditions))
        ]
        .groupby(subj_col, as_index=False)["Normalised_Contrast"].mean()
    )

    scatter_df = (
        cluster_df[[subj_col, "Cluster"]]
        .merge(mean_contrast, on=subj_col, how="left")
        .merge(personal_df[[subj_col] + list(top_moderators)], on=subj_col, how="left")
    )

    for moderator in top_moderators:
        valid = scatter_df.dropna(subset=[moderator, "Normalised_Contrast"])
        if len(valid) < 4:
            warnings.warn(f"  [Scatter] Too few valid rows for '{moderator}' — skipping.")
            continue
        print(f"\n  [Scatter] {moderator} × Mean Normalised Contrast  (n={len(valid)})")
        plot_scatter(
            x=valid[moderator].astype(float).values,
            y=valid["Normalised_Contrast"].astype(float).values,
            x_label=moderator,
            y_label="Mean Normalised Contrast (lvl_1)",
            category_list=valid["Cluster"].astype(str).tolist(),
            category_label="Cluster",
            save_dir=output_dir,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Top-level entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_heterogeneity_modelling(
    dep_vars: list[str],
    conditions_to_evaluate: dict[str, tuple[str, list[str]]],
    clustering_measures: list[str],
    plot_mi_categories: list[PlotKey],
    top_n_moderators: int,
    min_cluster_size: int,
    output_dir: Path,
    omnibus_results_dir: Path,
    experiment_results_dir: Path,
    n_subjects: int,
    analyse_mi_for_dfbetas: bool = True,
    alpha_omnibus: float = 0.05,
    subj_col: str = "Subject_ID",
    dep_var_col: str = "Dependent_Variable",
) -> None:
    """Run the full heterogeneity pipeline end-to-end."""
    import src.pipeline.data_integration as data_integration

    # ── Load data ─────────────────────────────────────────────────────────────
    personal_df = pd.DataFrame([
        data_integration.fetch_personal_data(experiment_results_dir / f"subject_{i:02d}")
        for i in range(n_subjects)
    ])
    personal_df.insert(0, subj_col, list(range(n_subjects)))
    personal_df["Is_Right-handed"] = (personal_df["Dominant hand"] == "Right").astype(int)
    personal_df["Is_Male"] = (personal_df["Gender"] == "Male").astype(int)

    attr_cols = [
        c for c in personal_df.columns
        if c != subj_col
        and personal_df[c].nunique(dropna=True) > 1
        and pd.api.types.is_numeric_dtype(personal_df[c])
    ]
    print(f"\n  Personal attribute columns ({len(attr_cols)}): {attr_cols}")

    influence_frame = pd.read_csv(filemgmt.most_recent_file(omnibus_results_dir, ".csv", ["Influence Analysis Combined"]))
    coefficient_frame = pd.read_csv(filemgmt.most_recent_file(omnibus_results_dir, ".csv", ["All Time Resolutions Results"]))
    contrast_frame = pd.read_csv(
        filemgmt.most_recent_file(omnibus_results_dir, ".csv", ["Subject Effect Summary Combined"])
    ).rename(columns={"Subject ID": subj_col})

    # ── Block 1 ───────────────────────────────────────────────────────────────
    responder_df = compute_responder_summary(contrast_frame, dep_vars, conditions_to_evaluate, subj_col, dep_var_col)
    print("\n  Responder Rate Summary:\n", responder_df.to_string(index=False))
    responder_df.to_csv(output_dir / filemgmt.file_title("Heterogeneity Responder Summary", ".csv"), index=False)

    # ── Block 2 ───────────────────────────────────────────────────────────────
    mi_df = compute_mi_results(
        dep_vars, influence_frame, contrast_frame, coefficient_frame, personal_df,
        attr_cols, conditions_to_evaluate, plot_mi_categories,
        alpha_omnibus=alpha_omnibus, analyse_dfbetas=analyse_mi_for_dfbetas,
        output_dir=output_dir, subj_col=subj_col, dep_var_col=dep_var_col,
    )
    mi_df.to_csv(output_dir / filemgmt.file_title("Heterogeneity MI Results Raw", ".csv"), index=False)
    print(f"\n✓ Raw MI results saved  ({len(mi_df)} rows)")

    # ── Block 3 ───────────────────────────────────────────────────────────────
    mi_summary = build_mi_summary(mi_df, dep_var_col)
    print("\n  MI Summary — Moderator Candidates:\n",
          mi_summary.loc[mi_summary["Moderator_Candidate"]].to_string(index=False))
    mi_summary.to_csv(output_dir / filemgmt.file_title("Heterogeneity MI Summary", ".csv"), index=False)

    # ── Block 4 ───────────────────────────────────────────────────────────────
    sig_pairs = coefficient_frame.loc[
        coefficient_frame[dep_var_col].isin(dep_vars)
        & (coefficient_frame["Model_Type"] == "LME")
        & (coefficient_frame["p_value_adjusted"] < alpha_omnibus),
        [dep_var_col, "Parameter"],
    ].drop_duplicates()

    combined_pivot = build_combined_pivot(
        influence_frame, contrast_frame, dep_vars, sig_pairs,
        conditions_to_evaluate, clustering_measures, subj_col, dep_var_col,
    )
    print(f"\n  [Clustering] Combined pivot: {combined_pivot.shape}")

    if combined_pivot.shape[1] < 2 or combined_pivot.shape[0] < 4:
        warnings.warn("  [Clustering] Insufficient data — skipped.")
        return

    cluster_df, _ = run_clustering(
        combined_pivot, personal_df, clustering_measures, dep_vars,
        min_cluster_size, output_dir, subj_col,
    )

    # ── Block 5 ───────────────────────────────────────────────────────────────
    plot_moderator_scatters(
        cluster_df, contrast_frame, personal_df, mi_summary,
        dep_vars, conditions_to_evaluate, top_n_moderators, output_dir,
        subj_col, dep_var_col,
    )

    print(f"\n✓ Heterogeneity modelling complete → {output_dir}")