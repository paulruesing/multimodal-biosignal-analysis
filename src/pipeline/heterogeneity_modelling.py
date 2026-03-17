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


def build_mi_summary(
    mi_df: pd.DataFrame,
    min_mi_score: float = 0.05,
) -> pd.DataFrame:
    """Pivot MI scores into a (Condition × Target) × Feature matrix.

    Filters out scores below ``min_mi_score``, takes the max across DVs per
    cell, and appends a summary column listing features that exceed the
    threshold in at least one combination.
    """
    mi_df = mi_df.copy()

    # ── Apply absolute minimum threshold ─────────────────────────────────────
    mi_df = mi_df.loc[mi_df["MI_Score"] >= min_mi_score]
    if mi_df.empty:
        warnings.warn(f"  [MI Summary] No scores >= {min_mi_score} — returning empty frame.")
        return pd.DataFrame()

    # ── Aggregate: max MI score across DVs per (Condition, Target, Feature) ──
    agg = (
        mi_df
        .groupby(["Condition", "Target", "Feature"], as_index=False)["MI_Score"]
        .max()
    )

    # ── Pivot: rows = (Condition, Target), columns = Feature ─────────────────
    pivoted = agg.pivot_table(
        index=["Condition", "Target"],
        columns="Feature",
        values="MI_Score",
        aggfunc="max",
    ).round(3)
    pivoted.columns.name = None
    pivoted = pivoted.reset_index()

    # ── Sort rows: Condition alphabetically, Target alphabetically ────────────
    pivoted = pivoted.sort_values(["Condition", "Target"]).reset_index(drop=True)

    # ── Summary column: features exceeding threshold in this row ─────────────
    feature_cols = [c for c in pivoted.columns if c not in ("Condition", "Target")]
    pivoted["Moderating_Candidates"] = pivoted[feature_cols].apply(
        lambda row: ", ".join(
            f"{feat} ({val:.2f})"
            for feat, val in row.items()
            if pd.notna(val)
        ),
        axis=1,
    )

    return pivoted


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
    """Fit Ward-linkage agglomerative clustering and save all outputs.

    Computes a Ward linkage matrix, selects the optimal number of clusters k
    via silhouette score (see :func:`select_best_k`), fits the final clustering,
    and delegates figure rendering to :func:`plot_clustering`. Three files are
    written to ``output_dir``:

    - ``Heterogeneity Subject Clusters`` (.csv) — one row per subject with
      cluster assignment and all personal attribute columns joined from
      ``personal_df``. Sorted by cluster label.
    - ``Heterogeneity Silhouette Scores`` (.csv) — silhouette score for each
      valid k that met the minimum cluster size constraint.
    - ``Heterogeneity Combined Clustering`` (.png) — Ward dendrogram +
      standardised feature heatmap (see :func:`plot_clustering`).

    Parameters
    ----------
    combined_pivot : pd.DataFrame
        Subject × feature matrix produced by :func:`build_combined_pivot`.
        Rows are subjects (indexed by ``subj_col`` values), columns are
        standardised heterogeneity features. Must have at least 4 rows and
        2 columns.
    personal_df : pd.DataFrame
        Personal attribute table with one row per subject, used to enrich the
        cluster assignment output. Must contain ``subj_col``.
    clustering_measures : list[str]
        Measure blocks present in ``combined_pivot`` (e.g.
        ``["contrast", "cooks_d"]``). Passed through to :func:`plot_clustering`
        for legend construction only — does not affect the clustering itself.
    dep_vars : list[str]
        Dependent variable names included in the analysis. Passed through to
        :func:`plot_clustering` for the figure title only.
    min_cluster_size : int
        Minimum number of subjects that every cluster must contain. k values
        that produce any smaller cluster are excluded from k selection.
    output_dir : Path
        Directory to which all three output files are written.
    subj_col : str, optional
        Subject identifier column name. Must be present as the index name of
        ``combined_pivot`` and as a column in ``personal_df``. Default
        ``"Subject_ID"``.

    Returns
    -------
    cluster_df : pd.DataFrame
        Cluster assignment table sorted by cluster label. Contains ``subj_col``,
        ``"Cluster"`` (integer label, 0-indexed), and all columns from
        ``personal_df``.
    sil_scores : dict[int, float]
        Silhouette score for each k that passed the minimum cluster size
        constraint. Empty if no valid k was found (in which case ``best_k``
        falls back to 2).
    """
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
    # Derive top moderators: mean score across all (Condition × Target) rows,
    # ignoring NaN (i.e. below-threshold) cells.
    feature_cols = [c for c in mi_summary.columns if c not in ("Condition", "Target", "Moderating_Candidates")]
    top_moderators = (
        mi_summary[feature_cols]
        .mean(skipna=True)
        .dropna()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )
    print(f"\n  [Scatter] Top moderators: {top_moderators}")

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
    analyse_mi_for_dfbetas: bool = True,
    alpha_omnibus: float = 0.05,
    subj_col: str = "Subject_ID",
    dep_var_col: str = "Dependent_Variable",
exclude_subjects: list[int] = [],
) -> None:
    """Run the full subject-heterogeneity modelling pipeline end-to-end.

    Executes five sequential analysis blocks:

        1. Responder Rate Summary — counts subjects whose normalised contrast
           exceeds zero (i.e. a positive response) per condition and DV,
           and saves a tidy summary CSV.

        2. Mutual Information (MI) Analysis — quantifies the association between
           each personal attribute (e.g. dancing habit, musical skill, age) and
           three targets: Cook's D influence, per-parameter DFBETA influence, and
           per-condition normalised contrast / responder flag. Results are saved
           as a raw long-format CSV and, optionally, as bar-plot figures.

        3. MI Summary with Tercile Ranking — aggregates raw MI scores across DVs,
           assigns within-group High / Medium / Low tercile bands, and flags
           features that reach "High" in at least one DV × condition combination
           as moderator candidates.

        4. Combined Subject Clustering — builds a subject × feature matrix from
           the selected measure blocks (contrast, DFBETA, and/or Cook's D),
           standardises each block independently, selects the optimal number of
           Ward-linkage clusters via silhouette score (subject to a minimum
           cluster size constraint), and renders a dendrogram + heatmap figure.

        5. Moderator Scatter Plots — for the top-N MI-ranked moderator candidates,
           plots each moderator against each subject's mean normalised contrast
           across all level-1 conditions, with points coloured by cluster
           assignment.

    All intermediate and final outputs are written to `output_dir` with
    timestamped filenames via `filemgmt.file_title`.

    Parameters
    ----------
    dep_vars : list[str]
        Dependent variable column names to analyse
        (e.g. ``["CMC_Flexor_mean_beta", "CMC_Extensor_mean_gamma"]``).
    conditions_to_evaluate : dict[str, tuple[str, list[str]]]
        Mapping of level key → (condition variable name, list of non-reference
        condition labels). Example::

            {
                "lvl_0": ("Music Listening", ["True"]),
                "lvl_1": ("Category or Silence", ["Happy", "Groovy", "Sad", "Classic"]),
            }

    clustering_measures : list[str]
        Subset of ``["contrast", "dfbeta", "cooks_d"]`` — determines which
        feature blocks are included in the clustering matrix.
    plot_mi_categories : list[PlotKey]
        Which MI targets to render as bar plots. Any subset of
        ``["contrast", "dfbeta", "cooks_d"]``. Pass an empty list to suppress
        all MI plots.
    top_n_moderators : int
        Number of highest-MI-ranked moderator candidates to carry into the
        cluster → scatter plots (Block 5).
    min_cluster_size : int
        Minimum number of subjects required in every cluster. Solutions with
        any cluster smaller than this are skipped during k selection.
    output_dir : Path
        Directory for all output files (figures and CSVs).
    omnibus_results_dir : Path
        Directory containing the pre-computed omnibus testing CSVs:
        ``"Influence Analysis Combined"``, ``"All Time Resolutions Results"``,
        and ``"Subject Effect Summary Combined"``.
    experiment_results_dir : Path
        Root directory of per-subject experiment folders, used to load personal
        attribute data via ``data_integration.fetch_personal_data``.
    n_subjects : int
        Total number of subjects; controls the range of subject folders loaded.
    analyse_mi_for_dfbetas : bool, optional
        If ``True`` (default), runs MI analysis for each significant LME
        parameter's DFBETA values in addition to Cook's D.
    alpha_omnibus : float, optional
        Significance threshold applied when selecting parameters for DFBETA MI
        analysis and for building the clustering feature matrix. Default 0.05.
    subj_col : str, optional
        Subject identifier column name. Default ``"Subject_ID"``.
    dep_var_col : str, optional
        Dependent variable column name. Default ``"Dependent_Variable"``.

    Returns
    -------
    None
        All outputs are written to disk; nothing is returned.

    Raises
    ------
    FileNotFoundError
        If any of the required CSVs are absent from ``omnibus_results_dir``.
    warnings.warn
        Emitted (rather than raised) when clustering is skipped due to
        insufficient data, or when a scatter moderator has too few valid rows.
    """
    import src.pipeline.data_integration as data_integration

    # ── Load data ─────────────────────────────────────────────────────────────
    subject_dirs = sorted(experiment_results_dir.glob("subject_*"))
    subject_ids = [int(d.name.split("_")[1]) for d in subject_dirs]

    # Apply exclusions
    subject_dirs = [d for d, i in zip(subject_dirs, subject_ids) if i not in exclude_subjects]
    subject_ids = [i for i in subject_ids if i not in exclude_subjects]

    if exclude_subjects:
        print(f"\n  [Exclusions] Skipping subjects: {exclude_subjects}")

    personal_df = pd.DataFrame([
        data_integration.fetch_personal_data(d) for d in subject_dirs
    ])
    personal_df.insert(0, subj_col, subject_ids)

    # binary flags:
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
    mi_summary = build_mi_summary(mi_df)
    print("\n  MI Summary:\n", mi_summary.to_string(index=False))
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