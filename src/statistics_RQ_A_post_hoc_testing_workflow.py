"""
cbpa_pipeline.py
────────────────────────────────────────────────────────────────────────────────
Cluster-Based Permutation Analysis (CBPA) — within-subject, multi-channel.

Compares 2 conditions, if categorical selects one as reference and one as treatment.
Averages trials for each condition per subject.

STATISTICAL RATIONALE
─────────────────────
LME (run on channel-ROI-averaged scalars) is the sole confirmatory test.
CBPA here is *exploratory*: it spatially/temporally decomposes effects that
LME already established. CBPA cluster p-values are used as cluster-selection
criteria for visualisation, NOT as primary inferential claims.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.pipeline.visualizations import EEG_CHANNEL_IND_DICT, EEG_CHANNELS_BY_AREA, EEG_CHANNELS
from src.pipeline.cbpa import CBPAConfig, run_batch
import src.pipeline.data_integration as data_integration
import src.utils.file_management as filemgmt



if __name__ == "__main__":

    ROOT = Path().resolve().parent
    OUTPUT = ROOT / "output" / "statistics_RQ_A" / "post_hoc_testing"
    EXPERIMENT_RESULTS = ROOT / "data" / "experiment_results"
    OMNIBUS_TESTING_RESULTS = ROOT / "output" / "statistics_RQ_A" / "omnibus_testing"
    N_SUBJECTS = 12  # needs to match statistical summary frame in pre-computed features.

    # workflow control:
    run_cbpa: bool = False

    run_heterogeneity_modelling: bool = True
    dep_vars_to_analyse: list[str] = [
        "CMC_Flexor_mean_beta", "CMC_Extensor_mean_beta",
        "CMC_Flexor_mean_gamma", "CMC_Extensor_mean_gamma",
    ]

    # Condition levels to scrutinise — level_key → (Condition_Variable, [non-reference conditions])
    conditions_to_evaluate: dict[str, tuple[str, list[str]]] = {
        "lvl_0": ("Music Listening", ["True"]),
        "lvl_1": ("Category or Silence", ["Happy", "Groovy", "Sad", "Classic"]),
    }

    analyse_dfbetas: bool = False
    # Which MI barplots to render — any subset of: "cooks_d", "contrast" + "dfbeta" (if analyse_dfbetas)
    # Empty list = no barplots (results still saved to CSV)
    mi_barplots: list[str] = ["cooks_d"]  # default: only per-condition contrast plots

    # Top-N MI-ranked moderators to carry into cluster→attribute scatter plots
    top_n_moderators: int = 3

    # Feature blocks to include in clustering (any subset of these three)
    clustering_measures: list[str] = ["dfbeta", "cooks_d", "contrasts"]

    # Minimum number of subjects per cluster — prevents trivial 1-subject clusters
    min_cluster_size: int = 2

    # ══════════════════════════════════════════════════════════════════════════════
    #  Cluster-based Permutation Analysis (CBPA)
    # ══════════════════════════════════════════════════════════════════════════════
    if run_cbpa:

        ### DEFINE CONTRASTS TO SCRUTINISE
        # Only run CBPA for effects already established by LME.
        # Add / remove / comment out entries here.

        CONTRASTS: list[CBPAConfig] = [



            # ── H1: CMC Extensor beta, Classic vs Silence ─────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Extensor",
                freq_band="beta",
                channels=None,
                condition_column="Category or Silence",
                condition_A="Classic",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                n_subjects=N_SUBJECTS,
                output_dir=OUTPUT,
                hypothesis_label="H1_CMC_Extensor_beta_Classic_vs_Silence",
            ),

            # ── H1: CMC Extensor beta, Happy vs Classic ───────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Extensor",
                freq_band="beta",
                channels=None,
                condition_column="Perceived Category",
                condition_A="Happy",
                condition_B="Classic",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                n_subjects=N_SUBJECTS,
                output_dir=OUTPUT,
                hypothesis_label="H1_CMC_Extensor_beta_Happy_vs_Classic",
            ),

            # ── H1: CMC Extensor beta, Groovy vs Classic ──────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Extensor",
                freq_band="beta",
                channels=None,
                condition_column="Perceived Category",
                condition_A="Groovy",
                condition_B="Classic",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                n_subjects=N_SUBJECTS,
                output_dir=OUTPUT,
                hypothesis_label="H1_CMC_Extensor_beta_Groovy_vs_Classic",
            ),

            #### HAPPY VS SILENCE
            # ── H1: CMC Flexor beta, Happy vs Silence ────────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Flexor",
                freq_band="beta",
                channels=None,
                condition_column="Category or Silence",
                condition_A="Happy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                n_subjects=N_SUBJECTS,
                output_dir=OUTPUT,
                hypothesis_label="H1_CMC_Flexor_beta_Happy_vs_Silence",
            ),

            # ── H1: CMC Extensor beta, Happy vs Silence ────────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Extensor",
                freq_band="beta",
                channels=None,
                condition_column="Category or Silence",
                condition_A="Happy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                n_subjects=N_SUBJECTS,
                output_dir=OUTPUT,
                hypothesis_label="H1_CMC_Extensor_beta_Happy_vs_Silence",
            ),


            # ── H1: CMC Extensor gamma, Happy vs Silence ──────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Extensor",
                freq_band="gamma",
                channels=None,
                condition_column="Category or Silence",
                condition_A="Happy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                n_subjects=N_SUBJECTS,
                output_dir=OUTPUT,
                hypothesis_label="H1_CMC_Extensor_gamma_Happy_vs_Silence",
            ),

            # ── H1: CMC Flexor gamma, Happy vs Silence ────────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Flexor",
                freq_band="gamma",
                channels=None,
                condition_column="Category or Silence",
                condition_A="Happy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                n_subjects=N_SUBJECTS,
                output_dir=OUTPUT,
                hypothesis_label="H1_CMC_Flexor_gamma_Happy_vs_Silence",
            ),

            #### Groovy VS SILENCE
            # ── H1: CMC Flexor beta, Groovy vs Silence ────────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Flexor",
                freq_band="beta",
                channels=None,
                condition_column="Category or Silence",
                condition_A="Groovy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                n_subjects=N_SUBJECTS,
                output_dir=OUTPUT,
                hypothesis_label="H1_CMC_Flexor_beta_Groovy_vs_Silence",
            ),

            # ── H1: CMC Extensor beta, Groovy vs Silence ────────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Extensor",
                freq_band="beta",
                channels=None,
                condition_column="Category or Silence",
                condition_A="Groovy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                n_subjects=N_SUBJECTS,
                output_dir=OUTPUT,
                hypothesis_label="H1_CMC_Extensor_beta_Groovy_vs_Silence",
            ),

            # ── H1: CMC Extensor gamma, Groovy vs Silence ──────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Extensor",
                freq_band="gamma",
                channels=None,
                condition_column="Category or Silence",
                condition_A="Groovy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                n_subjects=N_SUBJECTS,
                output_dir=OUTPUT,
                hypothesis_label="H1_CMC_Extensor_gamma_Groovy_vs_Silence",
            ),

            # ── H1: CMC Flexor gamma, Groovy vs Silence ────────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Flexor",
                freq_band="gamma",
                channels=None,
                condition_column="Category or Silence",
                condition_A="Groovy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                n_subjects=N_SUBJECTS,
                output_dir=OUTPUT,
                hypothesis_label="H1_CMC_Flexor_gamma_Groovy_vs_Silence",
            ),

            # ── H2: Fronto-central–centro-parietal–temporal theta, Happy vs Silence
            # Covers PSD_eeg_FC_CP_T_theta (Level 1 Happy effect)
            CBPAConfig(
                modality="PSD",
                modality_file_id="eeg",
                freq_band="theta",
                channels=(
                        EEG_CHANNELS_BY_AREA["Frontal"]
                        + EEG_CHANNELS_BY_AREA["Central"]
                        + EEG_CHANNELS_BY_AREA["Centro-Parietal"]
                        + EEG_CHANNELS_BY_AREA["Temporal"]
                ),
                condition_column="Category or Silence",
                condition_A="Happy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=0,
                n_permutations=1000,
                use_spatio_temporal=True,
                data_root=ROOT,
                n_subjects=N_SUBJECTS,
                output_dir=OUTPUT,
                hypothesis_label="H2_theta_FC_CP_T_Happy_vs_Silence",
            ),

            # ── H5: Global gamma, Happy vs Silence ────────────────────────────────
            # Covers PSD_eeg_Global_gamma (Level 1 Happy effect)
            CBPAConfig(
                modality="PSD",
                modality_file_id="eeg",
                freq_band="gamma",
                channels=EEG_CHANNELS,
                condition_column="Category or Silence",
                condition_A="Happy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=0,
                n_permutations=1000,
                use_spatio_temporal=True,
                data_root=ROOT,
                n_subjects=N_SUBJECTS,
                output_dir=OUTPUT,
                hypothesis_label="H5_gamma_Global_Happy_vs_Silence",
            ),
        ]



        ### RUN
        results = run_batch(CONTRASTS)

    # ══════════════════════════════════════════════════════════════════════════════
    #  Subject-Heterogeneity Modelling
    # ══════════════════════════════════════════════════════════════════════════════
    if run_heterogeneity_modelling:
        import warnings
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
        from matplotlib.patches import Patch

        from src.pipeline.signal_features import compute_feature_mi_importance
        from src.pipeline.visualizations import plot_scatter

        ALPHA_OMNIBUS = 0.05
        DEP_VAR_COL = "Dependent_Variable"
        SUBJ_COL = "Subject_ID"
        SUBJ_COL_CONTRAST = "Subject ID"

        # ── Personal attributes ───────────────────────────────────────────────────
        personal_data_dicts: list[dict] = [
            data_integration.fetch_personal_data(EXPERIMENT_RESULTS / f"subject_{s_ind:02d}")
            for s_ind in range(N_SUBJECTS)
        ]
        personal_df = pd.DataFrame(personal_data_dicts)
        personal_df.insert(0, SUBJ_COL, list(range(N_SUBJECTS)))

        attr_cols = [
            c for c in personal_df.columns
            if c != SUBJ_COL
               and personal_df[c].nunique(dropna=True) > 1
               and pd.api.types.is_numeric_dtype(personal_df[c])  # exclude string/categorical
        ]
        print(f"\n  Personal attribute columns ({len(attr_cols)}): {attr_cols}")

        # ── Load frames ───────────────────────────────────────────────────────────
        influence_measure_frame = pd.read_csv(
            filemgmt.most_recent_file(OMNIBUS_TESTING_RESULTS, ".csv", ["Influence Analysis Combined"])
        )
        coefficient_frame = pd.read_csv(
            filemgmt.most_recent_file(OMNIBUS_TESTING_RESULTS, ".csv", ["All Time Resolutions Results"])
        )
        subject_contrast_frame = pd.read_csv(
            filemgmt.most_recent_file(OMNIBUS_TESTING_RESULTS, ".csv", ["Subject Effect Summary Combined"])
        ).rename(columns={SUBJ_COL_CONTRAST: SUBJ_COL})

        # ══════════════════════════════════════════════════════════════════════════
        #  Block 1 — Responder Rate Summary
        # ══════════════════════════════════════════════════════════════════════════
        responder_rows: list[dict] = []

        for dep_var in dep_vars_to_analyse:
            contrast_subset = subject_contrast_frame.loc[
                subject_contrast_frame[DEP_VAR_COL] == dep_var
                ]
            for level_key, (cond_var, conditions) in conditions_to_evaluate.items():
                level_subset = contrast_subset.loc[
                    contrast_subset["Condition_Variable"] == cond_var
                    ]
                for condition in conditions:
                    cond_rows = level_subset.loc[level_subset["Condition"] == condition]
                    n_subj = cond_rows[SUBJ_COL].nunique()
                    n_resp = cond_rows.loc[cond_rows["Responder_Flag"], SUBJ_COL].nunique()
                    responder_rows.append({
                        DEP_VAR_COL: dep_var,
                        "Level": level_key,
                        "Condition_Variable": cond_var,
                        "Condition": condition,
                        "N_Subjects": n_subj,
                        "N_Responders": n_resp,
                        "Responder_Rate": round(n_resp / n_subj, 3) if n_subj > 0 else np.nan,
                    })

        responder_df = pd.DataFrame(responder_rows)
        print("\n  Responder Rate Summary:")
        print(responder_df.to_string(index=False))
        responder_df.to_csv(
            OUTPUT / filemgmt.file_title("Heterogeneity Responder Summary", ".csv"), index=False
        )

        # ══════════════════════════════════════════════════════════════════════════
        #  Block 2 — MI Analysis
        # ══════════════════════════════════════════════════════════════════════════
        mi_results: list[dict] = []


        def _run_mi(
                feature_df: pd.DataFrame,
                target_col: str,
                target_type: str,
                dep_var: str,
                level: str,
                cond_var: str,
                condition: str | None = None,
                plot_key: str | None = None,  # one of "cooks_d" | "dfbeta" | "contrast"
        ) -> None:
            """Run MI and append rows to mi_results.

            Parameters
            ----------
            plot_key : str or None
                Key checked against the outer-scope mi_barplots list to decide whether
                to render a barplot. None = never render.
            """
            valid = feature_df.dropna(subset=[target_col])
            if len(valid) < 4 or valid[target_col].nunique() < 2:
                return
            target_arr = (
                valid[target_col].astype(int).values
                if target_type == "discrete"
                else valid[target_col].astype(float).values
            )
            include_barplot = plot_key is not None and plot_key in mi_barplots
            mi_out = compute_feature_mi_importance(
                feature_array=valid[attr_cols].values,
                target_array=target_arr,
                feature_labels=attr_cols,
                target_label=f"{target_col}{f'[{condition}]' if condition is not None else ''} │ {dep_var}",
                target_type=target_type,
                include_barplot=include_barplot,
                plot_save_dir=OUTPUT if include_barplot else None,
            )
            scores: dict = mi_out[2] if isinstance(mi_out, tuple) else mi_out
            for feat, score in scores.items():
                mi_results.append({
                    DEP_VAR_COL: dep_var, "Level": level, "Condition_Variable": cond_var,
                    "Condition": condition, "Target": target_col,
                    "Feature": feat, "MI_Score": score,
                })


        for dep_var in dep_vars_to_analyse:
            print(f"\n{'=' * 72}")
            print(f"  MI Analysis │ {dep_var}")
            print(f"{'=' * 72}")

            influence_subset = influence_measure_frame.loc[
                influence_measure_frame[DEP_VAR_COL] == dep_var
                ].copy()
            contrast_subset = subject_contrast_frame.loc[
                subject_contrast_frame[DEP_VAR_COL] == dep_var
                ].copy()

            if influence_subset.empty:
                warnings.warn(f"  [skip] No influence data for '{dep_var}'.")
                continue

            # Cook's D MI — subject-level aggregate influence vs. personal attributes
            cooks_per_subj = (
                influence_subset
                .groupby(SUBJ_COL, as_index=False)["CooksD"].mean()
                .merge(personal_df, on=SUBJ_COL, how="left")
                .dropna(subset=attr_cols + ["CooksD"])
            )
            if len(cooks_per_subj) >= 4:
                print(f"\n  [Cook's D] n={len(cooks_per_subj)}")
                _run_mi(cooks_per_subj, "CooksD", "continuous",
                        dep_var, "influence", "—")

            # DFBeta MI per significant parameter
            if analyse_dfbetas:
                sig_params = coefficient_frame.loc[
                    (coefficient_frame[DEP_VAR_COL] == dep_var)
                    & (coefficient_frame["Model_Type"] == "LME")
                    & (coefficient_frame["p_value_adjusted"] < ALPHA_OMNIBUS),
                    "Parameter",
                ].unique()
                if len(sig_params) == 0:
                    print(f"  [DFBeta] No significant parameters for '{dep_var}'.")
                for param in sig_params:
                    param_rows = (
                        influence_subset.loc[influence_subset["Parameter"] == param]
                        .merge(personal_df, on=SUBJ_COL, how="left")
                        .dropna(subset=attr_cols + ["DFBETA"])
                    )
                    if len(param_rows) < 4:
                        continue
                    print(f"\n  [DFBeta] '{param}'  n={len(param_rows)}")
                    _run_mi(param_rows, "DFBETA", "continuous",
                            dep_var, "influence", "—", param)

            # Per-condition MI: Responder_Flag (discrete) + Normalised_Contrast (continuous)
            for level_key, (cond_var, conditions) in conditions_to_evaluate.items():
                level_subset = contrast_subset.loc[
                    contrast_subset["Condition_Variable"] == cond_var
                    ]
                for condition in conditions:
                    cond_rows = (
                        level_subset.loc[level_subset["Condition"] == condition]
                        .merge(personal_df, on=SUBJ_COL, how="left")
                        .dropna(subset=attr_cols)
                    )
                    if len(cond_rows) < 4:
                        warnings.warn(f"  [{level_key}|{condition}] Too few subjects.")
                        continue
                    print(f"\n  [{level_key}] '{condition}'  n={len(cond_rows)}")
                    # Responder_Flag: no barplot — summary table captures signal
                    _run_mi(cond_rows, "Responder_Flag", "discrete",
                            dep_var, level_key, cond_var, condition)
                    # Normalised_Contrast: barplot for visual inspection
                    _run_mi(cond_rows, "Normalised_Contrast", "continuous",
                            dep_var, level_key, cond_var, condition)

        mi_df = pd.DataFrame(mi_results)
        mi_df.to_csv(OUTPUT / filemgmt.file_title("Heterogeneity MI Results Raw", ".csv"), index=False)
        print(f"\n✓ Raw MI results saved  ({len(mi_df)} rows)")


        # ══════════════════════════════════════════════════════════════════════════
        #  Block 3 — MI Summary with Relative Tercile Ranking
        # ══════════════════════════════════════════════════════════════════════════
        # Within each (DV × Level × Condition × Target) group rank features by MI
        # score into tercile bands. Ties at quantile boundaries get the upper band.

        def _assign_tercile_band(grp: pd.DataFrame) -> pd.Series:
            """Assign High/Medium/Low by within-group MI_Score tercile."""
            scores = grp["MI_Score"]
            t33, t67 = scores.quantile([1 / 3, 2 / 3])
            # When all scores are equal assign "Medium" to avoid misleading extremes
            if t33 == t67:
                return pd.Series(["Medium"] * len(scores), index=scores.index)
            return scores.apply(
                lambda s: "High" if s >= t67 else ("Medium" if s >= t33 else "Low")
            )


        mi_df["MI_Band"] = (
            mi_df
            .groupby([DEP_VAR_COL, "Level", "Condition", "Target"], group_keys=False)
            .apply(_assign_tercile_band)
        )

        # Moderator candidate: feature reaches "High" in ≥1 combination across ANY DV
        high_features = set(mi_df.loc[mi_df["MI_Band"] == "High", "Feature"].unique())
        mi_df["Moderator_Candidate"] = mi_df["Feature"].isin(high_features)

        # Aggregate across DVs: max MI, band at max, count of DV×condition combos rated High
        mi_summary = (
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

        print("\n  MI Summary — Moderator Candidates:")
        candidates = mi_summary.loc[mi_summary["Moderator_Candidate"]]
        print(candidates.to_string(index=False))
        mi_summary.to_csv(
            OUTPUT / filemgmt.file_title("Heterogeneity MI Summary", ".csv"), index=False
        )

        # ══════════════════════════════════════════════════════════════════════════
        #  Block 4 — Combined Subject Clustering
        # ══════════════════════════════════════════════════════════════════════════
        sig_pairs = coefficient_frame.loc[
            coefficient_frame[DEP_VAR_COL].isin(dep_vars_to_analyse)
            & (coefficient_frame["Model_Type"] == "LME")
            & (coefficient_frame["p_value_adjusted"] < ALPHA_OMNIBUS),
            [DEP_VAR_COL, "Parameter"],
        ].drop_duplicates()

        # Each block is standardised independently before joining — prevents blocks
        # with larger absolute values (contrasts) from dominating the distance metric.
        pivot_blocks: list[pd.DataFrame] = []
        block_col_groups: dict[str, list[str]] = {}


        def _scaled_pivot(long_df, index_col, col_col, val_col, prefix) -> pd.DataFrame:
            """Pivot → drop incomplete columns → standardise."""
            piv = long_df.pivot_table(
                index=index_col, columns=col_col, values=val_col, aggfunc="mean"
            ).dropna(axis=1, how="any")
            block_col_groups[prefix] = piv.columns.tolist()
            return pd.DataFrame(
                StandardScaler().fit_transform(piv.values),
                index=piv.index, columns=piv.columns,
            )


        if "dfbeta" in clustering_measures:
            df_long = influence_measure_frame.merge(sig_pairs, on=[DEP_VAR_COL, "Parameter"], how="inner").copy()
            df_long["col_key"] = (
                    "DFBETA│" + df_long[DEP_VAR_COL].str.replace("CMC_", "", regex=False)
                    + "│" + df_long["Parameter"]
            )
            pivot_blocks.append(_scaled_pivot(df_long, SUBJ_COL, "col_key", "DFBETA", "DFBETA"))

        if "cooks_d" in clustering_measures:
            ck_long = influence_measure_frame.loc[
                influence_measure_frame[DEP_VAR_COL].isin(dep_vars_to_analyse)
            ].copy()
            ck_long["col_key"] = "CooksD│" + ck_long[DEP_VAR_COL].str.replace("CMC_", "", regex=False)
            pivot_blocks.append(_scaled_pivot(ck_long, SUBJ_COL, "col_key", "CooksD", "CooksD"))

        if "contrasts" in clustering_measures:
            ct_rows = pd.concat([
                subject_contrast_frame.loc[
                    subject_contrast_frame[DEP_VAR_COL].isin(dep_vars_to_analyse)
                    & (subject_contrast_frame["Condition_Variable"] == cond_var)
                    & (subject_contrast_frame["Condition"].isin(conditions))
                    ]
                for _, (cond_var, conditions) in conditions_to_evaluate.items()
            ], ignore_index=True)
            ct_rows["col_key"] = (
                    "Contrast│" + ct_rows[DEP_VAR_COL].str.replace("CMC_", "", regex=False)
                    + "│" + ct_rows["Condition"].astype(str)
            )
            pivot_blocks.append(_scaled_pivot(ct_rows, SUBJ_COL, "col_key", "Normalised_Contrast", "Contrast"))

        combined_pivot = pivot_blocks[0].copy()
        for blk in pivot_blocks[1:]:
            combined_pivot = combined_pivot.join(blk, how="inner")

        print(f"\n  [Clustering] Combined pivot: {combined_pivot.shape}")
        for grp, cols in block_col_groups.items():
            n_kept = sum(1 for c in cols if c in combined_pivot.columns)
            print(f"    {grp}: {n_kept} features")

        if combined_pivot.shape[1] < 2 or combined_pivot.shape[0] < 4:
            warnings.warn("  [Clustering] Insufficient data — skipped.")
        else:
            X = combined_pivot.values
            Z = linkage(X, method="ward", metric="euclidean")

            k_range = range(2, min(6, combined_pivot.shape[0]))


            # Select best k: silhouette score, restricted to solutions where every
            # cluster has at least min_cluster_size subjects.
            def _all_clusters_meet_min_size(labels: np.ndarray, min_size: int) -> bool:
                """Return True iff every cluster label has at least min_size members."""
                counts = np.bincount(labels)
                return bool((counts >= min_size).all())


            valid_sil_scores: dict[int, float] = {}
            for k in k_range:
                labels_k = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
                if not _all_clusters_meet_min_size(labels_k, min_cluster_size):
                    print(f"  [Clustering] k={k} skipped — has cluster < {min_cluster_size} subjects")
                    continue
                valid_sil_scores[k] = silhouette_score(X, labels_k)

            if not valid_sil_scores:
                warnings.warn(
                    f"  [Clustering] No valid k found with min_cluster_size={min_cluster_size} "
                    f"in range {list(k_range)}. Falling back to k=2 without size constraint."
                )
                best_k = 2
            else:
                best_k = max(valid_sil_scores, key=valid_sil_scores.get)
                print(f"  [Clustering] Valid silhouette scores: "
                      f"{ {k: f'{v:.3f}' for k, v in valid_sil_scores.items()} }")

            print(f"  [Clustering] Selected k = {best_k}")
            cluster_labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)

            row_order = leaves_list(Z)
            ordered_data = combined_pivot.iloc[row_order]

            col_colors = [
                "#d62728" if c.startswith("DFBETA") else
                "#ff7f0e" if c.startswith("CooksD") else
                "#1f77b4"
                for c in combined_pivot.columns
            ]

            fig, axes = plt.subplots(
                1, 2, figsize=(max(8, combined_pivot.shape[1] * 0.4), 7),
                gridspec_kw={"width_ratios": [1, 4]},
            )
            dendrogram(Z, labels=combined_pivot.index.tolist(), orientation="left",
                       ax=axes[0], color_threshold=Z[-(best_k - 1), 2])
            axes[0].set_title("Ward Dendrogram")
            axes[0].set_xlabel("Distance")

            vlim = np.nanpercentile(np.abs(ordered_data.values), 97)
            im = axes[1].imshow(ordered_data.values, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
            axes[1].set_xticks(range(len(combined_pivot.columns)))
            axes[1].set_xticklabels(combined_pivot.columns.tolist(), rotation=45, ha="right", fontsize=6)
            for tick, col in zip(axes[1].get_xticklabels(), col_colors):
                tick.set_color(col)
            axes[1].set_yticks(range(len(ordered_data)))
            axes[1].set_yticklabels(ordered_data.index.tolist(), fontsize=8)
            axes[1].set_title("Combined heterogeneity features (ordered by cluster)")
            plt.colorbar(im, ax=axes[1], label="Standardised value", shrink=0.7)

            legend_handles = (
                    ([Patch(color="#d62728", label="DFBeta")] if "dfbeta" in clustering_measures else [])
                    + ([Patch(color="#ff7f0e", label="Cook's D")] if "cooks_d" in clustering_measures else [])
                    + ([Patch(color="#1f77b4", label="Contrast")] if "contrasts" in clustering_measures else [])
            )
            axes[1].legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.7)
            fig.suptitle(
                f"Subject Heterogeneity Clustering │ {' + '.join(clustering_measures)}\n"
                f"DVs: {', '.join(dep_vars_to_analyse)}  │  k={best_k} clusters  │  Ward linkage",
                fontsize=10,
            )
            plt.tight_layout()
            fig.savefig(OUTPUT / filemgmt.file_title("Heterogeneity Combined Clustering", ".png"),
                        dpi=150, bbox_inches="tight")
            plt.show()
            plt.close(fig)

            cluster_assign_df = (
                pd.DataFrame({SUBJ_COL: combined_pivot.index, "Cluster": cluster_labels})
                .sort_values("Cluster")
                .merge(personal_df, on=SUBJ_COL, how="left")
            )
            cluster_assign_df.to_csv(
                OUTPUT / filemgmt.file_title("Heterogeneity Subject Clusters", ".csv"), index=False
            )
            pd.DataFrame([{"k": k, "Silhouette": v} for k, v in valid_sil_scores.items()]).to_csv(
                OUTPUT / filemgmt.file_title("Heterogeneity Silhouette Scores", ".csv"), index=False
            )

            # ══════════════════════════════════════════════════════════════════════
            #  Block 5 — Cluster → Moderator Scatter Plots
            # ══════════════════════════════════════════════════════════════════════
            # Top moderators ranked by Max_MI_Score across all conditions and DVs.
            top_moderators = (
                mi_summary.loc[mi_summary["Moderator_Candidate"]]
                .sort_values("Max_MI_Score", ascending=False)
                ["Feature"].unique()[:top_n_moderators]
            )
            print(f"\n  [Scatter] Top moderators: {list(top_moderators)}")

            # y-axis: mean Normalised_Contrast per subject across all lvl_1 conditions and DVs
            lvl1_cond_var, lvl1_conditions = conditions_to_evaluate["lvl_1"]
            mean_contrast_per_subj = (
                subject_contrast_frame.loc[
                    subject_contrast_frame[DEP_VAR_COL].isin(dep_vars_to_analyse)
                    & (subject_contrast_frame["Condition_Variable"] == lvl1_cond_var)
                    & (subject_contrast_frame["Condition"].isin(lvl1_conditions))
                    ]
                .groupby(SUBJ_COL, as_index=False)["Normalised_Contrast"].mean()
            )

            scatter_df = (
                cluster_assign_df[[SUBJ_COL, "Cluster"]]
                .merge(mean_contrast_per_subj, on=SUBJ_COL, how="left")
                .merge(personal_df[[SUBJ_COL] + list(top_moderators)], on=SUBJ_COL, how="left")
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
                    save_dir=OUTPUT,
                )

            print(f"\n✓ Heterogeneity modelling complete → {OUTPUT}")
