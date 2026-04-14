from collections.abc import Callable
from pathlib import Path
import functools
import pandas as pd
import numpy as np

import src.pipeline.statistical_modelling as statistics
import src.pipeline.visualizations as visualizations
import src.utils.file_management as filemgmt


# ============================================================================
# FEATURE COLUMN LISTS
# Mirror columns produced by statistics_data_preparation_workflow.py
# ============================================================================

_CMC_COLS: list[str] = [
    'CMC_Flexor_max_beta',    'CMC_Flexor_max_gamma',
    'CMC_Flexor_mean_beta',   'CMC_Flexor_mean_gamma',
    'CMC_Extensor_max_beta',  'CMC_Extensor_max_gamma',
    'CMC_Extensor_mean_beta', 'CMC_Extensor_mean_gamma',
]

_PSD_COLS: list[str] = [
    'PSD_eeg_FC_CP_T_theta',
    'PSD_eeg_F_C_beta',
    'PSD_eeg_P_PO_alpha',
    'PSD_eeg_Global_gamma',
    'PSD_emg_1_flexor_Global_all',   # excluded when include_emg_psd=False
    'PSD_emg_2_extensor_Global_all', # excluded when include_emg_psd=False
]

_BASE_COVARIATES: list[str] = [
    'Task Frequency',
]


def fetch_accuracy_level_definitions(
    multi_segments_per_trial: bool,
        include_emg_psd: bool = True,
        include_max_cmc: bool = True,
) -> list[dict]:
    """
    Build comparison-level definitions for RQ2: neural/motor features predicting accuracy.

    Parameters
    ----------
    multi_segments_per_trial : bool
        If True, 'Segment ID' is appended to temporal controls.
    include_emg_psd : bool, default True
        Whether to include EMG PSD columns at Level 1.
        Set False when EMG PSD is suspected to be collinear with Force Level.

    Returns
    -------
    list[dict]
        Level dicts compatible with statistics.run_model_levels.

        Level 0 — CMC features only (cortico-muscular coherence as sole neural block).
        Level 1 — CMC + EEG PSD (+ optionally EMG PSD); full neural predictor set.
    """
    # Segment ID added only when trial is split into multiple windows
    temporal_controls = (
        (['Trial ID', 'Segment ID'] if multi_segments_per_trial else ['Trial ID']) +
        (['Median Scaled Force [0-1]', 'Median Unscaled Force [% MVC]',] if multi_segments_per_trial else [
            'Median Unscaled Force [% MVC]'])
    )
    base = _BASE_COVARIATES + temporal_controls

    # Conditionally remove EMG PSD columns to avoid collinearity with Force
    psd_cols = _PSD_COLS if include_emg_psd else [
        c for c in _PSD_COLS if 'emg' not in c
    ]

    # Conditionally remove CMC Max cols to avoid collinearity with Mean
    cmc_cols = _CMC_COLS if include_max_cmc else [
        c for c in _CMC_COLS if 'max' not in c
    ]

    return [
        # Level 0 — CMC only
        {
            'df_filter': None,
            'condition_vars': {},
            'reference_categories': {},
            'explanatory_vars': base + cmc_cols,
            'moderation_pairs': [],
        },
        # Level 1 — CMC + PSD (EMG inclusion controlled by flag)
        {
            'df_filter': None,
            'condition_vars': {},
            'reference_categories': {},
            'explanatory_vars': base + cmc_cols + psd_cols,
            'moderation_pairs': [],
        },
    ]


if __name__ == '__main__':

    ######## PREPARATION #########
    ROOT = Path().resolve().parent
    DATA = ROOT / "data"
    OUTPUT = ROOT / 'output'
    # input:
    EXPERIMENT_DATA = DATA / "experiment_results"
    FEATURE_OUTPUT_DATA = DATA / "precomputed_features"
    # output:
    STATISTICS_OUTPUT_DATA = OUTPUT / 'statistics_RQ_B' / 'omnibus_testing'
    filemgmt.assert_dir(STATISTICS_OUTPUT_DATA)

    ##### PARAMETERS
    n_within_trial_segments_list: list[int] = [1, 2, 5, 10]

    # operate on subset of subjects?
    exclude_subjects: list[int] = []

    # Accuracy column name as written by statistics_data_preparation_workflow.py
    ACCURACY_COL_RAW: str = 'RMS_Accuracy'
    log_transform_accuracy: bool = True  # log-transform RMSE to reduce outlier influence

    # Neural predictor configuration
    include_emg_psd: bool = False    # toggle False to exclude EMG PSD from Level 1
    include_max_cmc: bool = False  # toggle False to exclude max CMC predictors in all levels

    ACCURACY_COL: str = f"log_{ACCURACY_COL_RAW}" if log_transform_accuracy else ACCURACY_COL_RAW

    # Single RQ; extend list here if further accuracy-related dependent vars are added
    statistical_hypotheses_var_tuples: list[tuple[str, str]] = [
        ('Task RMSE', ACCURACY_COL),
    ]

    # Levels to model (0 = CMC only, 1 = CMC + PSD)
    lvl_inds_to_include: list[int] = [0]
    lvls_to_include: list[str] = [f"lvl_{i}" for i in lvl_inds_to_include]

    ## Analysis flags
    conduct_analysis: bool = True

    save_single_time_res_summaries: bool = False

    render_ols_effect_plots: bool = False
    show_ols_effect_plots: bool = False

    render_lme_effect_plots: bool = True
    show_lme_effect_plots: bool = True

    parameter_rename_dict: dict[str, str] = {
        'CMC_Extensor_mean_beta': 'CMC extensor mean (beta)',
        'CMC_Extensor_mean_gamma': 'CMC extensor mean (gamma)',
        'CMC_Flexor_mean_beta': 'CMC flexor mean (beta)',
        'CMC_Flexor_mean_gamma': 'CMC flexor mean (gamma)',
        'Median Scaled Force [0-1]': 'Task-wise scaled force [0-1]',
        'Median Unscaled Force [% MVC]': 'Force [% MVC]',
        'PSD_eeg_FC_CP_T_theta': 'PSD FC/CP/T (theta)',
        'PSD_eeg_F_C_beta': 'PSD F/C (beta)',
        'PSD_eeg_Global_gamma': 'PSD Global (gamma)',
        'PSD_eeg_P_PO_alpha': 'PSD P/O (alpha)',
        'Task Frequency': 'Task frequency',
        'Trial ID': 'Trial number',
        'Segment ID': 'Intra-trial time'

    }

    # Across-time-resolution parameter comparisons
    plot_time_resolution_comparisons: bool = False
    parameter_comp_lvl_tuples_to_plot_across_time: list[tuple[str, int]] = [
        # ('CMC_Flexor_max_beta',    0), # insignificant at primary scale...
        ('CMC_Flexor_mean_beta',   0),
        ('CMC_Flexor_mean_gamma', 0),
        # ('CMC_Extensor_max_beta',  0),  # insignificant at primary scale...
        ('CMC_Extensor_mean_beta', 0),
        ('CMC_Extensor_mean_gamma', 0),

        ('Median Scaled Force [0-1]', 0),
        ('Median Unscaled Force [% MVC]', 0),
    ]

    # Robustness / influence checks
    conduct_robustness_checks: bool = False
    dep_var_comp_lvl_n_segments_tuples_to_robustness_check: list[tuple[str, int, int]] = [
        (ACCURACY_COL, 0, 5),  # 0 is important: PSD features are insignificant almost always
    ]

    # Power analysis — populate after inspecting initial effect sizes
    conduct_power_analysis: bool = False
    power_configs: list[statistics.PowerConfig] = [
        statistics.PowerConfig(
            dependent_var=ACCURACY_COL,
            comp_lvl=0,  # <- important, PSD features are insignificant almost always
            n_segments=5,
            target_parameters=[
                "Q('Task Frequency')",
                "CMC_Flexor_mean_beta",
                "CMC_Flexor_mean_gamma",
                "CMC_Extensor_mean_beta",
                "CMC_Extensor_mean_gamma",
                "Q('Median Scaled Force [0-1]')",
                "Q('Median Unscaled Force [0-1]')",
                "Q('Trial ID')",
            ],
        ),
    ]

    # Bind include_emg_psd so the function signature matches what statistics
    # utilities expect (they call level_def_fn(multi_segments_per_trial=...))
    _level_def_fn = functools.partial(
        fetch_accuracy_level_definitions,
        include_emg_psd=include_emg_psd,
        include_max_cmc=include_max_cmc,
    )





    #######################################################
    ################ LOOP OVER N_SEGMENTS #################
    #######################################################

    all_time_resolutions_results_list: list[pd.DataFrame] = []
    all_diagnostics_list: list[pd.DataFrame] = []

    for n_within_trial_segments in n_within_trial_segments_list:

        # Rebuild level definitions for this time resolution
        level_definitions = _level_def_fn(
            multi_segments_per_trial=n_within_trial_segments > 1,
        )


        #####################################################
        ################ LOAD SUMMARY FRAME #################
        #####################################################

        all_subject_data_frame = pd.read_csv(
            filemgmt.most_recent_file(
                FEATURE_OUTPUT_DATA, ".csv",
                [f"Combined Statistics {n_within_trial_segments}seg"],
            )
        )
        print(f"Loaded summary frame for n_segments={n_within_trial_segments} "
              f"({len(all_subject_data_frame)} rows)\n")

        # ── Apply subject exclusions ──────────────────────────────────────────────
        if exclude_subjects:
            before = all_subject_data_frame["Subject ID"].nunique()
            all_subject_data_frame = all_subject_data_frame.loc[
                ~all_subject_data_frame["Subject ID"].isin(exclude_subjects)
            ].reset_index(drop=True)
            after = all_subject_data_frame["Subject ID"].nunique()
            print(f"  [Exclusions] Removed subjects {exclude_subjects}: {before} → {after} subjects remaining.\n")


        # Skip frame silently if accuracy column absent (pre-dates RQ2 addition)
        if ACCURACY_COL_RAW not in all_subject_data_frame.columns:
            print(
                f"[WARNING] '{ACCURACY_COL_RAW}' not found — "
                f"re-run statistics_data_preparation.py to add accuracy. Skipping.\n"
            )
            continue

        # Optional log-transform: compresses right tail of RMSE, stabilises
        # variance, and makes LME coefficients interpretable as proportional
        # changes.  The raw column is preserved; a new log_ column is added.
        if log_transform_accuracy and ACCURACY_COL not in all_subject_data_frame.columns:
            raw = all_subject_data_frame[ACCURACY_COL_RAW]
            n_zero = (raw <= 0).sum()
            if n_zero:
                print(f"  [log-transform] {n_zero} rows with {ACCURACY_COL_RAW} <= 0 — "
                      f"these will become NaN after log transform.")
            all_subject_data_frame[ACCURACY_COL] = np.log(raw.where(raw > 0))
            print(f"  [log-transform] Created '{ACCURACY_COL}' from '{ACCURACY_COL_RAW}'.")


        ########################################################
        ################ STATISTICAL MODELLING #################
        ########################################################

        if conduct_analysis:
            all_model_results: list = []
            all_diagnostics: list = []

            for hypothesis, dependent_variable in statistical_hypotheses_var_tuples:

                print("\n")
                print("=" * 100)
                print(f"HYPOTHESIS:\t\t{hypothesis}")
                print(f"DEPENDENT VARIABLE:\t{dependent_variable}")
                print("=" * 100, "\n" * 3)

                # Run OLS + LME across both comparison levels
                statistics.run_model_levels(
                    base_df=all_subject_data_frame,
                    level_definitions=level_definitions,
                    levels_to_include=lvl_inds_to_include,
                    response_var=dependent_variable,
                    hypothesis_name=hypothesis,
                    n_windows_per_trial=n_within_trial_segments,
                    all_results_list=all_model_results,
                    diagnostics_list=all_diagnostics,
                )


            # ----------------------------------------------------------------
            # Summary tables
            # ----------------------------------------------------------------

            results_frame = pd.DataFrame(all_model_results)
            diagnostics_frame = pd.DataFrame(all_diagnostics)

            if save_single_time_res_summaries:
                statistics.generate_all_summary_tables(
                    results_df=results_frame,
                    output_dir=STATISTICS_OUTPUT_DATA,
                    diagnostics_df=diagnostics_frame,
                    file_identifier=f"{n_within_trial_segments}seg_{''.join(lvls_to_include)}",
                    generate_per_level_tables=False,
                    generate_thematic_tables=False,
                )


            # ----------------------------------------------------------------
            # Subject-level summary (coarsest resolution only to avoid redundancy)
            # ----------------------------------------------------------------

            if n_within_trial_segments == min(n_within_trial_segments_list):
                # Summarise per-subject accuracy distribution for inspection
                subject_accuracy_summary = (
                    all_subject_data_frame
                    .groupby('Subject ID')[ACCURACY_COL]
                    .agg(Mean='mean', Std='std', N='count')
                    .reset_index()
                )
                summary_path = STATISTICS_OUTPUT_DATA / filemgmt.file_title(
                    "RQ2 Subject Accuracy Summary", ".csv"
                )
                subject_accuracy_summary.to_csv(summary_path, index=False)
                print(f"\n✓ Saved subject accuracy summary → {summary_path}")


            # ----------------------------------------------------------------
            # Forest plots
            # ----------------------------------------------------------------

            hypotheses = results_frame['Hypothesis'].dropna().unique().tolist()

            if render_ols_effect_plots:
                visualizations.plot_hypothesis_forest_mosaic(
                    results_frame, hypotheses,
                    output_dir=STATISTICS_OUTPUT_DATA,
                    file_identifier_suffix=(
                        f"RQ2_{n_within_trial_segments}seg_{'_'.join(lvls_to_include)}"
                    ),
                    model_type='OLS',
                    hidden=not show_ols_effect_plots,
                    rename_dict=parameter_rename_dict,
                )

            if render_lme_effect_plots:
                visualizations.plot_hypothesis_forest_mosaic(
                    results_frame, hypotheses,
                    output_dir=STATISTICS_OUTPUT_DATA,
                    file_identifier_suffix=(
                        f"RQ2_{n_within_trial_segments}seg_{'_'.join(lvls_to_include)}"
                    ),
                    model_type='LME',
                    hidden=not show_lme_effect_plots,
                    rename_dict=parameter_rename_dict,
                )


            # ----------------------------------------------------------------
            # Tag with time-resolution metadata and accumulate
            # ----------------------------------------------------------------

            results_frame['Time Resolution'] = 40 / n_within_trial_segments
            diagnostics_frame['Time Resolution'] = 40 / n_within_trial_segments
            results_frame['N. Segments'] = n_within_trial_segments
            diagnostics_frame['N. Segments'] = n_within_trial_segments

            all_time_resolutions_results_list.append(results_frame.copy())
            all_diagnostics_list.append(diagnostics_frame.copy())


    #############################################################
    ################ SAVE ALL TIME-RES. RESULTS #################
    #############################################################

    if conduct_analysis and all_time_resolutions_results_list:

        all_time_resolutions_results_frame = pd.concat(all_time_resolutions_results_list)
        all_time_resolutions_results_frame.to_csv(
            STATISTICS_OUTPUT_DATA / filemgmt.file_title(
                "RQ2 All Time Resolutions Results", ".csv"
            ),
            index=False,
        )

        all_diagnostics_frame = pd.concat(all_diagnostics_list)
        all_diagnostics_frame.to_csv(
            STATISTICS_OUTPUT_DATA / filemgmt.file_title(
                "RQ2 All Time Resolutions Diagnostics", ".csv"
            ),
            index=False,
        )


    ########################################################
    ################ MULTI-TIME RES. PLOTS #################
    ########################################################

    if plot_time_resolution_comparisons:

        # Load saved results if the loop above was skipped
        try:
            _ = all_time_resolutions_results_frame
        except NameError:
            all_time_resolutions_results_frame = statistics.load_recent_results_frame(
                STATISTICS_OUTPUT_DATA
            )

        # Rebuild hypothesis list from saved frame (needed if loop was skipped)
        hypotheses = (
            all_time_resolutions_results_frame['Hypothesis'].dropna().unique().tolist()
        )

        for parameter, comparison_level in parameter_comp_lvl_tuples_to_plot_across_time:
            visualizations.plot_time_resolution_forest_mosaic(
                result_frame=all_time_resolutions_results_frame,
                hypotheses=hypotheses,
                parameter=parameter,
                comparison_level=comparison_level,
                model_type='LME',
                output_dir=STATISTICS_OUTPUT_DATA,
            )


    # Build optional transform so downstream steps that reload the CSV from
    # disk (influence analysis, power analysis) apply the same log-transform.
    _df_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    if log_transform_accuracy:
        def _df_transform(df: pd.DataFrame) -> pd.DataFrame:
            if ACCURACY_COL not in df.columns and ACCURACY_COL_RAW in df.columns:
                raw = df[ACCURACY_COL_RAW]
                df[ACCURACY_COL] = np.log(raw.where(raw > 0))
            return df


    #########################################################
    ################ INFLUENCE MEASURE COMP #################
    #########################################################

    if conduct_robustness_checks:

        try:
            _ = all_time_resolutions_results_frame
            # Validate that required n_segments were actually modelled
            n_segments_needed = {
                seg for _, _, seg
                in dep_var_comp_lvl_n_segments_tuples_to_robustness_check
            }
            for n_seg in n_segments_needed:
                if n_seg not in n_within_trial_segments_list:
                    raise ValueError(
                        f"n_segments={n_seg} required for robustness check is not in "
                        f"n_within_trial_segments_list={n_within_trial_segments_list}. "
                        f"Add it or adjust the robustness config."
                    )
        except NameError:
            all_time_resolutions_results_frame = statistics.load_recent_results_frame(
                STATISTICS_OUTPUT_DATA
            )

        influence_df = statistics.run_influence_analysis(
            configs=dep_var_comp_lvl_n_segments_tuples_to_robustness_check,
            full_results_df=all_time_resolutions_results_frame,
            feature_output_data=FEATURE_OUTPUT_DATA,
            statistics_output_data=STATISTICS_OUTPUT_DATA,
            fetch_level_definitions=_level_def_fn,
            run_model_levels=statistics.run_model_levels,
            file_title=filemgmt.file_title,
            df_transform=_df_transform,
        )
        # INTERPRETATION
        #   -> Cook's D: squared shift in all fitted values -> directionless
        #   -> DFBETA:   directional subject-level pull on each coefficient


    #############################################################
    ################ STATISTICAL POWER ANALYSIS #################
    #############################################################

    if conduct_power_analysis and power_configs:

        try:
            _ = all_time_resolutions_results_frame
        except NameError:
            all_time_resolutions_results_frame = statistics.load_recent_results_frame(
                STATISTICS_OUTPUT_DATA
            )

        power_curve_df, mde_df = statistics.run_power_analysis(
            configs=power_configs,
            results_df=all_time_resolutions_results_frame,
            feature_output_data=FEATURE_OUTPUT_DATA,
            statistics_output_data=STATISTICS_OUTPUT_DATA,
            fetch_level_definitions=_level_def_fn,
            file_title=filemgmt.file_title,
            df_transform=_df_transform,
        )