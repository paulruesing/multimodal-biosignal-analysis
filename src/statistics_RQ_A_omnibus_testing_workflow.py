from pathlib import Path
import pandas as pd

import src.pipeline.statistical_modelling as statistics
import src.pipeline.data_analysis as data_analysis
import src.pipeline.visualizations as visualizations
import src.utils.file_management as filemgmt


def fetch_level_definitions(multi_segments_per_trial: bool, always_include_scaled_force: bool = False) -> list[dict]:
    """Build the analysis-level configuration list for RQ-A omnibus modelling.

    Args:
        multi_segments_per_trial: If True, segment-wise models are configured (uses
            ``Segment ID`` in explanatory variables). If False, trial-wise models are
            configured (uses only ``Trial ID``).
        always_include_scaled_force: If True, ``Median Scaled Force [0-1]`` is kept in
            explanatory variables even for multi-segment models.

    Returns:
        A list of level-definition dictionaries consumed by
        ``statistics.run_model_levels``. Each dictionary can include keys such as:
        ``df_filter``, ``condition_vars``, ``reference_categories``,
        ``explanatory_vars``, and ``moderation_pairs``.

        Current levels are:
        - Level 0: all data, Music vs. Silence
        - Level 1: all data, Category-or-Silence comparison
        - Level 2: music trials only, subjective predictors
        - Level 3: music trials only, objective music-feature predictors
    """
    return [
        ######### CONFIRMATORY ANALYSES ############
        # Level 0 — all data, music vs. silence
        {
            'df_filter': None,
            'condition_vars': {'Music Listening': 'categorical'},
            'reference_categories': {'Music Listening': False},
            'explanatory_vars': (['Median Scaled Force [0-1]',
                                  'Median Unscaled Force [% MVC]'] if multi_segments_per_trial or always_include_scaled_force else [
                'Median Unscaled Force [% MVC]']) + (
                                    ['Trial ID'] if not multi_segments_per_trial else ['Trial ID', 'Segment ID']),
            'moderation_pairs': [('Music Listening', 'Musical skill [0-7]_centered'),
                                 ('Music Listening', 'Dancing habit [0-7]_centered')],
        },
        # Level 1 — all data, music category or silence
        {
            'df_filter': None,
            'condition_vars': {'Category or Silence': 'categorical'},
            'reference_categories': {'Category or Silence': 'Silence'},
            'explanatory_vars': (['Median Scaled Force [0-1]',
                                  'Median Unscaled Force [% MVC]'] if multi_segments_per_trial or always_include_scaled_force else [
                'Median Unscaled Force [% MVC]']) + (
                                    ['Trial ID'] if not multi_segments_per_trial else ['Trial ID', 'Segment ID']),
            'moderation_pairs': [('Category or Silence', 'Musical skill [0-7]_centered'),
                                 ('Category or Silence', 'Dancing habit [0-7]_centered')],
        },

        ######### EXPLORATORY ANALYSES #########
        # Level 2 — music trials only, subjective features
        {
            'df_filter': lambda df: df.loc[df['Music Listening']],
            'condition_vars': {'Perceived Category': 'categorical', 'Familiarity [0-7]': 'ordinal'},
            'reference_categories': {'Perceived Category': 'Classic'},
            'explanatory_vars': (['Median Scaled Force [0-1]',
                                  'Median Unscaled Force [% MVC]'] if multi_segments_per_trial or always_include_scaled_force else [
                'Median Unscaled Force [% MVC]']) + ['Liking_centered_squared'] + (
                                    ['Trial ID'] if not multi_segments_per_trial else ['Trial ID', 'Segment ID']),
            'moderation_pairs': [('Perceived Category', 'Musical skill [0-7]_centered'),
                                 ('Perceived Category', 'Dancing habit [0-7]_centered')],
        },
        # Level 3 — music trials only, objective features:
        #   Insights from music deep dive (distinctive features per perceived category)
        #       'Spectral Centroid Mean' for all but happy
        #       'Spectral Flux Std. ' for happy vs. sad
        #       'IOI Variance Coeff' for all but happy  ( + maybe a moderation with musical skill)
        {
            'df_filter': lambda df: df.loc[df['Music Listening']],
            'condition_vars': {'Familiarity [0-7]': 'ordinal'},  # REMOVED music category here to prevent redundancy!
            'explanatory_vars': (['Median Scaled Force [0-1]',
                                  'Median Unscaled Force [% MVC]'] if multi_segments_per_trial or always_include_scaled_force else [
                'Median Unscaled Force [% MVC]']) + ['Liking_centered_squared',
                                                     'Spectral Centroid Mean', 'Spectral Flux Std.',
                                                     'IOI Variance Coeff',] + (
                                    ['Trial ID'] if not multi_segments_per_trial else ['Trial ID', 'Segment ID']),
            'moderation_pairs': [('IOI Variance Coeff', 'Musical skill [0-7]_centered'),],
        },
    ]


if __name__ == '__main__':
    ######## PREPARATION #########
    ROOT = Path(__file__).resolve().parent.parent
    # system:
    DATA = ROOT / "data"
    OUTPUT = ROOT / 'output'
    # input:
    EXPERIMENT_DATA = DATA / "experiment_results"
    FEATURE_OUTPUT_DATA = DATA / "precomputed_features"
    # output:
    STATISTICS_OUTPUT_DATA = OUTPUT / 'statistics_RQ_A' / 'omnibus_testing'
    filemgmt.assert_dir(STATISTICS_OUTPUT_DATA)

    ##### PARAMETERS
    ## Time Resolution
    n_within_trial_segments_list: list[int] = [1, 2, 5, 10]  # of ~40sec trials

    # Subjects to exclude from all analyses (e.g. outliers or incomplete data)
    exclude_subjects: list[int] = []  # todo: always check!
    cmc_subject_subsets: list[list[int]] = [list(range(0, 6)), list(range(6, 12))]


    ## Data Exploration
    add_bin_features_dict: dict[str, int] = {'Median Unscaled Force [% MVC]': 4,
                                             'Liking': 4,
                                             'Liking_centered_squared': 4,
                                             'Familiarity [0-7]': 4,
                                             'Trial ID': 4}
    # creates bin index for per subject values to be used as categories (new col. will be named "{OLD_COL}_bin")
    cmc_plot_categories: list[str] = ['Category or Silence',# 'Liking_bin', 'Liking_centered_squared_bin', 'Trial ID_bin',
                                      #'Median Unscaled Force [% MVC]_bin', 'Familiarity [0-7]_bin'
                                      ]
    # color list:
    cmc_plot_colors = ['darkorange', 'red', 'green', 'blue', 'purple']
    cmc_plot_n_segments = 5
    # subject wise line plots:
    plot_cmc_lineplots_normalised: bool = False
    plot_cmc_lineplots_per_category: bool = False
    save_cmc_lineplots: bool = True
    # compound scatters:
    show_cmc_scatterplots: bool = False
    save_cmc_scatterplots: bool = True


    ## Statistical Analysis
    conduct_analysis: bool = True
    statistical_hypotheses_var_tuples: list[tuple[str, str]] = [
        # CMC Hypotheses:
        ('H1: Flexor Beta Peak CMC Increases with Music', "CMC_Flexor_max_beta"),
        ('H1: Flexor Beta Avg. CMC Increases with Music', "CMC_Flexor_mean_beta"),
        ('H1: Flexor Gamma Peak CMC Increases with Music', "CMC_Flexor_max_gamma"),
        ('H1: Flexor Gamma Avg. CMC Increases with Music', "CMC_Flexor_mean_gamma"),
        ('H1: Extensor Beta Peak CMC Increases with Music', "CMC_Extensor_max_beta"),
        ('H1: Extensor Beta Avg. CMC Increases with Music', "CMC_Extensor_mean_beta"),
        ('H1: Extensor Gamma Peak CMC Increases with Music', "CMC_Extensor_max_gamma"),
        ('H1: Extensor Gamma Avg. CMC Increases with Music', "CMC_Extensor_mean_gamma"),

        # EEG-PSD Hypotheses:
        ('H2: Temporal Prediction PSD Increases with Music', 'PSD_eeg_FC_CP_T_theta'),
        ('H3: Vigilance PSD Increases with Music', 'PSD_eeg_F_C_beta'),
        ('H4: Internal vs. External Attention PSD changes with Music', 'PSD_eeg_P_PO_alpha'),
        ('H5: Long Range Interactions PSD Increases with Music', 'PSD_eeg_Global_gamma'),

        # Validation Hypotheses (EMG PSD):
        ('VALIDATION: EMG Flexor PSD Increases with Force', 'PSD_emg_1_flexor_Global_all'),
        ('VALIDATION: EMG Extensor PSD Increases with Force', 'PSD_emg_2_extensor_Global_all'),

        # Possible Mediators: (NOW IN SEPARATE SCRIPT)
        # ('MEDIATION: Heart Rate', 'Median_Heart_Rate'),
        # ('MEDIATION: HRV', 'Median_HRV'),
        # ('MEDIATION: GSR', 'GSR'),
        # ('MEDIATION: Emotional Modulation', 'Emotional_State'),

    ]
    levels_for_fdr_correction: list[int] = [2, 3]  # exploratory changes, better don't change this
    save_single_time_res_summaries: bool = False

    # plotting:
    render_ols_effect_plots: bool = False
    render_lme_effect_plots: bool = False
    show_effect_plots: bool = True  # if False, either of the above will be hidden
    param_rename_dict: dict[str, str] = {
        # ── Category or Silence — main effects ───────────────────────────────────
        "Category or Silence[T.Classic]": "Classical vs. Silence",
        "Category or Silence[T.Groovy]": "Groovy vs. Silence",
        "Category or Silence[T.Happy]": "Happy vs. Silence",
        "Category or Silence[T.Sad]": "Sad vs. Silence",

        # ── Category or Silence — interactions ───────────────────────────────────
        "Category or Silence[T.Classic]:Dancing habit [0-7]_centered": "Classical vs. Silence x Dancing habit",
        "Category or Silence[T.Classic]:Musical skill [0-7]_centered": "Classical vs. Silence x Musical skill",
        "Category or Silence[T.Groovy]:Dancing habit [0-7]_centered": "Groovy vs. Silence x Dancing habit",
        "Category or Silence[T.Groovy]:Musical skill [0-7]_centered": "Groovy vs. Silence x Musical skill",
        "Category or Silence[T.Happy]:Dancing habit [0-7]_centered": "Happy vs. Silence x Dancing habit",
        "Category or Silence[T.Happy]:Musical skill [0-7]_centered": "Happy vs. Silence x Musical skill",
        "Category or Silence[T.Sad]:Dancing habit [0-7]_centered": "Sad vs. Silence x Dancing habit",
        "Category or Silence[T.Sad]:Musical skill [0-7]_centered": "Sad vs. Silence x Musical skill",

        # ── Music Listening — [T.True] dropped (boolean flag) ────────────────────
        "Music Listening[T.True]": "Music listening",
        "Music Listening[T.True]:Dancing habit [0-7]_centered": "Music listening x Dancing habit",
        "Music Listening[T.True]:Musical skill [0-7]_centered": "Music listening x Musical skill",

        # ── Covariates ────────────────────────────────────────────────────────────
        "Dancing habit [0-7]_centered": "Dancing habit",
        "Musical skill [0-7]_centered": "Musical skill",

        # ── Units ─────────────────────────────────────────────────────────────
        "Median Unscaled Force [% MVC]": "Force",
        "Median Scaled Force [0-1]": "Task-wise scaled force [0-1]",
        "Trial ID": "Trial number",
        "Segment ID": "Intra-trial time",
    }

    # comparison levels:
    lvl_inds_to_include: list[int] = [0, 1, 2, 3]  # defines below  # todo: good to remove 2, 3 for forest plots
    lvls_to_include: list[str] = [f"lvl_{lvl_ind}" for lvl_ind in lvl_inds_to_include]

    # across time resolution comparison:
    plot_time_resolution_comparisons: bool = True
    parameter_comp_lvl_tuples_to_plot_across_time: list[tuple[str, int]] = [
        ('Category or Silence[T.Happy]', 1),  # sig across all CMC DVs + EEG PSD
        ('Category or Silence[T.Groovy]', 1),  # sig: Flexor mean/max beta, Flexor mean gamma
        ('Category or Silence[T.Classic]', 1),  # sig: Extensor max beta (5-seg)
        ('Dancing habit [0-7]_centered', 1),  # main effect: Extensor max beta, Extensor mean gamma
        ('Segment ID', 1),  # sig: Extensor mean gamma, Flexor mean beta (L1 at 5-seg)
        # ('Perceived Category[T.Sad]:Dancing habit [0-7]_centered', 2),
        # ('Perceived Category[T.Happy]', 2),
        # ('Perceived Category[T.Groovy]', 2),
        # ('Perceived Category[T.Happy]:Musical skill [0-7]_centered', 2),
        # ('Perceived Category[T.Groovy]:Dancing habit [0-7]_centered', 2),
        ('Median Scaled Force [0-1]', 1),  # large effect on EMG PSD at 5-seg
        ('Median Unscaled Force [% MVC]', 1),
    ]


    # LME robustness checks / influence measures:
    conduct_robustness_checks: bool = False
    dep_var_comp_lvl_n_segments_tuples_to_robustness_check: list[tuple[str, int, int]] = [
        # CMC DVs: all 8 have significant effects at primary resolution
        ('CMC_Extensor_mean_beta', 1, 1),
        ('CMC_Extensor_max_beta', 1, 1),
        ('CMC_Extensor_mean_gamma', 1, 1),
        ('CMC_Extensor_max_gamma', 1, 1),
        ('CMC_Flexor_mean_beta', 1, 1),
        ('CMC_Flexor_max_beta', 1, 1),
        ('CMC_Flexor_mean_gamma', 1, 1),
        ('CMC_Flexor_max_gamma', 1, 1),

        # EEG PSD: H2, H3, H5 now have 1 significant effect each at 5-seg
        # todo: ponder, whether u should add the capability for multi-time-resolutions in this list, since the below
        #   are not significant for seg = 1
        ('PSD_eeg_FC_CP_T_theta', 1, 1),  # H2: Happy L1
        ('PSD_eeg_F_C_beta', 1, 1),  # H3: Happy L1, d=0.40
        ('PSD_eeg_Global_gamma', 1, 1),  # H5: Happy L1, d=0.39
        # NOTE: H4 (PSD_eeg_P_PO_alpha) still 0 significant effects at 5-seg

    ]

    # Statistical Power Analysis:
    conduct_power_analysis: bool = False  # todo: change, is very run-time extensive
    #### POWER CONFIG DEFINITION ###
    # ── Shared parameter sets ──────────────────────────────────────────────────
    _LVL0_PARAMS = [
        "C(Q('Music Listening'))[T.True]",
        "C(Q('Music Listening'))[T.True]:Q('Dancing habit [0-7]_centered')",
        "Q('Dancing habit [0-7]_centered')",
        "Q('Musical skill [0-7]_centered')",
    ]
    _LVL1_PARAMS = [
        "C(Q('Category or Silence'))[T.Happy]",
        "C(Q('Category or Silence'))[T.Groovy]",
        "C(Q('Category or Silence'))[T.Classic]",
        "C(Q('Category or Silence'))[T.Sad]",
        "C(Q('Category or Silence'))[T.Groovy]:Q('Dancing habit [0-7]_centered')",
        "C(Q('Category or Silence'))[T.Happy]:Q('Dancing habit [0-7]_centered')",
        "C(Q('Perceived Category'))[T.Happy]:Q('Musical skill [0-7]_centered')",
        "C(Q('Category or Silence'))[T.Classic]:Q('Musical skill [0-7]_centered')",
        "Q('Median Unscaled Force [% MVC]')",
        "Q('Trial ID')",
    ]
    _LVL1_5SEG_PARAMS = _LVL1_PARAMS + [
        "Q('Segment ID')",
        "Q('Median Scaled Force [0-1]')",
    ]
    _LVL2_PARAMS = [
        "C(Q('Perceived Category'))[T.Happy]",
        "C(Q('Perceived Category'))[T.Groovy]",
        "C(Q('Perceived Category'))[T.Sad]",
        "Liking_centered_squared",
        "Q('Familiarity [0-7]')",
    ]
    _STANDARD_DVS = [
        "CMC_Extensor_mean_beta",
        "CMC_Extensor_max_beta",
        "CMC_Extensor_mean_gamma",
        "CMC_Extensor_max_gamma",
        "CMC_Flexor_mean_beta",
        "CMC_Flexor_max_beta",
        "CMC_Flexor_mean_gamma",
        "CMC_Flexor_max_gamma",
        "PSD_eeg_FC_CP_T_theta",
        "PSD_eeg_F_C_beta",
        "PSD_eeg_Global_gamma",
        "PSD_eeg_P_PO_alpha",
    ]
    power_configs: list[statistics.PowerConfig] = [

        # ═══════════════════════════════════════════════════════════════════════
        #  Standardised entries — all 12 DVs, identical structure per level
        # ═══════════════════════════════════════════════════════════════════════

        *[
            cfg
            for dv in _STANDARD_DVS
            for cfg in [
                # ── n_segments = 1 ──────────────────────────────────────────
                statistics.PowerConfig(
                    dependent_var=dv,
                    comp_lvl=0,
                    n_segments=1,
                    target_parameters=_LVL0_PARAMS,
                ),
                statistics.PowerConfig(
                    dependent_var=dv,
                    comp_lvl=1,
                    n_segments=1,
                    target_parameters=_LVL1_PARAMS,
                ),
                statistics.PowerConfig(
                    dependent_var=dv,
                    comp_lvl=2,
                    n_segments=1,
                    target_parameters=_LVL2_PARAMS,
                ),
                # ── n_segments = 5 ──────────────────────────────────────────
                statistics.PowerConfig(
                    dependent_var=dv,
                    comp_lvl=0,
                    n_segments=5,
                    target_parameters=_LVL0_PARAMS,
                ),
                statistics.PowerConfig(
                    dependent_var=dv,
                    comp_lvl=1,
                    n_segments=5,
                    target_parameters=_LVL1_5SEG_PARAMS,  # +SegID, +ScaledForce
                ),
                statistics.PowerConfig(
                    dependent_var=dv,
                    comp_lvl=2,
                    n_segments=5,
                    target_parameters=_LVL2_PARAMS,
                ),
            ]
        ],
    ]

    #######################################################
    ################ LOOP OVER N_SEGMENTS #################
    #######################################################

    all_time_resolutions_results_list: list[pd.DataFrame] = []
    all_diagnostics_list: list[pd.DataFrame] = []
    for n_within_trial_segments in n_within_trial_segments_list:  # analyse different time resolutions

        # comparison levels:
        level_definitions = fetch_level_definitions(multi_segments_per_trial=n_within_trial_segments > 1)





        #####################################################
        ################ FEATURE EXTRACTION #################
        #####################################################

        ### TRY FETCHING EXISTING FRAME
        if plot_cmc_lineplots_per_category or show_cmc_scatterplots or conduct_analysis:
            all_subject_data_frame = pd.read_csv(filemgmt.most_recent_file(FEATURE_OUTPUT_DATA, ".csv", [f"Combined Statistics {n_within_trial_segments}seg"]))  # pd.read_csv(FEATURE_OUTPUT_DATA / "statistics_temp_eeg_alpha.csv")  # set to None to run computation
            print(f"Fetching existing statistical data frame for n_segments {n_within_trial_segments}...\n")

            # ── Apply subject exclusions ──────────────────────────────────────────────
            if exclude_subjects:
                before = all_subject_data_frame["Subject ID"].nunique()
                all_subject_data_frame = all_subject_data_frame.loc[
                    ~all_subject_data_frame["Subject ID"].isin(exclude_subjects)
                ].reset_index(drop=True)
                after = all_subject_data_frame["Subject ID"].nunique()
                print(f"  [Exclusions] Removed subjects {exclude_subjects}: {before} → {after} subjects remaining.\n")



        #######################################################
        ################ DATA EXPLORATION #####################
        #######################################################

        if cmc_plot_n_segments == n_within_trial_segments:
            if plot_cmc_lineplots_per_category or show_cmc_scatterplots:
                all_subject_data_frame = data_analysis.create_trial_bins(
                    df=all_subject_data_frame,
                    columns_to_bin=list(add_bin_features_dict.keys()),
                    n_bins_dict=add_bin_features_dict,

                )


            # normalised CMC line plot:
            if plot_cmc_lineplots_normalised:
                for subset_ids in cmc_subject_subsets:
                    subset_suffix = f"subjects_{min(subset_ids):02}_{max(subset_ids):02}"
                    subset_save_dir = None
                    if save_cmc_lineplots:
                        subset_save_dir = STATISTICS_OUTPUT_DATA / subset_suffix
                        subset_save_dir.mkdir(parents=True, exist_ok=True)

                    for muscle in ['Flexor', 'Extensor']:
                        visualizations.plot_cmc_lineplot_normalised(
                            all_subject_data_frame, muscle,
                            cmc_operator='mean', n_within_trial_segments=n_within_trial_segments,
                            cmc_plot_min=75.0, cmc_plot_max=125.0, n_yticks=5,
                            show_significance_threshold=False,
                            corridor_std_factor=1.0, corridor_alpha=.4,
                            corridor_color='lightblue',
                            save_dir=subset_save_dir,
                            alpha=.2,
                            line_width=1.2,
                            plot_size=(12, 6),
                            show_grid=True,
                            subject_ids_subset=subset_ids,
                        )




            if plot_cmc_lineplots_per_category:
                # plot CMC per subject and category:
                for subset_ids in cmc_subject_subsets:
                    subset_suffix = f"subjects_{min(subset_ids):02}_{max(subset_ids):02}"
                    subset_save_dir = None
                    if save_cmc_lineplots:
                        subset_save_dir = STATISTICS_OUTPUT_DATA / subset_suffix
                        subset_save_dir.mkdir(parents=True, exist_ok=True)

                    for muscle in ['Flexor', 'Extensor']:
                        # loop over categories (new plot per category)
                        for category_column in cmc_plot_categories:
                            visualizations.plot_cmc_lineplots_per_category(
                                all_subject_data_frame, category_column, muscle,
                                cmc_operator='mean', n_within_trial_segments=n_within_trial_segments,
                                cmc_plot_min=.7, cmc_plot_max=1.0, n_yticks=4,
                                include_std_dev=True, std_dev_factor=.2,
                                colormap=cmc_plot_colors,
                                show_significance_threshold=True,
                                save_dir=subset_save_dir,
                                alpha=.1,
                                subject_ids_subset=subset_ids,
                                plot_size=(12, 6),
                                show_grid=True,
                            )






            # dependent var (plot beta + gamma scatter for flexor + extensor) for each category:
            # [((x_column, x_label), (y_column, y_label), category_column), ...]
            if show_cmc_scatterplots:
                scatters_to_plot: list[tuple[tuple[str, str], tuple[str, str], str]] = []
                for category_column in cmc_plot_categories:
                    scatters_to_plot.append(
                        (('CMC_Flexor_mean_beta', 'CMC Flexor Avg. Beta'), ('CMC_Flexor_mean_gamma', 'CMC Flexor Avg. Gamma'),
                         category_column))
                    scatters_to_plot.append(
                        (('CMC_Extensor_mean_beta', 'CMC Extensor Avg. Beta'), ('CMC_Extensor_mean_gamma', 'CMC Extensor Avg. Gamma'),
                         category_column))


                # scatters:
                for (x, x_label), (y, y_label), category_column in scatters_to_plot:
                    dataframe_subset = all_subject_data_frame.dropna(subset=[x, y, category_column])
                    x_data = dataframe_subset[x]
                    y_data = dataframe_subset[y]
                    category_list = dataframe_subset[category_column]
                    visualizations.plot_scatter(x=x_data, x_label=x_label,
                                                y=y_data, y_label=y_label,
                                                category_list=category_list, category_label=category_column,
                                                save_dir=STATISTICS_OUTPUT_DATA if save_cmc_scatterplots else None,
                                                cmap=cmc_plot_colors,
                                                )








        ########################################################
        ################ STATISTICAL MODELLING #################
        ########################################################

        if conduct_analysis:
            # Store all results and diagnostics for summary tables
            all_model_results = []
            all_diagnostics = []

            for hypothesis, dependent_variable in statistical_hypotheses_var_tuples:

                # Intro String:
                print("\n")
                print("=" * 100)
                print("=" * 100)
                print(f"HYPOTHESIS:\t\t{hypothesis} ")
                print(f"DEPENDENT VARIABLE:\t{dependent_variable}")
                print("=" * 100)
                print("=" * 100, "\n"*3)


                ############# COMPARISON LEVELS #############

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


            # ============================================================================
            # Summary statistics
            # ============================================================================

            results_frame = pd.DataFrame(all_model_results)
            diagnostics_frame = pd.DataFrame(all_diagnostics)

            ### ADD DESCRIPTIVE TIME-RES VARIABLES:
            results_frame['Time Resolution'] = 40 / n_within_trial_segments
            diagnostics_frame['Time Resolution'] = 40 / n_within_trial_segments
            results_frame['N. Segments'] = n_within_trial_segments
            diagnostics_frame['N. Segments'] = n_within_trial_segments

            # ── FDR correction for exploratory levels (per time resolution) ───────────
            # Applied here so forest plots reflect corrected significance.
            # Levels 0–1 are confirmatory (pre-specified, directional) — no correction.
            # Levels 2–3 are exploratory — BH correction within each Level × DV stratum.
            results_frame = statistics.apply_fdr_correction(
                results_df=results_frame,
                levels_to_correct=levels_for_fdr_correction,
                alpha=0.05,
                group_by_dv=True,
            )

            # Generate all summary tables with one function call
            if save_single_time_res_summaries:
                statistics.generate_all_summary_tables(
                    results_df=results_frame,
                    output_dir=STATISTICS_OUTPUT_DATA,
                    diagnostics_df=diagnostics_frame,
                    file_identifier=f"{n_within_trial_segments}seg_{"".join(lvls_to_include)}",
                    generate_per_level_tables=False,
                    generate_thematic_tables=False,
                )



            # ============================================================================
            # Subject-specific analysis
            # ============================================================================

            # compute only once (since we average over trial-level) conditions for the coarsest time-resolution:
            if n_within_trial_segments == min(n_within_trial_segments_list):
                statistics.create_subject_effect_summary(
                    all_model_results=all_model_results,
                    original_data=all_subject_data_frame,
                    output_dir=STATISTICS_OUTPUT_DATA,
                    level_definitions=level_definitions,
                )

            # ============================================================================
            # Plotting
            # ============================================================================

            # derive unique hypotheses:
            hypotheses = results_frame['Hypothesis'].dropna().unique().tolist()

            # group hypotheses:
            cmc_flexor_hypotheses = [h for h in hypotheses if 'CMC' in h and 'Flexor' in h and not 'VALIDATION: ' in h]
            cmc_extensor_hypotheses = [h for h in hypotheses if 'CMC' in h and 'Extensor' in h and not 'VALIDATION: ' in h]
            psd_hypotheses = [h for h in hypotheses if 'PSD' in h and not 'VALIDATION: ' in h]
            validation_hypotheses = [h for h in hypotheses if 'VALIDATION: ' in h]

            if render_ols_effect_plots:
                visualizations.plot_hypothesis_forest_mosaic(results_frame, cmc_flexor_hypotheses, output_dir=STATISTICS_OUTPUT_DATA,
                                                             file_identifier_suffix=f"H1_Flexor_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                                             model_type='OLS', hidden=not show_effect_plots, significance_source='auto',
                                                             rename_dict=param_rename_dict)
                visualizations.plot_hypothesis_forest_mosaic(results_frame, cmc_extensor_hypotheses, output_dir = STATISTICS_OUTPUT_DATA,
                                                             file_identifier_suffix=f"H1_Extensor_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                                             model_type='OLS', hidden=not show_effect_plots, significance_source='auto',
                                                             rename_dict=param_rename_dict)
                visualizations.plot_hypothesis_forest_mosaic(results_frame, psd_hypotheses, output_dir = STATISTICS_OUTPUT_DATA,
                                                             file_identifier_suffix=f"H2-5_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                                             model_type='OLS', hidden=not show_effect_plots, significance_source='auto',
                                                             rename_dict=param_rename_dict)
                visualizations.plot_hypothesis_forest_mosaic(results_frame, validation_hypotheses, output_dir = STATISTICS_OUTPUT_DATA,
                                                             file_identifier_suffix=f"VAL_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                                             model_type='OLS', hidden=not show_effect_plots, significance_source='auto',
                                                             rename_dict=param_rename_dict)

            # LME Results:
            if render_lme_effect_plots:
                visualizations.plot_hypothesis_forest_mosaic(results_frame, cmc_flexor_hypotheses, output_dir=STATISTICS_OUTPUT_DATA,
                                                             file_identifier_suffix=f"H1_Flexor_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                                             model_type='LME', hidden=not show_effect_plots, significance_source='auto',
                                                             rename_dict=param_rename_dict)

                visualizations.plot_hypothesis_forest_mosaic(results_frame, cmc_extensor_hypotheses, output_dir = STATISTICS_OUTPUT_DATA,
                                                             file_identifier_suffix=f"H1_Extensor_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                                             model_type='LME', hidden=not show_effect_plots, significance_source='auto',
                                                             rename_dict=param_rename_dict)

                visualizations.plot_hypothesis_forest_mosaic(results_frame, psd_hypotheses, output_dir = STATISTICS_OUTPUT_DATA,
                                                             file_identifier_suffix=f"H2-5_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                                             model_type='LME', hidden=not show_effect_plots, significance_source='auto',
                                                             rename_dict=param_rename_dict)

                visualizations.plot_hypothesis_forest_mosaic(results_frame, validation_hypotheses, output_dir = STATISTICS_OUTPUT_DATA,
                                                             file_identifier_suffix=f"VAL_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                                             model_type='LME', hidden=not show_effect_plots, significance_source='auto',
                                                             rename_dict=param_rename_dict)





            ### SAVE TIME-RES FRAME
            # save:
            all_time_resolutions_results_list.append(results_frame.copy())
            all_diagnostics_list.append(diagnostics_frame.copy())




    #############################################################
    ################ SAVE ALL TIME-RES. RESULTS #################
    #############################################################

    if conduct_analysis:  # only then save results, otherwise will be loaded below
        # save all time resolutions' results
        all_time_resolutions_results_frame = pd.concat(all_time_resolutions_results_list)

        # save as CSVs
        all_time_resolutions_results_frame.to_csv(
            STATISTICS_OUTPUT_DATA / filemgmt.file_title("All Time Resolutions Results", ".csv")
        )
        all_diagnostics_frame = pd.concat(all_diagnostics_list)
        all_diagnostics_frame.to_csv(
            STATISTICS_OUTPUT_DATA / filemgmt.file_title("All Time Resolutions Diagnostics", ".csv")
        )









    ########################################################
    ################ MULTI-TIME RES. PLOTS #################
    ########################################################

    ### ANALYSIS ACROSS TIME-SCALES
    if plot_time_resolution_comparisons:
        try: _ = all_time_resolutions_results_frame
        except NameError:
            all_time_resolutions_results_frame = statistics.load_recent_results_frame(STATISTICS_OUTPUT_DATA)

        # Rebuild hypothesis groups from the loaded/combined frame so this block
        # also works when conduct_analysis=False in the current run.
        available_hypotheses = all_time_resolutions_results_frame['Hypothesis'].dropna().unique().tolist()
        cmc_flexor_hypotheses = [h for h in available_hypotheses if 'CMC' in h and 'Flexor' in h and 'VALIDATION: ' not in h]
        cmc_extensor_hypotheses = [h for h in available_hypotheses if 'CMC' in h and 'Extensor' in h and 'VALIDATION: ' not in h]
        psd_hypotheses = [h for h in available_hypotheses if 'PSD' in h and 'VALIDATION: ' not in h]
        validation_hypotheses = [h for h in available_hypotheses if 'VALIDATION: ' in h]

        for parameter, comparison_level in parameter_comp_lvl_tuples_to_plot_across_time:
            for hypotheses in [cmc_flexor_hypotheses, cmc_extensor_hypotheses, psd_hypotheses, validation_hypotheses]:
                if len(hypotheses) == 0:
                    continue
                visualizations.plot_time_resolution_forest_mosaic(
                    result_frame=all_time_resolutions_results_frame, hypotheses=hypotheses,
                    parameter=parameter, comparison_level=comparison_level,
                    model_type='LME', output_dir=STATISTICS_OUTPUT_DATA,
                    significance_source='auto',  # falls back to non-FDR for lvl 0 and 1
                    rename_dict=param_rename_dict,
                )






    #########################################################
    ################ INFLUENCE MEASURE COMP #################
    #########################################################

    if conduct_robustness_checks:
        try:
            _ = all_time_resolutions_results_frame
            n_segments = {seg for _, _, seg in dep_var_comp_lvl_n_segments_tuples_to_robustness_check}
            for n_seg in n_segments:
                if n_seg not in n_within_trial_segments_list: raise ValueError(f"Specified n_segments {n_seg} for robustness check wasn't modelled! Either change n_within_trial_segments_list ({n_within_trial_segments_list}) or pick one thereof.")
        except NameError:
            all_time_resolutions_results_frame = statistics.load_recent_results_frame(
                STATISTICS_OUTPUT_DATA
            )

        influence_df = statistics.run_influence_analysis(
            configs=dep_var_comp_lvl_n_segments_tuples_to_robustness_check,
            full_results_df=all_time_resolutions_results_frame,
            feature_output_data=FEATURE_OUTPUT_DATA,
            statistics_output_data=STATISTICS_OUTPUT_DATA,
            fetch_level_definitions=fetch_level_definitions,
            run_model_levels=statistics.run_model_levels,
            file_title=filemgmt.file_title,
        )
        # INTERPRETATION
        #   -> Cook's D: squared shift in all fitted values -> directionless
        #   -> DFBETA:   in which direction subject "pulls" coefficient -> directional

        # todo: eventually re-compute LME model for subset








    #############################################################
    ################ STATISTICAL POWER ANALYSIS #################
    #############################################################

    if conduct_power_analysis:
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
            fetch_level_definitions=fetch_level_definitions,
            file_title=filemgmt.file_title,
        )

    # todo: interpret also as outlook -> future data collection