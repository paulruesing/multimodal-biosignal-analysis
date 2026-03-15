from pathlib import Path
import pandas as pd

import src.pipeline.statistical_modelling as statistics
import src.pipeline.data_analysis as data_analysis
import src.pipeline.visualizations as visualizations
import src.utils.file_management as filemgmt


def fetch_level_definitions(multi_segments_per_trial: bool) -> list[dict]:
    return [
        # Level 0 — all data, music vs. silence
        {
            'df_filter': None,
            'condition_vars': {'Music Listening': 'categorical'},
            'reference_categories': {'Music Listening': 'False'},
            'explanatory_vars': ['Median Scaled Force [0-1]', 'Median Unscaled Force [% MVC]'] + (
                ['Trial ID'] if not multi_segments_per_trial else ['Trial ID', 'Segment ID']),
            'moderation_pairs': [('Music Listening', 'Musical skill [0-7]_centered'),
                                 ('Music Listening', 'Dancing habit [0-7]_centered')],
        },
        # Level 1 — all data, music category or silence
        {
            'df_filter': None,
            'condition_vars': {'Category or Silence': 'categorical'},
            'reference_categories': {'Category or Silence': 'Silence'},
            'explanatory_vars': ['Median Scaled Force [0-1]', 'Median Unscaled Force [% MVC]'] + (
                ['Trial ID'] if not multi_segments_per_trial else ['Trial ID', 'Segment ID']),
            'moderation_pairs': [('Category or Silence', 'Musical skill [0-7]_centered'),
                                 ('Category or Silence', 'Dancing habit [0-7]_centered')],
        },
        # Level 2 — music trials only, subjective features
        {
            'df_filter': lambda df: df.loc[df['Music Listening']],
            'condition_vars': {'Perceived Category': 'categorical', 'Familiarity [0-7]': 'ordinal'},
            'reference_categories': {'Perceived Category': 'Classic'},
            'explanatory_vars': ['Median Scaled Force [0-1]', 'Median Unscaled Force [% MVC]', 'Liking [0-7]'] + (
                ['Trial ID'] if not multi_segments_per_trial else ['Trial ID', 'Segment ID']),
            'moderation_pairs': [('Perceived Category', 'Musical skill [0-7]_centered'),
                                 ('Perceived Category', 'Dancing habit [0-7]_centered')],
        },
        # Level 3 — music trials only, add emotional state + biomarkers
        {
            'df_filter': lambda df: df.loc[df['Music Listening']],
            'condition_vars': {'Perceived Category': 'categorical', 'Familiarity [0-7]': 'ordinal'},
            'reference_categories': {'Perceived Category': 'Classic'},
            'explanatory_vars': ['Median Scaled Force [0-1]', 'Median Unscaled Force [% MVC]', 'Liking [0-7]',
                                 'Emotional State [0-7]', 'Median Heart Rate [bpm]', 'Median HRV [sec]',
                                 'GSR [0-3.3]'] + (
                                    ['Trial ID'] if not multi_segments_per_trial else ['Trial ID', 'Segment ID']),
            'moderation_pairs': [('Perceived Category', 'Musical skill [0-7]_centered'),
                                 ('Perceived Category', 'Dancing habit [0-7]_centered')],
        },
        # Level 4 — music trials only, add objective music features
        {
            'df_filter': lambda df: df.loc[df['Music Listening']],
            'condition_vars': {'Perceived Category': 'categorical', 'Familiarity [0-7]': 'ordinal'},
            'reference_categories': {'Perceived Category': 'Classic'},
            'explanatory_vars': ['Median Scaled Force [0-1]', 'Median Unscaled Force [% MVC]', 'Liking [0-7]',
                                 'Emotional State [0-7]', 'Median Heart Rate [bpm]', 'Median HRV [sec]',
                                 'GSR [0-3.3]',
                                 'BPM_manual', 'Spectral Flux Mean', 'Spectral Centroid Mean',
                                 'IOI Variance Coeff', 'Syncopation Ratio'] + (
                                    ['Trial ID'] if not multi_segments_per_trial else ['Trial ID', 'Segment ID']),
            'moderation_pairs': [('Perceived Category', 'Musical skill [0-7]_centered'),
                                 ('Perceived Category', 'Dancing habit [0-7]_centered')],
        },
    ]


if __name__ == '__main__':
    ######## PREPARATION #########
    ROOT = Path().resolve().parent
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
    n_within_trial_segments_list: list[int] = [1,2, 5, 10]  # of ~40sec trials


    ## Data Exploration
    add_bin_features_dict: dict[str, int] = {'Median Scaled Force [0-1]': 4, 'Familiarity [0-7]': 5,
                                             'GSR [0-3.3]': 4, 'Trial ID': 4}
    # creates bin index for per subject values to be used as categories (new col. will be named "{OLD_COL}_bin")
    cmc_plot_categories: list[str] = ['Subject ID', 'Category or Silence', 'Familiarity [0-7]_bin', 'Trial ID_bin',
                                      'Median Scaled Force [0-1]_bin', 'GSR [0-3.3]_bin']
    # subject wise line plots:
    plot_cmc_lineplots: bool = False
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
    ]
    save_single_time_res_summaries: bool = False

    # plotting:
    render_ols_effect_plots: bool = False
    render_lme_effect_plots: bool = False
    show_effect_plots: bool = True  # if False, either of the above will be hidden

    # comparison levels:
    lvl_inds_to_include: list[int] = [0, 1, 2, 3]  # defines below
    lvls_to_include: list[str] = [f"lvl_{lvl_ind}" for lvl_ind in lvl_inds_to_include]

    # across time resolution comparison:
    plot_time_resolution_comparisons: bool = False
    parameter_comp_lvl_tuples_to_plot_across_time: list[tuple[str, int]] = [
        ('Category or Silence[T.Happy]', 1),
        #('Category or Silence[T.Groovy]', 1),  # Tier 1 — not null for Flexor beta
        #('Perceived Category[T.Sad]:Dancing habit [0-7]_centered', 2),  # Tier 3 — robust interaction
        #('Perceived Category[T.Happy]', 2),
        #('Perceived Category[T.Groovy]', 2),
        ('Segment ID', 2),

        #('Dancing habit [0-7]_centered', 2),
        #('Perceived Category[T.Happy]:Musical skill [0-7]_centered', 2),
        #('Perceived Category[T.Groovy]:Dancing habit [0-7]_centered', 2),

        ('Median Scaled Force [0-1]', 1),
        ('Median Unscaled Force [% MVC]', 1)
    ]

    # LME robustness checks / influence measures:
    conduct_robustness_checks: bool = True
    dep_var_comp_lvl_n_segments_tuples_to_robustness_check: list[tuple[str, int, int]] = [
        ('CMC_Extensor_mean_beta', 1, 1),
        ('CMC_Extensor_max_beta', 1, 1),
        ('CMC_Extensor_mean_gamma', 1, 1),
        ('CMC_Extensor_max_gamma', 1, 1),
        ('CMC_Flexor_mean_beta', 1, 1),
        ('CMC_Flexor_max_beta', 1, 1),
        ('CMC_Flexor_mean_gamma', 1, 1),
        ('CMC_Flexor_max_gamma', 1, 1),
        ('PSD_eeg_FC_CP_T_theta', 1, 1),
        ('PSD_eeg_F_C_beta', 1, 1),
        ('PSD_eeg_P_PO_alpha', 1, 1),
        ('PSD_eeg_Global_gamma', 1, 1),

        # Level 2:
        ('CMC_Extensor_mean_beta', 2, 1),  # S01 drives Dancing habit / Musical skill interactions at L2
        ('CMC_Flexor_max_beta', 2, 1),  # S06/S07/S08 drive Groovy + Musical skill interactions at L2

        # todo: add capability for below, currently only one segment count possible
    ]

    """# all modulated by segment ID at segments = 5
    ('CMC_Extensor_mean_beta', 1, 5),
    ('CMC_Extensor_max_beta', 1, 5),
    ('CMC_Extensor_mean_gamma', 1, 5),
    ('CMC_Extensor_max_gamma', 1, 5),
    ('CMC_Flexor_mean_beta', 1, 5),
    ('CMC_Flexor_max_beta', 1, 5),
    ('CMC_Flexor_mean_gamma', 1, 5),
    ('CMC_Flexor_max_gamma', 1, 5),"""

    # Statistical Power Analysis:
    # todo: include n_segments = 2 for Segment ID power analysis!
    conduct_power_analysis: bool = False
    power_configs: list[statistics.PowerConfig] = [

        statistics.PowerConfig(
            dependent_var="CMC_Flexor_mean_beta",
            comp_lvl=1,
            n_segments=1,
            target_parameters=[
                "C(Q('Category or Silence'))[T.Happy]",  # near-sig, d=0.73
            ],
        ),
        statistics.PowerConfig(
            dependent_var="CMC_Flexor_mean_beta",
            comp_lvl=2,
            n_segments=1,
            target_parameters=[
                "C(Q('Perceived Category'))[T.Happy]",  # near-sig, d=0.71
            ],
        ),

        # ── CMC_Extensor_max_beta ── Level 2 + 3: Perceived Category = Happy
        statistics.PowerConfig(
            dependent_var="CMC_Extensor_max_beta",
            comp_lvl=2,
            n_segments=1,
            target_parameters=[
                "C(Q('Perceived Category'))[T.Happy]",
            ],
        ),
        statistics.PowerConfig(
            dependent_var="CMC_Extensor_max_beta",
            comp_lvl=3,
            n_segments=1,
            target_parameters=[
                "C(Q('Perceived Category'))[T.Happy]",
            ],
        ),

        # ── CMC_Extensor_max_gamma ── Level 1: Category or Silence + Force covariate
        statistics.PowerConfig(
            dependent_var="CMC_Extensor_max_gamma",
            comp_lvl=1,
            n_segments=1,
            target_parameters=[
                "C(Q('Category or Silence'))[T.Happy]",
                "Q('Median Unscaled Force [% MVC]')",
            ],
        ),

        # ── CMC_Extensor_max_gamma ── Level 2 + 3: Perceived Category = Happy
        statistics.PowerConfig(
            dependent_var="CMC_Extensor_max_gamma",
            comp_lvl=2,
            n_segments=1,
            target_parameters=[
                "C(Q('Perceived Category'))[T.Happy]",
            ],
        ),
        statistics.PowerConfig(
            dependent_var="CMC_Extensor_max_gamma",
            comp_lvl=3,
            n_segments=1,
            target_parameters=[
                "C(Q('Perceived Category'))[T.Happy]",
            ],
        ),

        # ── CMC_Extensor_mean_beta ── Level 1: Category + interactions + Force
        statistics.PowerConfig(
            dependent_var="CMC_Extensor_mean_beta",
            comp_lvl=1,
            n_segments=1,
            target_parameters=[
                "C(Q('Category or Silence'))[T.Classic]:Q('Dancing habit [0-7]_centered')",
                "C(Q('Category or Silence'))[T.Classic]:Q('Musical skill [0-7]_centered')",
                "C(Q('Category or Silence'))[T.Happy]",
                "Q('Median Unscaled Force [% MVC]')",
            ],
        ),

        # ── CMC_Extensor_mean_beta ── Level 2: Perceived Category + interactions + covariates
        statistics.PowerConfig(
            dependent_var="CMC_Extensor_mean_beta",
            comp_lvl=2,
            n_segments=1,
            target_parameters=[
                "C(Q('Perceived Category'))[T.Happy]",
                "C(Q('Perceived Category'))[T.Happy]:Q('Musical skill [0-7]_centered')",
                "C(Q('Perceived Category'))[T.Sad]:Q('Dancing habit [0-7]_centered')",
                "Q('Dancing habit [0-7]_centered')",
                "Q('Median Unscaled Force [% MVC]')",
            ],
        ),

        # ── CMC_Extensor_mean_beta ── Level 3: Perceived Category = Happy + Musical skill interaction
        statistics.PowerConfig(
            dependent_var="CMC_Extensor_mean_beta",
            comp_lvl=3,
            n_segments=1,
            target_parameters=[
                "C(Q('Perceived Category'))[T.Happy]",
                "C(Q('Perceived Category'))[T.Happy]:Q('Musical skill [0-7]_centered')",
            ],
        ),

        # ── CMC_Extensor_mean_gamma ── Level 1: Category or Silence + Force covariate
        statistics.PowerConfig(
            dependent_var="CMC_Extensor_mean_gamma",
            comp_lvl=1,
            n_segments=1,
            target_parameters=[
                "C(Q('Category or Silence'))[T.Happy]",
                "Q('Median Unscaled Force [% MVC]')",
            ],
        ),

        # ── CMC_Extensor_mean_gamma ── Level 2: Perceived Category + interactions + covariates
        statistics.PowerConfig(
            dependent_var="CMC_Extensor_mean_gamma",
            comp_lvl=2,
            n_segments=1,
            target_parameters=[
                "C(Q('Perceived Category'))[T.Happy]",
                "C(Q('Perceived Category'))[T.Happy]:Q('Musical skill [0-7]_centered')",
                "Q('Dancing habit [0-7]_centered')",
                "Q('Median Unscaled Force [% MVC]')",
            ],
        ),

        # ── CMC_Flexor_max_beta ── Level 1: Category + Groovy + interactions + covariates
        statistics.PowerConfig(
            dependent_var="CMC_Flexor_max_beta",
            comp_lvl=1,
            n_segments=1,
            target_parameters=[
                "C(Q('Category or Silence'))[T.Classic]:Q('Dancing habit [0-7]_centered')",
                "C(Q('Category or Silence'))[T.Groovy]",
                "C(Q('Category or Silence'))[T.Happy]",
                "C(Q('Category or Silence'))[T.Happy]:Q('Dancing habit [0-7]_centered')",
                "Q('Median Unscaled Force [% MVC]')",
                "Q('Musical skill [0-7]_centered')",
            ],
        ),

        # ── CMC_Flexor_max_beta ── Level 2: Perceived Category (Happy + Groovy) + interactions
        statistics.PowerConfig(
            dependent_var="CMC_Flexor_max_beta",
            comp_lvl=2,
            n_segments=1,
            target_parameters=[
                "C(Q('Perceived Category'))[T.Groovy]",
                "C(Q('Perceived Category'))[T.Groovy]:Q('Musical skill [0-7]_centered')",
                "C(Q('Perceived Category'))[T.Happy]",
                "C(Q('Perceived Category'))[T.Happy]:Q('Musical skill [0-7]_centered')",
            ],
        ),

        # ── CMC_Flexor_max_beta ── Level 3: Perceived Category (Happy + Groovy) + interactions
        statistics.PowerConfig(
            dependent_var="CMC_Flexor_max_beta",
            comp_lvl=3,
            n_segments=1,
            target_parameters=[
                "C(Q('Perceived Category'))[T.Groovy]",
                "C(Q('Perceived Category'))[T.Groovy]:Q('Musical skill [0-7]_centered')",
                "C(Q('Perceived Category'))[T.Happy]",
                "C(Q('Perceived Category'))[T.Happy]:Q('Musical skill [0-7]_centered')",
            ],
        ),

        # ── CMC_Flexor_max_gamma ── Level 1: Category or Silence = Happy
        statistics.PowerConfig(
            dependent_var="CMC_Flexor_max_gamma",
            comp_lvl=1,
            n_segments=1,
            target_parameters=[
                "C(Q('Category or Silence'))[T.Happy]",
            ],
        ),

        # ── CMC_Flexor_mean_gamma ── Level 1: Category + Groovy + interactions + Force
        statistics.PowerConfig(
            dependent_var="CMC_Flexor_mean_gamma",
            comp_lvl=1,
            n_segments=1,
            target_parameters=[
                "C(Q('Category or Silence'))[T.Classic]:Q('Dancing habit [0-7]_centered')",
                "C(Q('Category or Silence'))[T.Groovy]",
                "C(Q('Category or Silence'))[T.Happy]",
                "C(Q('Category or Silence'))[T.Happy]:Q('Dancing habit [0-7]_centered')",
                "Q('Median Unscaled Force [% MVC]')",
            ],
        ),

        # ── CMC_Flexor_mean_gamma ── Level 2: Perceived Category + interactions
        statistics.PowerConfig(
            dependent_var="CMC_Flexor_mean_gamma",
            comp_lvl=2,
            n_segments=1,
            target_parameters=[
                "C(Q('Perceived Category'))[T.Groovy]:Q('Dancing habit [0-7]_centered')",
                "C(Q('Perceived Category'))[T.Happy]",
                "C(Q('Perceived Category'))[T.Happy]:Q('Musical skill [0-7]_centered')",
            ],
        ),

        # ── PSD_emg_1_flexor_Global_all ── Level 2: Perceived Category + interactions + Force
        statistics.PowerConfig(
            dependent_var="PSD_emg_1_flexor_Global_all",
            comp_lvl=2,
            n_segments=1,
            target_parameters=[
                "C(Q('Perceived Category'))[T.Groovy]:Q('Musical skill [0-7]_centered')",
                "C(Q('Perceived Category'))[T.Happy]",
                "C(Q('Perceived Category'))[T.Happy]:Q('Musical skill [0-7]_centered')",
                "Q('Median Scaled Force [0-1]')",
                "Q('Median Unscaled Force [% MVC]')",
            ],
        ),

        # ── PSD_emg_1_flexor_Global_all ── Level 3: interactions + Force
        statistics.PowerConfig(
            dependent_var="PSD_emg_1_flexor_Global_all",
            comp_lvl=3,
            n_segments=1,
            target_parameters=[
                "C(Q('Perceived Category'))[T.Groovy]:Q('Musical skill [0-7]_centered')",
                "C(Q('Perceived Category'))[T.Happy]:Q('Musical skill [0-7]_centered')",
                "Q('Median Unscaled Force [% MVC]')",
            ],
        ),

        # ── PSD_emg_2_extensor_Global_all ── Level 0: Force covariates only
        statistics.PowerConfig(
            dependent_var="PSD_emg_2_extensor_Global_all",
            comp_lvl=0,
            n_segments=1,
            target_parameters=[
                "Q('Median Scaled Force [0-1]')",
                "Q('Median Unscaled Force [% MVC]')",
            ],
        ),

        # ── PSD_emg_2_extensor_Global_all ── Level 1: Force covariates only
        statistics.PowerConfig(
            dependent_var="PSD_emg_2_extensor_Global_all",
            comp_lvl=1,
            n_segments=1,
            target_parameters=[
                "Q('Median Scaled Force [0-1]')",
                "Q('Median Unscaled Force [% MVC]')",
            ],
        ),

        # ── PSD_emg_2_extensor_Global_all ── Level 2: Perceived Category + interactions + Force
        statistics.PowerConfig(
            dependent_var="PSD_emg_2_extensor_Global_all",
            comp_lvl=2,
            n_segments=1,
            target_parameters=[
                "C(Q('Perceived Category'))[T.Happy]",
                "C(Q('Perceived Category'))[T.Happy]:Q('Musical skill [0-7]_centered')",
                "Q('Median Scaled Force [0-1]')",
                "Q('Median Unscaled Force [% MVC]')",
            ],
        ),

        # ── PSD_emg_2_extensor_Global_all ── Level 3: Perceived Category + interactions + Force
        statistics.PowerConfig(
            dependent_var="PSD_emg_2_extensor_Global_all",
            comp_lvl=3,
            n_segments=1,
            target_parameters=[
                "C(Q('Perceived Category'))[T.Groovy]:Q('Musical skill [0-7]_centered')",
                "C(Q('Perceived Category'))[T.Happy]",
                "C(Q('Perceived Category'))[T.Happy]:Q('Musical skill [0-7]_centered')",
                "C(Q('Perceived Category'))[T.Sad]:Q('Musical skill [0-7]_centered')",
                "Q('Median Scaled Force [0-1]')",
                "Q('Median Unscaled Force [% MVC]')",
            ],
        ),
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
        if plot_cmc_lineplots or show_cmc_scatterplots or conduct_analysis:
            all_subject_data_frame = pd.read_csv(filemgmt.most_recent_file(FEATURE_OUTPUT_DATA, ".csv", [f"Combined Statistics {n_within_trial_segments}seg"]))  # pd.read_csv(FEATURE_OUTPUT_DATA / "statistics_temp_eeg_alpha.csv")  # set to None to run computation
            print(f"Fetching existing statistical data frame for n_segments {n_within_trial_segments}...\n")





        #######################################################
        ################ DATA EXPLORATION #####################
        #######################################################

        if plot_cmc_lineplots or show_cmc_scatterplots:
            all_subject_data_frame = data_analysis.create_trial_bins(
                df=all_subject_data_frame,
                columns_to_bin=list(add_bin_features_dict.keys()),
                n_bins_dict=add_bin_features_dict,

            )



        if plot_cmc_lineplots:
            # plot CMC per subject and category:
            for muscle in ['Flexor', 'Extensor']:
                # loop over categories (new plot per category)
                for category_column in cmc_plot_categories:
                    visualizations.plot_cmc_lineplots_per_category(
                        all_subject_data_frame, category_column, muscle,
                        cmc_operator='mean', n_within_trial_segments=n_within_trial_segments,
                        cmc_plot_min=.7, cmc_plot_max=1.0, n_yticks=4,
                        include_std_dev=True, std_dev_factor=.2, colormap='tab20',
                        show_significance_threshold=True,
                        save_dir=STATISTICS_OUTPUT_DATA if save_cmc_lineplots else None,
                        alpha=.1,
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
                                            cmap='tab20',
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
                                                             model_type='OLS', hidden=not show_effect_plots)
                visualizations.plot_hypothesis_forest_mosaic(results_frame, cmc_extensor_hypotheses, output_dir = STATISTICS_OUTPUT_DATA,
                                                             file_identifier_suffix=f"H1_Extensor_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                                             model_type='OLS', hidden=not show_effect_plots)
                visualizations.plot_hypothesis_forest_mosaic(results_frame, psd_hypotheses, output_dir = STATISTICS_OUTPUT_DATA,
                                                             file_identifier_suffix=f"H2-5_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                                             model_type='OLS', hidden=not show_effect_plots)
                visualizations.plot_hypothesis_forest_mosaic(results_frame, validation_hypotheses, output_dir = STATISTICS_OUTPUT_DATA,
                                                             file_identifier_suffix=f"VAL_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                                             model_type='OLS', hidden=not show_effect_plots)

            # LME Results:
            if render_lme_effect_plots:
                visualizations.plot_hypothesis_forest_mosaic(results_frame, cmc_flexor_hypotheses, output_dir=STATISTICS_OUTPUT_DATA,
                                                             file_identifier_suffix=f"H1_Flexor_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                                             model_type='LME', hidden=not show_effect_plots)

                visualizations.plot_hypothesis_forest_mosaic(results_frame, cmc_extensor_hypotheses, output_dir = STATISTICS_OUTPUT_DATA,
                                                             file_identifier_suffix=f"H1_Extensor_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                                             model_type='LME', hidden=not show_effect_plots)

                visualizations.plot_hypothesis_forest_mosaic(results_frame, psd_hypotheses, output_dir = STATISTICS_OUTPUT_DATA,
                                                             file_identifier_suffix=f"H2-5_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                                             model_type='LME', hidden=not show_effect_plots)

                visualizations.plot_hypothesis_forest_mosaic(results_frame, validation_hypotheses, output_dir = STATISTICS_OUTPUT_DATA,
                                                             file_identifier_suffix=f"VAL_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                                             model_type='LME', hidden=not show_effect_plots)




            # n. segment-dependent vars:
            results_frame['Time Resolution'] = 40 / n_within_trial_segments
            diagnostics_frame['Time Resolution'] = 40 / n_within_trial_segments
            results_frame['N. Segments'] = n_within_trial_segments
            diagnostics_frame['N. Segments'] = n_within_trial_segments

            # save:
            all_time_resolutions_results_list.append(results_frame.copy())
            all_diagnostics_list.append(diagnostics_frame.copy())




    #############################################################
    ################ SAVE ALL TIME-RES. RESULTS #################
    #############################################################

    if conduct_analysis:  # only then save results, otherwise will be loaded below
        # save all time resolutions' results
        all_time_resolutions_results_frame = pd.concat(all_time_resolutions_results_list)
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

        for parameter, comparison_level in parameter_comp_lvl_tuples_to_plot_across_time:
            for hypotheses in [cmc_flexor_hypotheses, cmc_extensor_hypotheses, psd_hypotheses]:
                visualizations.plot_time_resolution_forest_mosaic(
                    result_frame=all_time_resolutions_results_frame, hypotheses=hypotheses,
                    parameter=parameter, comparison_level=comparison_level,
                    model_type='LME', output_dir=STATISTICS_OUTPUT_DATA,
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