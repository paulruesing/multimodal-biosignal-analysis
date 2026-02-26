from pathlib import Path
import pandas as pd
import numpy as np
from typing import Literal
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

import src.pipeline.signal_features as features
import src.pipeline.statistical_modelling as statistics
import src.pipeline.data_integration as data_integration
import src.pipeline.data_analysis as data_analysis
import src.pipeline.visualizations as visualizations
from src.pipeline.visualizations import EEG_CHANNEL_IND_DICT, EEG_CHANNELS_BY_AREA
import src.utils.file_management as filemgmt







if __name__ == '__main__':
    ######## PREPARATION #########
    ROOT = Path().resolve().parent
    DATA = ROOT / "data"
    OUTPUT = ROOT / 'output'
    STUDY_PLOTS = OUTPUT / 'plots' / 'data_analysis_plots'
    EXPERIMENT_DATA = DATA / "experiment_results"
    FEATURE_OUTPUT_DATA = DATA / "precomputed_features"
    STATISTICS_OUTPUT_DATA = OUTPUT / 'statistical_analysis'


    ##### PARAMETERS
    # feature extraction:
    current_subject_count: int = 11  # todo: increase to consider new data!
    overwrite: bool = True  # ALWAYS compute new frame
    n_within_trial_segments_list: list[int] = [3, 4, 5, 10, 20]  # of ~40sec trials

    # data exploration:
    add_bin_features_dict: dict[str, int] = {'Median Force Level [0-1]': 4, 'Familiarity [0-7]': 5,
                                               'GSR [0-3.3]': 4, 'Trial ID': 4}
    # creates bin index for per subject values to be used as categories (new col. will be named "{OLD_COL}_bin")
    cmc_plot_categories: list[str] = ['Subject ID', 'Category or Silence', 'Familiarity [0-7]_bin', 'Trial ID_bin',
                                      'Median Force Level [0-1]_bin', 'GSR [0-3.3]_bin']
    # subject wise line plots:
    plot_cmc_lineplots: bool = False
    save_cmc_lineplots: bool = True
    # compound scatters:
    show_cmc_scatterplots: bool = False
    save_cmc_scatterplots: bool = True


    ## Statistical Analysis
    conduct_analysis: bool = True

    # plotting:
    render_ols_effect_plots: bool = False
    render_lme_effect_plots: bool = False
    show_effect_plots: bool = False  # otherwise either of the above will be hidden

    # comparison levels:
    lvl_inds_to_include: list[int] = [0, 1, 2, 3]  # defined below
    lvls_to_include: list[str] = [f"lvl_{lvl_ind}" for lvl_ind in lvl_inds_to_include]

    # across time resolution comparison:
    parameter_comp_lvl_tuples_to_plot_across_time: list[tuple[str, int]] = [

        # ── TIER 1: Primary category effects ──────────────────────────────────
        # L0: only level where Music Listening exists
        # ('Music Listening[T.True]', 0),  # ALWAYS INSIGNIFICANT

        # L1: only level with category-vs-silence contrast
        ('Category or Silence[T.Happy]', 1),
        # ('Category or Silence[T.Groovy]', 1),  # ALWAYS INSIGNIFICANT
        # ('Category or Silence[T.Sad]', 1),  # ALWAYS INSIGNIFICANT

        # L2: exists at 2 & 3; L2 has marginally more unadjusted-sig hits
        # across hypotheses AND is the richest model without physiological noise
        ('Perceived Category[T.Happy]', 2),
        ('Perceived Category[T.Groovy]', 2),  # only for beta peak at 2 secs
        # ('Perceived Category[T.Sad]', 2),  # ALWAYS INSIGNIFICANT

        # ── TIER 2: Temporal structure ────────────────────────────────────────
        # L0: Trial ID present at all levels; cleanest/most powerful at L0
        # ('Trial ID', 0),  # ALWAYS INSIGNIFICANT

        # L2: Segment ID present at all levels; slightly more sig at L2
        # because Perceived Category controls between-category drift
        ('Segment ID', 2),

        # ── TIER 3: Subjective moderators ─────────────────────────────────────
        # L2: only exists at 2 & 3; L2 has better power (fewer covariates)
        # ('Liking [0-7]', 2),  # ALWAYS INSIGNIFICANT
        # ('Familiarity [0-7]', 2),  # ALWAYS INSIGNIFICANT

        # ── TIER 4: Individual difference moderators ──────────────────────────
        # L0: present at all levels, but main effect clearest/strongest at L0
        # (H3/H4/H5 all show p<1e-7 unadjusted at L0)
        # ('Musical skill [0-7]_centered', 0),  # ALWAYS INSIGNIFICANT
        ('Dancing habit [0-7]_centered', 0),

        # ── TIER 5: Interaction effects ───────────────────────────────────────
        # L2: exists at 2 & 3; highest unadjusted sig count at L2
        ('Perceived Category[T.Happy]:Musical skill [0-7]_centered', 2),
        ('Perceived Category[T.Groovy]:Dancing habit [0-7]_centered', 2),

        # L1: these interactions only exist at L1
        # ('Category or Silence[T.Happy]:Musical skill [0-7]_centered', 1),  # ALWAYS INSIGNIFICANT
        # ('Category or Silence[T.Groovy]:Dancing habit [0-7]_centered', 1),  # ALWAYS INSIGNIFICANT

        # ── TIER 6: Physiological covariates ─────────────────────────────────
        # L1: Force present everywhere; L1 pairs it with category context
        # and has the highest effective N (127 vs 106 at L2/L3)
        ('Median Force Level [0-1]', 1),

        # L3: only level where physiological regressors appear
        # ('GSR [0-3.3]', 3),  # ALWAYS INSIGNIFICANT (todo: anymore...)
        # ('Median HRV [sec]', 3),  # ALWAYS INSIGNIFICANT
    ]

    #######################################################
    #######################################################
    ################ LOOP OVER N_SEGMENTS #################
    #######################################################
    #######################################################

    all_time_resolutions_results_list: list[pd.DataFrame] = []
    for n_within_trial_segments in n_within_trial_segments_list:  # analyse different time resolutions

        # comparison levels:
        level_definitions = [
            # Level 0 — all data, music vs. silence
            {
                'df_filter': None,
                'condition_vars': {'Music Listening': 'categorical'},
                'reference_categories': {'Music Listening': 'False'},
                'explanatory_vars': ['Median Force Level [0-1]'] + (
                    ['Trial ID'] if n_within_trial_segments == 1 else ['Trial ID', 'Segment ID']),
                'moderation_pairs': [('Music Listening', 'Musical skill [0-7]_centered'),
                                     ('Music Listening', 'Dancing habit [0-7]_centered')],
            },
            # Level 1 — all data, music category or silence
            {
                'df_filter': None,
                'condition_vars': {'Category or Silence': 'categorical'},
                'reference_categories': {'Category or Silence': 'Silence'},
                'explanatory_vars': ['Median Force Level [0-1]'] + (
                    ['Trial ID'] if n_within_trial_segments == 1 else ['Trial ID', 'Segment ID']),
                'moderation_pairs': [('Category or Silence', 'Musical skill [0-7]_centered'),
                                     ('Category or Silence', 'Dancing habit [0-7]_centered')],
            },
            # Level 2 — music trials only, subjective features
            {
                'df_filter': lambda df: df.loc[df['Music Listening']],
                'condition_vars': {'Perceived Category': 'categorical', 'Familiarity [0-7]': 'ordinal'},
                'reference_categories': {'Perceived Category': 'Classic'},
                'explanatory_vars': ['Median Force Level [0-1]', 'Liking [0-7]'] + (
                    ['Trial ID'] if n_within_trial_segments == 1 else ['Trial ID', 'Segment ID']),
                'moderation_pairs': [('Perceived Category', 'Musical skill [0-7]_centered'),
                                     ('Perceived Category', 'Dancing habit [0-7]_centered')],
            },
            # Level 3 — music trials only, add emotional state + biomarkers
            {
                'df_filter': lambda df: df.loc[df['Music Listening']],
                'condition_vars': {'Perceived Category': 'categorical', 'Familiarity [0-7]': 'ordinal'},
                'reference_categories': {'Perceived Category': 'Classic'},
                'explanatory_vars': ['Median Force Level [0-1]', 'Liking [0-7]',
                                     'Emotional State [0-7]', 'Median Heart Rate [bpm]', 'Median HRV [sec]',
                                     'GSR [0-3.3]'] + (
                                        ['Trial ID'] if n_within_trial_segments == 1 else ['Trial ID', 'Segment ID']),
                'moderation_pairs': [('Perceived Category', 'Musical skill [0-7]_centered'),
                                     ('Perceived Category', 'Dancing habit [0-7]_centered')],
            },
            # Level 4 — music trials only, add objective music features
            {
                'df_filter': lambda df: df.loc[df['Music Listening']],
                'condition_vars': {'Perceived Category': 'categorical', 'Familiarity [0-7]': 'ordinal'},
                'reference_categories': {'Perceived Category': 'Classic'},
                'explanatory_vars': ['Median Force Level [0-1]', 'Liking [0-7]',
                                     'Emotional State [0-7]', 'Median Heart Rate [bpm]', 'Median HRV [sec]',
                                     'GSR [0-3.3]',
                                     'BPM_manual', 'Spectral Flux Mean', 'Spectral Centroid Mean',
                                     'IOI Variance Coeff', 'Syncopation Ratio'] + (
                                        ['Trial ID'] if n_within_trial_segments == 1 else ['Trial ID', 'Segment ID']),
                'moderation_pairs': [('Perceived Category', 'Musical skill [0-7]_centered'),
                                     ('Perceived Category', 'Dancing habit [0-7]_centered')],
            },
        ]


        #####################################################
        ################ FEATURE EXTRACTION #################
        #####################################################

        ### TRY FETCHING EXISTING FRAME
        if not overwrite:
            try:
                all_subject_data_frame = pd.read_csv(filemgmt.most_recent_file(FEATURE_OUTPUT_DATA, ".csv", [f"Combined Statistics {n_within_trial_segments}seg"]))  # pd.read_csv(FEATURE_OUTPUT_DATA / "statistics_temp_eeg_alpha.csv")  # set to None to run computation
                print("Fetching existing statistical data frame...\n")
            except ValueError:
                overwrite = True  # triggers recomputation




        ### COMPUTE NEW STATISTICS FRAME
        if overwrite:  # if no all subject dataframe found
            print("Computing new statistical data frame...\n")
            ### PSD PARAMETERS
            # average over below bands and channels (region_label labels the channel group):
            modality_region_channels_band_psd_list: list[tuple[str, str, list[str], str]] = [
                # MODALITY, REGION_LABEL, REGION_CHANNELS, BAND
                ('eeg', 'FC_CP_T',
                 EEG_CHANNELS_BY_AREA['Fronto-Central'] + EEG_CHANNELS_BY_AREA['Centro-Parietal'] + EEG_CHANNELS_BY_AREA['Temporal'],
                'theta'),  # H2
                ('eeg', 'F_C',
                 EEG_CHANNELS_BY_AREA['Frontal'] + EEG_CHANNELS_BY_AREA['Central'], 'beta'),  # H3
                ('eeg', 'P_PO',
                 EEG_CHANNELS_BY_AREA['Parietal'] + EEG_CHANNELS_BY_AREA['Parieto-Occipital'], 'alpha'),  # H4
                ('eeg', 'Global', None, 'gamma'),
                ('emg_1_flexor', 'Global', None, 'all'),
                ('emg_2_extensor', 'Global', None, 'all')
            ]
            # select target band from freq_band_psd_per_segment_dict from
            #   - EMG   -> 'slow', 'fast'
            #   - EEG
            #       -> 'delta' (not sufficient frequencies)
            #       -> 'theta' (beat perception, entrainment)
            #       -> 'alpha' (auditory attention)
            #       -> 'beta' (motor control)
            #       -> 'gamma'

            # window lenghts:
            psd_time_window_size_sec = .25
            psd_is_log_scaled: bool = True  # define whether PSD was log scaled during feature extraction




            ### CMC PARAMETERS
            muscle_operator_band_cmc_list: list[tuple[str, str, str]] = [
                # MUSCLE, OPERATOR (max / mean / median), BAND
                ('Flexor', 'max', 'beta'),
                ('Flexor', 'max', 'gamma'),
                ('Flexor', 'mean', 'beta'),
                ('Flexor', 'mean', 'gamma'),
                ('Extensor', 'max', 'beta'),
                ('Extensor', 'max', 'gamma'),
                ('Extensor', 'mean', 'beta'),
                ('Extensor', 'mean', 'gamma'),
            ]
            # select target band from freq_band_psd_per_segment_dict from
            #   - EMG   -> 'slow', 'fast'
            #   - EEG
            #       -> 'delta' (not sufficient frequencies)
            #       -> 'theta' (beat perception, entrainment)
            #       -> 'alpha' (auditory attention)
            #       -> 'beta' (motor control)
            #       -> 'gamma'

            # window lengths:
            cmc_time_window_size_sec = 2.0





            ### DATA AGGREGATION PARAMETERS
            # how many segments:
            print(f"Will split 45sec trial into {n_within_trial_segments} segments (each ~{45/n_within_trial_segments:.1f}sec)")
            # below two are transformed via key-word (modality) search in columns:
            modalities_to_standardize_per_subject: list[str] = []  #['PSD', 'Force']  # will change that columns
            modalities_to_center_over_subjects: list[str] = [
                'Listening habit [0-3]', 'Dancing habit [0-7]',
                'Athleticism [0-7]', 'Musical skill [0-7]']  # will add new columns (COLUMN + _centered)

            music_features_to_fetch = ('BPM_manual', 'Spectral Flux Mean', 'Spectral Centroid Mean', 'IOI Variance Coeff',
                                       'Syncopation Ratio')







            ########### ITERATE OVER ALL PARTICIPANTS ###########
            all_subject_data_frame = pd.DataFrame(columns=['Subject ID'])
            for subject_ind in range(current_subject_count):
                print("\n")
                print("-" * 100)
                print(f"------------     Aggregating data for subject\t\t{subject_ind:02}     ------------- ")
                print("-" * 100)

                # dependent directories:
                subject_psd_save_dir = FEATURE_OUTPUT_DATA / f"subject_{subject_ind:02}"
                subject_cmc_save_dir = FEATURE_OUTPUT_DATA / f"subject_{subject_ind:02}"
                subject_experiment_data_dir = EXPERIMENT_DATA / f"subject_{subject_ind:02}"








                ### IMPORT LOG AND SERIAL DATAFRAMES
                log_df = data_integration.fetch_enriched_log_frame(subject_experiment_data_dir)
                serial_df = data_integration.fetch_enriched_serial_frame(subject_experiment_data_dir)
                # make time-zone aware:
                log_df.index = data_analysis.make_timezone_aware(log_df.index)
                serial_df.index = data_analysis.make_timezone_aware(serial_df.index)

                # slice towards qtc measurements:
                qtc_start, qtc_end = data_integration.get_qtc_measurement_start_end(log_df, False)
                sliced_log_df = log_df[qtc_start:qtc_end]
                sliced_serial_df = serial_df[qtc_start:qtc_end]











                ### DERIVE SEGMENT TIMESPANS
                # trial start end times:
                # (contains default cut-off seconds to prevent transients!)
                trial_start_end_dict = data_integration.get_all_task_start_ends(log_df, 'dict')
                # convert into segment start end times:
                seg_starts = []; seg_ends = []; seg_ids = []
                for start, end in trial_start_end_dict.values():
                    seg_starts_range = pd.date_range(start, end, periods=n_within_trial_segments+1, inclusive='both')
                    for ind, seg_start in enumerate(seg_starts_range.values[:-1]):
                        seg_ids.append(ind)
                        seg_starts.append(seg_start)
                        seg_ends.append(seg_starts_range.values[ind+1])






                ### PREPARE DATAFRAME
                single_subject_data_frame = pd.DataFrame(index=range(len(seg_starts)))







                ### IMPORT AND AGGREGATE PSD DATA PER HYPOTHESIS (modality_region_channels_band_psd_list)
                # loop over configurations:
                for modality, region_label, channels, band in modality_region_channels_band_psd_list:
                    # import PSD:
                    psd_spectrograms, psd_times, psd_freqs = features.fetch_stored_spectrograms(
                        subject_psd_save_dir, modality='PSD', file_identifier=modality)
                    #   -> shape: (n_windows, n_freqs, n_channels), (n_windows), (n_freqs)
                    # convert PSD second time-centers into timestamps:
                    psd_timestamps = data_analysis.add_time_index(
                        start_timestamp=qtc_start + pd.Timedelta(seconds=psd_time_window_size_sec / 2),
                        end_timestamp=qtc_end - pd.Timedelta(seconds=psd_time_window_size_sec / 2),
                        n_timesteps=len(psd_times)
                    )
                    psd_timestamps = data_analysis.make_timezone_aware(psd_timestamps)

                    # takes shapes (n_windows, n_freqs, n_channels)
                    psd_aggregated = features.aggregate_psd_spectrogram(psd_spectrograms, psd_freqs, normalize_mvc=False,
                                                               channel_indices=[EEG_CHANNEL_IND_DICT[ch] for ch in channels] if channels is not None else None,
                                                               is_log_scaled=psd_is_log_scaled, freq_slice=band,
                                                               aggregation_ops=[('mean', 1),   # mean within freq band
                                                                                # mean over EEG channels, max over EMG ones:
                                                                                ('mean' if 'eeg' in modality else 'max', 1),
                                                                                ],)
                    # returns shape (n_windows, )

                    # split per segment:
                    psd_per_segment = data_analysis.apply_window_operator(
                        window_timestamps=seg_starts,
                        window_timestamps_ends=seg_ends,

                        target_array=psd_aggregated,
                        target_timestamps=psd_timestamps,

                        operation='mean',
                        axis=0,  # time axis
                    )  # (n_trials, )

                    # save to dataframe:
                    single_subject_data_frame[f"PSD_{modality}_{region_label}_{band}"] = psd_per_segment








                    ### IMPORT AND AGGREGATE CMC DATA PER HYPOTHESIS (muscle_operator_band_cmc_list)
                    # loop over configurations:
                    for muscle, operator, band in muscle_operator_band_cmc_list:
                        # import CMC:
                        cmc_spectrograms, cmc_times, cmc_freqs = features.fetch_stored_spectrograms(
                            subject_cmc_save_dir, modality='CMC', file_identifier=muscle)
                        #   -> shape: (n_windows, n_freqs, n_channels), (n_windows), (n_freqs)
                        # convert CMC second time-centers into timestamps:
                        cmc_timestamps = data_analysis.add_time_index(
                            start_timestamp=qtc_start + pd.Timedelta(seconds=cmc_time_window_size_sec / 2),
                            end_timestamp=qtc_end - pd.Timedelta(seconds=cmc_time_window_size_sec / 2),
                            n_timesteps=len(cmc_times)
                        )
                        cmc_timestamps = data_analysis.make_timezone_aware(cmc_timestamps)

                        # takes shapes (n_windows, n_freqs, n_channels)
                        cmc_aggregated = features.aggregate_psd_spectrogram(cmc_spectrograms, cmc_freqs, normalize_mvc=False,
                                                                   is_log_scaled=False, freq_slice=band,
                                                                   aggregation_ops=[('max', 1),  # mean within freq band
                                                                                    # either take peak or average over channels
                                                                                    (operator, 1),
                                                                                    ]
                                                                   )
                        # returns shape (n_windows, )

                        # split per segment:
                        cmc_per_segment = data_analysis.apply_window_operator(
                            window_timestamps=seg_starts,
                            window_timestamps_ends=seg_ends,

                            target_array=cmc_aggregated,
                            target_timestamps=cmc_timestamps,

                            operation='mean',
                            axis=0,  # time axis
                        )  # (n_trials, )

                        # save to dataframe:
                        single_subject_data_frame[f"CMC_{muscle}_{operator}_{band}"] = cmc_per_segment






                ### SUBJECT-LEVEL VARIABLE AGGREGATION
                subject_level_data_dict = data_integration.fetch_personal_data(subject_experiment_data_dir)







                ### INDEPENDENT VARIABLE AGGREGATION
                # force level:
                force_level_per_segment = data_analysis.apply_window_operator(
                    window_timestamps=seg_starts,
                    window_timestamps_ends=seg_ends,
                    target_array=sliced_serial_df['Task-wise Scaled Force'],
                    operation='median',
                    axis=0,  # time axis
                )

                # trial category:
                song_id_per_segment = data_analysis.apply_window_operator(
                    window_timestamps=seg_starts,
                    window_timestamps_ends=seg_ends,
                    target_array=log_df['Song ID'],
                    operation='mode',  # most common string value
                    axis=0,  # time axis
                )
                silence_id_per_segment = data_analysis.apply_window_operator(
                    window_timestamps=seg_starts,
                    window_timestamps_ends=seg_ends,
                    target_array=log_df['Silence ID'],
                    operation='mode',  # most common string value
                    axis=0,  # time axis
                )
                is_music_trial = [not pd.isna(song_id) and pd.isna(silence_id) for song_id, silence_id in zip(song_id_per_segment, silence_id_per_segment)]

                # trial ID:
                trial_id_per_segment = data_analysis.apply_window_operator(
                    window_timestamps=seg_starts,
                    window_timestamps_ends=seg_ends,
                    target_array=log_df['Trial ID'],
                    operation='mode', axis=0,
                )

                # musical features:
                music_feature_tuples = [
                    data_integration.fetch_music_features(log_df, trial_id=trial_id,
                                                          features_to_return=music_features_to_fetch) for trial_id in trial_id_per_segment
                ]


                # song category:
                song_category_per_segment = data_analysis.apply_window_operator(
                    window_timestamps=seg_starts,
                    window_timestamps_ends=seg_ends,
                    target_array=log_df['Perceived Category'],
                    operation='mode', axis=0,
                )
                # song category with missing values replaced with: 'Silence'
                category_or_silence = pd.Series(song_category_per_segment).fillna('Silence')

                song_familiarity_per_segment = data_analysis.apply_window_operator(
                    window_timestamps=seg_starts,
                    window_timestamps_ends=seg_ends,
                    target_array=log_df['Familiarity'],
                    operation='mode', axis=0,
                )
                emotional_state_per_segment = data_analysis.apply_window_operator(
                    window_timestamps=seg_starts,
                    window_timestamps_ends=seg_ends,
                    target_array=log_df['Emotional State'],
                    operation='mode', axis=0,
                )

                song_liking_per_segment = data_analysis.apply_window_operator(
                    window_timestamps=seg_starts,
                    window_timestamps_ends=seg_ends,
                    target_array=log_df['Liking'],
                    operation='mode', axis=0,
                )

                task_frequency_per_segment = data_analysis.apply_window_operator(
                    window_timestamps=seg_starts,
                    window_timestamps_ends=seg_ends,
                    target_array=log_df['Task Frequency'],
                    operation='mode', axis=0,
                )

                # heart rate::
                bpm_per_segment = data_analysis.apply_window_operator(
                    window_timestamps=seg_starts,
                    window_timestamps_ends=seg_ends,
                    target_array=sliced_serial_df['bpm'],
                    operation='median',
                    axis=0,  # time axis
                )

                # heart rate variability:
                hrv_per_segment = data_analysis.apply_window_operator(
                    window_timestamps=seg_starts,
                    window_timestamps_ends=seg_ends,
                    target_array=sliced_serial_df['hrv'],
                    operation='median',
                    axis=0,  # time axis
                )

                # galvanic skin response:
                gsr_per_segment = data_analysis.apply_window_operator(
                    window_timestamps=seg_starts,
                    window_timestamps_ends=seg_ends,
                    target_array=sliced_serial_df['gsr'],
                    operation='median',
                    axis=0,
                )



                ### APPEND TO PER SUBJECT DATAFRAME
                # append to subject df:
                for column_name, data in [
                    ('Subject ID', [subject_ind] * len(seg_starts)),
                    ('Trial ID', trial_id_per_segment),
                    ('Music Listening', is_music_trial),
                    ('Median Force Level [0-1]', force_level_per_segment),
                    ('Task Frequency', task_frequency_per_segment),
                    ('Emotional State [0-7]', emotional_state_per_segment),
                    ('Median Heart Rate [bpm]', bpm_per_segment),
                    ('Median HRV [sec]', hrv_per_segment),
                    ('GSR [0-3.3]', gsr_per_segment),
                    # music features:
                    ('Perceived Category', song_category_per_segment),
                    ('Category or Silence', category_or_silence),
                    ('Liking [0-7]', song_liking_per_segment),
                    ('Familiarity [0-7]', song_familiarity_per_segment),
                    (list(music_features_to_fetch), music_feature_tuples),

                    # add segment counter:
                    ('Segment ID', seg_ids),

                    # subject-level features (moderating vars.)
                    ('Listening habit [0-3]', [subject_level_data_dict['Listening habit [0-3]']] * len(seg_starts)),
                    ('Dancing habit [0-7]', [subject_level_data_dict['Dancing habit']] * len(seg_starts)),
                    ('Athleticism [0-7]', [subject_level_data_dict['Athleticism']] * len(seg_starts)),
                    ('Musical skill [0-7]', [subject_level_data_dict['Musical skill']] * len(seg_starts)),

                ]:
                    single_subject_data_frame[column_name] = data

                # standardize:
                for modality in modalities_to_standardize_per_subject:
                    for column in [c for c in single_subject_data_frame.columns if modality in c]:
                        # columns to standardize:
                        print("Standardizing statistics for: ", column)
                        single_subject_data_frame[column] = single_subject_data_frame[column].transform(lambda x: (x - x.mean()) / x.std())




                ### CONCAT WITH ALL SUBJECT DATAFRAME
                all_subject_data_frame = pd.concat([all_subject_data_frame, single_subject_data_frame], axis=0)





            ######### CENTERING OVER ALL SUBJECTS #########
            # centering:
            for modality in modalities_to_center_over_subjects:
                for column in [c for c in all_subject_data_frame.columns if modality in c]:
                    # columns to standardize:
                    print("Centering statistics for: ", column)
                    all_subject_data_frame[f"{column}_centered"] = all_subject_data_frame[column].transform(
                        lambda x: (x - x.mean()))
                    print(f"Added new column: {column}_centered")




            ######### SAVE COMBINED STATISTICS #########
            all_subject_data_frame.to_csv(FEATURE_OUTPUT_DATA / filemgmt.file_title(f"Combined Statistics {int(n_within_trial_segments)}seg", ".csv"), index=False)





        #######################################################
        ################ DATA EXPLORATION #####################
        #######################################################

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
                                            save_dir=STATISTICS_OUTPUT_DATA, cmap='tab20',
                                            )




        #######################################################
        ################ STATISTICAL ANALYSIS #################
        #######################################################

        if conduct_analysis:
            # Store all results and diagnostics for summary tables
            all_model_results = []
            all_diagnostics = []

            for hypothesis, dependent_variable in [
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
                ('VALIDATION: EMG Extensor PSD Increases with Force', 'PSD_emg_2_extensor_Global_all'), ]:

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

            # Generate all summary tables with one function call
            statistics.generate_all_summary_tables(
                results_df=pd.DataFrame(all_model_results),
                output_dir=STATISTICS_OUTPUT_DATA,
                diagnostics_df=pd.DataFrame(all_diagnostics),
                file_identifier=f"{n_within_trial_segments}seg_{"".join(lvls_to_include)}",
                generate_per_level_tables=False,
                generate_thematic_tables=False,
            )



            # ============================================================================
            # Subject-specific analysis
            # ============================================================================

            # After all models are run
            filemgmt.assert_dir(STATISTICS_OUTPUT_DATA / f"subject_level_effects_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}")
            statistics.create_subject_effect_summary(
                all_model_results=all_model_results,
                original_data=all_subject_data_frame,
                output_dir=STATISTICS_OUTPUT_DATA / f"subject_level_effects_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}"
            )

            # ============================================================================
            # Plotting
            # ============================================================================

            results_frame = pd.DataFrame(all_model_results)

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


            # append to all time-resolution results:
            results_frame['Time Resolution'] = 40 / n_within_trial_segments
            all_time_resolutions_results_list.append(results_frame.copy())


    if conduct_analysis:
        ### ANALYSIS ACROSS TIME-SCALES
        all_time_resolutions_results_frame = pd.concat(all_time_resolutions_results_list)

        for parameter, comparison_level in parameter_comp_lvl_tuples_to_plot_across_time:
            for hypotheses in [cmc_flexor_hypotheses, cmc_extensor_hypotheses, psd_hypotheses]:
                visualizations.plot_time_resolution_forest_mosaic(
                    result_frame=all_time_resolutions_results_frame, hypotheses=hypotheses,
                    parameter=parameter, comparison_level=comparison_level,
                    model_type='LME', output_dir=STATISTICS_OUTPUT_DATA,
                )

