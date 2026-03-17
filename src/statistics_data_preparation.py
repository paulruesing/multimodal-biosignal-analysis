from pathlib import Path
import pandas as pd
import numpy as np
import src.pipeline.signal_features as features
import src.pipeline.data_integration as data_integration
import src.pipeline.data_analysis as data_analysis
from src.pipeline.visualizations import EEG_CHANNEL_IND_DICT, EEG_CHANNELS_BY_AREA
import src.utils.file_management as filemgmt


if __name__ == "__main__":
    ######## PREPARATION #########
    ROOT = Path().resolve().parent
    DATA = ROOT / "data"
    OUTPUT = ROOT / 'output'
    STUDY_PLOTS = OUTPUT / 'plots' / 'data_analysis_plots'
    EXPERIMENT_DATA = DATA / "experiment_results"
    FEATURE_OUTPUT_DATA = DATA / "precomputed_features"
    STATISTICS_OUTPUT_DATA = OUTPUT / 'statistics_omnibus_testing'


    # statistical frame parameters:
    current_subject_count: int = 12
    overwrite: bool = True  # True: ALWAYS compute new frame
    n_within_trial_segments_list: list[int] = [1, 2, 3, 4, 5, 8, 10, 20]  # of ~40sec trials





    ######## ITERATE OVER TIME-RESOLUTIONS #########
    for n_within_trial_segments in n_within_trial_segments_list:

        ### TRY FETCHING EXISTING FRAME
        recompute = False
        if not overwrite:
            try:
                all_subject_data_frame = pd.read_csv(filemgmt.most_recent_file(FEATURE_OUTPUT_DATA, ".csv",
                                                                               [f"Combined Statistics {n_within_trial_segments}seg"]))  # pd.read_csv(FEATURE_OUTPUT_DATA / "statistics_temp_eeg_alpha.csv")  # set to None to run computation
                print(f"Statistical data frame for n_segments {n_within_trial_segments} already exists!\n")
            except ValueError:
                recompute = True  # triggers recomputation




        ### COMPUTE NEW STATISTICS FRAME
        if overwrite or recompute:  # if no all subject dataframe found
            print("Computing new statistical data frame...\n")
            ### PSD PARAMETERS
            # average over below bands and channels (region_label labels the channel group):
            modality_region_channels_band_psd_list: list[tuple[str, str, list[str], str]] = [
                # MODALITY, REGION_LABEL, REGION_CHANNELS, BAND
                ('eeg', 'FC_CP_T',
                 EEG_CHANNELS_BY_AREA['Fronto-Central'] + EEG_CHANNELS_BY_AREA['Centro-Parietal'] + EEG_CHANNELS_BY_AREA[
                     'Temporal'],
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
            print(
                f"Will split 45sec trial into {n_within_trial_segments} segments (each ~{40 / n_within_trial_segments:.1f}sec)")
            # below two are transformed via key-word (modality) search in columns:
            modalities_to_standardize_per_subject: list[str] = []  # ['PSD', 'Force']  # will change that columns
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
                seg_starts = [];
                seg_ends = [];
                seg_ids = []
                for start, end in trial_start_end_dict.values():
                    seg_starts_range = pd.date_range(start, end, periods=n_within_trial_segments + 1, inclusive='both')
                    for ind, seg_start in enumerate(seg_starts_range.values[:-1]):
                        seg_ids.append(ind)
                        seg_starts.append(seg_start)
                        seg_ends.append(seg_starts_range.values[ind + 1])

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
                                                                        channel_indices=[EEG_CHANNEL_IND_DICT[ch] for ch in
                                                                                         channels] if channels is not None else None,
                                                                        is_log_scaled=psd_is_log_scaled, freq_slice=band,
                                                                        aggregation_ops=[('mean', 1),
                                                                                         # mean within freq band
                                                                                         # mean over EEG channels, max over EMG ones:
                                                                                         ('mean' if 'eeg' in modality else 'max',
                                                                                          1),
                                                                                         ], )
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
                        cmc_aggregated = features.aggregate_psd_spectrogram(cmc_spectrograms, cmc_freqs,
                                                                            normalize_mvc=False,
                                                                            is_log_scaled=False, freq_slice=band,
                                                                            aggregation_ops=[('max', 1),
                                                                                             # mean within freq band
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
                scaled_force_per_segment = data_analysis.apply_window_operator(
                    window_timestamps=seg_starts,
                    window_timestamps_ends=seg_ends,
                    target_array=sliced_serial_df['Task-wise Scaled Force'],
                    operation='median',
                    axis=0,  # time axis
                )

                # force level:
                force_per_segment = data_analysis.apply_window_operator(
                    window_timestamps=seg_starts,
                    window_timestamps_ends=seg_ends,
                    target_array=sliced_serial_df['Unscaled Force [% MVC]'],
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
                is_music_trial = [not pd.isna(song_id) and pd.isna(silence_id) for song_id, silence_id in
                                  zip(song_id_per_segment, silence_id_per_segment)]

                # trial ID:
                trial_id_per_segment = data_analysis.apply_window_operator(
                    window_timestamps=seg_starts,
                    window_timestamps_ends=seg_ends,
                    target_array=log_df['Trial ID'],
                    operation='mode', axis=0,
                )

                ### FETCH AND SEGMENT TASK ACCURACY
                # accuracy_array has no timestamps; assign uniform timestamps over the full
                # (no-cutoff) trial span so that apply_window_operator can exclude transient zone.
                # Group row-indices by trial_id for efficient per-trial processing
                trial_to_row_indices: dict[int, list[int]] = {}
                for row_idx, trial_id in enumerate(trial_id_per_segment):
                    if not pd.isna(trial_id):
                        trial_to_row_indices.setdefault(int(trial_id), []).append(row_idx)

                accuracy_per_segment: list[float] = [float('nan')] * len(seg_starts)

                for trial_id, row_indices in trial_to_row_indices.items():

                    # Fetch raw accuracy array (no timestamps, spans full trial incl. transients)
                    accuracy_array = data_integration.fetch_trial_accuracy(
                        subject_experiment_data_dir,
                        log_df=log_df,
                        trial_id=trial_id,
                        error_handling='continue',
                        verbose=True,
                    )
                    if accuracy_array is None:
                        print(f"[WARNING] Couldn't find trial accuracy for trial_id {trial_id}!")
                        continue  # leave NaN for all segments of this trial

                    # Get full trial span without cut-off to correctly anchor accuracy timestamps;
                    # log_df.index is already timezone-aware at this point in the pipeline
                    try:
                        full_start, full_end = data_integration.get_task_start_end(
                            log_df, trial_id=trial_id, cut_off_sec_to_prevent_transients=0.0
                        )
                    except ValueError: continue

                    # Assign uniform timestamps: assume taps are evenly distributed over full span
                    accuracy_timestamps = data_analysis.add_time_index(
                        start_timestamp=full_start,
                        end_timestamp=full_end,
                        n_timesteps=len(accuracy_array),
                    )

                    # Aggregate per segment using cutoff-aware windows;
                    # taps in the transient zone fall outside all windows and are silently excluded
                    trial_seg_starts = [seg_starts[i] for i in row_indices]
                    trial_seg_ends = [seg_ends[i] for i in row_indices]

                    accuracy_agg = np.sqrt(  # since logged accuracy is SQUARED error
                        data_analysis.apply_window_operator(
                            window_timestamps=trial_seg_starts,
                            window_timestamps_ends=trial_seg_ends,
                            target_array=accuracy_array,
                            target_timestamps=accuracy_timestamps,
                            operation='mean',
                            axis=0,
                        ).astype(float))  # shape: (n_trial_segments,)

                    for local_idx, row_idx in enumerate(row_indices):
                        val = accuracy_agg[local_idx]
                        accuracy_per_segment[row_idx] = float(val) if not pd.isna(val) else float('nan')

                # musical features:
                music_feature_tuples = [
                    data_integration.fetch_music_features(log_df, trial_id=trial_id,
                                                          features_to_return=music_features_to_fetch) for trial_id in
                    trial_id_per_segment
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
                    ('Median Scaled Force [0-1]', scaled_force_per_segment),
                    ('Median Unscaled Force [% MVC]', force_per_segment),
                    ('Task Frequency', task_frequency_per_segment),
                    ('Emotional_State_[0-7]', emotional_state_per_segment),
                    ('Median_Heart_Rate_[bpm]', bpm_per_segment),
                    ('Median_HRV_[sec]', hrv_per_segment),
                    ('GSR_[0-3.3]', gsr_per_segment),
                    # music features:
                    ('Perceived Category', song_category_per_segment),
                    ('Category or Silence', category_or_silence),
                    ('Liking [0-7]', song_liking_per_segment),
                    ('Familiarity [0-7]', song_familiarity_per_segment),
                    (list(music_features_to_fetch), music_feature_tuples),

                    # add segment counter:
                    ('Segment ID', seg_ids),

                    # accuracy:
                    ('RMS_Accuracy', accuracy_per_segment),

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
                        single_subject_data_frame[column] = single_subject_data_frame[column].transform(
                            lambda x: (x - x.mean()) / x.std())

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
            all_subject_data_frame.to_csv(
                FEATURE_OUTPUT_DATA / filemgmt.file_title(f"Combined Statistics {int(n_within_trial_segments)}seg", ".csv"),
                index=False)