from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import src.pipeline.signal_features as features
import src.pipeline.data_integration as data_integration
import src.pipeline.data_analysis as data_analysis
from src.pipeline.channel_layout import EEG_CHANNEL_IND_DICT, EEG_CHANNELS_BY_AREA
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
    n_within_trial_segments_list: list[int] = [1, 2, 5, 10]  # of ~40sec trials

    # Onset transient exclusion: discard this many seconds from the start of
    # each trial AFTER the latency correction already applied by
    # get_all_task_start_ends (assumed_latency_sec=3.25 s).  Affects all
    # modalities uniformly — PSD/CMC/accuracy/force timestamps stay unchanged;
    # only the segment boundaries shift forward.  Set to 0.0 for the original
    # behaviour.  Accuracy's own 5.5 s warm-up offset is independent and
    # remains in effect regardless of this parameter.
    n_onset_seconds_to_discard: float = 6.5

    # Print per-trial timing breakdown for every subject
    verbose_trial_timing: bool = False

    # Keep these explicit to make verbose timing diagnostics transparent.
    # Values need to match get_task_start_end / get_all_task_start_ends defaults in src/pipeline/data_integration.py
    task_latency_assumption_sec: float = 3.25
    # needs to match assumed_latency_sec default
    task_end_transient_cutoff_sec: float = 2.0
    # needs to match cut_off_sec_to_prevent_transients default





    ######## ITERATE OVER TIME-RESOLUTIONS #########
    for n_within_trial_segments in tqdm(n_within_trial_segments_list, desc='Time-Resolution Outer Loop'):

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
                f"Will split trials into {n_within_trial_segments} segments (each ~{(45 - task_end_transient_cutoff_sec - n_onset_seconds_to_discard) / n_within_trial_segments:.1f}sec)")
            # below two are transformed via key-word (modality) search in columns:
            modalities_to_standardize_per_subject: list[str] = []  # ['PSD', 'Force']  # will change that columns
            modalities_to_center_over_subjects: list[str] = [
                'Liking',
                'Listening habit [0-3]', 'Dancing habit [0-7]',
                'Athleticism [0-7]', 'Musical skill [0-7]']  # will add new columns (COLUMN + _centered)
            modalities_to_square: list[str] = [
                'Liking_centered'
            ]

            music_features_to_fetch = ('BPM_manual', 'Spectral Flux Mean', 'Spectral Centroid Mean', 'IOI Variance Coeff',
                                       'Syncopation Ratio', 'Spectral Flux Std.')

            ########### ITERATE OVER ALL PARTICIPANTS ###########
            all_subject_data_frame = pd.DataFrame(columns=['Subject ID'])
            for subject_ind in tqdm(range(current_subject_count), desc='Subject Inner Loop'):

                print("\n")
                # print("-" * 100)
                # print(f"------------     Aggregating data for subject\t\t{subject_ind:02}     ------------- ")
                # print("-" * 100)

                # dependent directories:
                subject_psd_save_dir = FEATURE_OUTPUT_DATA / f"subject_{subject_ind:02}"
                subject_cmc_save_dir = FEATURE_OUTPUT_DATA / f"subject_{subject_ind:02}"
                subject_experiment_data_dir = EXPERIMENT_DATA / f"subject_{subject_ind:02}"

                ### IMPORT LOG AND SERIAL DATAFRAMES
                log_df = data_integration.fetch_enriched_log_frame(subject_experiment_data_dir, verbose=False)
                serial_df = data_integration.fetch_enriched_serial_frame(subject_experiment_data_dir)
                # make time-zone aware:
                log_df.index = data_analysis.make_timezone_aware(log_df.index)
                serial_df.index = data_analysis.make_timezone_aware(serial_df.index)

                # slice towards qtc measurements:
                qtc_start, qtc_end = data_integration.get_qtc_measurement_start_end(log_df, False)
                sliced_log_df = log_df[qtc_start:qtc_end]
                sliced_serial_df = serial_df[qtc_start:qtc_end]

                if verbose_trial_timing:
                    print(
                        f"  Timing config | latency={task_latency_assumption_sec:.2f}s, "
                        f"task_end_cutoff={task_end_transient_cutoff_sec:.2f}s, "
                        f"extra_onset_discard={n_onset_seconds_to_discard:.2f}s, "
                        f"accuracy_onset_offset={data_integration.TRIAL_ACCURACY_START_OFFSET_SEC:.2f}s"
                    )
                    print(
                        f"  QTC span      | {qtc_start.strftime('%H:%M:%S.%f')[:12]} -> "
                        f"{qtc_end.strftime('%H:%M:%S.%f')[:12]} "
                        f"({(qtc_end - qtc_start).total_seconds():.1f}s)"
                    )

                ### DERIVE SEGMENT TIMESPANS
                # trial start end times:
                # (contains default cut-off seconds to prevent transients!)
                trial_start_end_dict = data_integration.get_all_task_start_ends(
                    log_df,
                    'dict',
                    assumed_latency_sec=task_latency_assumption_sec,
                    cut_off_sec_to_prevent_transients=task_end_transient_cutoff_sec,
                )
                # convert into segment start end times:
                seg_starts = []
                seg_ends = []
                seg_ids = []
                onset_delta = pd.Timedelta(seconds=n_onset_seconds_to_discard)
                for trial_id_key, (start, end) in trial_start_end_dict.items():
                    effective_start = start + onset_delta
                    if effective_start >= end:
                        print(
                            f"  [WARNING] Trial {trial_id_key}: onset discard "
                            f"({n_onset_seconds_to_discard:.1f}s) exceeds trial "
                            f"duration ({(end - start).total_seconds():.1f}s). Skipping."
                        )
                        continue
                    if verbose_trial_timing:
                        try:
                            raw_start, raw_end = data_integration.get_task_start_end(
                                log_df,
                                trial_id=int(trial_id_key),
                                assumed_latency_sec=0.0,
                                cut_off_sec_to_prevent_transients=0.0,
                            )
                            raw_dur = (raw_end - raw_start).total_seconds()
                            corrected_dur = (end - start).total_seconds()
                            effective_dur = (end - effective_start).total_seconds()
                            print(
                                f"  Trial {trial_id_key:>2} raw task span      | "
                                f"{raw_start.strftime('%H:%M:%S.%f')[:12]} -> "
                                f"{raw_end.strftime('%H:%M:%S.%f')[:12]} ({raw_dur:.1f}s)"
                            )
                            print(
                                f"             + latency {task_latency_assumption_sec:.2f}s, "
                                f"- end_cutoff {task_end_transient_cutoff_sec:.2f}s -> "
                                f"{start.strftime('%H:%M:%S.%f')[:12]} -> "
                                f"{end.strftime('%H:%M:%S.%f')[:12]} ({corrected_dur:.1f}s)"
                            )
                            print(
                                f"             + onset_discard {n_onset_seconds_to_discard:.2f}s -> "
                                f"{effective_start.strftime('%H:%M:%S.%f')[:12]} -> "
                                f"{end.strftime('%H:%M:%S.%f')[:12]} ({effective_dur:.1f}s)"
                            )
                        except ValueError:
                            print(
                                f"  Trial {trial_id_key:>2}: verbose raw-span reconstruction failed; "
                                f"using corrected span {start.strftime('%H:%M:%S.%f')[:12]} -> "
                                f"{end.strftime('%H:%M:%S.%f')[:12]}"
                            )
                    # segment boundaries:
                    seg_starts_range = pd.date_range(
                        effective_start, end,
                        periods=n_within_trial_segments + 1, inclusive='both',
                    )
                    for ind, seg_start in enumerate(seg_starts_range.values[:-1]):
                        seg_ids.append(ind)
                        seg_starts.append(pd.Timestamp(seg_start))
                        seg_ends.append(pd.Timestamp(seg_starts_range.values[ind + 1]))
                
                # Ensure seg_starts and seg_ends are timezone-aware for proper comparison with acc_start
                seg_starts = [data_analysis.make_timezone_aware(ts) for ts in seg_starts]
                seg_ends = [data_analysis.make_timezone_aware(ts) for ts in seg_ends]

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
                ### END PSD LOOP

                ### IMPORT AND AGGREGATE CMC DATA PER HYPOTHESIS (muscle_operator_band_cmc_list)
                for muscle, operator, band in muscle_operator_band_cmc_list:
                    # import CMC:
                    cmc_spectrograms, cmc_times, cmc_freqs = features.fetch_stored_spectrograms(
                        subject_cmc_save_dir, modality='CMC', file_identifier=muscle)
                    #   -> shape: (n_windows, n_freqs, n_channels), (n_windows), (n_freqs)

                    # Reconstruct timestamps uniformly across QTC span — identical
                    # to PSD handling; stored cmc_times are nominal and not used directly.
                    cmc_timestamps = data_analysis.add_time_index(
                        start_timestamp=qtc_start + pd.Timedelta(seconds=cmc_time_window_size_sec / 2),
                        end_timestamp=qtc_end - pd.Timedelta(seconds=cmc_time_window_size_sec / 2),
                        n_timesteps=len(cmc_times)
                    )
                    cmc_timestamps = data_analysis.make_timezone_aware(cmc_timestamps)

                    # takes shape (n_windows, n_freqs, n_channels)
                    cmc_aggregated = features.aggregate_psd_spectrogram(
                        cmc_spectrograms, cmc_freqs,
                        normalize_mvc=False,
                        is_log_scaled=False,
                        freq_slice=band,
                        aggregation_ops=[
                            ('max', 1),  # max within freq band
                            (operator, 1),  # mean or max over channels
                        ],
                    )
                    # returns shape (n_windows,)

                    # split per segment:
                    cmc_per_segment = data_analysis.apply_window_operator(
                        window_timestamps=seg_starts,
                        window_timestamps_ends=seg_ends,
                        target_array=cmc_aggregated,
                        target_timestamps=cmc_timestamps,
                        operation='mean',
                        axis=0,
                    )  # (n_segments,)

                    single_subject_data_frame[f"CMC_{muscle}_{operator}_{band}"] = cmc_per_segment
                # END CMC loop

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
                            log_df, trial_id=trial_id, cut_off_sec_to_prevent_transients=0.0,
                            assumed_latency_sec=task_latency_assumption_sec,  # assign explicitly to prevent mismatches
                        )
                    except ValueError: continue

                    # Accuracy sampling starts after a pre-phase (default 5s), so
                    # align samples to [trial_start+offset, trial_end].
                    acc_start = full_start + pd.Timedelta(
                        seconds=data_integration.TRIAL_ACCURACY_START_OFFSET_SEC
                    )
                    if acc_start >= full_end:
                        continue

                    if verbose_trial_timing:
                        print(
                            f"    Accuracy trial {trial_id:>2} base window | "
                            f"{full_start.strftime('%H:%M:%S.%f')[:12]} -> "
                            f"{full_end.strftime('%H:%M:%S.%f')[:12]} "
                            f"({(full_end - full_start).total_seconds():.1f}s)"
                        )
                        print(
                            f"                  + onset_offset {data_integration.TRIAL_ACCURACY_START_OFFSET_SEC:.2f}s -> "
                            f"{acc_start.strftime('%H:%M:%S.%f')[:12]} -> "
                            f"{full_end.strftime('%H:%M:%S.%f')[:12]} "
                            f"({(full_end - acc_start).total_seconds():.1f}s), n_samples={len(accuracy_array)}"
                        )

                    # Assign uniform timestamps over effective accuracy span
                    acc_t_rel = data_integration.build_accuracy_relative_time_axis(
                        n_samples=len(accuracy_array),
                        trial_dur_sec=(full_end - full_start).total_seconds(),
                        start_offset_sec=data_integration.TRIAL_ACCURACY_START_OFFSET_SEC,
                        endpoint=False,
                    )
                    if acc_t_rel.size == 0:
                        continue
                    accuracy_timestamps = full_start + pd.to_timedelta(acc_t_rel, unit='s')
                    acc_max = accuracy_timestamps.max()

                    # Aggregate per segment with clipped windows to include partial overlaps.
                    # This avoids dropping early segments and lets np.nanmean handle missing values.
                    valid_row_indices = []
                    trial_seg_starts = []
                    trial_seg_ends = []
                    for row_idx in row_indices:
                        seg_start = seg_starts[row_idx]
                        seg_end = seg_ends[row_idx]

                        # Skip segments without any overlap with available accuracy timestamps.
                        if seg_end < acc_start or seg_start > acc_max:
                            continue

                        valid_row_indices.append(row_idx)
                        trial_seg_starts.append(max(seg_start, acc_start))
                        trial_seg_ends.append(min(seg_end, acc_max))  # Clip to max accuracy timestamp, not full_end

                    if not valid_row_indices:
                        continue  # No overlap with available accuracy data in this trial

                    if verbose_trial_timing:
                        print(
                            f"                  overlap with segment windows: "
                            f"{len(valid_row_indices)}/{len(row_indices)} segments"
                        )

                    accuracy_agg = np.sqrt(  # since logged accuracy is SQUARED error
                        data_analysis.apply_window_operator(
                            window_timestamps=trial_seg_starts,
                            window_timestamps_ends=trial_seg_ends,
                            target_array=accuracy_array,
                            target_timestamps=accuracy_timestamps,
                            operation='mean',
                            axis=0,
                        ).astype(float))  # shape: (n_trial_segments,)

                    for local_idx, row_idx in enumerate(valid_row_indices):
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
                    ('Emotional_State', emotional_state_per_segment),
                    ('Median_Heart_Rate', bpm_per_segment),
                    ('Median_HRV', hrv_per_segment),
                    ('GSR', gsr_per_segment),
                    # music features:
                    ('Perceived Category', song_category_per_segment),
                    ('Category or Silence', category_or_silence),
                    ('Liking', song_liking_per_segment),
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

            for modality in modalities_to_square:
                for column in [c for c in all_subject_data_frame.columns if modality in c]:
                    print("Squaring statistics for: ", column)
                    all_subject_data_frame[f"{column}_squared"] = (
                        all_subject_data_frame[column].astype(float) ** 2
                    )
                    print(f"Added new column: {column}_squared")

            ######### SAVE COMBINED STATISTICS #########
            all_subject_data_frame.to_csv(
                FEATURE_OUTPUT_DATA / filemgmt.file_title(f"Combined Statistics {int(n_within_trial_segments)}seg", ".csv"),
                index=False)