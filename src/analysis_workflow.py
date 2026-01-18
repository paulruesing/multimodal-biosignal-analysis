import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Callable, Literal, Tuple

from pandas import DatetimeIndex

import src.pipeline.signal_features as features
import src.pipeline.preprocessing as preprocessing
from src.pipeline.preprocessing import BiosignalPreprocessor
import src.pipeline.visualizations as visualizations
import src.utils.file_management as filemgmt
from src.pipeline.visualizations import EEG_CHANNEL_IND_DICT, EEG_CHANNELS_BY_AREA

######## WORKFLOW #########
if __name__=="__main__":
    ROOT = Path().resolve().parent
    OUTPUT = ROOT / 'output'
    STUDY_PLOTS = OUTPUT / 'plots' / 'data_analysis_plots'
    QTC_DATA = ROOT / "data" / "qtc_data"
    EXPERIMENT_DATA = ROOT / "data" / "experiment_results"

    ### WORKFLOW CONTROL
    subject_plot_dir = STUDY_PLOTS / "subject_01"

    # EEG / EMG import behaviour:
    subject_qtc_data_dir = QTC_DATA / "subject_01"
    retrieve_latest_config: bool = True
    load_only_first_n_seconds: int | None = 1200
    eeg_channel_subset = EEG_CHANNELS_BY_AREA['Fronto-Central'] + EEG_CHANNELS_BY_AREA['Central'] + EEG_CHANNELS_BY_AREA['Centro-Parietal'] + EEG_CHANNELS_BY_AREA['Temporal']
    eeg_channel_subset_inds = [EEG_CHANNEL_IND_DICT[ch] for ch in eeg_channel_subset]
    print(f"Reducing EEG dataset to {len(eeg_channel_subset)} channels: {eeg_channel_subset}\n")

    # experiment results import behaviour:
    subject_experiment_data = EXPERIMENT_DATA / "subject_01"  # should have folders experiment_logs/, serial_measurements/, song_000/, ...

    # analysis behaviour:
    bad_channel_treatment: Literal['None', 'Zero'] = 'Zero'  # leads to setting to zero
    eeg_do_psd_log_transform: bool = False
    do_compute_psd: bool = True
    do_compute_cmc: bool = False
    psd_window_size_sec: float = .2
    cmc_window_size_sec: float = 2.0


    ### WORKFLOW
    # load experiment files:
    log_frame = preprocessing.fetch_experiment_log(subject_experiment_data)
    # log columns: ['Time', 'Music', 'Event', 'Questionnaire']
    serial_frame = preprocessing.fetch_serial_measurements(subject_experiment_data).set_index('Time')
    # serial columns: ['Time', 'fsr', 'ecg', 'gsr']

    # enrich log frame columns based on 'Event' and 'Questionnaire' data:
    log_frame = preprocessing.prepare_log_frame(log_frame, set_time_index=True)

    # select relevant log frame part:
    qtc_start = log_frame.loc[log_frame['Event'] == "Start Trigger"].index.item()
    qtc_end = log_frame.loc[log_frame['Event'] == "Stop Trigger"].index.item()
    print(f"EEG and EMG measurements last from {qtc_start} to {qtc_end}!")
    within_qtc_log_frame = log_frame[qtc_start:qtc_end]  # time slice
    within_qtc_serial_frame = serial_frame[qtc_start:qtc_end]
    if load_only_first_n_seconds is not None:
        within_qtc_log_frame = within_qtc_log_frame[:qtc_start + pd.Timedelta(seconds=load_only_first_n_seconds)]
        within_qtc_serial_frame = within_qtc_serial_frame[:qtc_start + pd.Timedelta(seconds=load_only_first_n_seconds)]

    # phase_series for visualization:
    phase_series = within_qtc_log_frame.Phase
    phases = phase_series.unique()
    print(f"Phases within data snippet: {phases}\n")

    # load qtc files:
    eeg_array, eeg_config = preprocessing.import_npy_with_config("eeg_full", subject_qtc_data_dir, load_only_first_n_seconds=load_only_first_n_seconds,
                                                                 channel_subset_inds=eeg_channel_subset_inds)
    emg_flexor_array, emg_flexor_config = preprocessing.import_npy_with_config("emg_1_flexor", subject_qtc_data_dir, load_only_first_n_seconds=load_only_first_n_seconds)
    emg_extensor_array, emg_extensor_config = preprocessing.import_npy_with_config("emg_2_extensor", subject_qtc_data_dir, load_only_first_n_seconds=load_only_first_n_seconds)



    ### PSD computation:
    if do_compute_psd:
        # compute EEG psd:
        eeg_psd, eeg_psd_times, eeg_psd_freqs = features.multitaper_psd(input_array=eeg_array,
                                                         sampling_freq=eeg_config['sampling_freq'], nw=3,
                                                         window_length_sec=psd_window_size_sec, overlap_frac=0.5, axis=0,
                                                         verbose=True, plot_result=False)
        visualizations.plot_spectrogram(np.mean(eeg_psd, axis=2), eeg_psd_times, eeg_psd_freqs,
                                        channels=eeg_channel_subset,
                                        log_scale=True,
                                        frequency_range=(0, 100), phase_series=phase_series,
                                        save_dir=subject_plot_dir,
                                        title="EEG Phase Spectrogram")

        # EMG1 psd:
        emg1_f_psd, emg1_f_psd_times, emg1_f_psd_freqs = features.multitaper_psd(input_array=emg_flexor_array,
                                                                        sampling_freq=emg_flexor_config['sampling_freq'], nw=3,
                                                                        window_length_sec=psd_window_size_sec, overlap_frac=0.5, axis=0,
                                                                        verbose=True, plot_result=False)
        visualizations.plot_spectrogram(np.mean(emg1_f_psd, axis=2), emg1_f_psd_times, emg1_f_psd_freqs,
                                        log_scale=True,
                                        frequency_range=(10, 250), phase_series=phase_series,
                                        save_dir=subject_plot_dir,
                                        title="EMG Flexor Phase Spectrogram")

        # EMG2 psd:
        emg2_e_psd, emg2_e_psd_times, emg2_e_psd_freqs = features.multitaper_psd(input_array=emg_extensor_array,
                                                                        sampling_freq=emg_extensor_config['sampling_freq'], nw=3,
                                                                        window_length_sec=psd_window_size_sec, overlap_frac=0.5, axis=0,
                                                                        verbose=True, plot_result=False)
        visualizations.plot_spectrogram(np.mean(emg2_e_psd, axis=2), emg2_e_psd_times, emg2_e_psd_freqs,
                                        log_scale=True,
                                        frequency_range=(10, 250), phase_series=phase_series,
                                        save_dir=subject_plot_dir,
                                        title="EMG Extensor Phase Spectrogram")



    ### CMC computation:
    if do_compute_cmc:
        # returns
        #       cmc_values (n_times, n_freqs, n_eeg_channels, n_emg_channels)
        #       time_centers (n_times, )
        #       frequencies (n_freqs, )
        # FLEXOR
        flexor_cmc_values, flexor_time_centers, flexor_freqs = features.multitaper_magnitude_squared_coherence(
            eeg_array, emg_flexor_array, sampling_freq=eeg_config['sampling_freq'], verbose=True,
            window_length_sec=cmc_window_size_sec,  # consider increasing
            overlap_frac=0.5,
            )
        # max. within frequency band:
        flexor_cmc_aggregates_per_band = features.aggregate_spectrogram_over_frequency_band(
            flexor_cmc_values, flexor_freqs, behaviour='max', frequency_axis=1,
            pre_aggregate_axis=(3, 'max'),  # further max. over EMG channels (index 3)
        )
        visualizations.plot_spectrogram(flexor_cmc_aggregates_per_band['beta'],
                                        flexor_time_centers, channels=eeg_channel_subset,
                                        plot_type='time-channel',
                                        phase_series=phase_series,
                                        save_dir=subject_plot_dir,
                                        title=f"Aggregated Flexor CMC (max-pooled) for {'beta'.capitalize()} Band",
                                        cbar_label="Magnitude Squared Coherence"
                                        )

        # EXTENSOR
        extensor_cmc_values, extensor_time_centers, extensor_freqs = features.multitaper_magnitude_squared_coherence(
            eeg_array, emg_extensor_array, sampling_freq=eeg_config['sampling_freq'], verbose=True,
            window_length_sec=cmc_window_size_sec,  # consider increasing
            overlap_frac=0.5,
        )
        # max. within frequency band:
        extensor_cmc_aggregates_per_band = features.aggregate_spectrogram_over_frequency_band(
            extensor_cmc_values, extensor_freqs, behaviour='max', frequency_axis=1,
            pre_aggregate_axis=(3, 'max'),  # further max. over EMG channels (index 3)
        )
        visualizations.plot_spectrogram(extensor_cmc_aggregates_per_band['beta'],
                                        extensor_time_centers, channels=eeg_channel_subset,
                                        plot_type='time-channel',
                                        phase_series=phase_series,
                                        save_dir=subject_plot_dir,
                                        title=f"Aggregated Extensor CMC (max-pooled) for {'beta'.capitalize()} Band",
                                        cbar_label="Magnitude Squared Coherence"
                                        )

        # todo: add other CMC approaches


    ### Statistical Tests
    if do_compute_psd:
        def info_per_window_sec(
                window_time_centers: np.ndarray,
                window_size: float,
                target_series: pd.Series,
                operation: Literal['min', 'max', 'mean', 'median', 'mode', 'std'] = 'mean',
        ) -> list:
            """ Target series needs to have time index (seconds or absolute). """

            # Convert to numeric only for non-mode operations
            if operation != 'mode' and target_series.dtype == 'object':
                target_series = pd.to_numeric(target_series, errors='coerce')

            # derive time in seconds from time index:
            if isinstance(target_series.index, pd.DatetimeIndex):
                time_seconds = (target_series.index - target_series.index[0]).total_seconds().values
            else:
                time_seconds = target_series.index.values.astype(float)

            # Create window boundaries
            starts = window_time_centers - window_size / 2
            ends = window_time_centers + window_size / 2

            # For each time point, find which window it belongs to
            window_indices = np.full(len(time_seconds), np.nan, dtype=float)

            for i, (start, end) in enumerate(zip(starts, ends)):
                mask = (time_seconds >= start) & (time_seconds < end)
                window_indices[mask] = i

            # Create dataframe with both the data and window indices
            df_with_windows = pd.DataFrame({
                'data': target_series.values,
                '_window': window_indices
            })

            # Filter out NaN groups BEFORE groupby
            df_with_windows_filtered = df_with_windows[df_with_windows['_window'].notna()]
            grouped = df_with_windows_filtered.groupby('_window', sort=False)['data']

            # Handle mode separately, use agg for others
            if operation == 'mode':
                result = grouped.apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan)
            else:
                result = grouped.agg(operation)

            all_windows = pd.RangeIndex(len(window_time_centers), name='_window')
            result = result.reindex(all_windows)  # Ensures all windows present
            result = result.fillna(0)  # or ffill(), but now consistent
            return result.tolist()


        def slow_info_per_window_sec(
                window_time_centers: np.ndarray,
                window_size: float,
                target_series: pd.DataFrame,
                operation: Literal['min', 'max', 'mean', 'median', 'mode', 'std'] = 'mean',
        ) -> list:
            """ Target series needs to have time index (seconds or absolute). """
            starts = window_time_centers - window_size / 2
            ends = window_time_centers + window_size / 2

            # derive time in seconds from time index:
            if isinstance(target_series.index, pd.DatetimeIndex):
                time_seconds = (target_series.index - target_series.index[0]).total_seconds().to_series().reset_index(drop=True)
            else:
                time_seconds = target_series.index.to_series().reset_index(drop=True)

            # compute target per window:
            target_list = []
            for ind, (start, end) in enumerate(zip(starts, ends)):
                relevant_slice = target_series.reset_index(drop=True).loc[(time_seconds >= start) & (time_seconds < end)]

                # simple:
                if len(relevant_slice) == 0: target = target_list[ind - 1]  # if no relevant slice within window, take previous
                elif len(relevant_slice) == 1: target = relevant_slice.item()

                #
                elif operation == 'min': target = relevant_slice.min()
                elif operation == 'max': target = relevant_slice.max()
                elif operation == 'mean': target = relevant_slice.mean()
                elif operation == 'median': target = relevant_slice.median()
                elif operation == 'mode': target = relevant_slice.mode().item()
                elif operation == 'std': target = relevant_slice.std()
                else: raise ValueError('Unknown operation {}'.format(operation))

                target_list.append(target)

            return target_list


        phase_per_eeg_psd = info_per_window_sec(eeg_psd_times, psd_window_size_sec, within_qtc_log_frame['Phase'], 'mode')
        rmse_per_eeg_psd = info_per_window_sec(eeg_psd_times, psd_window_size_sec, within_qtc_log_frame['Task RMSE'], 'mean')
        category_per_eeg_psd = info_per_window_sec(eeg_psd_times, psd_window_size_sec, within_qtc_log_frame['Music Category'], 'mode')
        force_per_eeg_psd = info_per_window_sec(eeg_psd_times, psd_window_size_sec, within_qtc_serial_frame['fsr'], 'mean')

        eeg_psd_aggregates_per_band = features.aggregate_spectrogram_over_frequency_band(
            eeg_psd, eeg_psd_freqs, behaviour='mean', frequency_axis=1,
            pre_aggregate_axis=(2, 'mean'),  # further max. over EMG channels (index 3)
        )

        features.compute_feature_mi_importance(np.array([phase_per_eeg_psd, rmse_per_eeg_psd, category_per_eeg_psd, force_per_eeg_psd]).T,
                                               eeg_psd_aggregates_per_band['beta'], feature_labels=['Phase', 'Task RMSE', 'Music Category', 'Force'])


        emg1_psd_aggregates_per_band = features.aggregate_spectrogram_over_frequency_band(
            emg1_f_psd, emg1_f_psd_freqs, behaviour='mean', frequency_axis=1,
            pre_aggregate_axis=(2, 'mean'),  # further max. over EMG channels (index 3)
        )
        features.compute_feature_mi_importance(
            np.array([phase_per_eeg_psd, rmse_per_eeg_psd, category_per_eeg_psd, force_per_eeg_psd]).T,
            emg1_psd_aggregates_per_band['beta'], feature_labels=['Phase', 'Task RMSE', 'Music Category', 'Force'])

