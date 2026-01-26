import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Callable, Literal, Tuple

from pandas import DatetimeIndex

import src.pipeline.signal_features as features
import src.pipeline.preprocessing as preprocessing
import src.pipeline.visualizations as visualizations
import src.pipeline.data_analysis as data_analysis
import src.pipeline.data_integration as data_integration
import src.utils.file_management as filemgmt

from src.pipeline.preprocessing import BiosignalPreprocessor
from src.pipeline.visualizations import EEG_CHANNEL_IND_DICT, EEG_CHANNELS_BY_AREA


if __name__=="__main__":
    ######## PREPARATION #########
    ROOT = Path().resolve().parent
    DATA = ROOT / "data"
    OUTPUT = ROOT / 'output'
    STUDY_PLOTS = OUTPUT / 'plots' / 'data_analysis_plots'
    QTC_DATA = DATA / "qtc_data"
    EXPERIMENT_DATA = DATA / "experiment_results"
    FEATURE_OUTPUT_DATA = DATA / "precomputed_features"






    ### WORKFLOW CONTROL
    # select subject:
    subject_ind: int = 4

    # EEG / EMG import behaviour:
    load_only_first_n_seconds: int | None = None  # if None, loads full data
    # select eeg subset (otherwise set to empty list):
    eeg_channel_subset = []  # EEG_CHANNELS_BY_AREA['Fronto-Central'] + EEG_CHANNELS_BY_AREA['Central'] + EEG_CHANNELS_BY_AREA['Centro-Parietal'] + EEG_CHANNELS_BY_AREA['Temporal']
    bad_channel_treatment: Literal['None', 'Zero'] = 'Zero'  # leads to setting to zero

    # PSD computation:
    do_compute_psd: bool = True
    psd_window_size_sec: float = .25  # -> 4 Hz resolution
    save_psd: bool = True

    # CMC computation:
    do_compute_cmc: bool = False
    cmc_window_size_sec: float = 2.0
    save_cmc: bool = True

    # Heart Rate and HRV computation:
    # todo: implement!






    ### Dependent Variables:
    # corresponding output dirs:
    if save_psd:
        subject_psd_save_dir = FEATURE_OUTPUT_DATA /  f"subject_{subject_ind:02}"; filemgmt.assert_dir(subject_psd_save_dir)
    else: subject_psd_save_dir = None
    if save_cmc:
        subject_cmc_save_dir = FEATURE_OUTPUT_DATA /  f"subject_{subject_ind:02}"; filemgmt.assert_dir(subject_cmc_save_dir)
    else: subject_cmc_save_dir = None
    subject_qtc_data_dir = QTC_DATA / f"subject_{subject_ind:02}"
    subject_plot_dir = STUDY_PLOTS / f"subject_{subject_ind:02}"; filemgmt.assert_dir(subject_plot_dir)
    subject_experiment_data = EXPERIMENT_DATA / f"subject_{subject_ind:02}"  # should have folders experiment_logs/, serial_measurements/, song_000/, ...

    # eeg channel subset indices for slicing:
    if len(eeg_channel_subset) > 0:
        eeg_channel_subset_inds = [EEG_CHANNEL_IND_DICT[ch] for ch in eeg_channel_subset]
        print(f"Reducing EEG dataset to {len(eeg_channel_subset)} channels: {eeg_channel_subset}\n")
    else: eeg_channel_subset_inds = None












    ######## WORKFLOW #########
    ### Data Import
    # load experiment files:
    log_frame = data_integration.fetch_enriched_log_frame(subject_experiment_data)
    qtc_start, qtc_end = data_integration.get_qtc_measurement_start_end(log_frame)

    # accuracies per trial:
    """
    accuracy_per_trial_dict, questionnaire_per_trial_dict = data_integration.fetch_all_accuracies_and_questionnaires(
        experiment_data_dir=subject_experiment_data,
        max_song_ind=log_frame['Song ID'].max().astype(int),
        max_silence_ind=log_frame['Silence ID'].max().astype(int), )"""

    # serial measurements:
    serial_frame = data_integration.fetch_serial_measurements(subject_experiment_data, set_time_index=True)
    within_qtc_serial_frame = serial_frame[qtc_start:qtc_end]

    # load qtc files:
    eeg_array, eeg_config = preprocessing.import_npy_with_config(f"sub_{subject_ind:02}_eeg", subject_qtc_data_dir,
                                                                 load_only_first_n_seconds=load_only_first_n_seconds,
                                                                 channel_subset_inds=eeg_channel_subset_inds)
    emg_flexor_array, emg_flexor_config = preprocessing.import_npy_with_config(f"sub_{subject_ind:02}_emg_1_flexor",
                                                                               subject_qtc_data_dir,
                                                                               load_only_first_n_seconds=load_only_first_n_seconds)
    emg_extensor_array, emg_extensor_config = preprocessing.import_npy_with_config(
        f"sub_{subject_ind:02}_emg_2_extensor", subject_qtc_data_dir,
        load_only_first_n_seconds=load_only_first_n_seconds)




    ### Time Slicing
    # select relevant log frame part:
    within_qtc_log_frame = log_frame[qtc_start:qtc_end]  # time slice

    if load_only_first_n_seconds is not None:
        within_qtc_log_frame = within_qtc_log_frame[:qtc_start + pd.Timedelta(seconds=load_only_first_n_seconds)]
        within_qtc_serial_frame = within_qtc_serial_frame[:qtc_start + pd.Timedelta(seconds=load_only_first_n_seconds)]

    # phase_series for visualization:
    phase_series = within_qtc_log_frame.Phase
    phases = phase_series.unique()
    print(f"Phases within data snippet: {phases}\n")




    ### PSD Computation
    if do_compute_psd:
        # compute EEG psd:
        eeg_psd, eeg_psd_times, eeg_psd_freqs = features.multitaper_psd(input_array=eeg_array,
                                                                        sampling_freq=eeg_config['sampling_freq'], nw=3,
                                                                        window_length_sec=psd_window_size_sec,
                                                                        overlap_frac=0.5, axis=0,
                                                                        plot_result=False,
                                                                        apply_log_scale=True,
                                                                        psd_save_dir=subject_cmc_save_dir,
                                                                        psd_file_suffix="eeg")
        visualizations.plot_spectrogram(np.mean(eeg_psd, axis=2), eeg_psd_times, eeg_psd_freqs,
                                        channels=eeg_channel_subset,
                                        log_scale=False,
                                        frequency_range=(0, 100), phase_series=phase_series,
                                        save_dir=subject_plot_dir,
                                        title="EEG Phase Spectrogram")

        # EMG1 psd:
        emg1_f_psd, emg1_f_psd_times, emg1_f_psd_freqs = features.multitaper_psd(input_array=emg_flexor_array,
                                                                                 sampling_freq=emg_flexor_config[
                                                                                     'sampling_freq'], nw=3,
                                                                                 window_length_sec=psd_window_size_sec,
                                                                                 overlap_frac=0.5, axis=0,
                                                                                 plot_result=False,
                                                                                 apply_log_scale=True,
                                                                                 psd_save_dir=subject_cmc_save_dir,
                                                                                 psd_file_suffix="emg_1_flexor")
        visualizations.plot_spectrogram(np.mean(emg1_f_psd, axis=2), emg1_f_psd_times, emg1_f_psd_freqs,
                                        frequency_range=(20, 250), phase_series=phase_series,
                                        save_dir=subject_plot_dir,
                                        log_scale=False,
                                        title="EMG Flexor Phase Spectrogram")

        # EMG2 psd:
        emg2_e_psd, emg2_e_psd_times, emg2_e_psd_freqs = features.multitaper_psd(input_array=emg_extensor_array,
                                                                                 sampling_freq=emg_extensor_config[
                                                                                     'sampling_freq'], nw=3,
                                                                                 window_length_sec=psd_window_size_sec,
                                                                                 overlap_frac=0.5, axis=0,
                                                                                 apply_log_scale=True,
                                                                                 plot_result=False,
                                                                                 psd_save_dir=subject_cmc_save_dir,
                                                                                 psd_file_suffix="emg_2_extensor")
        visualizations.plot_spectrogram(np.mean(emg2_e_psd, axis=2), emg2_e_psd_times, emg2_e_psd_freqs,
                                        log_scale=False,
                                        frequency_range=(20, 250), phase_series=phase_series,
                                        save_dir=subject_plot_dir,
                                        title="EMG Extensor Phase Spectrogram")


        # test loading:
        spec, times, freqs = features.fetch_stored_psd(subject_cmc_save_dir, "emg_1_flexor")
        print(spec.shape)
        print(times.shape)
        print(freqs.shape)
        quit()



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
        phase_per_eeg_psd = data_analysis.apply_window_operator(eeg_psd_times, psd_window_size_sec, within_qtc_log_frame['Phase'], 'mode')
        rmse_per_eeg_psd = data_analysis.apply_window_operator(eeg_psd_times, psd_window_size_sec, within_qtc_log_frame['Task RMSE'], 'mean')
        category_per_eeg_psd = data_analysis.apply_window_operator(eeg_psd_times, psd_window_size_sec, within_qtc_log_frame['Music Category'], 'mode')
        force_per_eeg_psd = data_analysis.apply_window_operator(eeg_psd_times, psd_window_size_sec, within_qtc_serial_frame['fsr'], 'mean')

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

