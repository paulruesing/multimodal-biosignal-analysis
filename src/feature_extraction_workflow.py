import numpy as np
import pandas as pd
import json
import gc
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Callable, Literal, Tuple
from tqdm import tqdm

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
    for subject_ind in [8]:  # only new
        handedness: Literal['left', 'right'] = 'right' if subject_ind != 3 else 'left'


        # EEG / EMG import behaviour:
        load_only_first_n_seconds: int | None = None  # if None, loads full data
        bad_channel_treatment: Literal['None', 'Zero'] = 'Zero'  # leads to setting to zero


        # PSD computation:
        do_compute_psd: bool = True  #(subject_ind == 6)  # only for subject 6
        fetch_precomputed_psd: bool = False
        psd_window_size_sec: float = .25  # -> 4 Hz resolution
        save_psd: bool = True
        plot_psd_results: bool = True  # False for workflow behavior


        # CMC computation:
        do_compute_cmc: bool = True
        only_extensor: bool = False
        fetch_precomputed_cmc: bool = False
        cmc_eeg_channel_subset = [  # will be mirrored for left-handed subject
            'C5', 'C3', 'C1',  # channels around C3
            'FC5', 'FC3', 'FC1', 'F3',  # channels around FC3
            'CP5', 'CP3', 'CP1', 'P3',  # channels around CP3
        ]
        cmc_line_visualization_channels = cmc_eeg_channel_subset  # should be included in the above
        cmc_window_size_sec: float = 2.0
        cmc_window_overlap_ratio: float = 0.5
        cmc_independence_threshold_alpha: float = .2  # significance level from which to derive confidence treshold
        filter_significant_cmc: bool = False  # filter out all CMCs lower than confidence threshold
        cmc_jackknife_alpha: float = .05
        save_cmc: bool = True
        plot_cmc_results: bool = False  # False for workflow behavior


        # Compute Heart Rate (HR), HR Variability and Task-wise Scaled Force -> aka. "Enriched Serial Frame"
        compute_enriched_serial_frame: bool = True
        heart_refractory_period: str = '300ms'
        min_bpm: float = 30.0
        max_bpm: float = 200.0
        max_hrv_seconds: float = 0.3
        output_smoothing_window_sec: float = 2.5
        save_enriched_serial_frame: bool = True




        ### Dependent Variables:
        if handedness == 'left':
            cmc_eeg_channel_subset = features.mirror_eeg_channel_list(cmc_eeg_channel_subset, input_is_left=True)
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
        subject_serial_dir = subject_experiment_data / "serial_measurements"  # to eventually store enriched serial frame














        ######## WORKFLOW #########
        ### Data Import
        # load experiment files:
        log_frame = data_integration.fetch_enriched_log_frame(subject_experiment_data)
        qtc_start, qtc_end = data_integration.get_qtc_measurement_start_end(log_frame)

        # serial measurements:
        serial_frame = data_integration.fetch_serial_measurements(subject_experiment_data, set_time_index=True)
        within_qtc_serial_frame = serial_frame[qtc_start:qtc_end]

        # load qtc files:
        if do_compute_psd or do_compute_cmc:
            eeg_array, eeg_config = preprocessing.import_npy_with_config(f"sub_{subject_ind:02}_eeg", subject_qtc_data_dir,
                                                                         load_only_first_n_seconds=load_only_first_n_seconds)
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
        if do_compute_psd or fetch_precomputed_psd:
            if do_compute_psd:
                # compute EEG psd:
                eeg_psd, eeg_psd_times, eeg_psd_freqs = features.multitaper_psd(input_array=eeg_array,
                                                                                sampling_freq=eeg_config['sampling_freq'], nw=3,
                                                                                window_length_sec=psd_window_size_sec,
                                                                                overlap_frac=0.5, axis=0,
                                                                                plot_result=False,
                                                                                apply_log_scale=True,
                                                                                psd_save_dir=subject_psd_save_dir,
                                                                                psd_file_suffix="eeg")
            else:
                eeg_psd, eeg_psd_times, eeg_psd_freqs = features.fetch_stored_spectrograms(subject_psd_save_dir,
                                                                                           "PSD", "eeg")

            if plot_psd_results:
                visualizations.plot_spectrogram(np.mean(eeg_psd, axis=2), eeg_psd_times, eeg_psd_freqs,
                                                is_log_scale=True, apply_log_scale=False,
                                                frequency_range=(0, 100), phase_series=phase_series,
                                                save_dir=subject_plot_dir,
                                                title="EEG Phase Spectrogram")

            # EMG1 psd:
            if do_compute_psd:
                emg1_f_psd, emg1_f_psd_times, emg1_f_psd_freqs = features.multitaper_psd(input_array=emg_flexor_array,
                                                                                         sampling_freq=emg_flexor_config[
                                                                                             'sampling_freq'], nw=3,
                                                                                         window_length_sec=psd_window_size_sec,
                                                                                         overlap_frac=0.5, axis=0,
                                                                                         plot_result=False,
                                                                                         apply_log_scale=True,
                                                                                         psd_save_dir=subject_psd_save_dir,
                                                                                         psd_file_suffix="emg_1_flexor")

            else:
                emg1_f_psd, emg1_f_psd_times, emg1_f_psd_freqs = features.fetch_stored_spectrograms(subject_psd_save_dir,
                                                                                           "PSD", "emg_1_flexor")
            if plot_psd_results:
                visualizations.plot_spectrogram(np.mean(emg1_f_psd, axis=2), emg1_f_psd_times, emg1_f_psd_freqs,
                                                frequency_range=(20, 250), phase_series=phase_series,
                                                save_dir=subject_plot_dir, is_log_scale=True, apply_log_scale=False,
                                                title="EMG Flexor Phase Spectrogram")

            if do_compute_psd:
                # EMG2 psd:
                emg2_e_psd, emg2_e_psd_times, emg2_e_psd_freqs = features.multitaper_psd(input_array=emg_extensor_array,
                                                                                         sampling_freq=emg_extensor_config[
                                                                                             'sampling_freq'], nw=3,
                                                                                         window_length_sec=psd_window_size_sec,
                                                                                         overlap_frac=0.5, axis=0,
                                                                                         apply_log_scale=True,
                                                                                         plot_result=False,
                                                                                         psd_save_dir=subject_psd_save_dir,
                                                                                         psd_file_suffix="emg_2_extensor")
            else:
                emg2_e_psd, emg2_e_psd_times, emg2_e_psd_freqs = features.fetch_stored_spectrograms(subject_psd_save_dir,
                                                                                           "PSD", "emg_2_extensor")

            if plot_psd_results:
                visualizations.plot_spectrogram(np.mean(emg2_e_psd, axis=2), emg2_e_psd_times, emg2_e_psd_freqs,
                                                is_log_scale=True, apply_log_scale=False,
                                                frequency_range=(20, 250), phase_series=phase_series,
                                                save_dir=subject_plot_dir,
                                                title="EMG Extensor Phase Spectrogram")

            del eeg_psd, emg1_f_psd, emg2_e_psd
            gc.collect()
            """# test loading:
            spec, times, freqs = features.fetch_stored_spectrograms(subject_cmc_save_dir, "PSD", "emg_1_flexor")
            print(spec.shape)
            print(times.shape)
            print(freqs.shape)"""










        ### CMC computation:
        if do_compute_cmc or fetch_precomputed_cmc:

            ####### FLEXOR #######
            ## Computation
            if not fetch_precomputed_cmc and not only_extensor:

                flexor_cmc_values, flexor_cmc_values_lower, flexor_cmc_values_upper, flexor_time_centers, flexor_freqs = features.compute_task_wise_aggregated_cmc(
                    eeg_array=eeg_array, emg_array=emg_flexor_array, sampling_freq=eeg_config['sampling_freq'],
                    log_frame=within_qtc_log_frame, eeg_channel_subset=cmc_eeg_channel_subset,
                    muscle_group='Flexor', window_size_sec=cmc_window_size_sec,
                    window_overlap_ratio=cmc_window_overlap_ratio,
                    enforce_independence_threshold=filter_significant_cmc, independence_threshold_alpha=cmc_independence_threshold_alpha,
                    use_jackknife=True, jackknife_alpha=cmc_jackknife_alpha,
                    save_dir=subject_cmc_save_dir,
                )


            ## Fetch Pre-computed
            elif not only_extensor:
                flexor_cmc_values, flexor_time_centers, flexor_freqs = features.fetch_stored_spectrograms(
                    subject_cmc_save_dir, "Flexor CMC", f"Channels_{"_".join(cmc_eeg_channel_subset)}"
                )


            if plot_cmc_results and not only_extensor:
                # max. within frequency band - value shapes: (n_windows, n_channels)
                flexor_cmc_aggregates_per_band = features.aggregate_spectrogram_over_frequency_band(
                    flexor_cmc_values, flexor_freqs, behaviour='max', frequency_axis=1,
                    lower_array=flexor_cmc_values_lower if do_compute_cmc else None,
                    upper_array=flexor_cmc_values_upper if do_compute_cmc else None,
                )  # if lower and upper provided, values are tuples (vals, lower_ci, upper_ci)

                ### DRAW LINEPLOT WITH CI
                band = 'beta'
                if do_compute_cmc:  # CIs given:
                    cmc_values, cmc_lower_ci, cmc_upper_ci = flexor_cmc_aggregates_per_band[band]
                else: cmc_values = flexor_cmc_aggregates_per_band[band]
                fig, ax = visualizations.plot_array_with_ci(
                    cmc_values[:, [cmc_eeg_channel_subset.index(ch) for ch in cmc_line_visualization_channels]],
                    time_axis=0,
                    hue_axis=1,
                    hue_name='EEG Channel',
                    hue_labels =cmc_line_visualization_channels,
                    input_lower_ci=cmc_lower_ci[:, [cmc_eeg_channel_subset.index(ch) for ch in cmc_line_visualization_channels]] if do_compute_cmc else None,
                    input_upper_ci=cmc_upper_ci[:, [cmc_eeg_channel_subset.index(ch) for ch in cmc_line_visualization_channels]] if do_compute_cmc else None,
                    sampling_freq=None,  # Change to your sampling rate if needed (Hz)
                    ci_alpha=0.2,
                    linewidth=2.0,
                    title=f'Flexor CMC ({band.capitalize()}) with {(1-cmc_jackknife_alpha)*100}% Confidence Intervals' if do_compute_cmc else f"Flexor CMC ({band.capitalize()})",
                    xlabel='Time (samples)',
                    ylabel='Magn. Sq. Coherence (CMC)',
                    legend=True,
                    phase_series=phase_series,
                )
                plt.tight_layout()
                plt.show()

                band = 'gamma'
                if do_compute_cmc:  # CIs given:
                    cmc_values, cmc_lower_ci, cmc_upper_ci = flexor_cmc_aggregates_per_band[band]
                else:
                    cmc_values = flexor_cmc_aggregates_per_band[band]
                fig, ax = visualizations.plot_array_with_ci(
                    cmc_values[:, [cmc_eeg_channel_subset.index(ch) for ch in cmc_line_visualization_channels]],
                    time_axis=0,
                    hue_axis=1,
                    hue_name='EEG Channel',
                    hue_labels=cmc_line_visualization_channels,
                    input_lower_ci=cmc_lower_ci[:, [cmc_eeg_channel_subset.index(ch) for ch in
                                                    cmc_line_visualization_channels]] if do_compute_cmc else None,
                    input_upper_ci=cmc_upper_ci[:, [cmc_eeg_channel_subset.index(ch) for ch in
                                                    cmc_line_visualization_channels]] if do_compute_cmc else None,
                    sampling_freq=None,  # Change to your sampling rate if needed (Hz)
                    ci_alpha=0.2,
                    linewidth=2.0,
                    title=f'Flexor CMC ({band.capitalize()}) with {(1 - cmc_jackknife_alpha) * 100}% Confidence Intervals' if do_compute_cmc else f"Flexor CMC ({band.capitalize()})",
                    xlabel='Time (samples)',
                    ylabel='Magn. Sq. Coherence (CMC)',
                    legend=True,
                    phase_series=phase_series,
                )
                plt.tight_layout()
                plt.show()


                visualizations.plot_spectrogram(flexor_cmc_aggregates_per_band[band][0] if do_compute_cmc else flexor_cmc_aggregates_per_band[band],
                                                flexor_time_centers, channels=cmc_eeg_channel_subset,
                                                is_log_scale=False,
                                                plot_type='time-channel',
                                                phase_series=phase_series,
                                                save_dir=subject_plot_dir,
                                                title=f"Aggregated Flexor CMC (max-pooled) for {band.capitalize()} Band",
                                                cbar_label="Magnitude Squared Coherence"
                                                )

            ####### EXTENSOR #######
            ## Computation
            if do_compute_cmc:
                extensor_cmc_values, extensor_cmc_values_lower, extensor_cmc_values_upper, extensor_time_centers, extensor_freqs = features.compute_task_wise_aggregated_cmc(
                    eeg_array=eeg_array, emg_array=emg_extensor_array, sampling_freq=eeg_config['sampling_freq'],
                    muscle_group='Extensor', window_size_sec=cmc_window_size_sec,
                    log_frame=within_qtc_log_frame, eeg_channel_subset=cmc_eeg_channel_subset,
                    window_overlap_ratio=cmc_window_overlap_ratio,
                    enforce_independence_threshold=filter_significant_cmc,
                    independence_threshold_alpha=cmc_independence_threshold_alpha,
                    use_jackknife=True, jackknife_alpha=cmc_jackknife_alpha,
                    save_dir=subject_cmc_save_dir,
                )


            ## Fetch Pre-computed
            else:
                extensor_cmc_values, extensor_time_centers, extensor_freqs = features.fetch_stored_spectrograms(
                    subject_cmc_save_dir, "Extensor CMC", f"Channels_{"_".join(cmc_eeg_channel_subset)}"
                )


            if plot_cmc_results:
                # max. within frequency band - value shapes: (n_windows, n_channels)
                extensor_cmc_aggregates_per_band = features.aggregate_spectrogram_over_frequency_band(
                    extensor_cmc_values, extensor_freqs, behaviour='max', frequency_axis=1,
                    lower_array=extensor_cmc_values_lower if do_compute_cmc else None,
                    upper_array=extensor_cmc_values_upper if do_compute_cmc else None,
                )  # if lower and upper provided, values are tuples (vals, lower_ci, upper_ci)

                ### DRAW LINEPLOT WITH CI
                band = 'beta'
                if do_compute_cmc:  # CIs given:
                    cmc_values, cmc_lower_ci, cmc_upper_ci = extensor_cmc_aggregates_per_band[band]
                else:
                    cmc_values = extensor_cmc_aggregates_per_band[band]
                fig, ax = visualizations.plot_array_with_ci(
                    cmc_values[:, [cmc_eeg_channel_subset.index(ch) for ch in cmc_line_visualization_channels]],
                    time_axis=0,
                    hue_axis=1,
                    hue_name='EEG Channel',
                    hue_labels=cmc_line_visualization_channels,
                    input_lower_ci=cmc_lower_ci[:, [cmc_eeg_channel_subset.index(ch) for ch in
                                                    cmc_line_visualization_channels]] if do_compute_cmc else None,
                    input_upper_ci=cmc_upper_ci[:, [cmc_eeg_channel_subset.index(ch) for ch in
                                                    cmc_line_visualization_channels]] if do_compute_cmc else None,
                    sampling_freq=None,  # Change to your sampling rate if needed (Hz)
                    ci_alpha=0.2,
                    linewidth=2.0,
                    title=f'Extensor CMC ({band.capitalize()}) with {(1 - cmc_jackknife_alpha) * 100}% Confidence Intervals' if do_compute_cmc else f"Extensor CMC ({band.capitalize()})",
                    xlabel='Time (samples)',
                    ylabel='Magn. Sq. Coherence (CMC)',
                    legend=True,
                    phase_series=phase_series,
                )
                plt.tight_layout()
                plt.show()

                band = 'gamma'
                if do_compute_cmc:  # CIs given:
                    cmc_values, cmc_lower_ci, cmc_upper_ci = extensor_cmc_aggregates_per_band[band]
                else:
                    cmc_values = extensor_cmc_aggregates_per_band[band]
                fig, ax = visualizations.plot_array_with_ci(
                    cmc_values[:, [cmc_eeg_channel_subset.index(ch) for ch in cmc_line_visualization_channels]],
                    time_axis=0,
                    hue_axis=1,
                    hue_name='EEG Channel',
                    hue_labels=cmc_line_visualization_channels,
                    input_lower_ci=cmc_lower_ci[:, [cmc_eeg_channel_subset.index(ch) for ch in
                                                    cmc_line_visualization_channels]] if do_compute_cmc else None,
                    input_upper_ci=cmc_upper_ci[:, [cmc_eeg_channel_subset.index(ch) for ch in
                                                    cmc_line_visualization_channels]] if do_compute_cmc else None,
                    sampling_freq=None,  # Change to your sampling rate if needed (Hz)
                    ci_alpha=0.2,
                    linewidth=2.0,
                    title=f'Extensor CMC ({band.capitalize()}) with {(1 - cmc_jackknife_alpha) * 100}% Confidence Intervals' if do_compute_cmc else f"Extensor CMC ({band.capitalize()})",
                    xlabel='Time (samples)',
                    ylabel='Magn. Sq. Coherence (CMC)',
                    legend=True,
                    phase_series=phase_series,
                )
                plt.tight_layout()
                plt.show()

                visualizations.plot_spectrogram(
                    extensor_cmc_aggregates_per_band[band][0] if do_compute_cmc else extensor_cmc_aggregates_per_band[band],
                    extensor_time_centers, channels=cmc_eeg_channel_subset,
                    is_log_scale=False,
                    plot_type='time-channel',
                    phase_series=phase_series,
                    save_dir=subject_plot_dir,
                    title=f"Aggregated Extensor CMC (max-pooled) for {band.capitalize()} Band",
                    cbar_label="Magnitude Squared Coherence"
                    )

            # todo: add other CMC approaches

        gc.collect()




        ### Enriched Serial Frame Computation
        if compute_enriched_serial_frame:
            # compute Heart Rate (HR) and HR Variability:
            bpm_series, hrv_series = features.compute_heart_rate_and_variability(within_qtc_serial_frame['ecg'])

            # scale force per task:
            scaled_force_series = features.compute_task_wise_scaled_force(within_qtc_serial_frame['fsr'], log_frame)

            # keep GSR:
            gsr_series = within_qtc_serial_frame['gsr']

            # merge to dataframe:
            bpm_series = data_analysis.make_timezone_aware(bpm_series)
            hrv_series = data_analysis.make_timezone_aware(hrv_series)
            scaled_force_series = data_analysis.make_timezone_aware(scaled_force_series)
            gsr_series = data_analysis.make_timezone_aware(gsr_series)

            enriched_serial_df = pd.concat(
                [bpm_series, hrv_series, scaled_force_series, gsr_series],
                axis=1  # concatenate as columns
            )

            if save_enriched_serial_frame:
                save_path = subject_serial_dir / filemgmt.file_title("Enriched Serial Frame", ".csv")
                enriched_serial_df.to_csv(save_path)
