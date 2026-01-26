from pathlib import Path
import pandas as pd
import numpy as np
from typing import Literal
import statsmodels.formula.api as smf

import src.pipeline.signal_features as features
import src.pipeline.data_integration as data_integration
import src.pipeline.data_analysis as data_analysis
import src.utils.file_management as filemgmt

if __name__ == '__main__':
    ######## PREPARATION #########
    ROOT = Path().resolve().parent
    DATA = ROOT / "data"
    OUTPUT = ROOT / 'output'
    STUDY_PLOTS = OUTPUT / 'plots' / 'data_analysis_plots'
    EXPERIMENT_DATA = DATA / "experiment_results"
    # QTC_DATA = DATA / "qtc_data"  # not necessary, since we import pre-computed features (below)
    FEATURE_OUTPUT_DATA = DATA / "precomputed_features"


    ################################################
    ################ PSD MODELLING #################
    ################################################

    # todo: run for multiple participants
    subject_ind = 0
    # define PSD modality
    modality: Literal['eeg', 'emg_1_flexor', 'emg_2_extensor'] = 'emg_1_flexor'
    is_log_scaled: bool = True  # define whether PSD was log scaled during feature extraction

    # dependent directories:
    subject_psd_save_dir = FEATURE_OUTPUT_DATA / f"subject_{subject_ind:02}"
    subject_experiment_data_dir = EXPERIMENT_DATA / f"subject_{subject_ind:02}"

    # import log and serial frame:
    log_df = data_integration.fetch_enriched_log_frame(subject_experiment_data_dir)
    serial_df = data_integration.fetch_serial_measurements(subject_experiment_data_dir)

    # slice towards qtc measurements:
    qtc_start, qtc_end = data_integration.get_qtc_measurement_start_end(log_df, False)
    sliced_log_df = log_df[qtc_start:qtc_end]
    sliced_serial_df = serial_df[qtc_start:qtc_end]

    # import PSD:
    psd_spectrograms, psd_times, psd_freqs = features.fetch_stored_psd(subject_psd_save_dir, file_identifier=modality)
    # shape: (n_times, n_freqs, n_channels), (n_times), (n_freqs)

    # window start end times:
    trial_start_end_dict = data_integration.get_all_task_start_ends(log_df, 'dict')
    trial_starts = [start for start, end in trial_start_end_dict.values()]
    trial_ends = [end for start, end in trial_start_end_dict.values()]


    ### MODALITY-DEPENDENT AGGREGATION
    if 'emg' in modality:
        # 1) MVC Normalization per Channel if not log-scaled!
        if not is_log_scaled:
            mvc = np.max(np.max(psd_spectrograms, axis=0, keepdims=True), axis=1, keepdims=True)  # maximum over time and frequencies
            psd_spectrograms = psd_spectrograms / mvc * 100  # * 100 for [%]

        # 2) Average over a: all frequencies (broad band) OR b: frequency bands
        freq_band_spec_dict: dict[str, np.ndarray] = features.aggregate_spectrogram_over_frequency_band(
            psd_spectrograms, psd_freqs, 'mean',
            frequency_bands={'slow': (0, 40), 'fast': (60, 250)},  # slow and fast twitching, deliberately omitting overlap zone (40-60)
        )  # value shape: (n_times, n_channels)

        # 3) Average over time (per trial)
        freq_band_agg_psd_dict: dict[str, np.ndarray] = {}
        for band, psd_arr in freq_band_spec_dict.items():
            # todo: still doesn't work
            psd_per_trial = data_analysis.apply_window_operator(
                window_timestamps=trial_starts,
                window_timestamps_ends=trial_ends,
                target_array=psd_arr,
                target_timestamps=psd_times,
                operation='mean',
                axis=0,  # time axis
            )  # (n_trials, n_channels)

            # 4) Max-Pool over Channels
            psd_per_trial = np.nanmax(psd_per_trial, axis=1)

            # 5) store in same dict:
            freq_band_agg_psd_dict[band] = psd_per_trial


    elif 'eeg' in modality:
        # 1) Average over Frequency Bands
        # 2) Average over Region-of-Interest (RoI)
        # 3) Average over time (per trial)
        pass

    # aggregate PSD over frequencies:




    quit()



    # todo: compute per trial observations

    # todo: run MLM

    ######## MODEL 1: MUSIC VS. SILENCE #########
    # Aggregate per trial
    agg_data = data.groupby(
        ['participant_id', 'trial_id', 'condition', 'force_level']
    ).agg({
        'PSD': 'mean'  # or 'median' if outlier-resistant preferred
    }).reset_index()

    # Model 1: Silence vs. Music (overall)
    model1 = smf.mixedlm(
        "PSD ~ C(condition, Treatment('silence')) + force_level",
        data=agg_data,
        groups=agg_data['participant_id']
    )
    result1 = model1.fit()
    print(result1.summary())
    # Output: Is music ≠ silence? (single p-value)
