import numpy as np
import pandas as pd
from pathlib import Path
import gc
from typing import Callable, Literal, Tuple

import src.pipeline.signal_features as features
from src.pipeline.preprocessing import BiosignalPreprocessor
import src.pipeline.visualizations as visualizations
import src.utils.file_management as filemgmt

######## WORKFLOW #########
if __name__=="__main__":
    ROOT = Path().resolve().parent
    OUTPUT = ROOT / 'output'
    STUDY_PLOTS = OUTPUT / 'preprocessing_plots'
    QTC_DATA = ROOT / "data" / "qtc_data"

    ### WORKFLOW CONTROL
    # change these to select subject and trial:
    for subject_ind in [0, 1, 3, 4]:
        print("\n" + "-"*100)
        print(f"Processing subject {subject_ind}")
        print("-" * 100)
        sampling_rate_Hz: int = 2048
        cut_off_first_n_seconds: int | None = None  # if given, starts after such (if there is corrupted data in such span)
        # should also be marked in data_integration as 'Actual Start Trigger'
        load_only_first_n_seconds: int | None = None  # if None, loads full file
        retrieve_latest_config: bool = True  # search .json config file in subject_data dir with matching file_title key

        # interactive workflow behavior:
        plot_input_plots: bool = False
        allow_manual_ic_input: bool = False  # only pertains to EEG data + if configured once, this isn't necessary anymore
        plot_output_plots: bool = False
        plot_psd_animation: bool = False
        conduct_validation: bool = False

        # saving behavior:
        save_new_config: bool = False
        save_result: bool = True

        # todo: add less conservative configuration, to judge whether this one foregoes too much information

        ### ITERATE OVER ALL CHANNEL SUBSETS
        for channel_set in [
            "eeg",
            "emg_1_flexor",
            "emg_2_extensor"
        ]:

            ## Dependent Variables:
            subject_data_dir = QTC_DATA / f"subject_{subject_ind:02}"
            subject_plot_dir = STUDY_PLOTS / f"subject_{subject_ind:02}"
            filemgmt.assert_dir(subject_plot_dir)
            file_title = f"sub_{subject_ind:02}_{channel_set}"
            save_dir: Path | str | None = subject_data_dir if save_result else None  # if None, nothing is saved


            ### DATA LOADING
            print('Loading data...')
            # mmap_mode='r': memory-mapped read-only access (would accelerate but sometimes deletes files)
            try:  # try npy
                input_file = np.load(subject_data_dir / f"{file_title}.npy").T
            except FileNotFoundError:  # try csv
                # read reduced slice:
                rows_to_read = sampling_rate_Hz*int(load_only_first_n_seconds) if load_only_first_n_seconds else None
                # consider cut-off offset if given:
                if rows_to_read is not None and cut_off_first_n_seconds is not None:
                    rows_to_read += sampling_rate_Hz * int(cut_off_first_n_seconds)

                input_file = pd.read_csv(subject_data_dir / f"{file_title}.csv", delimiter=",",
                                         nrows=rows_to_read,
                                         dtype=float).to_numpy()

                # csvs have time at column 0
                #timestamps_s = input_file[:, 0].copy()
                input_file = input_file[:, 1:65]

            # shorten file if desired:
            if cut_off_first_n_seconds: input_file = input_file[sampling_rate_Hz*int(cut_off_first_n_seconds):, :]
            if load_only_first_n_seconds: input_file = input_file[:sampling_rate_Hz*int(load_only_first_n_seconds), :]

            if retrieve_latest_config:
                try:  # search matching config, saved as e.g. "... eeg 64ch (FILE_TITLE).json"
                    config_file = filemgmt.most_recent_file(subject_data_dir, ".json", file_title)
                except ValueError:
                    print(f"No config file found for {file_title}")
                    config_file = None
            else: config_file = None

            print('Initialising BiosignalPreprocessor...')
            if config_file is not None:  # try initialising from config:
                prepper = BiosignalPreprocessor.init_from_config(config_file, input_file)
            else:
                data_modality: Literal['eeg', 'emg'] = 'emg' if 'emg' in file_title else 'eeg'
                sampling_freq = sampling_rate_Hz  # Hz
                prepper = BiosignalPreprocessor(
                    np_input_data=input_file,
                    sampling_freq=sampling_freq,
                    modality=data_modality,
                    band_pass_frequencies='auto',
                    amplitude_rejection_threshold=.01,
                    wavelet_type=None,  # seems to be over-conservative
                    laplacian_filter_neighbor_radius='auto',  # automatic: -> None for EMG

                )

            ### INPUT PLOTS
            if plot_input_plots:
                # fourier spectrum:
                features.discrete_fourier_transform(prepper.np_input_data,
                                                    sampling_freq=prepper.sampling_freq,
                                                    frequency_range=(0, 100),
                                                    plot_title=f'Raw Data - Fourier Spectrum ({file_title})',
                                                    save_dir=subject_plot_dir,
                                                    plot_result=True, )
                # PSD spectrogram:
                psd, psd_times, psd_freqs = features.multitaper_psd(input_array=prepper.np_input_data, sampling_freq=prepper.sampling_freq, nw=3,
                                        window_length_sec=1.0, overlap_frac=0.5, axis=0,
                                                                    apply_log_scale=True,
                                                                    verbose=True, plot_result=False)
                visualizations.plot_spectrogram(np.mean(psd, axis=2), psd_times, psd_freqs,
                                                frequency_range=(0, 100),
                                                save_dir=subject_plot_dir, title=f'Input Averaged PSD ({file_title})',
                                                log_scale=False,  # already transformed to log during computed
                                                )


            ### ARTEFACT REJECTION
            # automatic artefact rejection:
            _ = prepper.mne_artefact_free_data

            if prepper.modality == 'eeg' and allow_manual_ic_input:  # manual inspection of ICs (only for EEG!)
                if input("Do you want to visualize all ICs? Press enter if yes, else type anything: ") == "":
                    for ic_ind in range(prepper.n_ica_components):
                        prepper.plot_independent_component(ic_ind,
                                                           verbose=(ic_ind == 0),  # print only on first iteration
                                                           )
                # possibility for changes:
                manual_ics = input(
                    "Please enter additional independent components to exclude, separated by space (e.g. '10 13 7'): ")
                if manual_ics != '':
                    manual_ics = [int(ind_str.strip()) for ind_str in manual_ics.split(' ')]
                    prepper.manual_ics_to_exclude = manual_ics

            # save to config:
            if save_new_config: prepper.export_config(subject_data_dir, file_title)

            ### Output Plots
            if plot_output_plots:
                # fourier spectrum:
                features.discrete_fourier_transform(prepper.np_output_data,
                                                    sampling_freq=prepper.sampling_freq,
                                                    frequency_range=(0, 100),
                                                    plot_title=f'Preprocessed Data - Fourier Spectrum ({file_title})',
                                                    save_dir=subject_plot_dir,
                                                    plot_result=True)
                # PSD spectrogram:
                spectrograms, timestamps, freqs = features.multitaper_psd(input_array=prepper.np_output_data,
                                                                          sampling_freq=prepper.sampling_freq, nw=3,
                                                                          window_length_sec=.2, overlap_frac=0.5, axis=0,
                                                                          apply_log_scale=True,
                                                                    verbose=True, plot_result=False)
                visualizations.plot_spectrogram(np.mean(spectrograms, axis=2), timestamps, freqs,
                                                frequency_range=(0, 100),
                                                save_dir=subject_plot_dir, title=f'Output Averaged PSD ({file_title})',
                                                log_scale=False,  # already transformed to log during computed
                                                )

            ### PSD Animation
            if plot_psd_animation:
                if not plot_output_plots:
                    # PSD spectrogram:
                    spectrograms, _, freqs = features.multitaper_psd(input_array=prepper.np_output_data,
                                                                     sampling_freq=prepper.sampling_freq, nw=3,
                                                                     window_length_sec=.2, overlap_frac=0.5, axis=0,
                                                                     plot_result=False, frequency_range=(0, 100))

                # spectrograms shape: (n_channels, n_windows, n_frequencies)
                # timestamps shape: (n_windows), frequencies shape: (n_frequencies)
                psd_sampling_freq = spectrograms.shape[1] / (
                        len(prepper.np_output_data) / prepper.sampling_freq)  # new_timesteps / time_duration

                # average (and eventually log-transform) spectrogram across frequency bins:
                do_log_transform: bool = True
                freq_averaged_psd_dict = {}  # keys: band-label keys, values: np.ndarrays shaped (n_channels, n_windows)
                for band_label, band_range in features.FREQUENCY_BANDS.items():
                    frequency_mask = (freqs >= band_range[0]) & (freqs < band_range[1])  # select band frequencies
                    spectrogram_subset = spectrograms[:, :, frequency_mask]
                    if do_log_transform: spectrogram_subset = np.log10(spectrogram_subset + 1e-10)
                    freq_averaged_psd_dict[band_label] = np.squeeze(np.mean(spectrogram_subset, axis=2))  # average across freqs.

                # animation:
                band_to_scrutinize = 'beta'
                visualizations.animate_electrode_heatmap(
                    freq_averaged_psd_dict[band_to_scrutinize].T,  # requires shape (n_timesteps, n_channels)
                    positions=visualizations.EEG_POSITIONS if prepper.modality == 'eeg' else visualizations.EMG_POSITIONS,
                    add_head_shape=prepper.modality == 'eeg',
                    sampling_rate=psd_sampling_freq, animation_fps=psd_sampling_freq,
                    value_label="Power [V^2/Hz]" if not do_log_transform else "Power [V^2/Hz] [log10]",
                    plot_title=f"{prepper.modality.upper()} PSD ({band_to_scrutinize}-band)"
                )

            ### VALIDATION
            if conduct_validation:
                filt_snr_increase, filt_psd_diff = prepper.validate_filtering()
                if prepper.modality == 'eeg': ref_snr_increase = prepper.validate_referencing()
                specificity, selectivity = prepper.validate_amplitude_thresholding()
                spat_filt_local_coh_decrease = prepper.validate_spatial_filtering()
                denoise_snr_increase = prepper.validate_wavelet_denoising()


            ### SAVING
            if save_dir is not None:
                prepper.export_results(save_dir, file_title, with_config=True)

            # clear memory
            gc.collect()
            del prepper