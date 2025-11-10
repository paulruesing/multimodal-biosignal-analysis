import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable, Literal, Tuple
import os
from tqdm import tqdm
import mne
import mne_icalabel
import pywt
import mne
import mne_icalabel
import pywt
from scipy.signal import butter, filtfilt, iirnotch, welch, csd


from src.pipeline.visualizations import plot_eeg_heatmap, plot_freq_domain, EEG_CHANNELS

class BiosignalPreprocessor:
    def __init__(self,
                 np_input_array: np.ndarray,  # shape: (timesteps, channels)
                 sampling_freq: int,  # Hz
                 modality: Literal['eeg', 'emg'],
                 band_pass_frequencies: tuple[float, float] | Literal['auto'] = 'auto',
                 notch_frequency: float | None = 50,
                 notch_harmonics: int = 4,
                 notch_width: float | None = None,
                 reference_channels: str | Literal['average'] | None = 'average',
                 amplitude_rejection_threshold: float | None = .001,
                 ica_components: int | None = 25,
                 laplacian_filter_neighbor_radius: float | None = .05,
                 wavelet_type: Literal['db4', 'sym5', 'coif1'] | None = 'db4',
                 denoising_threshold_mode: Literal['soft', 'hard'] = 'soft',
                 ):
        """
        Properties follow hierarchical logic (higher ones containing filtering steps pertaining to lower ones):
        - np_output_data
        - np_denoised_data
        - np_smoothed_data
        - mne_artefact_free_data or np_artefact_free_data
        - mne_amplitude_compliant_data
        - mne_referenced_data
        - mne_filtered_data
        - mne_raw_data or np_input_array
        """
        # all private attributes accessible via properties, because setting recomputes filtered data
        # import parameters:
        assert np_input_array.shape[1] < np_input_array.shape[0], "Should be more timesteps (rows) than channels (columns)!"
        self._np_input_array = np_input_array
        self._sampling_freq = sampling_freq
        self._modality = modality
        self.n_timesteps, self.n_channels = np_input_array.shape

        # preprocessing parameters:
        self._band_pass_frequencies = band_pass_frequencies
        self._notch_frequency = notch_frequency  # None leads to no notch filter
        self._notch_harmonics = notch_harmonics
        self._notch_width = notch_width  # None leads to automatic setting
        self._reference_channels = reference_channels  # None leads to no re-referencing
        self._amplitude_rejection_threshold = amplitude_rejection_threshold  # None leads to no amplitude thresholding
        self._ica_components = ica_components  # None leads to no automatic artefact rejection
        self._manual_ics_to_exclude: list[int] = None
        self._laplacian_filter_neighbor_radius = laplacian_filter_neighbor_radius  # None -> no Laplacian filtering
        self._wavelet_type = wavelet_type  # None -> no wavelet denoising
        self._denoising_threshold_mode = denoising_threshold_mode

        ### result placeholders:
        # mne type:
        self._mne_amplitude_compliant_data = self._mne_filtered_data = None
        self._mne_referenced_data = self._mne_raw_data = None
        self._mne_ica_result = self._mne_artefact_free_data = None
        # np type:
        self._np_artefact_free_data = self._np_smoothed_data = None
        self._np_denoised_data = self._np_output_data = None
        # others:
        self._wavelet_coefficients = self._denoised_wavelet_coefficients = None


    ############# PROPERTIES #############
    ### input properties ###
    # todo: implement setters with clear_downstream_results
    @property
    def np_input_array(self) -> np.ndarray:
        return self._np_input_array

    @property
    def sampling_freq(self) -> int:
        return self._sampling_freq

    @property
    def modality(self) -> Literal['eeg', 'emg']:
        return self._modality

    @property
    def band_pass_frequencies(self) -> tuple[float, float]:
        if self._band_pass_frequencies == "auto":
            if self.modality == 'eeg': return (.1, 100)
            elif self.modality == 'emg': return (20, 500)
        else: return self._band_pass_frequencies

    @property
    def notch_frequency(self) -> float | None:
        """ None leads to no notch filtering being carried out. """
        return self._notch_frequency

    @property
    def notch_harmonics(self) -> int:
        return self._notch_harmonics

    @property
    def notch_width(self) -> float | None:
        """ None leads to automatic setting. """
        return self._notch_width

    @property
    def reference_channels(self) -> str | Literal['average'] | None:
        return self._reference_channels

    @property
    def amplitude_rejection_threshold(self) -> float | None:
        """ None leads to no amplitude rejection being carried out. """
        return self._amplitude_rejection_threshold

    @property
    def ica_components(self) -> int:
        """ None leads to no ICA being carried out. """
        return self._ica_components

    @property
    def manual_ics_to_exclude(self) -> list[int]:
        if self._manual_ics_to_exclude is None: return []
        return self._manual_ics_to_exclude

    @manual_ics_to_exclude.setter
    def manual_ics_to_exclude(self, value: list[int] | None) -> None:
        """ It is advisable to call plot_independent_component for this. """
        self._manual_ics_to_exclude = value
        self.clean_downstream_results('artefact rejection')

    @property
    def laplacian_filter_neighbor_radius(self) -> float | None:
        """ If None, no Laplacian spatial filtering is computed. """
        return self._laplacian_filter_neighbor_radius

    @property
    def wavelet_type(self) -> Literal['db4', 'sym5', 'coif1'] | None:
        """ If None, no wavelet coefficients are computed. """
        return self._wavelet_type

    @property
    def denoising_threshold_mode(self) -> Literal['soft', 'hard']:
        return self._denoising_threshold_mode

    ### calculation-based properties ###
    @property
    def mne_raw_data(self):
        if self._mne_raw_data is not None: return self._mne_raw_data

        # initialise and return:
        data_info = mne.create_info(ch_names=EEG_CHANNELS,
                                    sfreq=self.sampling_freq,
                                    ch_types=self.modality,)
        raw_data = mne.io.RawArray(self.np_input_array.T, data_info)

        if self.modality == 'eeg': raw_data.set_montage(mne.channels.make_standard_montage('standard_1020'))
        # todo: implement custom sEMG montage for Laplacian spatial filtering

        self._mne_raw_data = raw_data
        return self._mne_raw_data

    @property
    def mne_filtered_data(self):
        if self._mne_filtered_data is not None: return self._mne_filtered_data

        # compute:
        filtered_data = self.mne_raw_data.copy()
        filtered_data.filter(l_freq=self.band_pass_frequencies[0],
                             h_freq=self.band_pass_frequencies[1],
                             fir_design='firwin')
        self._mne_filtered_data = filtered_data
        if self.notch_frequency is not None: self._apply_notch_filter()

    @property
    def mne_referenced_data(self):
        if self._mne_referenced_data is not None: return self._mne_referenced_data

        # compute:
        if self.reference_channels is None: return self.mne_filtered_data
        temp_data = self.mne_filtered_data.copy()
        temp_data.set_eeg_reference(ref_channels=self.reference_channels,
                                    ch_type='auto' if self.modality == 'eeg' else 'emg')
        self._mne_referenced_data = temp_data
        return self._mne_referenced_data

    @property
    def mne_amplitude_compliant_data(self):
        if self._mne_amplitude_compliant_data is not None: return self._mne_amplitude_compliant_data

        # initialise:
        if self.amplitude_rejection_threshold is None: return self.mne_referenced_data

        self._mne_amplitude_compliant_data = self.mne_referenced_data.copy()
        # annotate bad segments and discover bad channels based on amplitude thresholding:
        self._annotate_amplitude_based_artefacts()
        return self._mne_amplitude_compliant_data

    @property
    def bad_channels(self) -> list[str]:
        return self.mne_amplitude_compliant_data.info['bads']

    @property
    def mne_ica_result(self):
        if self._mne_ica_result is not None: return self._mne_ica_result

        if self.ica_components is None: raise ValueError("ica_components needs to be defined!")
        # fit ICA:
        ica = mne.preprocessing.ICA(n_components=self.ica_components,
                                    max_iter='auto',
                                    method='infomax',
                                    # switches between nonlinearities, appears more robust but takes much longer (than FastICA)
                                    fit_params=dict(extended=True))
        # convergence is difficult with too short datasets, rule of thumb appears to be n-components x 20-30 = required_seconds
        ica.fit(self.mne_amplitude_compliant_data)
        self._mne_ica_result = ica
        return self._mne_ica_result

    @property
    def mne_artefact_free_data(self):
        if self._mne_artefact_free_data is not None: return self._mne_artefact_free_data

        # otherwise compute
        if self.ica_components is None: return self.mne_amplitude_compliant_data

        # label ica components:
        ica_label_output = mne_icalabel.label_components(self.mne_amplitude_compliant_data,
                                                         self.mne_ica_result, method='iclabel')
        probs, labels = ica_label_output.values()
        print("Found the following IC labels:\n", labels)
        print("Will exclude 'heart beat' and 'muscle artifact'.")

        # exclude such. we can access the private variable _mne_ica... here because we ensured it's computed by accessing the property above
        exclusion_list = [idx for idx, label in enumerate(labels) if label in ('heart beat', 'muscle artifact')] + self.manual_ics_to_exclude
        print(f'Also excluding manual set ICs: {self.manual_ics_to_exclude}\n(change this selection via manual_ics_to_exclude parameter)')
        # set conversion interim to prevent duplicates
        self._mne_ica_result.exclude = list(set(exclusion_list))

        # save and return:
        self._mne_artefact_free_data = self.mne_ica_result.apply(self.mne_amplitude_compliant_data.copy())

        return self._mne_artefact_free_data

    @property
    def np_artefact_free_data(self):
        if self._np_artefact_free_data is not None: return self._np_artefact_free_data

        # compute:
        self._np_artefact_free_data = self.mne_artefact_free_data.get_data().T
        print(f'np data also contains bad channels ({self.bad_channels}).\nConsider excluding such manually!')

        return self._np_artefact_free_data

    @property
    def np_smoothed_data(self) -> np.ndarray:
        if self._np_smoothed_data is not None: return self._np_smoothed_data

        # reasons to omit Laplacian spatial filtering:
        if self.laplacian_filter_neighbor_radius is None: return self.np_artefact_free_data
        if self.modality == 'emg':
            print("Laplacian spatial filtering for EMG data is not implemented yet (requires 3D positional electrode mapping). Data remains unchanged.")
            return self.np_artefact_free_data

        # compute:
        _ = self.mne_artefact_free_data.get_montage()  # checks whether montage was provided
        pos = np.array([self.mne_artefact_free_data.info['chs'][i]['loc'][:3] for i in range(len(self.mne_artefact_free_data.ch_names))])  # 3D coords of channels

        # identify proximal EEG channels based on positional mapping:
        neighbors = []
        for i, pos_i in enumerate(pos):
            dists = np.linalg.norm(pos - pos_i,
                                   axis=1)  # compute Euclidean dist from current electrode to all others
            # find neighbors by: excluding self and keeping channels within radius
            neigh = np.where((dists > 0) & (dists < self.laplacian_filter_neighbor_radius))[0]
            neighbors.append(neigh)

        # Compute Laplacian-filtered data:
        laplacian_data = np.zeros_like(self.np_artefact_free_data)  # prepare output array
        for i, neigh_idx in enumerate(neighbors):  # iterate over channels
            if len(neigh_idx) > 0:  # subtract neighbors' average:
                laplacian_data[:, i] = self.np_artefact_free_data[:, i] - self.np_artefact_free_data[:, neigh_idx].mean(axis=1)
            else:  # no neighbors: keep original:
                laplacian_data[:, i] = self.np_artefact_free_data[:, i]

        self._np_smoothed_data = laplacian_data
        return self._np_smoothed_data

    @property
    def wavelet_coefficients(self) ->  list[np.ndarray[np.ndarray[float]]]:
        if self._wavelet_coefficients is not None: return self._wavelet_coefficients
        if self.wavelet_type is None: raise ValueError("wavelet_type needs to be defined.")

        # implement computation:
        max_level = pywt.dwt_max_level(len(self.np_smoothed_data),
                                       pywt.Wavelet(self.wavelet_type).dec_len)  # up to which level to decompose

        # compute wavelet coefficients:
        self._wavelet_coefficients = pywt.wavedec(self.np_smoothed_data, self.wavelet_type,
                            level=max_level)
        return self._wavelet_coefficients

    @property
    def denoised_wavelet_coefficients(self) ->  list[np.ndarray[np.ndarray[float]]]:
        if self._denoised_wavelet_coefficients is not None: return self._denoised_wavelet_coefficients
        if self.wavelet_type is None: raise ValueError("wavelet_type needs to be defined.")

        ### compute:
        # select finest scale coefficient (highest frequency noise) to estimate noise level:
        detail_coeffs = self.wavelet_coefficients[-1]

        # compute noise threshold:
        sigma = np.median(np.abs(
            detail_coeffs)) / .6745  # factor .6745 converts median absolute deviation to standard deviation assuming Gaussian noise
        uthresh = sigma * np.sqrt(2 * np.log(
            len(detail_coeffs)))  # universal threshold (Donoho's), common for balancing signal preservation and noise reduction
        # other (adaptive) options: SURE, Minimax, BayesShrink

        # denoise coefficients:
        n_coeffs_to_skip = 1
        denoised_coeffs = [np.array(*self.wavelet_coefficients[
            :n_coeffs_to_skip])]  # initialise denoised list by keeping approximation coeffs (index 0) unaltered
        for coeff in self.wavelet_coefficients[n_coeffs_to_skip:]:
            denoised_coeff = pywt.threshold(coeff, value=uthresh,
                                            mode=self.denoising_threshold_mode)  # apply soft thresholding to shrink coefficients below threshold
            denoised_coeffs.append(denoised_coeff)

        # save and return:
        self._denoised_wavelet_coefficients = denoised_coeffs
        return self._denoised_wavelet_coefficients

    @property
    def np_denoised_data(self):
        if self._np_denoised_data is not None: return self._np_denoised_data

        # skip computation if:
        if self.wavelet_type is None: return self.np_smoothed_data

        # compute and return:
        self._np_denoised_data = pywt.waverec(self.denoised_wavelet_coefficients, self.wavelet_type)
        return self._np_denoised_data

    @property
    def np_output_data(self):
        """ This equals np_denoised_data (currently the last step of the pipeline) but displays a progress bar upon computation."""
        if self._np_output_data is not None: return self._np_output_data

        # else compute with progress bar:
        print('Running full preprocessing pipeline...')
        for step in tqdm(['import', 'filtering', 'referencing', 'amplitude thesholding',
                                 'artefact rejection', 'smoothing', 'denoising']):
            # accessing properties leads to computation:
            if step == 'import': _ = self.mne_raw_data
            elif step == 'filtering': _ = self.mne_filtered_data
            elif step == 'referencing': _ = self.mne_referenced_data
            elif step == 'amplitude thesholding': _ = self.mne_amplitude_compliant_data
            elif step == 'artefact rejection': _ = self.np_artefact_free_data
            elif step == 'smoothing': _ = self.np_smoothed_data
            elif step == 'denoising': _ = self.np_denoised_data

        self._np_output_data = self.np_denoised_data
        return self._np_output_data

    ############# PREPROCESSING METHODS #############
    def _apply_notch_filter(self):
        """ Apply notch filters to remove a fundamental frequency and its harmonics from a signal. """
        if self.notch_frequency is None: raise ValueError("notch_frequency needs to be defined!")
        self._mne_filtered_data = self.mne_filtered_data.copy().notch_filter(
            freqs=[self.notch_frequency * i for i in range(1, self.notch_harmonics+1)],
            picks='all', notch_widths=self.notch_width)

    def _annotate_amplitude_based_artefacts(self):
        if self.amplitude_rejection_threshold is None: raise ValueError("amplitude_rejection_threshold needs to be defined!")
        reject_criteria = dict(eeg=self.amplitude_rejection_threshold, emg=self.amplitude_rejection_threshold)

        # derive annotations:
        annotations, bad_channels = mne.preprocessing.annotate_amplitude(
            self._mne_amplitude_compliant_data,
            peak=reject_criteria,
            min_duration=.025,  # minimum duration for consecutive samples to exceed or fall below threshold
            bad_percent=5,  # channels with more bad segments will be marked as complete bad channels
        )
        print(f"Found {len(bad_channels)} bad channels.")

        # save result:
        self._mne_amplitude_compliant_data.set_annotations(annotations)
        self._mne_amplitude_compliant_data.info['bads'].extend(bad_channels)

    def clean_downstream_results(self,
                                 change_in: Literal['import', 'filtering', 'referencing', 'amplitude thesholding',
                                 'artefact rejection', 'smoothing', 'denoising']):
        """
        Force recalculation (aka clear) downstream results based on parameter changes in lower hierarchy levels.
        Properties follow hierarchical logic (higher ones containing filtering steps pertaining to lower ones):
        - np_output_data
        - np_denoised_data
        - np_smoothed_data
        - mne_artefact_free_data or np_artefact_free_data
        - mne_amplitude_compliant_data
        - mne_referenced_data
        - mne_filtered_data
        - mne_raw_data or np_input_array
        """
        if change_in.lower() == 'import':
            self._mne_raw_data = None
            self._mne_filtered_data = None
            self._mne_referenced_data = None
            self._mne_amplitude_compliant_data = None
            self._mne_artefact_free_data = None
            self._np_artefact_free_data = None
            self._mne_ica_results = None
            self._np_smoothed_data = None
            self._np_denoised_data = None
            self._denoised_wavelet_coefficients = None
            self._wavelet_coefficients = None
            self._np_output_data = None
        elif change_in.lower() == 'filtering':
            self._mne_filtered_data = None
            self._mne_referenced_data = None
            self._mne_amplitude_compliant_data = None
            self._mne_artefact_free_data = None
            self._mne_ica_results = None
            self._np_smoothed_data = None
            self._np_denoised_data = None
            self._denoised_wavelet_coefficients = None
            self._wavelet_coefficients = None
            self._np_output_data = None
        elif change_in.lower() == 'referencing':
            self._mne_referenced_data = None
            self._mne_amplitude_compliant_data = None
            self._mne_artefact_free_data = None
            self._np_artefact_free_data = None
            self._mne_ica_results = None
            self._np_smoothed_data = None
            self._np_denoised_data = None
            self._denoised_wavelet_coefficients = None
            self._wavelet_coefficients = None
            self._np_output_data = None
        elif change_in.lower() == 'amplitude thesholding':
            self._mne_amplitude_compliant_data = None
            self._mne_artefact_free_data = None
            self._np_artefact_free_data = None
            self._mne_ica_results = None
            self._np_smoothed_data = None
            self._np_denoised_data = None
            self._denoised_wavelet_coefficients = None
            self._wavelet_coefficients = None
            self._np_output_data = None
        elif change_in.lower() == 'artefact rejection':
            self._mne_artefact_free_data = None
            self._np_artefact_free_data = None
            self._mne_ica_results = None
            self._np_smoothed_data = None
            self._np_denoised_data = None
            self._denoised_wavelet_coefficients = None
            self._wavelet_coefficients = None
            self._np_output_data = None
        elif change_in.lower() == 'smoothing':
            self._np_smoothed_data = None
            self._np_denoised_data = None
            self._denoised_wavelet_coefficients = None
            self._wavelet_coefficients = None
            self._np_output_data = None
        elif change_in.lower() == 'denoising':
            self._np_denoised_data = None
            self._denoised_wavelet_coefficients = None
            self._wavelet_coefficients = None
            self._np_output_data = None
        else: raise ValueError("change_in category is undefined!")

    @staticmethod
    def discrete_fourier_transform(input_array: np.ndarray, axis: Literal[0, 1] | None = None, plot_result: bool = True, **plot_kwargs) -> tuple[np.ndarray, np.ndarray]:
        """ To be commented. """
        # input sanity checks:
        if len(input_array.shape) == 1:
            input_array = input_array[:, np.newaxis]
            if axis is None: axis = 0
        else:
            if axis is None: raise AttributeError("For 2D signal arrays, axis needs to be defined!")

        # descriptive parameters:
        n_samples = input_array.shape[axis]

        # compute discrete FT with FFT algorithm:
        fft_result = np.fft.fft(input_array, axis=axis)  # shape is as input
        freqs_fft = np.fft.fftfreq(n_samples, d=1/sampling_freq)    # frequency bin labels [Hz]

        # retain only positive frequencies (FFT of real-valued signals is symmetric)
        freqs_pos = freqs_fft[freqs_fft >= 0]
        fft_pos = fft_result[freqs_fft >= 0, :] if axis == 0 else fft_result[:, freqs_fft >= 0]

        # compute magnitude:
        amplitude_spectrum = np.abs(fft_pos) * 2 / n_samples  # normalize amplitude by 2/n_samples

        # eventually plot:
        if plot_result: plot_freq_domain(amplitude_spectrum, freqs_pos, **plot_kwargs)

        return amplitude_spectrum, freqs_pos

    ############# PLOTTING METHODS #############
    def plot_independent_component(self, ic_index: int):
        self.mne_ica_result.plot_properties(self.mne_amplitude_compliant_data, ic_index)

    def plot_data_overview(self):
        """ Can also be used to manually annotate bad channels. """
        temp_bad_channels = self.bad_channels.copy()

        # open interactive plot:
        self.mne_amplitude_compliant_data.plot()

        # if new bad channels have been (de-)selected:
        if temp_bad_channels != self.mne_amplitude_compliant_data.info['bads']:
            print('New bad channels (de-)selected, will clean downstream results.')
            self.clean_downstream_results(change_in='amplitude thesholding')

if __name__ == '__main__':
    ROOT = Path().resolve().parent.parent
    QTC_DATA = ROOT / "data" / "qtc_measurements" / "2025_06"
    subject_data_dir = QTC_DATA / "sub-10"

    # example run:
    input_file = np.load(subject_data_dir / "motor_eeg_full.npy").T
    data_modality: Literal['eeg', 'emg'] = 'eeg'
    sampling_freq = 2048  # Hz

    prepper = BiosignalPreprocessor(
        np_input_array=input_file,
        sampling_freq=sampling_freq,
        modality=data_modality,
        band_pass_frequencies='auto',
    )

    print(prepper.np_output_data)