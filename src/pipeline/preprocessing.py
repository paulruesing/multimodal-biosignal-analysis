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
import src.utils.file_management as filemgmt

class BiosignalPreprocessor:
    """
    Class for preprocessing biosignal data such as EEG and EMG.

    Parameters
    ----------
    np_input_array : np.ndarray
        Input data array of shape (timesteps, channels).
    sampling_freq : int
        Sampling frequency of the input data in Hz.
    modality : {'eeg', 'emg'}
        Type of biosignal modality.
    band_pass_frequencies : tuple of two floats or 'auto', optional
        Band-pass filter frequency range (low, high) in Hz.
        Default is 'auto' which sets frequencies based on modality.
    notch_frequency : float or None, optional
        Notch filter fundamental frequency in Hz; None disables notch filtering.
        Default is 50.
    notch_harmonics : int, optional
        Number of harmonics of notch frequency to filter.
        Default is 4.
    notch_width : float or None, optional
        Width of the notch filter; None enables automatic width.
        Default is None.
    reference_channels : str, 'average', or None, optional
        Reference channels for re-referencing; None disables re-referencing.
        Default is 'average'.
    amplitude_rejection_threshold : float or None, optional
        Amplitude threshold for rejection; None disables amplitude rejection.
        Default is 0.001.
    n_ica_components : int or None, optional
        Number of ICA components for artifact rejection; None disables ICA.
        Default is 25.
    laplacian_filter_neighbor_radius : float or None, optional
        Radius for Laplacian spatial filtering neighbor channels; None disables filtering.
        Default is 0.05.
    wavelet_type : {'db4', 'sym5', 'coif1'} or None, optional
        Wavelet type used for denoising; None disables wavelet denoising.
        Default is 'db4'.
    denoising_threshold_mode : {'soft', 'hard'}, optional
        Thresholding mode for denoising wavelet coefficients.
        Default is 'soft'.

    Attributes
    ----------
    np_output_data : np.ndarray
        The final preprocessed output data.
    np_denoised_data : np.ndarray
        Wavelet denoised data.
    np_smoothed_data : np.ndarray
        Smoothed data after Laplacian filtering.
    mne_artefact_free_data : mne.io.Raw
        ICA-filtered artefact-free MNE data.
    np_artefact_free_data : np.ndarray
        ICA-filtered artefact-free numpy data.
    mne_amplitude_compliant_data : mne.io.Raw
        Data after amplitude-based artifact annotation.
    mne_referenced_data : mne.io.Raw
        Re-referenced data.
    mne_filtered_data : mne.io.Raw
        Band-pass and notch filtered data.
    mne_raw_data : mne.io.Raw
        Raw input data as MNE object.
    """
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
                 n_ica_components: int | None = 25,
                 automatic_ic_labelling: bool = True,
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

        # preprocessing parameters:
        self._band_pass_frequencies = band_pass_frequencies
        self._notch_frequency = notch_frequency  # None leads to no notch filter
        self._notch_harmonics = notch_harmonics
        self._notch_width = notch_width  # None leads to automatic setting
        self._reference_channels = reference_channels  # None leads to no re-referencing
        self._amplitude_rejection_threshold = amplitude_rejection_threshold  # None leads to no amplitude thresholding
        self._n_ica_components = n_ica_components  # None leads to no automatic artefact rejection
        self._automatic_ic_labelling = automatic_ic_labelling
        self._manual_ics_to_exclude: list[int] = None
        self._laplacian_filter_neighbor_radius = laplacian_filter_neighbor_radius  # None -> no Laplacian filtering
        self._wavelet_type = wavelet_type  # None -> no wavelet denoising
        self._denoising_threshold_mode = denoising_threshold_mode

        ### result placeholders:
        # mne type:
        self._mne_amplitude_compliant_data = self._mne_filtered_data = None
        self._mne_referenced_data = self._mne_raw_data = None
        self._mne_ica_result = self._ica_automatic_labels = self._mne_artefact_free_data = None
        # np type:
        self._np_artefact_free_data = self._np_smoothed_data = None
        self._np_denoised_data = self._np_output_data = None
        # others:
        self._wavelet_coefficients = self._denoised_wavelet_coefficients = None

    ############# MAGIC FUNCTIONS #############
    def describe(self) -> str:
        """
        Return a detailed description of the BiosignalPreprocessor instance,
        including its general purpose and hierarchical preprocessing steps.
        """
        description = (
            "BiosignalPreprocessor: preprocesses EEG or EMG biosignals with filtering, referencing, "
            "artifact rejection, smoothing, and denoising steps.\n"
            "Access properties in hierarchical order for each step:\n"
            "- np_output_data (final output)\n"
            "- np_denoised_data\n"
            "- np_smoothed_data\n"
            "- mne_artefact_free_data / np_artefact_free_data\n"
            "- mne_amplitude_compliant_data\n"
            "- mne_referenced_data\n"
            "- mne_filtered_data\n"
            "- mne_raw_data / np_input_array (raw input)\n\n"
            f"Instance details:\n"
            f"- modality: {self.modality}\n"
            f"- sampling frequency: {self.sampling_freq} Hz\n"
            f"- data shape: {self.n_timesteps} timesteps x {self.n_channels} channels"
        )
        return description

    def __str__(self):
        """
        Return a user-friendly string summary by using describe().
        """
        return self.describe()

    def __repr__(self):
        """
        Return a detailed string representation by using describe().
        """
        return self.describe()

    ############# PROPERTIES #############
    ### input properties ###
    @property
    def np_input_array(self) -> np.ndarray:
        """
        Original input data array.

        Returns
        -------
        np.ndarray
            Input data with shape (timesteps, channels).
        """
        return self._np_input_array

    @np_input_array.setter
    def np_input_array(self, np_input_array: np.ndarray):
        self._np_input_array = np_input_array
        self.clean_downstream_results(change_in='import')

    @property
    def sampling_freq(self) -> int:
        """
        Sampling frequency of the input data.

        Returns
        -------
        int
            Sampling frequency in Hz.
        """
        return self._sampling_freq

    @sampling_freq.setter
    def sampling_freq(self, sampling_freq: int):
        self._sampling_freq = sampling_freq
        self.clean_downstream_results(change_in='import')

    @property
    def modality(self) -> Literal['eeg', 'emg']:
        """
        Biosignal modality type.

        Returns
        -------
        {'eeg', 'emg'}
            Data modality.
        """
        return self._modality

    @modality.setter
    def modality(self, modality: Literal['eeg', 'emg']):
        self._modality = modality
        self.clean_downstream_results(change_in='import')

    @property
    def n_timesteps(self) -> int:
        """ Number of timesteps (rows) of the provided input_array. """
        return self.np_input_array.shape[0]

    @property
    def n_channels(self) -> int:
        """ Number of channels (columns) of the provided input_array. """
        return self.np_input_array.shape[1]

    @property
    def band_pass_frequencies(self) -> tuple[float, float]:
        """
        Band-pass filter frequency range.

        Returns
        -------
        tuple of float
            (low_freq, high_freq) in Hz; auto-set based on modality if 'auto'.
        """
        if self._band_pass_frequencies == "auto":
            if self.modality == 'eeg': return (.1, 100)
            elif self.modality == 'emg': return (20, 500)
        else: return self._band_pass_frequencies

    @band_pass_frequencies.setter
    def band_pass_frequencies(self, band_pass_frequencies: tuple[float, float] | Literal['auto']):
        self._band_pass_frequencies = band_pass_frequencies
        self.clean_downstream_results(change_in='filtering')

    @property
    def notch_frequency(self) -> float | None:
        """
        Fundamental notch filter frequency.

        Returns
        -------
        float or None
            Frequency in Hz. None disables notch filtering.
        """
        return self._notch_frequency

    @notch_frequency.setter
    def notch_frequency(self, notch_frequency: float | None) -> None:
        self._notch_frequency = notch_frequency
        self.clean_downstream_results(change_in='filtering')

    @property
    def notch_harmonics(self) -> int:
        """
        Number of notch filter harmonics.

        Returns
        -------
        int
            Number of multiples of the notch frequency to be filtered.
        """
        return self._notch_harmonics

    @notch_harmonics.setter
    def notch_harmonics(self, notch_harmonics: int) -> None:
        self._notch_harmonics = notch_harmonics
        self.clean_downstream_results(change_in='filtering')

    @property
    def notch_width(self) -> float | None:
        """
        Width of each notch filter.

        Returns
        -------
        float or None
            Width of notch filter in Hz; None enables automatic width.
        """
        return self._notch_width

    @notch_width.setter
    def notch_width(self, notch_width: float | None) -> None:
        self._notch_width = notch_width
        self.clean_downstream_results(change_in='filtering')

    @property
    def reference_channels(self) -> str | Literal['average'] | None:
        """
        Channels used for referencing.

        Returns
        -------
        str, 'average', or None
            Reference channel specifier. None disables re-referencing.
        """
        return self._reference_channels

    @reference_channels.setter
    def reference_channels(self, reference_channels: str | Literal['average'] | None) -> None:
        self._reference_channels = reference_channels
        self.clean_downstream_results(change_in='referencing')

    @property
    def amplitude_rejection_threshold(self) -> float | None:
        """
        Threshold for amplitude-based artifact rejection.

        Returns
        -------
        float or None
            Amplitude threshold; None disables rejection.
        """
        return self._amplitude_rejection_threshold

    @amplitude_rejection_threshold.setter
    def amplitude_rejection_threshold(self, amplitude_rejection_threshold: float | None) -> None:
        self._amplitude_rejection_threshold = amplitude_rejection_threshold
        self.clean_downstream_results(change_in='amplitude thresholding')

    @property
    def n_ica_components(self) -> int:
        """
        Number of ICA components for artifact correction.

        Returns
        -------
        int or None
            Number of ICA components; None disables ICA processing.
        """
        return self._n_ica_components

    @n_ica_components.setter
    def n_ica_components(self, ica_components: int) -> None:
        self._n_ica_components = ica_components
        self.clean_downstream_results(change_in='artefact rejection')

    @property
    def automatic_ic_labelling(self) -> bool:
        return self._automatic_ic_labelling

    @automatic_ic_labelling.setter
    def automatic_ic_labelling(self, automatic_ic_labelling: bool) -> None:
        self._automatic_ic_labelling = automatic_ic_labelling
        self.clean_downstream_results(change_in='artefact rejection')

    @property
    def manual_ics_to_exclude(self) -> list[int]:
        """
        List of manually excluded ICA components.

        Returns
        -------
        list of int
            ICA components indices to exclude manually.
        """
        if self._manual_ics_to_exclude is None: return []
        return self._manual_ics_to_exclude

    @manual_ics_to_exclude.setter
    def manual_ics_to_exclude(self, value: list[int] | None) -> None:
        """
        Set manually excluded ICA components.

        Parameters
        ----------
        value : list of int or None
            List of ICA component indices to exclude.

        Notes
        -----
        It is recommended to use `plot_independent_component` to select components before exclusion.
        """
        self._manual_ics_to_exclude = value
        self.clean_downstream_results('artefact rejection')

    @property
    def laplacian_filter_neighbor_radius(self) -> float | None:
        """
        Radius to define neighboring channels for Laplacian filtering.

        Returns
        -------
        float or None
            Radius in same units as channel locations; None disables Laplacian filtering.
        """
        return self._laplacian_filter_neighbor_radius

    @laplacian_filter_neighbor_radius.setter
    def laplacian_filter_neighbor_radius(self, laplacian_filter_neighbor_radius: float | None) -> None:
        self._laplacian_filter_neighbor_radius = laplacian_filter_neighbor_radius
        self.clean_downstream_results(change_in='smoothing')

    @property
    def wavelet_type(self) -> Literal['db4', 'sym5', 'coif1'] | None:
        """
        Wavelet type for wavelet denoising.

        Returns
        -------
        str or None
            Wavelet name; None disables wavelet processing.
        """
        return self._wavelet_type

    @wavelet_type.setter
    def wavelet_type(self, wavelet_type: Literal['db4', 'sym5', 'coif1'] | None) -> None:
        self._wavelet_type = wavelet_type
        self.clean_downstream_results(change_in='denoising')

    @property
    def denoising_threshold_mode(self) -> Literal['soft', 'hard']:
        """
        Thresholding mode used for wavelet denoising.

        Returns
        -------
        {'soft', 'hard'}
            Thresholding method.
        """
        return self._denoising_threshold_mode

    @denoising_threshold_mode.setter
    def denoising_threshold_mode(self, denoising_threshold_mode: Literal['soft', 'hard']) -> None:
        self._denoising_threshold_mode = denoising_threshold_mode
        self.clean_downstream_results(change_in='denoising')

    ### calculation-based properties ###
    @property
    def mne_raw_data(self):
        """
        Raw biosignal data as an MNE Raw object.

        Returns
        -------
        mne.io.Raw
            Raw data formatted for MNE processing.
        """
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
        """
        Band-pass and notch filtered data in MNE Raw format.

        Returns
        -------
        mne.io.Raw
            Filtered biosignal data.
        """
        if self._mne_filtered_data is not None: return self._mne_filtered_data

        # compute:
        filtered_data = self.mne_raw_data.copy()
        filtered_data.filter(l_freq=self.band_pass_frequencies[0],
                             h_freq=self.band_pass_frequencies[1],
                             fir_design='firwin')
        self._mne_filtered_data = filtered_data
        if self.notch_frequency is not None: self._apply_notch_filter()
        return self._mne_filtered_data

    @property
    def mne_referenced_data(self):
        """
        Re-referenced MNE data.

        Returns
        -------
        mne.io.Raw
            Data after applying referencing scheme.
        """
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
        """
        Data after amplitude-based artifact detection and annotation.

        Returns
        -------
        mne.io.Raw
            Data annotated with bad segments and channels.
        """
        if self._mne_amplitude_compliant_data is not None: return self._mne_amplitude_compliant_data

        # initialise:
        if self.amplitude_rejection_threshold is None: return self.mne_referenced_data

        self._mne_amplitude_compliant_data = self.mne_referenced_data.copy()
        # annotate bad segments and discover bad channels based on amplitude thresholding:
        self._annotate_amplitude_based_artefacts()
        return self._mne_amplitude_compliant_data

    @property
    def bad_channels(self) -> list[str]:
        """
        List of bad channels detected based on amplitude thresholding.

        Returns
        -------
        list of str
            Names of channels marked as bad.
        """
        return self.mne_amplitude_compliant_data.info['bads']

    @property
    def mne_ica_result(self):
        """
        ICA decomposition result object.

        Returns
        -------
        mne.preprocessing.ICA
            Fitted ICA object for artifact removal.

        Raises
        ------
        ValueError
            If `n_ica_components` is not defined.
        """
        if self._mne_ica_result is not None: return self._mne_ica_result

        if self.n_ica_components is None: raise ValueError("n_ica_components needs to be defined!")
        # fit ICA:
        ica = mne.preprocessing.ICA(n_components=self.n_ica_components,
                                    max_iter='auto',
                                    method='infomax',
                                    random_state=42,
                                    # switches between nonlinearities, appears more robust but takes much longer (than FastICA)
                                    fit_params=dict(extended=True))
        # convergence is difficult with too short datasets, rule of thumb appears to be n-components x 20-30 = required_seconds
        ica.fit(self.mne_amplitude_compliant_data)
        self._mne_ica_result = ica
        return self._mne_ica_result

    @property
    def mne_artefact_free_data(self):
        """
        Artefact-free data after ICA component exclusion.

        Returns
        -------
        mne.io.Raw
            ICA cleaned data.
                """
        if self._mne_artefact_free_data is not None: return self._mne_artefact_free_data

        # otherwise compute
        if self.n_ica_components is None: return self.mne_amplitude_compliant_data

        # label ica components:
        if self._ica_automatic_labels is None:  # store internally
            self._ica_automatic_labels = mne_icalabel.label_components(
                self.mne_amplitude_compliant_data,
                                                             self.mne_ica_result, method='iclabel')
        probs, labels = self._ica_automatic_labels.values()
        print("Found the following IC labels:\n", labels)
        labels_to_exclude = ('heart beat', 'muscle artifact', 'channel noise')
        automatically_excluded_ics = [idx for idx, label in enumerate(labels) if label in labels_to_exclude]
        print(f"Will exclude {labels_to_exclude} ICs, that are: ", automatically_excluded_ics)

        # exclude such. we can access the private variable _mne_ica... here because we ensured it's computed by accessing the property above
        exclusion_list = automatically_excluded_ics + self.manual_ics_to_exclude
        print(f'Also excluding manual set ICs: {self.manual_ics_to_exclude}\n(change this selection via manual_ics_to_exclude parameter)')
        # set conversion interim to prevent duplicates
        self._mne_ica_result.exclude = list(set(exclusion_list))

        # save and return:
        self._mne_artefact_free_data = self.mne_ica_result.apply(self.mne_amplitude_compliant_data.copy())

        return self._mne_artefact_free_data

    @property
    def np_artefact_free_data(self):
        """
        Artefact-free data as numpy array.

        Returns
        -------
        np.ndarray
            ICA cleaned data array of shape (timesteps, channels).

        Notes
        -----
        May still contain bad channels which should be manually excluded.
        """
        if self._np_artefact_free_data is not None: return self._np_artefact_free_data

        # compute:
        self._np_artefact_free_data = self.mne_artefact_free_data.get_data().T
        print(f'np data also contains bad channels ({self.bad_channels}).\nConsider excluding such manually!')

        return self._np_artefact_free_data

    @property
    def np_smoothed_data(self) -> np.ndarray:
        """
        Smoothed data after Laplacian spatial filtering.

        Returns
        -------
        np.ndarray
            Smoothed signal data.

        Notes
        -----
        Laplacian filtering is only applied for EEG modality when a neighbor radius is set;
        otherwise returns artefact-free data unchanged.
        """
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
        """
        Computed wavelet decomposition coefficients of the smoothed data.

        Returns
        -------
        list of np.ndarray
            Wavelet coefficients at multiple decomposition levels.

        Raises
        ------
        ValueError
            If `wavelet_type` is not defined.
        """
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
        """
        Wavelet coefficients after threshold-based denoising.

        Returns
        -------
        list of np.ndarray
            Denoised wavelet coefficients.

        Raises
        ------
        ValueError
            If `wavelet_type` is not defined.
        """
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
        """
        Reconstructed signal from denoised wavelet coefficients.

        Returns
        -------
        np.ndarray
            Denoised biosignal data.

        Notes
        -----
        Returns smoothed data if wavelet denoising is disabled.
        """
        if self._np_denoised_data is not None: return self._np_denoised_data

        # skip computation if:
        if self.wavelet_type is None: return self.np_smoothed_data

        # compute and return:
        self._np_denoised_data = pywt.waverec(self.denoised_wavelet_coefficients, self.wavelet_type)
        return self._np_denoised_data

    @property
    def np_output_data(self):
        """
        Final output data after full preprocessing pipeline.

        Returns
        -------
        np.ndarray
            Fully preprocessed biosignal data.

        Notes
        -----
        Triggers full pipeline processing with progress indication.
        """
        if self._np_output_data is not None: return self._np_output_data

        # else compute with progress bar:
        print('Running full preprocessing pipeline...')
        for step in tqdm(['import', 'filtering', 'referencing', 'amplitude thresholding',
                                 'artefact rejection', 'smoothing', 'denoising']):
            # accessing properties leads to computation:
            if step == 'import': _ = self.mne_raw_data
            elif step == 'filtering': _ = self.mne_filtered_data
            elif step == 'referencing': _ = self.mne_referenced_data
            elif step == 'amplitude thresholding': _ = self.mne_amplitude_compliant_data
            elif step == 'artefact rejection': _ = self.np_artefact_free_data
            elif step == 'smoothing': _ = self.np_smoothed_data
            elif step == 'denoising': _ = self.np_denoised_data

        self._np_output_data = self.np_denoised_data
        return self._np_output_data

    ############# PREPROCESSING METHODS #############
    def _apply_notch_filter(self):
        """
        Apply notch filters to remove electrical interference frequencies.

        Raises
        ------
        ValueError
            If `notch_frequency` is not defined.
        """
        if self.notch_frequency is None: raise ValueError("notch_frequency needs to be defined!")
        self._mne_filtered_data = self.mne_filtered_data.copy().notch_filter(
            freqs=[self.notch_frequency * i for i in range(1, self.notch_harmonics+1)],
            picks='all', notch_widths=self.notch_width)

    def _annotate_amplitude_based_artefacts(self,
                                            min_duration: float = .025,
                                            max_bad_segments_percent: float = 5):
        """
        Annotate data segments and channels with amplitude-based artefacts.

        Raises
        ------
        ValueError
            If `amplitude_rejection_threshold` is not defined.
        """
        if self.amplitude_rejection_threshold is None: raise ValueError("amplitude_rejection_threshold needs to be defined!")
        reject_criteria = dict(eeg=self.amplitude_rejection_threshold, emg=self.amplitude_rejection_threshold)

        # derive annotations:
        annotations, bad_channels = mne.preprocessing.annotate_amplitude(
            self._mne_amplitude_compliant_data,
            peak=reject_criteria,
            min_duration=min_duration,  # minimum duration for consecutive samples to exceed or fall below threshold
            bad_percent=max_bad_segments_percent,  # channels with more bad segments will be marked as complete bad channels
        )
        print(f"Found {len(bad_channels)} bad channels.")

        # save result:
        self._mne_amplitude_compliant_data.set_annotations(annotations)
        self._mne_amplitude_compliant_data.info['bads'].extend(bad_channels)

    def clean_downstream_results(self,
                                 change_in: Literal['import', 'filtering', 'referencing', 'amplitude thresholding',
                                 'artefact rejection', 'smoothing', 'denoising']):
        """
        Clear cached intermediate results downstream of a specified processing step.

        Parameters
        ----------
        change_in : str
            Processing stage where changes occurred; must be one of
            {'import', 'filtering', 'referencing', 'amplitude thresholding',
             'artefact rejection', 'smoothing', 'denoising'}.

        Raises
        ------
        ValueError
            If the specified `change_in` category is invalid.

        Notes
        ------
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
            self._ica_automatic_labels = None
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
            self._np_artefact_free_data = None
            self._mne_ica_results = None
            self._ica_automatic_labels = None
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
            self._ica_automatic_labels = None
            self._np_smoothed_data = None
            self._np_denoised_data = None
            self._denoised_wavelet_coefficients = None
            self._wavelet_coefficients = None
            self._np_output_data = None
        elif change_in.lower() == 'amplitude thresholding':
            self._mne_amplitude_compliant_data = None
            self._mne_artefact_free_data = None
            self._np_artefact_free_data = None
            self._mne_ica_results = None
            self._ica_automatic_labels = None
            self._np_smoothed_data = None
            self._np_denoised_data = None
            self._denoised_wavelet_coefficients = None
            self._wavelet_coefficients = None
            self._np_output_data = None
        elif change_in.lower() == 'artefact rejection':
            self._mne_artefact_free_data = None
            self._np_artefact_free_data = None
            self._mne_ica_results = None
            self._ica_automatic_labels = None
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
        else: raise ValueError(f"change_in category: '{change_in}' is undefined!")

    def export_results(self, save_dir: Path | str, identifier: str = None) -> None:
        """ Exports results to .npy """
        # prepare save-title:
        title = f"Preprocessed {self.modality} {self.n_channels}ch {self.n_timesteps/self.sampling_freq}sec"
        if identifier is not None: title += f" ({identifier})"
        save_path = save_dir / filemgmt.file_title(f"{title}", ".npy")
        # save:
        np.save(save_path, self.np_output_data)
        print('Saved results to ', save_path)

    @staticmethod
    def discrete_fourier_transform(input_array: np.ndarray,
                                   sampling_freq: int,
                                   axis: Literal[0, 1] = 0, plot_result: bool = True, **plot_kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the discrete Fourier transform (DFT) of the input signal.

        Parameters
        ----------
        input_array : np.ndarray
            Input signal array (can be 1D or 2D).
        sampling_freq : int
            Data sampling frequency (Hz).
        axis : {0, 1}, optional
            Axis along which to compute the DFT for 2D input. Default is 0.
        plot_result : bool, optional
            Whether to plot the amplitude spectrum. Default is True.
        **plot_kwargs
            Additional keyword arguments to pass to the plotting function.

        Returns
        -------
        amplitude_spectrum : np.ndarray
            Magnitude of the DFT for positive frequencies.
        freqs_pos : np.ndarray
            Corresponding positive frequency bins in Hz.

        Raises
        ------
        AttributeError
            If axis is not specified for 2D input.
        """
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
    def plot_independent_component(self, ic_index: int, verbose: bool = True):
        """
        Plot properties of a specified independent component from ICA.

        Parameters
        ----------
        ic_index : int
            Index of the independent component to plot.
        """
        if verbose:
            print("Consider manually selecting ICs to exclude based on the following comments:")
            print("Muscle artifacts are localized spatially, show high-frequency power spectra with positive slopes, and have irregular spike-like time courses.")
            print("Heartbeat components have bilateral topographies, clear spectral peaks at heartbeat frequency, and periodic variance across time.")
            print("Noise shows non-physiological spatial patterns with narrow-band frequency peaks or broadband noise and variable variance patterns.")
        self.mne_ica_result.plot_properties(self.mne_amplitude_compliant_data, ic_index)

    def plot_data_overview(self):
        """
        Display an interactive overview plot of the amplitude-compliant data.

        Notes
        -----
        Can be used to manually mark or unmark bad channels.
        Changing bad channels triggers cleaning of downstream results.
        """
        temp_bad_channels = self.bad_channels.copy()

        # open interactive plot:
        self.mne_amplitude_compliant_data.plot()

        # if new bad channels have been (de-)selected:
        if temp_bad_channels != self.mne_amplitude_compliant_data.info['bads']:
            print('New bad channels (de-)selected, will clean downstream results.')
            self.clean_downstream_results(change_in='amplitude thresholding')

if __name__ == '__main__':
    ROOT = Path().resolve().parent.parent
    QTC_DATA = ROOT / "data" / "qtc_measurements" / "2025_06"
    subject_data_dir = QTC_DATA / "sub-10"

    # load data:
    print('Loading data...')
    input_file = np.load(subject_data_dir / "motor_eeg_full.npy").T
    data_modality: Literal['eeg', 'emg'] = 'eeg'
    sampling_freq = 2048  # Hz

    # define prepper:
    print('Initialising BiosignalPreprocessor...')
    prepper = BiosignalPreprocessor(
        np_input_array=input_file,
        sampling_freq=sampling_freq,
        modality=data_modality,
        band_pass_frequencies='auto',
    )

    #prepper.plot_data_overview()

    # automatic artefact rejection:
    _ = prepper.mne_artefact_free_data

    # manual inspection of ICs:
    for ic_ind in range(prepper.n_ica_components):
        prepper.plot_independent_component(ic_ind,
                                           verbose=(ic_ind==0),  # print only on first iteration
                                           )


    #prepper.discrete_fourier_transform(prepper.np_output_data)