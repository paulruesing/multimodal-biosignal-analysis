import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable, Literal, Tuple
import os
import json
from tqdm import tqdm
import mne
import mne_icalabel
import pywt
from scipy import signal

import src.pipeline.signal_features as features
import src.pipeline.data_surrogation as surrogation
import src.pipeline.visualizations as visualizations
from src.pipeline.visualizations import plot_freq_domain, EEG_CHANNELS, EEG_CHANNEL_IND_DICT
import src.utils.file_management as filemgmt

# constant parameters:
EMG_CHANNELS = [f"EMG{i:02d}" for i in range(64)]

class BiosignalPreprocessor:
    """
    Class for preprocessing biosignal data such as EEG and EMG.

    Parameters
    ----------
    np_input_data : np.ndarray
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
        Amplitude threshold for rejection in peak-to-peak voltage; None disables amplitude rejection.
        Default is 0.003.
    n_ica_components : int or None, optional
        Number of ICA components for artifact rejection; None disables ICA.
        Default is 25.
    laplacian_filter_neighbor_radius : float or 'auto', optional
        Radius for Laplacian spatial filtering neighbor channels; None disables filtering.
        Default is 0.05 for 'eeg' and 0.015 for 'emg' (returned for 'auto').
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
    np_spatially_filtered_data : np.ndarray
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
                 np_input_data: np.ndarray,  # shape: (timesteps, channels)
                 sampling_freq: int,  # Hz
                 modality: Literal['eeg', 'emg'],
                 band_pass_frequencies: tuple[float, float] | Literal['auto'] = 'auto',
                 notch_frequency: float | None = 50,
                 notch_harmonics: int = 4,
                 notch_width: float | None = None,
                 reference_channels: str | Literal['average'] | None = 'average',
                 amplitude_rejection_threshold: float | None = .003,
                 n_ica_components: int | None = 25,
                 automatic_ic_labelling: bool = True,
                 laplacian_filter_neighbor_radius: float | None | Literal['auto'] = 'auto',
                 wavelet_type: Literal['db4', 'sym5', 'coif1'] | None = None,
                 denoising_threshold_mode: Literal['soft', 'hard'] = 'soft',
                 ):
        """
        Properties follow hierarchical logic (higher ones containing filtering steps pertaining to lower ones):
        - np_output_data
        - np_denoised_data
        - np_spatially_filtered_data
        - mne_artefact_free_data or np_artefact_free_data
        - mne_amplitude_compliant_data
        - mne_referenced_data
        - mne_filtered_data
        - mne_raw_data or np_input_data
        """
        # all private attributes accessible via properties, because setting recomputes filtered data
        # import parameters:
        assert np_input_data.shape[1] < np_input_data.shape[0], "Should be more timesteps (rows) than channels (columns)!"
        self._np_input_data = np_input_data
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
        self._np_artefact_free_data = self._np_spatially_filtered_data = None
        self._np_denoised_data = self._np_output_data = None
        # others:
        self._wavelet_coefficients = self._denoised_wavelet_coefficients = None

    ############# DESCRIPTIVE FUNCTIONS #############
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
            "- np_spatially_filtered_data\n"
            "- mne_artefact_free_data / np_artefact_free_data\n"
            "- mne_amplitude_compliant_data\n"
            "- mne_referenced_data\n"
            "- mne_filtered_data\n"
            "- mne_raw_data / np_input_data (raw input)\n\n"
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

    ############# CLASS INITIALISATION METHODS #############
    @classmethod
    def init_from_config(cls, config_file_path: Path | str, np_input_data: np.ndarray):
        """ Returns class instance from config file (.json path) and input data (np.ndarray). """
        if str(config_file_path)[-5:] != ".json": raise ValueError("Provided file path must be .json")

        # read json
        with open(config_file_path, "r") as f:
            config_dict = json.load(f)

        try:  # read out non-initialisation attributes
            manual_ics_to_exclude = config_dict.pop('manual_ics_to_exclude')
        except KeyError: manual_ics_to_exclude = None
        try: _ = config_dict.pop('bad_channels')  # this doesn't matter for Preprocessor but data analysis, hence remove
        except KeyError: pass

        # initialise (from input array and config_dict) and return instance:
        instance = cls(np_input_data=np_input_data, **config_dict)

        # set (not-initialisation) attributes
        if manual_ics_to_exclude is not None:
            instance.manual_ics_to_exclude = manual_ics_to_exclude

        return instance

    ############# EXPORT METHODS #############
    def export_config(self, save_dir: Path | str, identifier: str = None) -> None:
        """ Exports config-file with all attributes as .json """
        # prepare save-title:
        title = f"Preprocessor Config {self.modality} {self.n_channels}ch"
        if identifier is not None: title += f" ({identifier})"
        save_path = save_dir / filemgmt.file_title(f"{title}", ".json")

        # construct json:
        attr_list = ['sampling_freq', 'modality', 'band_pass_frequencies', 'notch_frequency',
                     'notch_harmonics', 'notch_width', 'reference_channels', 'amplitude_rejection_threshold',
                     'n_ica_components', 'automatic_ic_labelling', 'laplacian_filter_neighbor_radius', 'wavelet_type',
                     'denoising_threshold_mode', 'manual_ics_to_exclude', 'bad_channels']
        config_dict = {attr_name: getattr(self, attr_name) for attr_name in attr_list}

        # save:
        with open(save_path, "w") as json_file:
            json.dump(config_dict, json_file, indent=4)  # Pretty print with indent=4
        print('Saved config to ', save_path)

    def export_results(self, save_dir: Path | str, identifier: str = None, with_config: bool = True) -> None:
        """ Exports results (np_output_data) to .npy """
        # prepare save-title:
        title = f"Preprocessed {self.modality} {self.n_channels}ch {int(self.n_timesteps/self.sampling_freq)}sec"
        if identifier is not None: title += f" ({identifier})"
        save_path = save_dir / filemgmt.file_title(f"{title}", ".npy")
        # save:
        np.save(save_path, self.np_output_data)
        print('Saved results to ', save_path)

        if with_config:
            self.export_config(save_dir, identifier=identifier)

    ############# PROPERTIES #############
    ### input properties ###
    @property
    def np_input_data(self) -> np.ndarray:
        """
        Original input data array.

        Returns
        -------
        np.ndarray
            Input data with shape (timesteps, channels).
        """
        return self._np_input_data

    @np_input_data.setter
    def np_input_data(self, np_input_data: np.ndarray):
        self._np_input_data = np_input_data
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
        return self.np_input_data.shape[0]

    @property
    def n_channels(self) -> int:
        """ Number of channels (columns) of the provided input_array. """
        return self.np_input_data.shape[1]

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
        self.clean_downstream_results(change_in='ica computation')

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
        if self._laplacian_filter_neighbor_radius == 'auto':
            if self.modality == 'eeg': return .05
            elif self.modality == 'emg': return None
            else: raise ValueError(f"Unknown modality: {self.modality}")
        else:
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
    def mne_raw_data(self) -> mne.io.RawArray:
        """
        Raw biosignal data as an MNE Raw object.

        Returns
        -------
        mne.io.Raw
            Raw data formatted for MNE processing.
        """
        if self._mne_raw_data is not None: return self._mne_raw_data

        # initialise and return:
        ch_names = EEG_CHANNELS if self.modality == 'eeg' else EMG_CHANNELS
        data_info = mne.create_info(ch_names=ch_names,
                                    sfreq=self.sampling_freq,
                                    ch_types=self.modality,)
        raw_data = mne.io.RawArray(self.np_input_data.T, data_info)

        # electrode positions:
        if self.modality == 'eeg':
            raw_data.set_montage(mne.channels.make_standard_montage('standard_1020'))

        elif self.modality == 'emg':
            # GR10MM0808 OT Bioelettronica EMG electrode:
            #   8x8 with 3mm electrode diameter and 10mm inter-electrode distance -> total 8cm*8cm
            # positions: array of shape (64, 3) in METERS
            #   hence add a third dimension (=0) and
            #   scale the positions from visualizations.py to 8cm*8cm
            pos_array = np.array(list(visualizations.EMG_POSITIONS.values()))  # (64,2)
            pos_array = np.concatenate([pos_array, np.zeros_like(pos_array[:, 0:1])], axis=1)  # (64,3)
            current_x_span = np.max(pos_array[:, 0]) - np.min(pos_array[:, 0])
            scale_factor = .08 / current_x_span
            pos_array *= scale_factor

            # save for latter accessing with electrode_positions property:
            self._electrode_positions = pos_array
            # we don't use set_montage here because MNE doesn't foresee montages for EMG data

        self._mne_raw_data = raw_data
        return self._mne_raw_data

    @property
    def electrode_positions(self) -> np.ndarray:
        """ Returns numpy array of shape (n_channels, 3). """
        if self.modality == 'emg':  # for EMG, MNE doesn't store montages
            # regularly this private attribute is defined in mne_raw_data property, therefore access such:
            _ = self.mne_raw_data
            if hasattr(self, "_electrode_positions"):
                return self._electrode_positions
            else:
                raise ValueError("For EMG data private attribute _electrode_positions needs to be defined before accessing electrode_positions property.")

        # else for EEG data:
        _ = self.mne_artefact_free_data.get_montage()  # checks whether montage was provided
        return np.array([self.mne_artefact_free_data.info['chs'][i]['loc'][:3] for i in
                        range(len(self.mne_artefact_free_data.ch_names))])  # 3D coords of channels

    @property
    def mne_filtered_data(self) -> mne.io.RawArray:
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
                             fir_design='firwin', picks='all')
        self._mne_filtered_data = filtered_data
        if self.notch_frequency is not None: self._apply_notch_filter()
        return self._mne_filtered_data

    @property
    def mne_referenced_data(self) -> mne.io.RawArray:
        """
        Re-referenced MNE data. Currently only re-references EEG data, EMG data remains unaltered.

        Returns
        -------
        mne.io.Raw
            Data after applying referencing scheme.
        """
        if self._mne_referenced_data is not None: return self._mne_referenced_data

        # compute:
        if self.reference_channels is None or self.modality == 'emg': return self.mne_filtered_data
        temp_data = self.mne_filtered_data.copy()
        temp_data.set_eeg_reference(ref_channels=self.reference_channels,
                                    ch_type='auto')
        self._mne_referenced_data = temp_data
        return self._mne_referenced_data

    @property
    def mne_amplitude_compliant_data(self) -> mne.io.RawArray:
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
        _ = self._annotate_amplitude_based_artefacts()  # returns bad channels -> not needed
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
        if self.modality == 'emg': raise ValueError("ica fitting only works (and is only intended) for EEG data.")
        # fit ICA:
        ica = mne.preprocessing.ICA(n_components=self.n_ica_components,
                                    max_iter='auto',
                                    method='infomax',
                                    random_state=42,
                                    # switches between nonlinearities, appears more robust but takes much longer (than FastICA)
                                    fit_params=dict(extended=True))
        # convergence is difficult with too short datasets, rule of thumb appears to be n-components x 20-30 = required_seconds
        ica.fit(self.mne_amplitude_compliant_data, picks="all")
        self._mne_ica_result = ica
        return self._mne_ica_result

    @property
    def mne_artefact_free_data(self) -> mne.io.RawArray:
        """
        Artefact-free data after ICA component exclusion.
        This step is skipped if n_ica_components=None or for EMG data!

        Returns
        -------
        mne.io.Raw
            ICA cleaned data.
                """
        if self._mne_artefact_free_data is not None: return self._mne_artefact_free_data

        # otherwise compute
        if self.n_ica_components is None or self.modality == 'emg': return self.mne_amplitude_compliant_data

        # label ica components:
        if self._ica_automatic_labels is None:  # store internally
            self._ica_automatic_labels = mne_icalabel.label_components(
                self.mne_amplitude_compliant_data,
                                                             self.mne_ica_result, method='iclabel')
        probs, labels = self._ica_automatic_labels.values()
        mne.utils.logger.info("Found the following IC labels:\n" + str(labels))
        labels_to_exclude = ('heart beat', 'muscle artifact', 'channel noise', 'eye blink')
        automatically_excluded_ics = [idx for idx, label in enumerate(labels) if label in labels_to_exclude]
        mne.utils.logger.info(f"Will exclude {labels_to_exclude} ICs, that are: " + str(automatically_excluded_ics))

        # exclude such. we can access the private variable _mne_ica... here because we ensured it's computed by accessing the property above
        exclusion_list = automatically_excluded_ics + self.manual_ics_to_exclude
        mne.utils.logger.info(f'Also excluding manual set ICs: {self.manual_ics_to_exclude}\n(change this selection via manual_ics_to_exclude parameter)')
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

        if len(self.bad_channels) > 0:  # if there are bad channels
            if self.modality == 'eeg':
                bad_channel_inds = [EEG_CHANNEL_IND_DICT[ch] for ch in self.bad_channels]
            else:
                bad_channel_inds = [ch[-2:] for ch in self.bad_channels]
            print(f'np data also contains bad channels ({self.bad_channels}, corresponding to {bad_channel_inds}).\nConsider excluding such manually!')

        return self._np_artefact_free_data

    @property
    def np_spatially_filtered_data(self) -> np.ndarray:
        """
        Spatially sharpened data after Laplacian spatial filtering.

        Returns
        -------
        np.ndarray
            Smoothed signal data.

        Notes
        -----
        Laplacian filtering is only applied for EEG modality when a neighbor radius is set;
        otherwise returns artefact-free data unchanged.
        """
        if self._np_spatially_filtered_data is not None: return self._np_spatially_filtered_data

        # reasons to omit Laplacian spatial filtering:
        if self.laplacian_filter_neighbor_radius is None: return self.np_artefact_free_data

        neighbors = self.get_neighboring_electrodes_mapping()

        # Compute Laplacian-filtered data:
        laplacian_data = np.zeros_like(self.np_artefact_free_data)  # prepare output array
        for i, neigh_idx in enumerate(neighbors):  # iterate over channels
            if len(neigh_idx) > 0:  # subtract neighbors' average:
                laplacian_data[:, i] = self.np_artefact_free_data[:, i] - self.np_artefact_free_data[:, neigh_idx].mean(axis=1)
            else:  # no neighbors: keep original:
                laplacian_data[:, i] = self.np_artefact_free_data[:, i]

        self._np_spatially_filtered_data = laplacian_data
        return self._np_spatially_filtered_data

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
        max_level = pywt.dwt_max_level(len(self.np_spatially_filtered_data),
                                       pywt.Wavelet(self.wavelet_type).dec_len)  # up to which level to decompose

        # compute wavelet coefficients:
        self._wavelet_coefficients = pywt.wavedec(self.np_spatially_filtered_data, self.wavelet_type,
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
        if self.wavelet_type is None: return self.np_spatially_filtered_data

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
            elif step == 'smoothing': _ = self.np_spatially_filtered_data
            elif step == 'denoising': _ = self.np_denoised_data

        self._np_output_data = self.np_denoised_data
        return self._np_output_data

    ############# PREPROCESSING METHODS #############
    @staticmethod
    def mne_to_numpy(mne_data: mne.io.RawArray) -> np.ndarray:
        return mne_data.get_data().T

    @staticmethod
    def numpty_to_mne(np_data: np.ndarray,
                      sampling_freq: float,
                      modality: Literal['eeg', 'emg'],
                      ) -> mne.io.RawArray:
        data_info = mne.create_info(ch_names=EEG_CHANNELS if modality == 'eeg' else EMG_CHANNELS,
                                    sfreq=sampling_freq,
                                    ch_types=modality, )
        return mne.io.RawArray(np_data.T, data_info)

    def get_neighboring_electrodes_mapping(self) -> list[list[int]]:
        """
        Derives neighboring electrodes based on mne-montage (3d montage information) and
        laplacian_filter_neighbor_radius radius threshold.

        Returns a list (length = n_channels) containing lists of neighbors per channel (electrode).
        """
        # sanity checks
        if self.laplacian_filter_neighbor_radius is None:
            raise ValueError("laplacian_filter_neighbor_radius needs to be defined!")

        # fetch electrode positions:
        pos = self.electrode_positions  # 3D coords of channels

        # identify proximal EEG channels based on positional mapping:
        neighbors = []
        for i, pos_i in enumerate(pos):
            dists = np.linalg.norm(pos - pos_i,
                                   axis=1)  # compute Euclidean dist from current electrode to all others
            # find neighbors by: excluding self and keeping channels within radius
            neigh = np.where((dists > 0) & (dists < self.laplacian_filter_neighbor_radius))[0]
            neighbors.append(neigh.tolist())  # store as list
        return neighbors

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
                                            input_mne_data: mne.io.RawArray | None = None,
                                            min_duration: float = .025,
                                            max_bad_segments_percent: float = 5,
                                            inplace: bool = True, ) -> list[int]:
        """
        Annotate data segments and channels with amplitude-based artefacts.
        Initialise self._mne_amplitude_compliant_data beforehand because this will be annotated.

        Raises
        ------
        ValueError
            If `amplitude_rejection_threshold` is not defined.
        """
        if self.amplitude_rejection_threshold is None: raise ValueError("amplitude_rejection_threshold needs to be defined!")
        # if inplace: uses _mne_amplitude_compliant_data even though input_mne_data provided
        use_internal_data = input_mne_data is None or inplace
        if self._mne_amplitude_compliant_data is None and use_internal_data:
            raise ValueError("If inplace operation is desired, _mne_amplitude_compliant_data needs to be initialised beforehand!")
        reject_criteria = dict(eeg=self.amplitude_rejection_threshold, emg=self.amplitude_rejection_threshold)

        # derive annotations:
        annotations, bad_channels = mne.preprocessing.annotate_amplitude(
            self._mne_amplitude_compliant_data if use_internal_data else input_mne_data,
            peak=reject_criteria,
            min_duration=min_duration,  # minimum duration for consecutive samples to exceed or fall below threshold
            bad_percent=max_bad_segments_percent,  # channels with more bad segments will be marked as complete bad channels
            picks='all',
        )
        mne.utils.logger.info(f"Found {len(bad_channels)} bad channels.")

        # save result:
        if inplace:
            self._mne_amplitude_compliant_data.set_annotations(annotations)
            self._mne_amplitude_compliant_data.info['bads'].extend(bad_channels)

        # convert to indices (starting at 1 and return), for EMG we can just cut off the first 3 letters (EMG13 -> 13)
        bad_channel_inds = [EEG_CHANNEL_IND_DICT[ch] for ch in bad_channels] if self.modality == 'eeg' else [int(ch[3:]) for ch in bad_channels]
        if len(bad_channel_inds) == self.n_channels: raise ValueError("current amplitude_rejection_threshold causes all channels to be marked as bad!")
        return bad_channel_inds

    def clean_downstream_results(self,
                                 change_in: Literal['import', 'filtering', 'referencing', 'amplitude thresholding',
                                 'ica computation', 'artefact rejection', 'smoothing', 'denoising']):
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
        - np_spatially_filtered_data
        - mne_artefact_free_data or np_artefact_free_data
        - mne_amplitude_compliant_data
        - mne_referenced_data
        - mne_filtered_data
        - mne_raw_data or np_input_data
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
            self._np_spatially_filtered_data = None
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
            self._np_spatially_filtered_data = None
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
            self._np_spatially_filtered_data = None
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
            self._np_spatially_filtered_data = None
            self._np_denoised_data = None
            self._denoised_wavelet_coefficients = None
            self._wavelet_coefficients = None
            self._np_output_data = None
        elif change_in.lower() == 'ica computation':
            self._mne_ica_results = None
            self._ica_automatic_labels = None
            self._np_artefact_free_data = None
            self._mne_artefact_free_data = None
            self._np_spatially_filtered_data = None
            self._np_denoised_data = None
            self._denoised_wavelet_coefficients = None
            self._wavelet_coefficients = None
            self._np_output_data = None
        elif change_in.lower() == 'artefact rejection':
            self._mne_artefact_free_data = None
            self._np_artefact_free_data = None
            self._np_spatially_filtered_data = None
            self._np_denoised_data = None
            self._denoised_wavelet_coefficients = None
            self._wavelet_coefficients = None
            self._np_output_data = None
        elif change_in.lower() == 'smoothing':
            self._np_spatially_filtered_data = None
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

    ############# VALIDATION METHODS #############
    def validate_filtering(self,
                           target_freq: float = 21.5,
                           freq_window: float = 8.5,
                           verbose: bool = True) -> tuple[float, float]:
        """ To be commented.
        Returns increase in SNR (based on target freq.) due to filtering.
        Default arguments are CMC-reelvant beta band: 21.5±8.5 Hz
        """
        with mne.utils.use_log_level('warning'):  # context manager to keep console output clean
            ### SNR improvement:
            # compute SNR for np.input_data:
            input_snr = features.compute_spectral_snr(self.np_input_data,
                                                self.sampling_freq,
                                                target_freq=target_freq, freq_window=freq_window)

            # compute SNR for mne_filtered_data (band pass and notch)
            filtered_snr = features.compute_spectral_snr(self.mne_to_numpy(self.mne_filtered_data),
                                                self.sampling_freq,
                                                target_freq=target_freq, freq_window=freq_window)
            snr_improvement = filtered_snr - input_snr
            if verbose: print(f'[VALIDATION] Target-band SNR improvement due to filtering: {snr_improvement:.3f} dB (now {filtered_snr:.3f} dB)')

            ### PSD consistency:
            # calculate PSDs:
            raw_freqs, raw_psd = signal.welch(self.np_input_data, axis=0,
                                              fs=self.sampling_freq, nperseg=self.sampling_freq * 4,  # 4-second window
                                              )
            filtered_freqs, filtered_psd = signal.welch(self.mne_to_numpy(self.mne_filtered_data), axis=0,
                                              fs=self.sampling_freq, nperseg=self.sampling_freq * 4,  # 4-second window
                                              )

            # define target band:
            assert np.array_equal(raw_freqs,filtered_freqs), "PSD result frequencies should be equivalent between un- and pre-processed data."
            target_band = (raw_freqs < target_freq + freq_window) & (raw_freqs > target_freq - freq_window)

            # compare mean PSD in target band:
            raw_mean_psd = 10 * np.log10(np.mean(raw_psd[target_band]))
            filtered_mean_psd = 10 * np.log10(np.mean(filtered_psd[target_band]))
            psd_difference = filtered_mean_psd - raw_mean_psd
            if verbose: print(f'[VALIDATION] Target-band PSD difference due to filtering: {psd_difference:.3f} dB')

            # return both differences:
            return snr_improvement, psd_difference

    def validate_referencing(self,
                             target_freq: float = 21.5,
                             freq_window: float = 8.5,
                             verbose: bool = True,
                            ) -> float:
        with mne.utils.use_log_level('warning'):  # context manager to keep console output clean
            # compute SNR for mne_filtered_data (band pass and optionally notch):
            input_snr = features.compute_spectral_snr(self.mne_to_numpy(self.mne_filtered_data),
                                                      self.sampling_freq,
                                                      target_freq=target_freq, freq_window=freq_window)

            # compute SNR for mne_referenced_data (re-referenced):
            filtered_snr = features.compute_spectral_snr(self.mne_to_numpy(self.mne_referenced_data),
                                                         self.sampling_freq,
                                                         target_freq=target_freq, freq_window=freq_window)
            snr_improvement = filtered_snr - input_snr
            if verbose: print(f'[VALIDATION] Target-band SNR improvement due to referencing: {snr_improvement:.3f} dB (now {filtered_snr:.3f} dB)')
            return snr_improvement

    def validate_amplitude_thresholding(self,
                                        n_runs: int = 10,
                                        verbose: bool = True) -> tuple[float, float]:
        """ Returns specificity (= true neg. rate) and selectivity (= true pos. rate) for surrogate bad channel recognition based on amplitude thresholding. """
        with mne.utils.use_log_level('warning'):  # context manager to keep console output clean
            all_channels = [i for i in range(self.n_channels)]
            specificity_list = []; selectivity_list = []
            # iterative procedure, to average specificity / selectivity metrics:
            for trial in range(n_runs):
                # create surrogate:
                surrogate, amended_channels = surrogation.insert_bad_channels(self.mne_to_numpy(self.mne_referenced_data),
                                                            axis=0, scale_range=(5, 15))
                unchanged_channels = [ch for ch in all_channels if ch not in amended_channels]

                # detect bad channels:
                detected_bad_channels = self._annotate_amplitude_based_artefacts(input_mne_data=self.numpty_to_mne(surrogate,
                                                                                                                   self.sampling_freq,
                                                                                                                   self.modality),
                                                                                 inplace=False)

                # compute metrics (positive = detected as bad)
                false_positives = [ch for ch in unchanged_channels if ch in detected_bad_channels]
                true_positives = [ch for ch in amended_channels if ch in detected_bad_channels]
                false_negatives = [ch for ch in amended_channels if ch not in detected_bad_channels]
                true_negatives = [ch for ch in unchanged_channels if ch not in detected_bad_channels]

                # specificity = true_neg / (true_neg + false_pos) = 1 - false_pos_rate
                specificity_list.append(len(true_negatives) / (len(true_negatives) + len(false_positives)))
                # selectivity = true_pos / (true_pos + false_neg) = 1 - false_neg_rate
                selectivity_list.append(len(true_positives) / (len(true_positives) + len(false_negatives)))

            # average and return metrics:
            specificity = np.nanmean(specificity_list).item(); selectivity = np.nanmean(selectivity_list).item()
            if verbose: print(f'[VALIDATION] Amplitude-Thresholding for Bad Channel Detection:\n\tSpecificity (true neg.): {specificity:.3f}\n\tSelectivity (true pos.): {selectivity:.3f}')
            return np.nanmean(specificity_list).item(), np.nanmean(selectivity_list).item()

    # todo: validate_artefact_rejection (trial-to-trial variability)
    # later, when multi-trial data is processed
    def validate_spatial_filtering(self, verbose: bool = True) -> float:
        """ Averages coherence over all frequencies. """
        with mne.utils.use_log_level('warning'):  # context manager to keep console output clean
            # derive neighbors (based on 3d-positional information from MNE and laplacian_filter_neighbor_radius):
            neighbor_mapping = self.get_neighboring_electrodes_mapping()

            # compute average magn. squared coherence with neighboring electrodes before and after spatial filtering:
            if verbose: print(f"[VALIDATION] Computing local coherence for all electrodes before and after spatial filtering...")
            for step_ind, data in enumerate([self.np_artefact_free_data, self.np_spatially_filtered_data]):
                ch_local_coherences = []  # average neighbor coherence list
                for ch_ind, neighbors in enumerate(tqdm(neighbor_mapping)):  # takes ~2-5s per electrode:
                    pairwise_coherences = []  # pairwise coherence list
                    for neighbor_ind in neighbors:
                        # compute pairwise coherence (magn. squared -> stationary, non-time-lagged):
                        freqs, coh = signal.coherence(x=data[:, ch_ind],
                                                      y=data[:, neighbor_ind],
                                                      fs=self.sampling_freq, axis=0)
                        # average over all frequencies (might be changed down the road):
                        pairwise_coherences.append(np.nanmean(coh))

                    # average over neighbors:
                    ch_local_coherences.append(np.nanmean(pairwise_coherences))

                # average over electrodes and store depending on which data was used:
                if step_ind == 0:  # before
                    local_coherence_before = np.nanmean(ch_local_coherences).item()
                    if verbose: print(f"[VALIDATION] Local Mag.Sq. Coherence BEFORE spatial filtering: {local_coherence_before:.3f}")
                elif step_ind == 1:  # after
                    local_coherence_after = np.nanmean(ch_local_coherences).item()
                    if verbose: print(f"[VALIDATION] Local Mag.Sq. Coherence AFTER spatial filtering: {local_coherence_after:.3f}")

            # compute coerence diff:
            coh_diff = local_coherence_after - local_coherence_before
            if verbose: print(f"[VALIDATION] Local coherence (~cross talk) hence changed by: {coh_diff:.3f}")
            return coh_diff

    # todo: think how to include surrogate data here
    def validate_wavelet_denoising(self,
                                   target_freq: float = 21.5,
                                   freq_window: float = 8.5,
                                   verbose: bool = True, ) -> float:
        """ Noise is defined as frequencies outside the target band. Default target freq. is 21.5 ±8.5 Hz. """
        with mne.utils.use_log_level('warning'):  # context manager to keep console output clean
            # compute SNR for spatially filtered data (step before):
            input_snr = features.compute_spectral_snr(self.np_spatially_filtered_data,
                                                      self.sampling_freq,
                                                      target_freq=target_freq, freq_window=freq_window)

            # compute SNR for wavelet denoised data:
            filtered_snr = features.compute_spectral_snr(self.np_denoised_data,
                                                         self.sampling_freq,
                                                         target_freq=target_freq, freq_window=freq_window)
            snr_improvement = filtered_snr - input_snr
            if verbose: print(
                f'[VALIDATION] Target-band SNR improvement due to wavelet denoising: {snr_improvement:.3f} dB (now {filtered_snr:.3f} dB)')
            return snr_improvement

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


########## AUXILIARY METHODS ##########
def import_npy_with_config(file_title: str, data_dir: str | Path,
                           load_only_first_n_seconds: int | None = None,
                           sampling_rate_Hz: int = 2048,
                           retrieve_latest_config: bool = True,
                           bad_channel_treatment: Literal['None', 'Zero'] = 'Zero',
                           channel_subset_inds: list[int] | None = None,
                           ) -> tuple[np.ndarray, dict]:
    """ Returns loaded data and config dict. Requires file title to contain "Preprocessed" """
    print(f'Searching most recent file {file_title} in {data_dir}...')
    # mmap_mode='r': memory-mapped read-only access (would accelerate but sometimes deletes files)
    file_path = filemgmt.most_recent_file(data_dir, ".npy", [file_title, "Preprocessed"])
    file = np.load(file_path)
    # shorten file for testing purposes, if desired:
    if load_only_first_n_seconds is not None: file = file[:sampling_rate_Hz * int(load_only_first_n_seconds), :]

    if retrieve_latest_config:
        try:  # search matching config, saved as e.g. "... eeg 64ch (FILE_TITLE).json"
            config_file = filemgmt.most_recent_file(data_dir, ".json", [file_title])
            with open(config_file, "r") as f:
                config_dict = json.load(f)
        except ValueError:
            print(f"No config file found for {file_title}")
            config_dict = None
    else:
        config_dict = None

    if config_dict is None:  # manually enter important properties
        config_dict = {'sampling_freq': 2048, 'bad_channels': []}
        assert config_dict[
                   'sampling_freq'] == sampling_rate_Hz, "sampling_rate_Hz parameter doesn't match sampling frequency found in config file!"

    # remove bad channels: (todo: ponder, set to zero?)
    if bad_channel_treatment == 'Zero':
        if len(config_dict['bad_channels']) > 0: print(
            f"Setting the following channels to 0: {config_dict['bad_channels']}")
        if config_dict['modality'] == "eeg":
            channel_inds_to_remove = [EEG_CHANNEL_IND_DICT[ch] for ch in config_dict['bad_channels']]
        else:
            channel_inds_to_remove = [ch[-2:] for ch in config_dict['bad_channels']]

        file[:, channel_inds_to_remove] = np.zeros_like(file[:, channel_inds_to_remove])

    # select subset:
    if channel_subset_inds is not None:
        file = file[:, channel_subset_inds]
        print("Selecting channel subset: ", channel_subset_inds)

    print("Resulting file shape: ", file.shape, "\n")
    return file, config_dict