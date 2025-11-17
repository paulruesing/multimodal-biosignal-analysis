import pytest

from unittest.mock import patch, MagicMock
import numpy as np
from typing import Literal
import mne
import re
from src.pipeline.preprocessing import BiosignalPreprocessor

################### Test BiosignalPreprocessor ###################
class TestBiosignalPreprocessor:
    def setup_method(self):
        # Create dummy input data with shape (timesteps, channels)
        timesteps = 10000; channels = 64
        self.input_data = np.random.randn(timesteps, channels)  # 1000 timesteps, 8 channels
        self.sampling_freq = 2048
        self.modality: Literal['eeg', 'emg'] = 'eeg'

        # define dummy test instance:
        self.instance = BiosignalPreprocessor(
            np_input_data=self.input_data,
            sampling_freq=self.sampling_freq,
            modality=self.modality,
        )

    def teardown_method(self):
        pass

    def test_biosignal_preprocessor_init(self):
        # try different input configurations
        for modality, band_pass_freqs, notch_freq, notch_harmonics, notch_width, ref_channel, amp_thresh, n_ics, automatic_ic_labelling, neighbor_r, wavelet, denoising_thresh_mode in zip(
                ['eeg', 'emg'],
                [[10, 100], 'auto'],
                [40, None],
                [3, 0],
                [2, 5],
                ['average', None],
                [0.04, None],
                [40, None],  # n_ics
                [True, False],  # allow automatic labelling
                [.3, None],  # neighbor radius
                ['sym5', None],
                ['hard', 'soft'],
        ):
            # Initialize instance
            processor = BiosignalPreprocessor(
                np_input_data=self.input_data,
                sampling_freq=self.sampling_freq,
                modality=modality,
                band_pass_frequencies=band_pass_freqs,
                notch_frequency=notch_freq,
                notch_harmonics=notch_harmonics,
                notch_width=notch_width,
                reference_channels=ref_channel,
                amplitude_rejection_threshold=amp_thresh,
                n_ica_components=n_ics,
                laplacian_filter_neighbor_radius=neighbor_r,
                wavelet_type=wavelet,
                denoising_threshold_mode=denoising_thresh_mode,
            )

            # Check if attributes are correctly set
            assert np.array_equal(processor.np_input_data, self.input_data)
            assert processor.band_pass_frequencies == band_pass_freqs if band_pass_freqs != 'auto' else (0.1, 100)

            # this could be done with a loop and assert getattr(processor, VAR_NAME) == VALUE
            assert processor.sampling_freq == self.sampling_freq
            assert processor.modality == modality
            assert processor.n_timesteps == self.input_data.shape[0]
            assert processor.n_channels == self.input_data.shape[1]
            assert processor.notch_frequency == notch_freq
            assert processor.notch_harmonics == notch_harmonics
            assert processor.notch_width == notch_width
            assert processor.reference_channels == ref_channel
            assert processor.amplitude_rejection_threshold == amp_thresh
            assert processor.n_ica_components == n_ics
            assert processor.laplacian_filter_neighbor_radius == neighbor_r
            assert processor.wavelet_type == wavelet
            assert processor.denoising_threshold_mode == denoising_thresh_mode


    def test_biosignal_preprocessor_amenable_properties(self):
        """ Setting these properties should trigger clean_downstream_results with the respective category. """
        # iterate over amenable properties:
        for var_name, new_value, category in [
            ('np_input_data', self.input_data, 'import'),
            ('sampling_freq', 300, 'import'),
            ('modality', 'eeg', 'import'),
            ('band_pass_frequencies', (20, 80), 'filtering'),
            ('notch_frequency', 50, 'filtering'),
            ('notch_harmonics', 4, 'filtering'),
            ('notch_width', None, 'filtering'),
            ('reference_channels', 'average', 'referencing'),
            ('amplitude_rejection_threshold', 0.001, 'amplitude thresholding'),
            ('n_ica_components', 25, 'ica computation'),
            ('automatic_ic_labelling', False, 'artefact rejection'),
            ('manual_ics_to_exclude', [1, 4], 'artefact rejection'),
            ('laplacian_filter_neighbor_radius', 0.05, 'smoothing'),
            ('wavelet_type', 'db4', 'denoising'),
            ('denoising_threshold_mode', 'soft', 'denoising'),
        ]:
            # this replaces the method by a mock method:
            with (patch.object(self.instance, 'clean_downstream_results') as mock_clean_downstream_results):
                # set value:
                setattr(self.instance, var_name, new_value)
                # assert that clean_downstream_results was called:
                assert mock_clean_downstream_results.call_count == 1, f"'clean_downstream_results' was not called when setting {var_name}."

                # check for positional or keyword arguments:
                assert (
                    # positional arg
                    category in mock_clean_downstream_results.call_args[0]
                    or  # kwarg
                    category == mock_clean_downstream_results.call_args[1]['change_in']
                ), f"'clean_downstream_results' was not called with correct 'category' parameter: '{category}'."

                # assert that value is set:
                current_val = getattr(self.instance, var_name)
                if isinstance(current_val, np.ndarray):
                    assert np.array_equal(current_val, new_value), f"{var_name} not set successfully."
                else:
                    assert getattr(self.instance, var_name) == new_value, f"{var_name} not set successfully."

    def test_clean_downstream_results(self):
        """ Should clean private result attributes based on level of change in result hierarchy. """
        result_hierarchy = [
            '_mne_raw_data',  # 0: import
            '_mne_filtered_data',  # 1: filtering
            '_mne_referenced_data',  # 2: referencing
            '_mne_amplitude_compliant_data',  # 3: amplitude thresholding
            '_mne_ica_results',  # 4: ica computation
            '_ica_automatic_labels',
            '_mne_artefact_free_data',  # 6: artefact rejection
            '_np_artefact_free_data',
            '_np_spatially_filtered_data',  # 8: smoothing
            '_np_denoised_data',  # 9: denoising
            '_np_output_data',
        ]
        for change_in_str, result_level in [
            ('import', 0),
            ('filtering', 1),
            ('referencing', 2),
            ('amplitude thresholding', 3),
            ('ica computation', 4),
            ('artefact rejection', 6),
            ('smoothing', 8),
            ('denoising', 9),
        ]:
            # set attribute that should be reset:
            for attr in result_hierarchy[result_level:]:
                setattr(self.instance, attr, np.random.rand())

            # reset:
            self.instance.clean_downstream_results(change_in_str)

            # check whether now None:
            for attr in result_hierarchy[result_level:]:
                assert getattr(self.instance,
                               attr) is None, f"Reset of BiosignalPreprocessor.{attr} during clean_downstream_results (change_in='{change_in_str}') didn't work."

        # finally check error raising:
        with pytest.raises(ValueError, match="change_in category: 'blablu' is undefined!"):
            self.instance.clean_downstream_results('blablu')


    def test_annotate_amplitude_based_artefacts_raises_error(self):
        """ Check whether it raises ValueError for missing input. """
        self.instance.amplitude_rejection_threshold = None
        # mock object (magic because automatically fits type and structure)
        with pytest.raises(ValueError, match="amplitude_rejection_threshold needs to be defined!"):
            self.instance._annotate_amplitude_based_artefacts()

    # todo: test annotate amplitude based artefacts inplace=True!

    def test_apply_notch_filter_raises_error(self):
        """ Raises ValueError for missing input. """
        # test error raising:
        self.instance.notch_frequency = None
        with pytest.raises(ValueError, match="notch_frequency needs to be defined!"):
            self.instance._apply_notch_filter()

    def test_mne_to_numpy(self):
        np_data = self.instance.mne_to_numpy(self.instance.mne_raw_data)
        assert isinstance(np_data, np.ndarray), "mne_to_numpy method didn't return a numpy array."

    def test_numpy_to_mne(self):
        mne_data = self.instance.numpty_to_mne(self.input_data, sampling_freq=self.sampling_freq, modality=self.modality)
        assert isinstance(mne_data, (mne.io.RawArray, mne.io.Raw)), "numpy_to_mne method didn't return an mne object."

    def test_get_neighboring_electrodes_mapping(self):
        """ Should raise errors for laplacian_filter_neighbor_radius=None and modality='emg'. Should return list of lists of ints."""
        self.instance.n_ica_components = None  # ensures ICA is skipped

        # test EMG sanity check:
        self.instance.modality = 'emg'
        with pytest.raises(ValueError, match=re.escape("Laplacian spatial filtering for EMG data is not implemented yet (requires 3D positional electrode mapping). Data remains unchanged.")):
            # parentheses are grouping expressions in RE -> use re.escape
            self.instance.get_neighboring_electrodes_mapping()
        self.instance.modality = 'eeg'

        # test neighbor radius sanity check:
        self.instance.laplacian_filter_neighbor_radius = None
        with pytest.raises(ValueError, match="laplacian_filter_neighbor_radius needs to be defined!"):
            self.instance.get_neighboring_electrodes_mapping()
        self.instance.laplacian_filter_neighbor_radius = .05  # default value

        # test return value:
        neighbor_mapping = self.instance.get_neighboring_electrodes_mapping()
        neighbor_mapping = [entry for entry in neighbor_mapping if entry is not None and len(entry) != 0]  # skip NaNs
        assert isinstance(neighbor_mapping, list) and isinstance(neighbor_mapping[0], list) and isinstance(neighbor_mapping[0][0], int)