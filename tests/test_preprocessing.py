import pytest

from unittest.mock import patch
import numpy as np
from src.pipeline.preprocessing import BiosignalPreprocessor


################### Test BiosignalPreprocessor ###################
def test_biosignal_preprocessor_init():
    # Create dummy input data with shape (timesteps, channels)
    timesteps = 1000; channels = 8
    input_data = np.random.randn(timesteps, channels)  # 1000 timesteps, 8 channels
    sampling_freq = 256
    notch_harmonics = 3
    denoising_thresh_mode = 'hard'

    for modality, band_pass_freqs, notch_freq, notch_width, ref_channel, amp_thresh, n_ics, neighbor_r, wavelet in zip(
            ['eeg', 'emg'],
            [[10, 100], 'auto'],
            [40, None],
            [2, 5],
            ['average', None],
            [0.04, None],
            [40, None], [.3, None],
            ['sym5', None], ):
        # Initialize instance
        processor = BiosignalPreprocessor(
            np_input_array=input_data,
            sampling_freq=sampling_freq,
            modality=modality,
            band_pass_frequencies=band_pass_freqs,
            notch_frequency=notch_freq,
            notch_harmonics=notch_harmonics,
            notch_width=notch_width,
            reference_channels=ref_channel,
            amplitude_rejection_threshold=amp_thresh,
            ica_components=n_ics,
            laplacian_filter_neighbor_radius=neighbor_r,
            wavelet_type=wavelet,
            denoising_threshold_mode=denoising_thresh_mode,
        )

        # Check if attributes are correctly set
        assert np.array_equal(processor.np_input_array, input_data)
        assert processor.band_pass_frequencies == band_pass_freqs if band_pass_freqs != 'auto' else (0.1, 100)

        # this could be done with a loop and assert getattr(processor, VAR_NAME) == VALUE
        assert processor.sampling_freq == sampling_freq
        assert processor.modality == modality
        assert processor.n_timesteps == timesteps
        assert processor.n_channels == channels
        assert processor.notch_frequency == notch_freq
        assert processor.notch_harmonics == notch_harmonics
        assert processor.notch_width == notch_width
        assert processor.reference_channels == ref_channel
        assert processor.amplitude_rejection_threshold == amp_thresh
        assert processor.ica_components == n_ics
        assert processor.laplacian_filter_neighbor_radius == neighbor_r
        assert processor.wavelet_type == wavelet
        assert processor.denoising_threshold_mode == denoising_thresh_mode

# todo: tests for amenable properties with asserting execution of clean_downstream_results()
# todo: test clean downstream results

def test_biosignal_preprocessor_amenable_properties():
    """ Setting these properties should trigger clean_downstream_results """
    # Create dummy input data with shape (timesteps, channels)
    timesteps = 1000; channels = 8
    input_data = np.random.randn(timesteps, channels)  # 1000 timesteps, 8 channels

    # object to test:
    processor = BiosignalPreprocessor(
        np_input_array=input_data,
        sampling_freq=256,
        modality='eeg',
    )

    # iterate over amenable properties:
    for var_name, new_value in [
        ('np_input_array', input_data),
        ('sampling_freq', 300),
        ('modality', 'eeg'),
        ('band_pass_frequencies', (20, 80)),
        ('notch_frequency', 50),
        ('notch_harmonics', 4),
        ('notch_width', None),
        ('reference_channels', 'average'),
        ('amplitude_rejection_threshold', 0.001),
        ('ica_components', 25),
        ('laplacian_filter_neighbor_radius', 0.05),
        ('wavelet_type', 'db4'),
        ('denoising_threshold_mode', 'soft'),
    ]:
        # this replaces the method by a mock method:
        with patch.object(processor, 'clean_downstream_results') as mock_clean_downstream_results:
            # set value:
            setattr(processor, var_name, new_value)
            # assert that clean_downstream_results was called:
            assert mock_clean_downstream_results.call_count == 1, f"'clean_downstream_results' was not called when setting {var_name}."

            # assert that value is set:
            current_val = getattr(processor, var_name)
            if isinstance(current_val, np.ndarray):
                assert np.array_equal(current_val, new_value), f"{var_name} not set successfully."
            else:
                assert getattr(processor, var_name) == new_value, f"{var_name} not set successfully."