import pytest

import numpy as np
import src.pipeline.signal_features as features

class TestSignalFeatures:
    def setup_method(self):
        # Create dummy input data with shape (timesteps, channels)
        self.n_timesteps = 1000; self.n_channels = 64
        self.input_data = np.random.randn(self.n_timesteps, self.n_channels)  # 1000 timesteps, 8 channels
        self.sampling_frequency = 500

    def test_compute_spectral_snr(self):
        """ should return 0 SNR difference for scaled input data. """
        # compute snr for dummy input:
        input_snr = features.compute_spectral_snr(self.input_data,
                                                  self.sampling_frequency)

        # compute SNR for scaled input:
        scaled_snr = features.compute_spectral_snr(self.input_data * .5,
                                                   self.sampling_frequency)

        assert input_snr == scaled_snr, "SNR is not scale-invariant as should be."

    def test_resample_data(self):
        """ Should return same n_channels as provided and altered n_timesteps. """
        resampling_factor = 2  # n_timesteps should increase proportionately

        # call function:
        result = features.resample_data(self.input_data, axis=0,
                               original_sampling_freq=self.sampling_frequency,
                               new_sampling_freq=self.sampling_frequency*resampling_factor)
        assert result.shape[1] == self.n_channels, "Number of channels mustn't be altered during resampling."
        assert result.shape[0] == self.n_timesteps * resampling_factor, "Number of timesteps should increase proportionately to new_sampling_freq/original_sampling_freq."


    # todo: test equivalent input file format for all methods
