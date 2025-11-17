import pytest

import numpy as np
import src.pipeline.signal_features as features

class TestSignalFeatures:
    def setup_method(self):
        # Create dummy input data with shape (timesteps, channels)
        timesteps = 1000; channels = 64
        self.input_data = np.random.randn(timesteps, channels)  # 1000 timesteps, 8 channels
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

    # todo: test equivalent input file format for all methods
