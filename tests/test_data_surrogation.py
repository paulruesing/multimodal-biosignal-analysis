import pytest

import numpy as np
import src.pipeline.data_surrogation as surrogation

class TestDataSurrogation:
    def setup_method(self):
        # Create dummy input data with shape (timesteps, channels)
        self.timesteps = 1000; self.channels = 64
        self.input_data = np.random.randn(self.timesteps, self.channels)  # 1000 timesteps, 8 channels
        self.sampling_frequency = 500

    def test_insert_bad_channels(self):
        """
        a) Result should be unchanged input for scaling_range = (1.0, 1.0).
        b) Only channels defined by 2nd return var should be changed.
        c) Anything is changed.
        """
        ### assert remains unchanged for scaling_range = (1.0, 1.0):
        assert np.array_equal(self.input_data,
                              surrogation.insert_bad_channels(self.input_data, axis=0,
                                                              scale_range=(1.0, 1.0))[0]
                              ), "Scale factor of 1.0, i.e. scale_range=(1.0, 1.0), still causes changes in output!"

        ### assert only defined channels change:
        result, amended_channels = surrogation.insert_bad_channels(self.input_data, axis=0)
        remaining_channels = [i for i in range(self.channels) if i+1 not in amended_channels]
        assert np.array_equal(self.input_data[:, remaining_channels],
                              result[:, remaining_channels],
                              ), "Channels beyond amended channels also differ from input data!"

        ### anything changed:
        assert not np.array_equal(self.input_data, result), "No bad channels inserted with default arguments..."

# todo: test equivalent input and output types