from typing import Literal
from scipy import signal
import numpy as np
import random
import src.pipeline.signal_features as features


def check_2d_numpy_array(input_array: np.ndarray,
                         axis: Literal[0, 1] = None) -> tuple[np.ndarray, Literal[0, 1]]:
    # input sanity checks:
    if len(input_array.shape) == 1:
        input_array = input_array[:, np.newaxis]
        if axis is None: axis = 0
    else:
        if axis is None: raise AttributeError("For 2D signal arrays, axis needs to be defined!")
    return input_array, axis


def insert_bad_channels(input_array: np.ndarray,
                        axis: Literal[0, 1] = None,
                        n_channels: int = 5,
                        scale_range: tuple[float, float] = (10.0, 15.0)) -> tuple[np.ndarray, list[int]]:
    """
    Insert surrogate bad channels by scaling some channels in a 2D array.

    Parameters
    ----------
    input_array : np.ndarray
        2D numpy array representing the input data with shape (time, channels) or (channels, time).
    axis : Literal[0, 1], optional
        Axis representing channels in `input_array` (0 or 1). If None, defaults to a checked value.
    n_channels : int, default=5
        Number of channels to scale (simulate bad channels).
    scale_range : tuple of float, default=(10.0, 15.0)
        Range of scaling factors. Each bad channel is multiplied by a random factor within this range.

    Returns
    -------
    tuple
        - np.ndarray: A copy of the input array with `n_channels` randomly scaled channels to simulate bad channels.
        - list of int: List of the indices (1-based) of channels that were scaled (amended).

    Notes
    -----
    The function scales n (n_channels) random channels along the specified axis by a randomly sampled
    factor within `scale_range`. The output array is a copy and does not modify the input array in place.
    The indices of changed channels are returned starting at 1.
    """
    # input sanity check:
    input_array, axis = check_2d_numpy_array(input_array, axis)

    # iteratively amend channels
    output_array = input_array.copy()  # prepare output array
    amended_channel_inds = []
    for channel_ind in random.sample(range(1,
                                           input_array.shape[axis+1%2]),  # use the channel axis
                                     k=n_channels):  # pull n_channels samples
        # random scale factor within range:
        scale_factor = scale_range[0] + np.random.rand() * (scale_range[1] - scale_range[0])

        # amend channel:
        output_array[:, channel_ind] = input_array[:, channel_ind] * scale_factor
        amended_channel_inds.append(channel_ind+1)

    return output_array, amended_channel_inds  # returned ind is channel ind (starting at 1!)


# todo: noise insertion at custom dB levels