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


# todo: validate whether the below methods work correctly:
def add_noise_to_channels(input_array: np.ndarray,
                          noise_db: float,
                          channels: list[int],
                          axis: Literal[0, 1] = 0,
                          noise_type: Literal["white", "pink"] = "white",
                          random_seed: int = None) -> np.ndarray:
    """
    Add specified noise levels to selected channels in a 2D array.

    Parameters
    ----------
    input_array : np.ndarray
        2D array where noise will be added (samples × channels or channels × samples).
    noise_db : float
        Signal-to-Noise Ratio in dB. Negative values create more noise.
    channels : list[int]
        List of channel indices to add noise to.
    axis : Literal[0, 1]
        Axis defining time dimension (0 = samples in rows, 1 = samples in columns).
    noise_type : Literal["white", "pink"]
        Type of noise to generate. "white" for Gaussian; "pink" for 1/f noise.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array with added noise on specified channels.

    Raises
    ------
    ValueError
        If channel indices are out of bounds.
    """

    # set random seed if provided:
    if random_seed is not None:
        np.random.seed(random_seed)

    # validate and reshape input:
    array, axis = check_2d_numpy_array(input_array, axis)

    # validate channels:
    noise_axis = 1 - axis  # Opposite axis from time
    max_channels = array.shape[noise_axis]
    if not all(0 <= ch < max_channels for ch in channels):
        raise ValueError(f"Channel indices must be in range [0, {max_channels - 1}]")

    # Create output copy
    noisy_array = array.copy()

    # Process each channel
    for ch in channels:
        # Extract signal on this channel
        if axis == 0:
            signal = noisy_array[:, ch]
        else:
            signal = noisy_array[ch, :]

        # Calculate signal power
        signal_power = np.mean(signal ** 2)
        signal_rms = np.sqrt(signal_power)

        # Calculate noise amplitude from SNR in dB
        # SNR_dB = 10 * log10(signal_power / noise_power)
        # => noise_power = signal_power / 10^(SNR_dB/10)
        snr_linear = 10 ** (noise_db / 10)
        noise_power = signal_power / snr_linear
        noise_rms = np.sqrt(noise_power)

        # Generate noise
        noise = generate_noise(signal.shape, noise_type, noise_rms)

        # Add noise to signal
        if axis == 0:
            noisy_array[:, ch] = signal + noise
        else:
            noisy_array[ch, :] = signal + noise

    return noisy_array


def generate_noise(shape: tuple, noise_type: str, amplitude: float) -> np.ndarray:
    """
    Generate noise of specified type.

    Parameters
    ----------
    shape : tuple
        Shape of noise array to generate.
    noise_type : str
        "white" for Gaussian white noise; "pink" for 1/f pink noise.
    amplitude : float
        RMS amplitude of generated noise.

    Returns
    -------
    np.ndarray
        Noise array with specified amplitude.
    """

    if noise_type == "white":
        # Generate Gaussian white noise
        noise = np.random.normal(0, 1, shape)

    elif noise_type == "pink":
        # Generate pink (1/f) noise using spectral method
        # Create white noise in frequency domain
        white_fft = np.fft.rfft(np.random.normal(0, 1, shape[0]))

        # Apply 1/f filter (amplitude decreases with frequency)
        freqs = np.fft.rfftfreq(shape[0])
        freqs[0] = 1  # Avoid division by zero at DC
        pink_filter = 1 / np.sqrt(freqs)
        pink_fft = white_fft * pink_filter

        # Convert back to time domain
        noise = np.fft.irfft(pink_fft, n=shape[0])

        # If 2D, repeat for all channels
        if len(shape) > 1:
            noise = np.tile(noise[:, np.newaxis], (1, shape[1]))

    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    # Normalize to desired amplitude
    noise_current_rms = np.sqrt(np.mean(noise ** 2))
    noise = noise * (amplitude / noise_current_rms)

    return noise