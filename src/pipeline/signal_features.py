import numpy as np
from typing import Literal
from scipy import signal

from src.pipeline.visualizations import plot_freq_domain

def check_2d_numpy_array(input_array: np.ndarray,
                         axis: Literal[0, 1] | None = None) -> tuple[np.ndarray, Literal[0, 1]]:
    # input sanity checks:
    if len(input_array.shape) == 1:
        input_array = input_array[:, np.newaxis]
        if axis is None: axis: Literal[0, 1] = 0
    else:
        if axis is None: raise AttributeError("For 2D signal arrays, axis needs to be defined!")
    return input_array, axis


# todo: later, add cross-trial variability
def compute_spectral_snr(input_array: np.ndarray,
                         sampling_freq: int,
                         target_freq: float = 21.5,
                         freq_window: float = 8.5,
                         target_band_ratio: float = .5,
                         axis: Literal[0, 1] = 0,
                         return_psd: bool = False) -> float | tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the spectral signal-to-noise ratio (SNR) at a target frequency from time-series data.

    Parameters
    ----------
    input_array : np.ndarray
        1D or 2D numpy array containing the signal data. For 2D, specify the axis along which to operate.
    sampling_freq : int
        Sampling frequency of the input signals (Hz).
    target_freq : float, default=21.5
        Center frequency of interest for SNR calculation (Hz).
    freq_window : float, default=8.5
        Frequency window around `target_freq` to define noise band boundaries (Hz).
    target_band_ratio : float, default=0.5
        Ratio determining bandwidth of target band relative to `freq_window`. Target band width = freq_window * target_band_ratio.
    axis : Literal[0, 1], default=0
        Axis along which to compute the power spectral density and SNR.
    return_psd : bool, default=False
        If True, returns a tuple containing SNR, frequency array, and PSD estimates; otherwise returns only SNR value.

    Returns
    -------
    float or tuple
        If `return_psd` is False, returns:
            - float: SNR in decibels (dB).
        If `return_psd` is True, returns:
            - float: SNR in decibels (dB).
            - np.ndarray: Frequencies corresponding to PSD estimates.
            - np.ndarray: Power spectral density values.

    Notes
    -----
    The function uses Welch's method to estimate the power spectral density (PSD). SNR is computed as the mean power
    within a narrow target band around `target_freq` divided by the mean power in the surrounding noise band, then
    converted to dB scale. Defaults are tuned for beta band analysis in cortico-muscular coherence studies.
    """
    # input sanity check:
    input_array, axis = check_2d_numpy_array(input_array, axis=axis)

    # Compute PSD using Welch's method
    freqs, psd = signal.welch(input_array, axis=axis,
                              fs=sampling_freq,
                              nperseg=sampling_freq * 4,  # 4-second window
                              )

    # Find indices for target frequency and noise bands
    target_freq_window = freq_window * target_band_ratio
    target_band = (freqs < target_freq + target_freq_window) & (freqs > target_freq - target_freq_window)
    noise_band = (freqs >= target_freq - freq_window) & (freqs <= target_freq + freq_window)

    # SNR = power at precise target freq / mean power in noise band
    snr_linear = np.mean(psd[target_band]) / np.mean(psd[noise_band])
    snr_db = 10 * np.log10(snr_linear)  # log transform

    return snr_db if not return_psd else (snr_db, freqs, psd)


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
    input_array, axis = check_2d_numpy_array(input_array, axis=axis)

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