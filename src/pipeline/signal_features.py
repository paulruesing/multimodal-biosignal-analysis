import numpy as np
from scipy.interpolate import interp1d
from typing import Literal
from scipy import signal
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import src.pipeline.visualizations as visualizations
from src.pipeline.visualizations import smart_save_fig

FREQUENCY_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (13, 30),
    'gamma': (30, 100),  # EEG gamma range
    #'emg_alpha': (8, 12),
    #'emg_beta': (15, 35),
    #'emg_gamma': (35, 60)
}

########### HELPER FUNCTIONS ###########
def check_2d_numpy_array(input_array: np.ndarray,
                         axis: Literal[0, 1] | None = None) -> tuple[np.ndarray, Literal[0, 1]]:
    # input sanity checks:
    if len(input_array.shape) == 1:
        input_array = input_array[:, np.newaxis]
        if axis is None: axis: Literal[0, 1] = 0
    else:
        if axis is None: raise AttributeError("For 2D signal arrays, axis needs to be defined!")
    return input_array, axis


def resample_data(data: np.ndarray,
                  original_sampling_freq, new_sampling_freq,
                  axis: Literal[0, 1] = None,):
    """ Resample an np.ndarray, e.g. for animation. """
    # input sanity checks and descriptive attributes:
    input_array, axis = check_2d_numpy_array(data, axis=axis)
    n_timesteps = input_array.shape[axis]; n_channels = input_array.shape[axis+1%2]

    original_duration = n_timesteps / original_sampling_freq
    new_n_timesteps = int(round(original_duration * new_sampling_freq))

    original_times = np.linspace(0, original_duration, n_timesteps)
    new_times = np.linspace(0, original_duration, new_n_timesteps)

    interpolator = interp1d(original_times, input_array, axis=axis, kind='linear', fill_value='extrapolate')
    resampled_data = interpolator(new_times)
    return resampled_data


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
    if plot_result: visualizations.plot_freq_domain(amplitude_spectrum, freqs_pos, **plot_kwargs)

    return amplitude_spectrum, freqs_pos


########### BIOLOGICAL FEATURES ###########
def multitaper_psd(input_array: np.ndarray,
                      sampling_freq: float,
                      nw: float = 3,
                      window_length_sec: float = 1.0,
                      overlap_frac: float = 0.5,
                      axis: Literal[0, 1] = None,
                      verbose: bool = False,
                      plot_result: bool = False, **plot_kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized across windows, preserves exact output behavior."""

    input_array, axis = check_2d_numpy_array(input_array, axis=axis)

    n_samples = input_array.shape[axis]
    n_channels = input_array.shape[axis + 1 % 2]
    window_samples = int(window_length_sec * sampling_freq)
    hop_samples = int(window_samples * (1 - overlap_frac))
    k = int(2 * nw - 1)

    # Generate tapers once
    tapers = signal.windows.dpss(M=window_samples, NW=nw, Kmax=k)  # shape: (k, window_samples)

    # Pre-compute window indices (vectorized)
    window_starts = np.arange(0, n_samples - window_samples, hop_samples)
    n_windows = len(window_starts)

    # Pre-compute time centers (no longer deferred)
    time_centers = (window_starts + window_samples / 2) / sampling_freq

    # Transpose for faster axis access if needed
    if axis == 1:
        input_array = input_array.T

    spectrograms = []

    for ch_ind in tqdm(range(n_channels)):
        # Extract all windows at once via fancy indexing
        # Shape: (n_windows, window_samples)
        windows = np.array([input_array[start:start + window_samples, ch_ind]
                            for start in window_starts])

        # Apply all tapers to all windows: (n_windows, k, n_freqs)
        psd_list = []
        for taper in tapers:
            # Vectorized periodogram for all windows with same taper
            freqs, pxx = signal.periodogram(windows, fs=sampling_freq,
                                            axis=1, window=taper)
            # pxx shape: (n_windows, n_freqs)
            psd_list.append(pxx)

        # Mean across tapers: (n_windows, n_freqs)
        channel_spec = np.mean(psd_list, axis=0)
        spectrograms.append(channel_spec)

    # Convert to output format: (n_windows, n_freqs, n_channels)
    spectrograms = np.transpose(np.array(spectrograms), axes=[1, 2, 0])

    if plot_result:
        fig_title = 'All-Channel Average PSD Spectrogram' if "title" not in plot_kwargs.keys() else plot_kwargs['title']
        _ = plot_kwargs.pop("title", None)
        visualizations.plot_spectrogram(spectrogram=np.squeeze(np.mean(spectrograms, axis=2)),
                                        title=fig_title,
                                        timestamps=time_centers,
                                        frequencies=freqs,
                                        **plot_kwargs)

    return spectrograms, time_centers, freqs

def multitaper_magnitude_squared_coherence(
        eeg_array: np.ndarray,
        emg_array: np.ndarray,
        sampling_freq: float,
        nw: float = 3,
        window_length_sec: float = 1.0,
        overlap_frac: float = 0.5,
        eeg_axis: Literal[0, 1] = 0,
        emg_axis: Literal[0, 1] = 0,
        verbose: bool = False,) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute multitaper magnitude squared coherence between EEG and EMG signals.
    MEMORY-OPTIMIZED VERSION: Pre-allocates output array, avoids intermediate lists.
    """

    # Normalize input arrays to (n_samples, n_channels) format
    eeg_array = _normalize_to_time_first(eeg_array, axis=eeg_axis)
    emg_array = _normalize_to_time_first(emg_array, axis=emg_axis)

    # Extract dimensions
    n_samples_eeg, n_eeg_channels = eeg_array.shape
    n_samples_emg, n_emg_channels = emg_array.shape

    # Validate sample alignment
    if n_samples_eeg != n_samples_emg:
        raise ValueError(
            f"EEG and EMG must have same number of samples. "
            f"Got EEG: {n_samples_eeg}, EMG: {n_samples_emg}"
        )
    n_samples = n_samples_eeg

    # Window parameters
    window_samples = int(window_length_sec * sampling_freq)
    hop_samples = int(window_samples * (1 - overlap_frac))
    k = int(2 * nw - 1)  # number of tapers

    # Generate DPSS tapers
    tapers = signal.windows.dpss(M=window_samples, NW=nw, Kmax=k)
    # tapers shape: (k, window_samples)

    # Compute frequency array once before loop
    freqs = np.fft.rfftfreq(window_samples, d=1/sampling_freq)
    n_freqs = len(freqs)

    # Calculate number of windows
    n_windows = (n_samples - window_samples) // hop_samples + 1

    # PRE-ALLOCATE OUTPUT ARRAY (instead of accumulating in list)
    coherences = np.zeros(
        (n_windows, n_freqs, n_eeg_channels, n_emg_channels),
        dtype=np.float32  # Use float32 instead of float64 to save 50% memory
    )
    time_centers = np.zeros(n_windows, dtype=np.float64)

    if verbose:
        print(
            f"Computing magnitude squared coherence between {n_eeg_channels} EEG "
            f"and {n_emg_channels} EMG channels."
        )
        print(
            f"Window length: {window_length_sec:.3f}s, "
            f"Overlap: {overlap_frac*100:.1f}%, "
            f"Tapers: {k}"
        )
        print(f"Output array shape: {coherences.shape}, dtype: {coherences.dtype}")

    # Slide window across signal
    for win_idx in tqdm(range(n_windows), disable=not verbose):
        start_idx = win_idx * hop_samples
        end_idx = start_idx + window_samples

        # Extract windows for this timestep
        eeg_window = eeg_array[start_idx:end_idx, :]  # (window_samples, n_eeg_ch)
        emg_window = emg_array[start_idx:end_idx, :]  # (window_samples, n_emg_ch)

        # Initialize accumulators for this window (will accumulate across tapers)
        psd_eeg_sum = np.zeros((n_freqs, n_eeg_channels), dtype=np.float32)
        psd_emg_sum = np.zeros((n_freqs, n_emg_channels), dtype=np.float32)
        csd_sum = np.zeros((n_freqs, n_eeg_channels, n_emg_channels), dtype=np.complex64)

        # Process all tapers and accumulate (no intermediate list storage)
        for taper in tapers:
            # Apply taper to both signals
            eeg_tapered = eeg_window * taper[:, np.newaxis]  # (window_samples, n_eeg_ch)
            emg_tapered = emg_window * taper[:, np.newaxis]  # (window_samples, n_emg_ch)

            # Compute FFTs
            eeg_fft = np.fft.rfft(eeg_tapered, axis=0)  # (n_freqs, n_eeg_ch)
            emg_fft = np.fft.rfft(emg_tapered, axis=0)  # (n_freqs, n_emg_ch)

            # Compute one-sided PSD (power spectral density)
            psd_eeg = np.abs(eeg_fft) ** 2 / (sampling_freq * window_samples)  # (n_freqs, n_eeg_ch)
            psd_emg = np.abs(emg_fft) ** 2 / (sampling_freq * window_samples)  # (n_freqs, n_emg_ch)

            # ACCUMULATE directly into sum arrays (no list append)
            psd_eeg_sum += psd_eeg
            psd_emg_sum += psd_emg

            # Compute and accumulate cross-spectral density
            csd = (
                np.conj(eeg_fft)[:, :, np.newaxis] * emg_fft[:, np.newaxis, :]
                / (sampling_freq * window_samples)
            )  # (n_freqs, n_eeg_ch, n_emg_ch)
            csd_sum += csd

        # Average across tapers (divide by k)
        psd_eeg_mean = psd_eeg_sum / k
        psd_emg_mean = psd_emg_sum / k
        csd_mean = csd_sum / k

        # Compute magnitude squared coherence
        # MSC = |CSD|² / (PSD_EEG * PSD_EMG)
        numerator = np.abs(csd_mean) ** 2  # (n_freqs, n_eeg_ch, n_emg_ch)

        # Denominator with broadcasting
        denominator = (
            psd_eeg_mean[:, :, np.newaxis] * psd_emg_mean[:, np.newaxis, :]
        )

        # Compute coherence (clip to [0, 1] to handle numerical errors)
        # Store directly into pre-allocated array
        coherences[win_idx, :, :, :] = np.clip(numerator / denominator, 0, 1)

        # Save time center
        time_centers[win_idx] = (start_idx + window_samples / 2) / sampling_freq

    return coherences, time_centers, freqs


def oom_multitaper_magnitude_squared_coherence(
        eeg_array: np.ndarray,
        emg_array: np.ndarray,
        sampling_freq: float,
        nw: float = 3,
        window_length_sec: float = 1.0,
        overlap_frac: float = 0.5,
        eeg_axis: Literal[0, 1] = 0,
        emg_axis: Literal[0, 1] = 0,
        verbose: bool = False,) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute multitaper magnitude squared coherence between EEG and EMG signals.

    Coherence is computed per frequency, per time window, and per EEG-EMG channel pair
    using the multitaper method. Output preserves all pairwise relationships.

    Parameters
    ----------
    eeg_array : ndarray
        EEG input signal, shape (n_samples, n_eeg_channels) or (n_eeg_channels, n_samples)
    emg_array : ndarray
        EMG input signal, shape (n_samples, n_emg_channels) or (n_emg_channels, n_samples)
    sampling_freq : float
        Sampling frequency (Hz)
    nw : float
        Time-half-bandwidth product (affects frequency smoothing). Default: 3
    window_length_sec : float
        Window length in seconds (affects time resolution). Default: 1.0
    overlap_frac : float
        Overlap between windows (0.5 = 50%, standard). Default: 0.5
    eeg_axis : Literal[0, 1]
        Time axis of EEG array. 0 if (n_samples, n_channels), 1 if (n_channels, n_samples). Default: 0
    emg_axis : Literal[0, 1]
        Time axis of EMG array. 0 if (n_samples, n_channels), 1 if (n_channels, n_samples). Default: 0
    verbose : bool
        Print progress information. Default: False

    Returns
    -------
    coherences : ndarray
        Magnitude squared coherence. Shape: (n_windows, n_freqs, n_eeg_channels, n_emg_channels)
        Value range: [0, 1] where 1 = perfect coherence, 0 = no coherence
    time_centers : ndarray
        Time center of each window (seconds). Shape: (n_windows,)
    freqs : ndarray
        Frequency array (Hz). Shape: (n_freqs,)

    Notes
    -----
    Magnitude squared coherence is defined as:
        MSC(f) = |CSD(f)|² / (PSD_EEG(f) * PSD_EMG(f))

    where CSD is the cross-spectral density and PSD is power spectral density. Auto-spectral density and PSD
    only sometimes differ in terms of normalization, but if computation is coherent across nominator and denominator any
    is compatible.

    The multitaper method uses Slepian sequences (DPSS tapers) to reduce spectral
    leakage and improve frequency resolution.
    """

    # Normalize input arrays to (n_samples, n_channels) format
    eeg_array = _normalize_to_time_first(eeg_array, axis=eeg_axis)
    emg_array = _normalize_to_time_first(emg_array, axis=emg_axis)

    # Extract dimensions
    n_samples_eeg, n_eeg_channels = eeg_array.shape
    n_samples_emg, n_emg_channels = emg_array.shape

    # Validate sample alignment
    if n_samples_eeg != n_samples_emg:
        raise ValueError(
            f"EEG and EMG must have same number of samples. "
            f"Got EEG: {n_samples_eeg}, EMG: {n_samples_emg}"
        )
    n_samples = n_samples_eeg

    # Window parameters
    window_samples = int(window_length_sec * sampling_freq)
    hop_samples = int(window_samples * (1 - overlap_frac))
    k = int(2 * nw - 1)  # number of tapers

    # Generate DPSS tapers
    tapers = signal.windows.dpss(M=window_samples, NW=nw, Kmax=k)
    # tapers shape: (k, window_samples)

    # Initialize output storage
    coherence_list = []  # Will accumulate (n_windows, n_freqs, n_eeg_ch, n_emg_ch)
    time_centers = []

    if verbose:
        print(
            f"Computing magnitude squared coherence between {n_eeg_channels} EEG "
            f"and {n_emg_channels} EMG channels."
        )
        print(
            f"Window length: {window_length_sec:.3f}s, "
            f"Overlap: {overlap_frac*100:.1f}%, "
            f"Tapers: {k}"
        )

    # Compute frequency array once before loop (same for all windows)
    freqs = np.fft.rfftfreq(window_samples, d=1/sampling_freq)

    # Slide window across signal
    n_windows = (n_samples - window_samples) // hop_samples + 1

    for win_idx in tqdm(range(n_windows), disable=not verbose):
        start_idx = win_idx * hop_samples
        end_idx = start_idx + window_samples

        # Extract windows for this timestep
        eeg_window = eeg_array[start_idx:end_idx, :]  # (window_samples, n_eeg_ch)
        emg_window = emg_array[start_idx:end_idx, :]  # (window_samples, n_emg_ch)

        # Compute PSD for EEG and EMG, and CSD between all pairs using all tapers
        psd_eeg_tapered = []  # Will be (k, n_freqs, n_eeg_ch)
        psd_emg_tapered = []  # Will be (k, n_freqs, n_emg_ch)
        csd_tapered = []      # Will be (k, n_freqs, n_eeg_ch, n_emg_ch)

        for taper in tapers:
            # Apply taper to both signals
            eeg_tapered = eeg_window * taper[:, np.newaxis]  # (window_samples, n_eeg_ch)
            emg_tapered = emg_window * taper[:, np.newaxis]  # (window_samples, n_emg_ch)

            # Compute FFTs
            eeg_fft = np.fft.rfft(eeg_tapered, axis=0)  # (n_freqs, n_eeg_ch)
            emg_fft = np.fft.rfft(emg_tapered, axis=0)  # (n_freqs, n_emg_ch)

            # Compute one-sided PSD (power spectral density)
            psd_eeg = np.abs(eeg_fft) ** 2 / (sampling_freq * window_samples)  # (n_freqs, n_eeg_ch)
            psd_emg = np.abs(emg_fft) ** 2 / (sampling_freq * window_samples)  # (n_freqs, n_emg_ch)

            psd_eeg_tapered.append(psd_eeg)
            psd_emg_tapered.append(psd_emg)

            # Compute cross-spectral density (CSD) between all EEG-EMG pairs
            # CSD[f, eeg_ch, emg_ch] = conj(EEG_FFT[f, eeg_ch]) * EMG_FFT[f, emg_ch] / (sampling_freq * window_samples)
            csd = (
                np.conj(eeg_fft)[:, :, np.newaxis] * emg_fft[:, np.newaxis, :]
                / (sampling_freq * window_samples)
            )  # (n_freqs, n_eeg_ch, n_emg_ch)
            csd_tapered.append(csd)

        # Average across tapers
        psd_eeg_mean = np.mean(psd_eeg_tapered, axis=0)  # (n_freqs, n_eeg_ch)
        psd_emg_mean = np.mean(psd_emg_tapered, axis=0)  # (n_freqs, n_emg_ch)
        csd_mean = np.mean(csd_tapered, axis=0)          # (n_freqs, n_eeg_ch, n_emg_ch)

        # Compute magnitude squared coherence
        # MSC = |CSD|² / (PSD_EEG * PSD_EMG)
        # Numerator: |CSD|²
        numerator = np.abs(csd_mean) ** 2  # (n_freqs, n_eeg_ch, n_emg_ch)

        # Denominator: PSD_EEG[:, :, None] * PSD_EMG[:, None, :]
        # Broadcast to (n_freqs, n_eeg_ch, n_emg_ch)
        denominator = (
            psd_eeg_mean[:, :, np.newaxis] * psd_emg_mean[:, np.newaxis, :]
        )

        # Compute coherence (clip to [0, 1] to handle numerical errors)
        coherence_window = np.clip(numerator / denominator, 0, 1)  # (n_freqs, n_eeg_ch, n_emg_ch)

        coherence_list.append(coherence_window)

        # Save time center
        time_center = (start_idx + window_samples / 2) / sampling_freq
        time_centers.append(time_center)

    # Convert to numpy arrays
    coherences = np.array(coherence_list)  # (n_windows, n_freqs, n_eeg_ch, n_emg_ch)
    time_centers = np.array(time_centers)

    return coherences, time_centers, freqs


def _normalize_to_time_first(
        array: np.ndarray,
        axis: Literal[0, 1]) -> np.ndarray:
    """
    Normalize input array to (n_samples, n_channels) format.

    Parameters
    ----------
    array : ndarray
        Input signal, either (n_samples, n_channels) or (n_channels, n_samples)
    axis : Literal[0, 1]
        Time axis. 0 if already (n_samples, n_channels), 1 if (n_channels, n_samples)

    Returns
    -------
    normalized : ndarray
        Array in (n_samples, n_channels) format
    """
    if array.ndim != 2:
        raise ValueError(f"Input array must be 2D. Got shape {array.shape}")

    if axis == 0:
        return array
    elif axis == 1:
        return array.T
    else:
        raise ValueError(f"axis must be 0 or 1. Got {axis}")


def aggregate_spectrogram_over_frequency_band(
        spectrograms: np.ndarray,
        freqs: np.ndarray,
        behaviour: Literal['max', 'mean'] = 'mean',
        frequency_bands: dict | None = None,
        log_transform: bool = False,
        log_epsilon: float = 1e-10,
        frequency_axis: int = 1,
        pre_aggregate_axis: tuple[int, Literal['max', 'mean']] | None = None,
) -> dict[str, np.ndarray]:
    """
    Average spectrogram over defined frequency bands.

    Parameters
    ----------
    spectrograms : np.ndarray
        Spectrogram data of default shape (n_windows, n_freqs, n_channels, ).
        Other shapes are manageable but frequency axis needs to be specified.
    freqs : np.ndarray
        Frequency values corresponding to spectrogram axis. Shape (n_freqs,).
    behaviour : 'max' or 'mean'
        How to aggregate values within a frequency band, average or max-pool.
    frequency_axis : int, optional
        Axis along which to aggregate (frequencies). Default is 1.
    frequency_bands : dict, optional
        Mapping of band labels to (min_freq, max_freq) tuples.
        Example: {'alpha': (8, 12), 'beta': (12, 30)}.
        If None, uses DEFAULT_FREQUENCY_BANDS.
    log_transform : bool, optional
        Whether to apply log10 transform to spectrograms. Default is True.
    log_epsilon : float, optional
        Small constant added before log transform to avoid log(0). Default is 1e-10.
    pre_aggregate_axis: tuple[int, Literal['max', 'mean']] | None, optional
        Another axis along which to aggregate (frequencies). Default is None.
        Applied before freq. aggregation.
        E.g. (1, 'mean') would lead to averaging across axis one and further reducing output dim.

    Returns
    -------
    eeg_freq_averaged_psd_dict : dict
        Dictionary with band labels as keys and np.ndarrays as values.
        Value array shape matches spectrogram input shape without frequency_axis (and eventually further_aggregate_axis).

    Raises
    ------
    ValueError
        If spectrograms shape is invalid, frequency ranges exceed available frequencies,
        or required inputs are missing.
    """

    # Use default frequency bands if not provided
    if frequency_bands is None:
        frequency_bands = FREQUENCY_BANDS

    # Input validation
    if spectrograms.ndim < 2 + int(pre_aggregate_axis is not None):
        raise ValueError(
            f"spectrograms must have at least {2 + int(pre_aggregate_axis is not None)} dimensions, got shape {spectrograms.shape}"
        )

    n_frequencies = spectrograms.shape[frequency_axis]
    # n_channels, n_windows, n_frequencies = spectrograms.shape

    if len(freqs) != n_frequencies:
        raise ValueError(
            f"freqs length ({len(freqs)}) must match spectrograms frequency axis ({n_frequencies})"
        )

    if not frequency_bands:
        raise ValueError("frequency_bands dict cannot be empty")

    # Initialize output dictionary
    freq_aggregated_dict = {}

    # pre-aggregation:
    if pre_aggregate_axis is not None:
        if pre_aggregate_axis[1] == 'max': spectrograms = np.max(spectrograms, axis=pre_aggregate_axis[0], keepdims=True)
        elif pre_aggregate_axis[1] == 'mean': spectrograms = np.mean(spectrograms, axis=pre_aggregate_axis[0], keepdims=True)
        else: raise ValueError(f"Unknown behavior for pre_aggregate_axis '{pre_aggregate_axis}'")
        # axis is only squeezed later to not confuse frequency axis

    # Process each frequency band
    for band_label, (min_freq, max_freq) in frequency_bands.items():
        # Validate frequency range
        if min_freq < freqs.min() or max_freq > freqs.max():
            raise ValueError(
                f"Band '{band_label}' range ({min_freq}, {max_freq}) exceeds available "
                f"frequencies ({freqs.min():.2f}, {freqs.max():.2f})"
            )

        # create frequency mask for this band:
        frequency_mask = (freqs >= min_freq) & (freqs < max_freq)

        if not frequency_mask.any():
            print(f"No frequencies found for band '{band_label}' in range ({min_freq}, {max_freq})")

        # Extract spectrogram subset for this band
        spectrogram_subset = np.take(spectrograms, frequency_mask, axis=frequency_axis)

        # Apply log transform if requested
        if log_transform:
            spectrogram_subset = np.log10(spectrogram_subset + log_epsilon)

        # aggregate across frequencies:
        if behaviour == 'max': condensed = np.max(spectrogram_subset, axis=frequency_axis, keepdims=True)
        elif behaviour == 'mean': condensed = np.mean(spectrogram_subset, axis=frequency_axis, keepdims=True)
        else: raise ValueError(f"Unknown behaviour '{behaviour}'")

        # squeeze arrays:
        if pre_aggregate_axis is not None:  # remove both axes:
            condensed = np.squeeze(condensed, axis=(frequency_axis, pre_aggregate_axis[0]))
        else:  # remove only frequency axis
            condensed = np.squeeze(condensed, axis=frequency_axis)

        # store in output dict and remove freq. axis:
        freq_aggregated_dict[band_label] = condensed

    return freq_aggregated_dict



########### STATISTICAL FEATURES ###########
def compute_feature_mi_importance(feature_array, target_array, feature_labels,
                                  target_label: str = 'Target', target_type: str = 'auto',
                                  feature_type: str = 'auto', random_state: int = 42,
                                  figsize: tuple = (10, 6), sort_by_importance: bool = True,
                                  include_barplot: bool = True, plot_save_dir: str | Path | None = None):
    """Compute and plot mutual information feature importance.

    Parameters
    ----------
    feature_array : array-like or DataFrame
        Feature matrix (n_samples, n_features). Can be continuous or discrete/categorical.
    target_array : array-like
        Target values (n_samples,). Can be continuous or discrete.
    feature_labels : list[str]
        Names of features.
    target_label : str, default 'Target'
        Label for target variable in plot title.
    target_type : str, default 'auto'
        'discrete', 'continuous', or 'auto' (infers from data).
    feature_type : str, default 'auto'
        'discrete', 'continuous', or 'auto' (infers per-feature).
    random_state : int, default 42
        Random state for MI computation.
    figsize : tuple, default (10, 6)
        Figure size.
    sort_by_importance : bool, default True
        Whether to sort bars by MI score.
    include_barplot : bool, default True
        Whether to create and display barplot.
    plot_save_dir : str | Path, optional
        Directory to save plot. If None, doesn't save.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if include_barplot=True, else None.
    ax : matplotlib.axes.Axes or None
        Axes object if include_barplot=True, else None.
    feature_importance : dict
        Mapping of feature names to MI scores.

    Notes
    -----
    - String/object dtype columns are automatically detected as discrete
    - Mutual information scores are normalized [0, 1]
    - Uses sklearn's mutual_info_classif for discrete targets,
      mutual_info_regression for continuous targets
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import LabelEncoder
    import src.pipeline.visualizations as visualizations

    # =========================================================================
    # CONVERT TO NUMPY ARRAYS
    # =========================================================================
    if hasattr(feature_array, 'values'):
        feature_array = feature_array.values

    feature_array = np.asarray(feature_array)
    target_array_original = np.asarray(target_array)  # Preserve original for reference

    # =========================================================================
    # HELPER FUNCTIONS FOR TYPE DETECTION
    # =========================================================================

    def is_categorical_dtype(arr):
        """Check if array contains categorical/string data.

        Parameters
        ----------
        arr : array-like
            Input array to check.

        Returns
        -------
        bool
            True if array is object/string dtype, False otherwise.
        """
        arr = np.asarray(arr)
        return arr.dtype == object or arr.dtype.kind in ('U', 'S')

    def infer_discrete_vs_continuous(arr, unique_threshold_ratio=0.05):
        """Infer if numeric array should be treated as discrete or continuous.

        Parameters
        ----------
        arr : array-like
            Numeric array.
        unique_threshold_ratio : float
            If unique_values / n_samples < threshold_ratio, treat as discrete.

        Returns
        -------
        str
            'discrete' or 'continuous'.
        """
        arr = np.asarray(arr, dtype=float)
        n_unique = len(np.unique(arr))
        ratio = n_unique / len(arr)
        return 'discrete' if ratio < unique_threshold_ratio else 'continuous'

    # =========================================================================
    # ENCODE TARGET
    # =========================================================================

    if target_type == 'auto':
        # Detect categorical first (before numeric conversion)
        if is_categorical_dtype(target_array_original):
            target_type = 'discrete'
            target_array_encoded = LabelEncoder().fit_transform(target_array_original)
        else:
            # Numeric data: infer based on unique value ratio
            try:
                target_array_numeric = target_array_original.astype(float)
                target_type = infer_discrete_vs_continuous(target_array_numeric)
                target_array_encoded = target_array_numeric
            except (ValueError, TypeError):
                # Fallback: treat as discrete if conversion fails
                target_type = 'discrete'
                target_array_encoded = LabelEncoder().fit_transform(target_array_original)
    else:
        # Manual type specification provided
        if target_type == 'discrete':
            if is_categorical_dtype(target_array_original):
                target_array_encoded = LabelEncoder().fit_transform(target_array_original)
            else:
                target_array_encoded = target_array_original.astype(int)
        else:  # continuous
            target_array_encoded = target_array_original.astype(float)

    # =========================================================================
    # ENCODE FEATURES
    # =========================================================================

    encoded_features = np.zeros((feature_array.shape[0], feature_array.shape[1]), dtype=float)
    feature_is_categorical = np.zeros(feature_array.shape[1], dtype=bool)

    # Encode each feature column
    for col_idx in range(feature_array.shape[1]):
        col_data = feature_array[:, col_idx]

        # Check if categorical
        if is_categorical_dtype(col_data):
            feature_is_categorical[col_idx] = True
            le = LabelEncoder()
            encoded_features[:, col_idx] = le.fit_transform(col_data)
        else:
            # Try to convert to numeric
            try:
                encoded_features[:, col_idx] = col_data.astype(float)
                feature_is_categorical[col_idx] = False
            except (ValueError, TypeError):
                # If numeric conversion fails, treat as categorical
                feature_is_categorical[col_idx] = True
                le = LabelEncoder()
                encoded_features[:, col_idx] = le.fit_transform(col_data)

    # =========================================================================
    # INFER FEATURE TYPE (if auto)
    # =========================================================================

    if feature_type == 'auto':
        # Determine feature type based on per-feature analysis
        # A feature is discrete if: (a) categorical dtype OR (b) numeric with low cardinality
        feature_types = []
        for col_idx in range(encoded_features.shape[1]):
            if feature_is_categorical[col_idx]:
                feature_types.append('discrete')
            else:
                # Numeric feature: check cardinality
                feature_types.append(infer_discrete_vs_continuous(encoded_features[:, col_idx]))

        # Overall feature type: 'discrete' if majority are discrete, else 'continuous'
        n_discrete = sum(1 for ft in feature_types if ft == 'discrete')
        feature_type = 'discrete' if n_discrete > len(feature_types) / 2 else 'continuous'

    # =========================================================================
    # COMPUTE MUTUAL INFORMATION
    # =========================================================================

    # Select MI function based on feature and target types
    if feature_type == 'discrete' and target_type == 'discrete':
        mi_scores = mutual_info_classif(encoded_features, target_array_encoded.astype(int),
                                        random_state=random_state)
    elif feature_type == 'discrete' and target_type == 'continuous':
        mi_scores = mutual_info_regression(encoded_features, target_array_encoded.astype(float),
                                           random_state=random_state)
    elif feature_type == 'continuous' and target_type == 'discrete':
        mi_scores = mutual_info_classif(encoded_features, target_array_encoded.astype(int),
                                        random_state=random_state)
    else:  # continuous features, continuous target
        mi_scores = mutual_info_regression(encoded_features, target_array_encoded.astype(float),
                                           random_state=random_state)

    # Create importance dictionary
    feature_importance = dict(zip(feature_labels, mi_scores))

    # Print scores for inspection
    print(f"Feature MI scores ({feature_type} features <-> {target_type} target):")
    print({k: f"{v:.4f}" for k, v in feature_importance.items()})

    # =========================================================================
    # SORT IF REQUESTED
    # =========================================================================

    if sort_by_importance:
        feature_importance = dict(sorted(feature_importance.items(),
                                         key=lambda x: x[1], reverse=True))

    # =========================================================================
    # CREATE BARPLOT (if requested)
    # =========================================================================

    if include_barplot:
        fig, ax = plt.subplots(figsize=figsize)
        features = list(feature_importance.keys())
        scores = list(feature_importance.values())

        # Create bars
        bars = ax.bar(range(len(features)), scores, color='steelblue', alpha=0.7, edgecolor='navy')

        # Configure axes
        ax.set_xlabel('Features', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mutual Information Score', fontsize=11, fontweight='bold')
        ax.set_title(f'Feature Importance (MI: Feature ↔ {target_label})',
                     fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha='right')

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + max(scores) * 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # Save if directory provided
        if plot_save_dir is not None:
            visualizations.smart_save_fig(plot_save_dir, "Mutual_Information_Barplot")

        plt.show()

        return fig, ax, feature_importance

    return feature_importance




if __name__ == '__main__':
    from pathlib import Path
    from matplotlib import pyplot as plt
    import matplotlib as mpl

    mpl.use('MacOSX')

    # global vars:
    ROOT = Path().resolve().parent.parent
    QTC_DATA = ROOT / "data" / "qtc_measurements" / "2025_06"
    subject_data_dir = QTC_DATA / "sub-10"

    # area to scrutinize:
    use_ch_subset = False
    ch_subset = visualizations.EEG_CHANNELS_BY_AREA['Fronto-Central']
    ch_subset_inds = [visualizations.EEG_CHANNEL_IND_DICT[ch]-1 for ch in ch_subset]  # -1 to convert to computer-indices (0 = start)

    ### load data:
    print('Loading data...')
    # mmap_mode='r' leads to memory-mapped read-only access -> only loads data then accessed through slicing (more efficient!)
    input_file = np.load(subject_data_dir / "motor_eeg_full.npy", mmap_mode='r').T[:2048*20, ch_subset_inds if use_ch_subset else list(range(64))]  # 1 minute
    sampling_freq = 2048  # Hz
    n_channels = input_file.shape[1]; n_timesteps = input_file.shape[0]

    # compute dynamic multi-taper PSD:
    window_length_psd = .2
    spectrograms, timestamps, freqs = multitaper_psd(input_array=input_file, sampling_freq=sampling_freq, nw=3,
                                                     window_length_sec=window_length_psd, overlap_frac=0.5, axis=0, verbose=True,
                                                     # plot spectrogram for channel-average:
                                                     plot_result=False, frequency_range=(0, 100),)
    # spectrograms shape: (n_channels, n_windows, n_frequencies)
    # timestamps shape: (n_windows), frequencies shape: (n_frequencies)
    n_windows = spectrograms.shape[1]
    psd_sampling_freq = n_windows / (n_timesteps / sampling_freq)  # new_timesteps / time_duration

    # average (and eventually log-transform) spectrogram across frequency bins:
    do_log_transform: bool = True
    freq_averaged_psd_dict = {}  # keys: band-label keys, values: np.ndarrays shaped (n_channels, n_windows)
    for band_label, band_range in FREQUENCY_BANDS.items():
        frequency_mask = (freqs >= band_range[0]) & (freqs < band_range[1])  # select band frequencies
        spectrogram_subset = spectrograms[:, :, frequency_mask]
        if do_log_transform: spectrogram_subset = np.log10(spectrogram_subset + 1e-10)
        freq_averaged_psd_dict[band_label] = np.squeeze(np.mean(spectrogram_subset, axis=2))  # average across freqs.

    # lineplot overview:
    """
    fig, ax = plt.subplots()
    for ch in range(n_channels):
        ax.plot(timestamps, freq_averaged_psd_dict['beta'][ch, :],
                 label=visualizations.EEG_CHANNELS[ch])
    plt.xlabel('Time [s]'); plt.ylabel('Power [V^2/Hz]')
    plt.legend()
    plt.show()
    """

    # todo: ponder visualization of subset of EEG channels (e.g. np.zeros for the others)

    # animation:
    print(psd_sampling_freq)
    visualizations.animate_electrode_heatmap(
        freq_averaged_psd_dict['beta'].T,  # requires shape (n_timesteps, n_channels)
        positions=visualizations.EMG_POSITIONS, add_head_shape=False,
        sampling_rate=psd_sampling_freq, animation_fps=psd_sampling_freq,
        value_label="Power [V^2/Hz]" if not do_log_transform else "Power [V^2/Hz] [log10]",
        plot_title="EEG PSD (Beta-Band)"
    )