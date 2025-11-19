import numpy as np
from scipy.interpolate import interp1d
from typing import Literal
from scipy import signal
from tqdm import tqdm

import src.pipeline.visualizations as visualizations

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


def multitaper_psd(input_array: np.ndarray,
                   sampling_freq: float,
                   nw: float = 3,
                   window_length_sec: float = 1.0,
                   overlap_frac: float = 0.5,
                   axis: Literal[0, 1] = None,
                   verbose: bool = False,
                   plot_result: bool = False, **plot_kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute multitaper spectrogram (time-varying PSD) via sliding_window.

    Parameters
    ----------
    input_array : ndarray
        Input signal
    sampling_freq : float
        Sampling frequency (Hz)
    nw : float
        Time-half-bandwidth product (affects frequency smoothing)
    window_length_sec : float
        Window length in seconds (affects time resolution)
    overlap_frac : float
        Overlap between windows (0.5 = 50%, standard)

    Returns
    -------
    spectrogram : ndarray
        Shape (n_freqs, n_times) - power at each time-frequency point
    time_centers : ndarray
        Time center of each window (seconds)
    freqs : ndarray
        Frequency array (Hz)
    """
    # input sanity check:
    input_array, axis = check_2d_numpy_array(input_array, axis=axis)

    # input properties (axis is time-axis):
    n_samples = input_array.shape[axis]; n_channels = input_array.shape[axis+1%2]
    window_samples = int(window_length_sec * sampling_freq)
    hop_samples = int(window_samples * (1 - overlap_frac))  # = step size
    k = int(2 * nw - 1)  # number of tapers

    # generate tapers based on parameters:
    tapers = signal.windows.dpss(M=window_samples, NW=nw, Kmax=k)  # set return_ratios=True for eigenvalue scrutiny
    # tapers (output) shape is: (Kmax, M)

    channel_spectrogram_list: list[list[np.ndarray[float]]] = []
    channel_time_centers: list[list[float]] = []
    if verbose: print(f"Computing PSD for {n_channels} channels with {overlap_frac*100:.1f}% overlapping windows of size {window_length_sec:.3f}s.")
    for ch_ind in tqdm(range(n_channels)):  # iterate over channels for separate PSD computation
        # time-window result lists:
        spectrogram_list = []; time_centers = []

        # slide window across signal:
        for start_idx in range(0, n_samples - window_samples, hop_samples):
            end_idx = start_idx + window_samples
            # select window signal based on axis and window indices:
            window_signal = input_array[start_idx:end_idx, ch_ind] if axis == 0 else input_array[ch_ind, start_idx:end_idx]

            # compute PSD for this window with ALL tapers
            psd_window = []
            for taper in tapers:
                freqs, pxx = signal.periodogram(
                    window_signal, fs=sampling_freq, window=taper
                )  # shapes of freqs and pxx are (n_freqs)
                psd_window.append(pxx)

            # average across tapers:
            psd_mean = np.mean(psd_window, axis=0), # shape of psd_mean: (n_freqs)
            spectrogram_list.append(np.squeeze(psd_mean))  # squeez removes averaged dimension

            if ch_ind == 0:  # time_center computation is equivalent for all channels:
                # save time center (= timestamp):
                time_center = (start_idx + window_samples / 2) / sampling_freq
                time_centers.append(time_center)  # shape of time_center:

        # results have shapes:
        #   spectrogram_list: (n_windows, n_freqs), time_centers: (n_windows)

        # store in channel arrays:
        channel_spectrogram_list.append(spectrogram_list)
        if ch_ind == 0: channel_time_centers = time_centers  # timestamps are equivalent for all channels

    # convert to numpy:
    spectrograms = np.array(channel_spectrogram_list)
    timestamps = np.array(channel_time_centers)

    # plot if desired:
    if plot_result:
        fig_title = 'All-Channel Average PSD Spectrogram' if "title" not in plot_kwargs.keys() else plot_kwargs['title']
        _ = plot_kwargs.pop("title", None)  # remove title to prevent multiple values for kw argument
        visualizations.plot_spectrogram(spectrogram=np.squeeze(np.mean(spectrograms, axis=0)),
                         title=fig_title,
                         timestamps=timestamps,
                         frequencies=freqs,
                         **plot_kwargs)

    return spectrograms, timestamps, freqs


if __name__ == '__main__':
    from pathlib import Path
    from matplotlib import pyplot as plt
    import matplotlib as mpl

    mpl.use('MacOSX')
    """
    ####### DEV
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection
    from matplotlib.animation import FuncAnimation
    import numpy as np

    fig, ax = plt.subplots()
    circles = [patches.Circle((i, 0), 0.05) for i in range(10)]
    circle_collection = PatchCollection(circles, cmap='viridis')
    ax.add_collection(circle_collection)
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 1)

    values = np.random.rand(20, 10)
    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = plt.cm.viridis

    def update(frame):
        print(f"Updating frame {frame}")
        colors = cmap(norm(values[frame]))
        ax.set_title('Hi'+str(frame))
        for circle, color in zip(circles, colors):
            circle.set_facecolor(color)
        fig.canvas.draw_idle()
        return circle_collection,


    ani = FuncAnimation(fig, update, frames=20, interval=200, blit=False)
    plt.show()
    #### END DEV
    quit()"""





    # global vars:
    ROOT = Path().resolve().parent.parent
    QTC_DATA = ROOT / "data" / "qtc_measurements" / "2025_06"
    subject_data_dir = QTC_DATA / "sub-10"

    # area to scrutinize:
    use_ch_subset = False
    ch_subset = visualizations.EEG_CHANNELS_BY_AREA['Fronto-Central']
    ch_subset_inds = [visualizations.EEG_CHANNEL_IND_DICT[ch]-1 for ch in ch_subset]  # -1 to convert to computer-indices (0 = start)

    # load data:
    print('Loading data...')
    input_file = np.load(subject_data_dir / "motor_eeg_full.npy").T[:2048*20, ch_subset_inds if use_ch_subset else list(range(64))]  # 1 minute
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
    visualizations.animate_eeg_heatmap(
        freq_averaged_psd_dict['beta'].T,  # requires shape (n_timesteps, n_channels)
        sampling_rate=psd_sampling_freq, animation_fps=psd_sampling_freq,
        value_label="Power [V^2/Hz]" if not do_log_transform else "Power [V^2/Hz] [log10]",
        plot_title="EEG PSD (Beta-Band)"
    )
