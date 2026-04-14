import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Literal
from scipy import signal
from scipy.stats import beta, t as t_dist
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import src.pipeline.visualizations as visualizations
from src.pipeline.channel_layout import EEG_CHANNELS, EEG_CHANNELS_BY_AREA, EEG_CHANNEL_IND_DICT
import src.pipeline.data_integration as data_integration
import src.pipeline.data_analysis as data_analysis
import src.utils.file_management as filemgmt

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

###################### HELPER FUNCTIONS ######################
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


def mirror_eeg_channel_list(channels: list[str], input_is_left: bool = True) -> list[str]:
    mirrored_channels = []
    for channel in channels:
        if channel[-1] == 'z':
            mirrored_channels.append(channel)
        else:  # if not midline:
            if channel[-2:].isnumeric():  # two digits
                channel_ind = int(channel[-2:])
                channel_area = channel[:-2]
            elif channel[-1].isnumeric():  # one digit
                channel_ind = int(channel[-1])
                channel_area = channel[:-1]
            else:
                raise ValueError("Unrecognizable EEG channel name: ", channel)
            channel_ind += 1 if input_is_left else -1  # mirroring
            mirrored_channels.append(f"{channel_area}{channel_ind}")

    return mirrored_channels


###################### BIOSTATISTICAL FEATURES ######################
def multitaper_psd(input_array: np.ndarray,
                   sampling_freq: float,
                   nw: float = 3,
                   window_length_sec: float = 1.0,
                   overlap_frac: float = 0.5,
                   axis: Literal[0, 1] = None,
                   apply_log_scale: bool = True,
                   psd_save_dir: str | Path | None = None,
                   psd_file_suffix: str = "",
                   plot_result: bool = False, **plot_kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Compute multi-taper power spectral density using sliding windows.

    Applies DPSS (Slepian) tapers to overlapping time windows across input
    channels. Tapers are averaged to produce variance-reduced spectrograms
    suitable for motor cortex analysis, CMC computation, and dynamic tracking
    of oscillatory activity. Outputs may be log-scaled for statistical analysis
    and visualization.

    Parameters
    ----------
    input_array : np.ndarray
        Input time series data. Must be 2D. Shape is either
        (n_samples, n_channels) if axis=0 or (n_channels, n_samples) if axis=1.
        Contains EEG, EMG, or other time-domain biosignals.

    sampling_freq : float
        Sampling frequency in Hz. Used to define the time window length
        and control Nyquist frequency (fs/2). Must be > 0.

    nw : float, default 3
        Time-bandwidth product (Shannon number). Controls the number of
        orthogonal DPSS tapers generated: k = int(2*nw - 1).

        - nw=3 generates k=5 tapers (conservative, low frequency leakage)
        - nw=4 generates k=7 tapers (moderate variance reduction)
        - nw=5 generates k=9 tapers (aggressive variance reduction, higher leakage)

        Higher nw produces more tapers (averaging more uncorrelated estimates,
        lower variance) but reduces frequency resolution. Recommended range:
        3-5 for motor cortex analysis.

    window_length_sec : float, default 1.0
        Length of the sliding time window in seconds. Determines frequency
        resolution: Δf = 1/window_length_sec. Directly controls the time-
        frequency tradeoff.

        - 0.5 sec → 2 Hz resolution (higher temporal, coarser frequency)
        - 1.0 sec → 1 Hz resolution (standard)
        - 2.0 sec → 0.5 Hz resolution (better frequency, coarser temporal)

        For motor cortex beta-band analysis (15-35 Hz), 0.25-1.0 sec windows
        are typical. Should be ≥ 1/(2*tapsmofrq) per Shannon-Nyquist.

    overlap_frac : float, default 0.5
        Fraction of window overlap between consecutive sliding windows.
        Range: [0, 1).

        - 0.5 → 50% overlap, hop_samples = window_samples * 0.5
        - 0.75 → 75% overlap, denser time sampling (more computation)

        For dynamic tracking of motor changes, 0.5-0.75 is typical.
        Recommended: 0.5 for balance between resolution and efficiency.

    axis : Literal[0, 1], default None
        Axis along which samples are oriented.

        - axis=0: input_array shape = (n_samples, n_channels)
        - axis=1: input_array shape = (n_channels, n_samples)

        If None, automatically detected via check_2d_numpy_array().
        Input is transposed internally if axis=1 for faster indexing.

    apply_log_scale : bool, default True
        If True, convert output power spectral density from linear units (V²/Hz)
        to log₁₀ scale. Output shape, dimensions, and time/frequency vectors
        remain unchanged; only power values are transformed.

        **Important for statistical analysis:** EEG/EMG power follows a log-normal
        distribution. Log transformation is **strongly recommended** for:

        - Parametric statistical tests (ANOVA, t-tests, linear regression)
        - Baseline-corrected analysis (ERSP computation)
        - Normalization to meet normality assumptions
        - Visualization of power across multiple orders of magnitude

        The log transformation converts multiplicative motor effects to additive
        effects, aligning with motor cortex physiology. For example, a doubling
        of power (linear: 1→2) becomes a +3 dB change (log: 0→3 dB).

        **Output units when True:**
        - Linear (False): V²/Hz
        - Log (True): dimensionless log₁₀ units (equivalent to 10×log₁₀(V²/Hz) dB,
          offset by 10 dB from strict dB definition but preserves all information)

        **Statistical workflow with motor tasks:**

        1. Set apply_log_scale=True (default)
        2. Extract baseline period (e.g., 5 sec before movement)
        3. Baseline-correct in log space: ERSP = psd_log - baseline_mean
        4. Run parametric tests (ANOVA, t-tests) on ERSP
        5. This ordering ensures proper normality and statistical validity

        **Validation:** Log-transformed EEG power passes Anderson-Darling normality
        tests (p > 0.05), whereas raw power fails (p < 0.001).

        Note: A small positive constant (1e-10) is added before log transformation
        to avoid log(0) errors, but this has negligible effect on non-zero power
        values.

    psd_save_dir : str | Path | None, default None
        Directory path for saving output arrays as NumPy .npy files.
        If None, arrays are not saved to disk.

        When provided, three files are created:
        - PSD Spectrograms {int(n_channels)}ch {window_length_sec:.2f}sec_window[log10 scaled].npy
        - PSD Timecenters {len(time_centers)}windows.npy
        - PSD Frequencies {len(freqs)}freqs.npy

        Filename automatically includes "[log10 scaled]" suffix when
        apply_log_scale=True, for clarity in file names.

    psd_file_suffix : str, default ""
        Custom suffix appended to saved filenames. Useful for distinguishing
        different preprocessing or parameter configurations.
        Example: psd_file_suffix="left_hand_task" produces
        "... left_hand_task.npy" filenames.

    plot_result : bool, default False
        If True, generate a spectrogram visualization showing the average PSD
        across all channels, with time (seconds) and frequency (Hz) axes.
        Requires matplotlib and access to visualization.plot_spectrogram().

        The plot will display log-scaled power (if apply_log_scale=True) or
        linear power, with appropriate colorbar labeling.

    **plot_kwargs : dict
        Additional keyword arguments passed to visualization.plot_spectrogram().
        Common options:

        - title : str - Figure title (auto-generated if not provided)
        - cmap : str - Colormap name (default: 'viridis')
        - vlim : tuple - (vmin, vmax) for color scaling in log units if
          apply_log_scale=True
        - figsize : tuple - (width, height) in inches

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:

        spectrograms : np.ndarray, shape (n_times, n_freqs, n_channels)
            Time-frequency-channel power spectral density estimates.

            **Output units depend on apply_log_scale:**

            - If apply_log_scale=False: [V²/Hz] linear power spectral density
            - If apply_log_scale=True: [log₁₀(V²/Hz)] dimensionless log units
              (equivalent to 10×log₁₀(V²/Hz) dB, offset by 10 from dB convention
              but preserves power relationships)

            Array dimensions:

            - n_times : Number of sliding windows (floor((n_samples - window_samples) / hop_samples) + 1)
            - n_freqs : Number of frequency bins (typically n_samples / 2 + 1 for one-sided spectrum)
            - n_channels : Number of input channels

            Values are averaged across all k DPSS tapers. Suitable for
            visualization, statistical analysis, or downstream coherence
            (CMC) calculation.

        time_centers : np.ndarray, shape (n_times,)
            Center times of each sliding window in seconds [float].
            Computed as (window_start + window_samples/2) / sampling_freq.

            Aligned with spectrograms[:, :, :] first dimension. Use for
            temporal axis in plots or time-resolved analysis.

        freqs : np.ndarray, shape (n_freqs,)
            Frequency bins in Hz [float]. Ranges from 0 to sampling_freq/2
            (one-sided spectrum). Computed by scipy.signal.periodogram.

            Frequency resolution: freqs[1] - freqs[0] ≈ 1/window_length_sec.

    Notes
    -----
    **Implementation Details:**

    This function applies the multi-taper method (Thomson, 1982) for variance
    reduction in spectral estimation. The process:

    1. Generates k orthogonal DPSS tapers using scipy.signal.windows.dpss()
    2. Extracts overlapping windows via fancy indexing (vectorized)
    3. For each taper, computes periodogram across all windows and channels
    4. Averages periodograms across tapers → final PSD estimate
    5. Optionally applies log₁₀ transformation for statistical analysis

    **Frequency Resolution:**

    The frequency grid is determined by the FFT of window_length_sec data:

        Δf = sampling_freq / (sampling_freq * window_length_sec) = 1 / window_length_sec

    This cannot be changed without altering window_length_sec. Zero-padding
    (increasing FFT size) would interpolate but not improve resolution.

    **Tapers and Variance Reduction:**

    DPSS tapers are orthogonal sequences optimized for time-bandwidth product.
    Averaging k uncorrelated estimates reduces variance by factor of k while
    preserving bias. For motor cortex:

    - k=5 tapers (nw=3): ~5× variance reduction, minimal spectral leakage
    - k=7 tapers (nw=4): ~7× variance reduction, slight leakage trade-off

    **Log-Scale Transformation and Normality:**

    EEG/EMG power follows a log-normal distribution (right-skewed). Raw power
    values violate the normality assumption required by parametric tests. Log
    transformation converts the distribution to approximately normal, enabling
    valid hypothesis testing. Empirical evidence (Anderson-Darling test):

    - Raw power: fails normality (p < 0.001)
    - Log-transformed power: passes normality (p > 0.05)

    The log transformation also accounts for multiplicative motor effects: when
    motor drive increases, power tends to multiply rather than add. Log
    transformation converts multiplication to addition, aligning with the
    additive assumptions of ANOVA and linear models.

    **Baseline Correction with Log-Scaled Data:**

    When using apply_log_scale=True for baseline-corrected power (ERSP):

    1. This function returns log-transformed PSD
    2. Extract baseline period: baseline = psd[:baseline_idx, :, :].mean(axis=0)
    3. Baseline-correct: ersp = psd - baseline  # Subtraction in log space
    4. This equals 10×log₁₀(PSD/baseline) in log units
    5. Run statistics (ANOVA, t-tests) on ersp—normality is now satisfied

    For visualization, the result (ersp) represents log power relative to
    baseline, in the same units as the output spectrograms.

    **Edge Effects:**

    Windows near the start (t < 2 sec) and end (t > duration - 2 sec) may
    exhibit edge artifacts, especially for low frequencies. Consider excluding
    ±2 seconds from analytical boundaries for robust estimates.

    **Motor Cortex Recommendations:**

    For CMC and motor task analysis with 45-second snippets:

    - window_length_sec = 0.25-0.5 sec for beta-band dynamics (15-35 Hz)
    - nw = 3-4 for balanced taper count
    - overlap_frac = 0.5-0.75 for smooth time tracking
    - apply_log_scale = True for statistical analysis (strongly recommended)
    - Time window should be ≥ 5 cycles at lowest frequency of interest

    Example: Beta band (15 Hz minimum), 5 cycles → min window = 5/15 = 0.33 sec.

    **Statistical Properties:**

    When apply_log_scale=False: Output values are unbiased estimators of the
    true power spectral density under stationarity assumptions. Variance scales
    as O(1/window_length) and O(1/k_tapers).

    When apply_log_scale=True: Output approximates a normal distribution (validated
    by Anderson-Darling test). This enables:
    - Parametric statistics (ANOVA, t-tests, linear mixed-effects models)
    - Confidence intervals from normal assumptions
    - Proper hypothesis testing without nonparametric corrections

    Confidence intervals can be estimated via jackknife or bootstrap across
    tapers for either linear or log-scaled output.

    **Relationship to Other Methods:**

    - Single-taper Welch: This method generalizes Welch by using k > 1 tapers
    - FFT spectrogram: Multi-taper is less noisy but slower
    - Wavelet spectrograms: Multi-taper has higher frequency resolution at
      fixed temporal resolution
    - Baseline-corrected ERSP: Equivalent to (apply_log_scale=True, then subtract
      baseline), the standard in motor neurophysiology

    **Comparison: apply_log_scale=True vs. False:**

    | Aspect | Linear (False) | Log (True) |
    |--------|--------|-------|
    | Output units | V²/Hz | log₁₀(V²/Hz) |
    | Distribution | Right-skewed | ~Normal |
    | Parametric tests | Invalid without transform | Valid |
    | Visualization | Dominated by low freq | Full dynamic range visible |
    | Motor effect semantics | Additive | Multiplicative (true) |
    | File sizes | Same | Same (no compression) |

    Raises
    ------
    ValueError
        If window_length_sec is not positive or if overlap_frac ≥ 1.
    RuntimeError
        If save directory does not exist and psd_save_dir is not None.
    """

    input_array, axis = check_2d_numpy_array(input_array, axis=axis)

    n_samples = input_array.shape[axis]
    n_channels = input_array.shape[(axis + 1) % 2]
    window_samples = int(window_length_sec * sampling_freq)
    hop_samples = int(window_samples * (1 - overlap_frac))
    k = int(2 * nw - 1)  # Shannon number: floor(2*NW - 1) tapers
    # For nw=3: k=5, nw=4: k=7, nw=5: k=9

    # Generate tapers once
    tapers = signal.windows.dpss(M=window_samples, NW=nw, Kmax=k)  # shape: (k, window_samples)

    # Pre-compute window indices (vectorized)
    window_starts = np.arange(0, n_samples - window_samples, hop_samples)

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
            tapered_windows = windows * taper[np.newaxis, :]  # Explicit taper application
            freqs, pxx = signal.periodogram(tapered_windows, fs=sampling_freq,
                                            axis=1, window=None)  # No double-windowing
            """  # alternative, but above is lcearer
            # Vectorized periodogram for all windows with same taper
            freqs, pxx = signal.periodogram(windows, fs=sampling_freq,
                                            axis=1, window=taper)"""
            # pxx shape: (n_windows, n_freqs)
            psd_list.append(pxx)

        # Mean across tapers: (n_windows, n_freqs)
        channel_spec = np.mean(psd_list, axis=0)
        spectrograms.append(channel_spec)

    # Convert to output format: (n_windows, n_freqs, n_channels)
    spectrograms = np.transpose(np.array(spectrograms), axes=[1, 2, 0])

    # apply log scale:
    if apply_log_scale:
        spectrograms = np.log10(np.abs(spectrograms) + 1e-10)

    if psd_save_dir is not None:
        # save all relevant arrays:
        save_spectrograms(spectrograms, time_centers, freqs, "PSD", save_dir=psd_save_dir, identifier_suffix=psd_file_suffix)



    if plot_result:
        fig_title = 'All-Channel Average PSD Spectrogram' if "title" not in plot_kwargs.keys() else plot_kwargs['title']
        _ = plot_kwargs.pop("title", None)
        visualizations.plot_spectrogram(spectrogram=np.squeeze(np.mean(spectrograms, axis=2)),
                                        title=fig_title,
                                        timestamps=time_centers,
                                        frequencies=freqs,
                                        **plot_kwargs)

    return spectrograms, time_centers, freqs



###################### COHERENCE COMPUTATION ######################
def fisher_atanh_transform(coherence: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Forward Fisher atanh: C² → z. Stabilizes variance to ≈ 1/(K-1)."""
    coherence_safe = np.clip(coherence, eps, 1 - eps)
    return 0.5 * np.log((1 + coherence_safe) / (1 - coherence_safe))


def inverse_fisher_atanh(z: np.ndarray) -> np.ndarray:
    """Inverse Fisher atanh: z → C². Maps z-space back to [0,1]."""
    return np.tanh(z) ** 2


def compute_cmc_independence_threshold(K: int, alpha: float = 0.05) -> float:
    """
    Compute CMC independence threshold from null distribution Beta(K-2, K-2). Returns (1-alpha) quantile.

    K: number of tapers
    alpha: significance level

    Returns:
        CMC independence threshold, values below can be due to chance.
    """
    a = b = K - 2
    return beta.ppf(1 - alpha, a, b)


def jackknife_coherence_and_ci(
        tapers_filtered: np.ndarray,
        eeg_window: np.ndarray,
        emg_window: np.ndarray,
        sampling_freq: float,
        window_samples: int,
        jackknife_alpha: float = 0.05
) -> tuple:
    """
    Jackknife leave-one-out resampling.

    Key insight: Compute mean in COHERENCE space (for unbiased point estimate),
    but variance in FISHER Z-space (for stable, normalized intervals).
    """
    K = len(tapers_filtered)
    n_freqs = len(np.fft.rfftfreq(window_samples, d=1 / sampling_freq))
    n_eeg_ch = eeg_window.shape[1]
    n_emg_ch = emg_window.shape[1]

    jackknife_replicates_coherence = np.zeros((K, n_freqs, n_eeg_ch, n_emg_ch), dtype=np.float32)
    jackknife_replicates_z = np.zeros((K, n_freqs, n_eeg_ch, n_emg_ch), dtype=np.float32)

    # Leave-one-out loop: recompute PSD/CSD excluding each taper
    for leave_out_idx in range(K):
        psd_eeg_sum = np.zeros((n_freqs, n_eeg_ch), dtype=np.float32)
        psd_emg_sum = np.zeros((n_freqs, n_emg_ch), dtype=np.float32)
        csd_sum = np.zeros((n_freqs, n_eeg_ch, n_emg_ch), dtype=np.complex64)

        # Accumulate across K-1 tapers
        for k, taper in enumerate(tapers_filtered):
            if k == leave_out_idx:
                continue

            eeg_tapered = eeg_window * taper[:, np.newaxis]
            emg_tapered = emg_window * taper[:, np.newaxis]

            eeg_fft = np.fft.rfft(eeg_tapered, axis=0)
            emg_fft = np.fft.rfft(emg_tapered, axis=0)

            psd_eeg = np.abs(eeg_fft) ** 2 / (sampling_freq * window_samples)
            psd_emg = np.abs(emg_fft) ** 2 / (sampling_freq * window_samples)

            psd_eeg_sum += psd_eeg
            psd_emg_sum += psd_emg

            csd = (
                    np.conj(eeg_fft)[:, :, np.newaxis] * emg_fft[:, np.newaxis, :]
                    / (sampling_freq * window_samples)
            )
            csd_sum += csd

        # Average across K-1 tapers
        psd_eeg_jk = psd_eeg_sum / (K - 1)
        psd_emg_jk = psd_emg_sum / (K - 1)
        csd_jk = csd_sum / (K - 1)

        # Compute coherence
        numerator_jk = np.abs(csd_jk) ** 2
        denominator_jk = psd_eeg_jk[:, :, np.newaxis] * psd_emg_jk[:, np.newaxis, :]

        eps = np.finfo(np.float64).tiny
        denominator_jk_safe = np.maximum(denominator_jk, eps)

        coherence_jk = numerator_jk / denominator_jk_safe
        coherence_jk = np.clip(coherence_jk, 0, 1)

        # Store both coherence and z-transformed versions
        jackknife_replicates_coherence[leave_out_idx, :, :, :] = coherence_jk
        jackknife_replicates_z[leave_out_idx, :, :, :] = fisher_atanh_transform(coherence_jk)

    # *** KEY CHANGE: Compute mean in COHERENCE space ***
    coherence_mean = np.mean(jackknife_replicates_coherence, axis=0)
    coherence_mean = np.clip(coherence_mean, 0, 1)

    # *** Compute variance from Z-space deviations (stabilized) ***
    z_jack_mean = np.mean(jackknife_replicates_z, axis=0)
    z_deviations = jackknife_replicates_z - z_jack_mean[np.newaxis, :, :, :]
    z_var = ((K - 1) / K) * np.sum(z_deviations ** 2, axis=0)
    z_se = np.sqrt(z_var)

    # Student-t CI in z-space, centered on z-transform of the coherence mean
    t_crit = t_dist.ppf(1-(jackknife_alpha/2), K - 1)
    z_mean_of_coherence = fisher_atanh_transform(coherence_mean)
    z_ci_lower = z_mean_of_coherence - t_crit * z_se
    z_ci_upper = z_mean_of_coherence + t_crit * z_se

    # Transform back to coherence space [0,1]
    coherence_ci_lower = inverse_fisher_atanh(z_ci_lower)
    coherence_ci_upper = inverse_fisher_atanh(z_ci_upper)

    # Enforce bounds explicitly to handle any remaining numerical artifacts
    coherence_ci_lower = np.minimum(coherence_ci_lower, coherence_mean)
    coherence_ci_upper = np.maximum(coherence_ci_upper, coherence_mean)

    return coherence_mean, coherence_ci_lower, coherence_ci_upper


def apply_threshold_filtering(
        coherence_values: np.ndarray,
        K: int,
        alpha: float = 0.05,
        n_comparisons: int = None,
        apply_bonferroni: bool = False,
) -> tuple:
    """
    Apply IT filtering with optional Bonferroni correction.

    Computes IT from Beta(K-2,K-2). If apply_bonferroni=True, adjusts alpha
    by dividing by n_comparisons. Returns: (significant_mask, IT_threshold).
    """
    if apply_bonferroni and n_comparisons is not None:
        alpha_adjusted = alpha / n_comparisons
        if alpha_adjusted < 1e-10:
            alpha_adjusted = 1e-10
    else:
        alpha_adjusted = alpha

    IT = compute_cmc_independence_threshold(K, alpha=alpha_adjusted)
    significant_mask = coherence_values > IT

    return significant_mask, IT


def _normalize_to_time_first(array: np.ndarray, axis: int) -> np.ndarray:
    """Transpose array if needed to ensure (n_samples, n_channels) format."""
    if axis == 1:
        array = array.T
    return array


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def multitaper_magnitude_squared_coherence(
        eeg_array: np.ndarray,
        emg_array: np.ndarray,
        sampling_freq: float,
        nw: float = 3,
        window_length_sec: float = 1.0,
        overlap_frac: float = 0.5,
        eeg_axis: Literal[0, 1] = 0,
        emg_axis: Literal[0, 1] = 0,
        taper_eigenvalue_threshold: float = 0.90,
        use_jackknife: bool = True,
        jackknife_alpha: float = 0.05,
        apply_independence_threshold: bool = True,
        apply_bonferroni_correction: bool = False,
        significance_level: float = 0.05,
        window_mask: np.ndarray | None = None,   # <-- NEW: bool array (n_windows,)
        verbose: bool = False,
) -> dict:
    """
    ... (existing docstring, add one entry) ...

    window_mask : np.ndarray of bool, shape (n_windows,), optional
        If provided, only windows where ``window_mask[i] is True`` are
        computed.  All other windows are left as zeros in every output
        array.  The mask must have exactly ``n_windows`` elements; a
        ValueError is raised otherwise.

        This is the correct mechanism for task-selective computation:
        build the mask once from task timestamps (see
        ``_build_task_window_mask``), pass it here, and the global
        window grid remains intact — no slicing, no stitching.
    """

    # ---- INPUT VALIDATION & NORMALIZATION ----
    eeg_array = _normalize_to_time_first(eeg_array, axis=eeg_axis)
    emg_array = _normalize_to_time_first(emg_array, axis=emg_axis)

    n_samples_eeg, n_eeg_channels = eeg_array.shape
    n_samples_emg, n_emg_channels = emg_array.shape

    if n_samples_eeg != n_samples_emg:
        raise ValueError(
            f"EEG and EMG must have same number of samples. "
            f"Got EEG: {n_samples_eeg}, EMG: {n_samples_emg}"
        )
    n_samples = n_samples_eeg

    # ---- WINDOW PARAMETERS ----
    window_samples = int(window_length_sec * sampling_freq)
    hop_samples    = int(window_samples * (1 - overlap_frac))
    k              = int(2 * nw - 1)

    # ---- INITIALIZE TAPERS ----
    tapers, taper_eigs = signal.windows.dpss(
        M=window_samples, NW=nw, Kmax=k, return_ratios=True
    )
    keep_mask         = taper_eigs > taper_eigenvalue_threshold
    tapers_filtered   = tapers[keep_mask]
    tapers_normalized = [t / np.sqrt(np.sum(t ** 2)) for t in tapers_filtered]
    K                 = len(tapers_normalized)

    freqs    = np.fft.rfftfreq(window_samples, d=1 / sampling_freq)
    n_freqs  = len(freqs)
    n_windows = (n_samples - window_samples) // hop_samples + 1

    # ---- VALIDATE window_mask ----
    if window_mask is not None:
        window_mask = np.asarray(window_mask, dtype=bool)
        if window_mask.shape != (n_windows,):
            raise ValueError(
                f"window_mask must have shape ({n_windows},), "
                f"got {window_mask.shape}"
            )
        n_active = int(window_mask.sum())
        if verbose:
            print(f"window_mask: {n_active}/{n_windows} windows will be computed "
                  f"({100 * n_active / n_windows:.1f}%)")
    else:
        n_active = n_windows

    if verbose:
        print(f"Using {K} high-quality tapers (λ > {taper_eigenvalue_threshold})")
        print(
            f"Computing MSC: {n_eeg_channels} EEG × {n_emg_channels} EMG channels"
        )
        print(
            f"Window: {window_length_sec:.3f}s, Overlap: {overlap_frac * 100:.1f}%, "
            f"Tapers: {K}"
        )

    # ---- PRE-ALLOCATE OUTPUT ARRAYS (full grid, zeros for skipped windows) ----
    coherences_raw = np.zeros(
        (n_windows, n_freqs, n_eeg_channels, n_emg_channels), dtype=np.float32
    )
    time_centers = np.zeros(n_windows, dtype=np.float64)

    if use_jackknife:
        coherences_ci_lower = np.zeros_like(coherences_raw)
        coherences_ci_upper = np.zeros_like(coherences_raw)

    if apply_independence_threshold:
        coherences_significant = np.zeros(
            (n_windows, n_freqs, n_eeg_channels, n_emg_channels), dtype=bool
        )

    # ---- MAIN WINDOW LOOP ----
    for win_idx in tqdm(range(n_windows), disable=not verbose, desc="Window"):

        # Always fill time_centers — callers rely on this even for skipped windows.
        start_idx = win_idx * hop_samples
        time_centers[win_idx] = (start_idx + window_samples / 2) / sampling_freq

        # *** SKIP non-task windows ***
        if window_mask is not None and not window_mask[win_idx]:
            continue

        end_idx    = start_idx + window_samples
        eeg_window = eeg_array[start_idx:end_idx, :]
        emg_window = emg_array[start_idx:end_idx, :]

        psd_eeg_sum = np.zeros((n_freqs, n_eeg_channels), dtype=np.float64)
        psd_emg_sum = np.zeros((n_freqs, n_emg_channels), dtype=np.float64)
        csd_sum     = np.zeros((n_freqs, n_eeg_channels, n_emg_channels), dtype=np.complex128)

        for taper in tapers_normalized:
            eeg_tapered = eeg_window * taper[:, np.newaxis]
            emg_tapered = emg_window * taper[:, np.newaxis]

            eeg_fft = np.fft.rfft(eeg_tapered, axis=0)
            emg_fft = np.fft.rfft(emg_tapered, axis=0)

            psd_eeg = np.abs(eeg_fft) ** 2 / (sampling_freq * window_samples)
            psd_emg = np.abs(emg_fft) ** 2 / (sampling_freq * window_samples)

            psd_eeg_sum += psd_eeg
            psd_emg_sum += psd_emg

            csd = (
                np.conj(eeg_fft)[:, :, np.newaxis] * emg_fft[:, np.newaxis, :]
                / (sampling_freq * window_samples)
            )
            csd_sum += csd

        psd_eeg_mean = psd_eeg_sum / K
        psd_emg_mean = psd_emg_sum / K
        csd_mean     = csd_sum / K

        # ---- STEP 1: RAW COHERENCE ----
        numerator      = np.abs(csd_mean) ** 2
        denominator    = psd_eeg_mean[:, :, np.newaxis] * psd_emg_mean[:, np.newaxis, :]
        eps            = np.finfo(np.float64).tiny
        coherence_raw  = np.clip(numerator / np.maximum(denominator, eps), 0, 1)

        # ---- STEP 2: JACKKNIFE CI ----
        if use_jackknife:
            coherence_mean, ci_lower, ci_upper = jackknife_coherence_and_ci(
                tapers_normalized, eeg_window, emg_window,
                sampling_freq, window_samples,
                jackknife_alpha=jackknife_alpha,
            )
            coherences_raw[win_idx]        = coherence_mean
            coherences_ci_lower[win_idx]   = ci_lower
            coherences_ci_upper[win_idx]   = ci_upper
        else:
            coherences_raw[win_idx] = coherence_raw

        # ---- STEP 3: IT FILTERING ----
        if apply_independence_threshold:
            n_comparisons = (
                n_eeg_channels * n_emg_channels
                if apply_bonferroni_correction else None
            )
            significant_mask, _ = apply_threshold_filtering(
                coherences_raw[win_idx], K=K,
                alpha=significance_level,
                n_comparisons=n_comparisons,
                apply_bonferroni=apply_bonferroni_correction,
            )
            coherences_significant[win_idx] = significant_mask

    # ---- BUILD RESULTS (unchanged from original) ----
    result = {
        "coherence_raw":  coherences_raw,
        "time_centers":   time_centers,
        "freqs":          freqs,
        "metadata": {
            "K_tapers":                     K,
            "n_windows":                    n_windows,
            "n_active_windows":             n_active,
            "window_length_sec":            window_length_sec,
            "overlap_frac":                 overlap_frac,
            "use_jackknife":                use_jackknife,
            "apply_independence_threshold": apply_independence_threshold,
            "apply_bonferroni_correction":  apply_bonferroni_correction,
            "significance_level":           significance_level,
        },
    }

    if use_jackknife:
        result["coherence_ci_lower"] = coherences_ci_lower
        result["coherence_ci_upper"] = coherences_ci_upper

    if apply_independence_threshold:
        result["coherence_significant"] = coherences_significant
        IT_unadjusted = compute_cmc_independence_threshold(K, alpha=significance_level)
        result["metadata"]["IT_unadjusted"] = float(IT_unadjusted)
        if apply_bonferroni_correction:
            n_comp = n_eeg_channels * n_emg_channels
            result["metadata"]["IT_bonferroni"] = float(
                compute_cmc_independence_threshold(K, alpha=significance_level / n_comp)
            )
            result["metadata"]["n_comparisons"] = n_comp
        result["metadata"]["n_significant"] = int(np.sum(coherences_significant))

    if verbose:
        print(f"\n✓ Done!")
        if apply_independence_threshold:
            print(f"  IT (unadjusted): {result['metadata']['IT_unadjusted']:.3f}")
            print(f"  Significant: {result['metadata']['n_significant']}")

    return result


def _build_task_window_mask(
        time_centers_sec: np.ndarray,
        log_frame: pd.DataFrame,
        pre_buffer_sec: float,
        post_buffer_sec: float,
) -> np.ndarray:
    """
    Return a boolean mask (shape: n_windows) marking windows whose centre
    falls within any task period expanded by the requested buffers.

    Works entirely in floating-point seconds-since-recording-start space,
    which is the same space as ``time_centers_sec``.  No DataFrame or
    Timestamp arithmetic at window granularity.

    Parameters
    ----------
    time_centers_sec : np.ndarray, shape (n_windows,)
        Centre time of each window in seconds from recording start,
        as returned by ``multitaper_magnitude_squared_coherence``.
    log_frame : pd.DataFrame
        Task log with timing columns (same format used elsewhere).
    pre_buffer_sec, post_buffer_sec : float
        Seconds to expand each task period before / after its edges.
        Mirrors the buffer semantics of the old task-wise approach.

    Returns
    -------
    mask : np.ndarray of bool, shape (n_windows,)
    """
    measurement_start, _ = data_integration.get_qtc_measurement_start_end(log_frame)
    measurement_start_aware = data_analysis.make_timezone_aware(
        pd.Timestamp(measurement_start)
    )
    trial_start_ends = data_integration.get_all_task_start_ends(
        log_frame, output_type='list'
    )

    mask = np.zeros(len(time_centers_sec), dtype=bool)

    for trial_start, trial_end in trial_start_ends:
        # Convert absolute timestamps → seconds from recording start.
        # This is the same reference frame as time_centers_sec.
        t0 = (trial_start - measurement_start_aware).total_seconds() - pre_buffer_sec
        t1 = (trial_end   - measurement_start_aware).total_seconds() + post_buffer_sec
        mask |= (time_centers_sec >= t0) & (time_centers_sec <= t1)

    n_active = int(mask.sum())
    print(
        f"Task window mask: {n_active}/{len(mask)} windows selected "
        f"({100 * n_active / len(mask):.1f}%) across "
        f"{len(trial_start_ends)} trials "
        f"[±{pre_buffer_sec}s / +{post_buffer_sec}s buffers]"
    )
    return mask


def compute_task_wise_aggregated_cmc(
        eeg_array: np.ndarray,
        emg_array: np.ndarray,
        sampling_freq: int,
        muscle_group: str,
        log_frame: pd.DataFrame | None = None,
        eeg_channel_subset: list[str] | None = None,
        window_size_sec: float = 2.0,
        window_overlap_ratio: float = 0.5,
        enforce_independence_threshold: bool = False,
        independence_threshold_alpha: float = 0.2,
        use_jackknife: bool = True,
        jackknife_alpha: float = 0.05,
        save_dir: str | Path | None = None,
        pre_trial_computation_buffer_sec: float = 3.0,
        post_trial_computation_buffer_sec: float = 3.0,
) -> tuple:
    """
    Compute channel-aggregated CMC between EEG and EMG signals.

    Uses a single global sliding window grid (identical to multitaper_psd).
    When log_frame is provided, a boolean window mask marks only windows
    whose centre falls within a task period ± the requested buffers; all
    other windows are skipped and left as zeros.  This eliminates the
    slice-and-stitch approach used previously and fixes all geometry
    mismatch / index drift issues.
    """
    # ---- channel subset ----
    if eeg_channel_subset:
        eeg_channel_subset_inds = [EEG_CHANNEL_IND_DICT[ch] for ch in eeg_channel_subset]
        print(f"Reducing EEG to {len(eeg_channel_subset)} channels: {eeg_channel_subset}")
        eeg_array = eeg_array[:, eeg_channel_subset_inds]

    n_samples_eeg, n_eeg_channels = eeg_array.shape
    n_samples_emg, n_emg_channels = emg_array.shape
    if n_samples_eeg != n_samples_emg:
        raise ValueError(
            f"EEG and EMG must have same number of samples. "
            f"Got EEG: {n_samples_eeg}, EMG: {n_samples_emg}"
        )

    # ---- build task mask (same global grid as multitaper_psd) ----
    if log_frame is not None:
        # Pre-compute the global time_centers so we can build the mask
        # before calling the coherence function.  This mirrors how
        # multitaper_psd builds window_starts up front.
        window_samples  = int(window_size_sec * sampling_freq)
        hop_samples     = int(window_samples * (1 - window_overlap_ratio))
        if hop_samples <= 0:
            raise ValueError("window_overlap_ratio too high: hop_samples becomes <= 0")
        n_windows       = (n_samples_eeg - window_samples) // hop_samples + 1
        window_starts   = np.arange(n_windows) * hop_samples
        time_centers_preview = (window_starts + window_samples / 2) / sampling_freq

        window_mask = _build_task_window_mask(
            time_centers_sec=time_centers_preview,
            log_frame=log_frame,
            pre_buffer_sec=pre_trial_computation_buffer_sec,
            post_buffer_sec=post_trial_computation_buffer_sec,
        )
    else:
        window_mask = None  # compute every window

    # ---- single coherence call over the full arrays ----
    output_dict = multitaper_magnitude_squared_coherence(
        eeg_array, emg_array,
        sampling_freq=sampling_freq,
        window_length_sec=window_size_sec,
        overlap_frac=window_overlap_ratio,
        significance_level=independence_threshold_alpha,
        apply_independence_threshold=enforce_independence_threshold,
        use_jackknife=use_jackknife,
        jackknife_alpha=jackknife_alpha,
        window_mask=window_mask,
        verbose=True,
    )

    time_centers = output_dict['time_centers']
    freqs        = output_dict['freqs']

    # ---- mask by significance (if requested) ----
    values = (
        np.where(output_dict['coherence_significant'], output_dict['coherence_raw'], 0.0)
        if enforce_independence_threshold
        else output_dict['coherence_raw']
    )

    # ---- sanity-check CI bounds ----
    if use_jackknife:
        assert np.all(output_dict['coherence_raw'] >= output_dict['coherence_ci_lower']), \
            "CI lower bound exceeded coherence mean"
        assert np.all(output_dict['coherence_raw'] <= output_dict['coherence_ci_upper']), \
            "CI upper bound below coherence mean"

    # ---- aggregate over EMG channels (max across EMG axis) ----
    if use_jackknife:
        values, values_lower, values_upper = max_cmc_spectrograms_over_channels(
            values,
            output_dict['coherence_ci_lower'],
            output_dict['coherence_ci_upper'],
            channel_ax=3,
            verbose=True,
        )
    else:
        values = max_cmc_spectrograms_over_channels(
            values, channel_ax=3, verbose=True,
        )

    # ---- save ----
    if save_dir is not None:
        channel_suffix = (
            f"Channels_{'_'.join(eeg_channel_subset)}"
            if eeg_channel_subset else "All_Channels"
        )
        label = (
            f"{muscle_group.capitalize()} CMC"
            f"{' Trial-wise' if log_frame is not None else ''}"
        )
        save_spectrograms(
            values, time_centers, freqs,
            save_dir=save_dir,
            modality=label,
            identifier_suffix=channel_suffix,
        )

    # ---- return ----
    if use_jackknife:
        return values, values_lower, values_upper, time_centers, freqs
    return values, time_centers, freqs





###################### SPECTROGRAM HANDLING ######################
def save_spectrograms(spectrograms: np.ndarray, time_centers: np.ndarray, frequencies: np.ndarray, modality: str,
                      save_dir: str | Path, identifier_suffix: str = ""):
    print(f"Saving {modality} spectrograms of shape {spectrograms.shape} alongside time-centers and frequencies to:\n\t{save_dir}")
    time_center_diffs = np.diff(time_centers)
    window_length_sec = np.nanmin(np.where(time_center_diffs > 0, time_center_diffs, np.nan))
    for obj, title in [
        (spectrograms,
         f"{modality} Spectrograms {spectrograms.shape[2]}ch {window_length_sec:.2f}sec_step{f' {identifier_suffix}' if identifier_suffix != "" else ""}"),
        (time_centers,
         f"{modality} Timecenters {len(time_centers)}windows{f' {identifier_suffix}' if identifier_suffix != "" else ""}"),
        (frequencies, f"{modality} Frequencies {len(frequencies)}freqs{f' {identifier_suffix}' if identifier_suffix != "" else ""}"),
    ]:
        save_path = save_dir / filemgmt.file_title(title, ".npy")
        np.save(save_path, obj)



def fetch_stored_spectrograms(dir: Path | str,
                              modality: str,
                              file_identifier: str | list[str] | None = None,
                              expected_n_channels: int | None = None,
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    If no file_identifier is provided, just fetches the most recent fitting .npy file. Expects stored file titles
    to be 'PSD Spectrograms' for spectrograms, 'PSD Timecenters' for timecenters and 'PSD Frequencies' for frequencies.

    Parameters
    ----------
    file_identifier : str, list[str], or None
        Additional keyword(s) that must appear in the filename.  A list
        adds multiple independent substring requirements (useful when the
        identifiers are non-contiguous in the filename, e.g. muscle group
        and channel subset).
    expected_n_channels : int or None
        If provided, asserts that the channel axis (axis 2) of the loaded
        spectrogram has exactly this many channels.  Protects against
        silently loading a file with a different channel configuration
        (e.g. 11-channel CMC subset vs. 64-channel full CMC).

    Returns tuple with three np.ndarrays:
    1) spectrograms (n_times, n_freqs, n_channels)
    2) timecenters (n_times)
    3) frequencies (n_freqs)
    """
    ids = ([file_identifier] if isinstance(file_identifier, str)
           else file_identifier if file_identifier is not None
           else [])

    spec_kws = [f"{modality}", "Spectrograms"] + ids
    spectrograms = np.load(filemgmt.most_recent_file(dir, ".npy", spec_kws))

    if expected_n_channels is not None and spectrograms.ndim >= 3:
        actual = spectrograms.shape[2]
        if actual != expected_n_channels:
            raise ValueError(
                f"fetch_stored_spectrograms: expected {expected_n_channels} channels "
                f"on axis 2 but loaded spectrogram has {actual} "
                f"(modality={modality!r}, file_identifier={file_identifier!r}). "
                f"Check that the correct file is being loaded."
            )

    times_kws = [f"{modality}", "Timecenters"] + ids
    timecenters = np.load(filemgmt.most_recent_file(dir, ".npy", times_kws))

    freq_kws = [f"{modality}", "Frequencies"] + ids
    frequencies = np.load(filemgmt.most_recent_file(dir, ".npy", freq_kws))

    return spectrograms, timecenters, frequencies


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


def max_cmc_spectrograms_over_channels(
        cmc_array: np.ndarray,
        cmc_array_lower_ci: np.ndarray | None = None,
        cmc_array_upper_ci: np.ndarray | None = None,
        channel_ax: int = 3,
        verbose: bool = True,):
    """ Jointly max CMC with confidence interval to take coherent EMG channels. """

    if verbose: print("Maxing CMC values over EMG channels (aligned)...")

    # Step 1: Find which EMG channel has the max coherence
    max_emg_idx = np.argmax(cmc_array, axis=channel_ax)  # Shape: (time, freq, eeg)

    # Step 2: Use those SAME indices for all three arrays
    cmc_maxed = np.take_along_axis(
        cmc_array,
        max_emg_idx[..., np.newaxis],
        axis=channel_ax
    ).squeeze(axis=channel_ax)  # (time, freq, eeg)

    if cmc_array_lower_ci is None or cmc_array_upper_ci is None:
        if verbose: print(f"  Shapes: {cmc_maxed.shape}")
        return cmc_maxed

    cmc_maxed_lower = np.take_along_axis(
        cmc_array_lower_ci,
        max_emg_idx[..., np.newaxis],
        axis=channel_ax
    ).squeeze(axis=channel_ax)

    cmc_maxed_upper = np.take_along_axis(
        cmc_array_upper_ci,
        max_emg_idx[..., np.newaxis],
        axis=channel_ax
    ).squeeze(axis=channel_ax)

    if verbose: print(f"  Shapes: {cmc_maxed.shape}")
    # Now all three come from the SAME EMG channel!

    return cmc_maxed, cmc_maxed_lower, cmc_maxed_upper


def aggregate_spectrogram_over_frequency_band(
        spectrograms: np.ndarray,
        freqs: np.ndarray,
        behaviour: Literal['max', 'mean'] = 'mean',
        frequency_bands: dict | None = None,
        log_transform: bool = False,
        log_epsilon: float = 1e-10,
        frequency_axis: int = 1,
        pre_aggregate_axis: tuple[int, Literal['max', 'mean']] | None = None,
        lower_array: np.ndarray | None = None,
        upper_array: np.ndarray | None = None,
) -> dict[str, np.ndarray] | dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Aggregate spectrogram over defined frequency bands with optional confidence bounds.

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
    lower_array : np.ndarray | None, optional
        Lower confidence bounds with same shape as spectrograms. Default is None.
    upper_array : np.ndarray | None, optional
        Upper confidence bounds with same shape as spectrograms. Default is None.
        When behaviour='max', indices are determined from main spectrogram and applied coherently.

    Returns
    -------
    freq_aggregated_dict : dict
        If lower_array and upper_array are None:
            Dictionary with band labels as keys and np.ndarrays as values.
        Otherwise:
            Dictionary with band labels as keys and tuples (main, lower, upper) as values.
        Value array shape matches spectrogram input shape without frequency_axis (and eventually pre_aggregate_axis).

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

    # Validate lower/upper arrays match spectrograms shape
    if lower_array is not None and lower_array.shape != spectrograms.shape:
        raise ValueError(
            f"lower_array shape {lower_array.shape} must match spectrograms shape {spectrograms.shape}"
        )
    if upper_array is not None and upper_array.shape != spectrograms.shape:
        raise ValueError(
            f"upper_array shape {upper_array.shape} must match spectrograms shape {spectrograms.shape}"
        )

    # Validate bounds coherence
    has_bounds = (lower_array is not None) or (upper_array is not None)
    if (lower_array is None) != (upper_array is None):
        raise ValueError("lower_array and upper_array must both be provided or both be None")

    n_frequencies = spectrograms.shape[frequency_axis]

    if len(freqs) != n_frequencies:
        raise ValueError(
            f"freqs length ({len(freqs)}) must match spectrograms frequency axis ({n_frequencies})"
        )

    if not frequency_bands:
        raise ValueError("frequency_bands dict cannot be empty")

    # Initialize output dictionary
    freq_aggregated_dict = {}

    # Pre-aggregation on main array and bounds
    if pre_aggregate_axis is not None:
        if pre_aggregate_axis[1] == 'max':
            spectrograms = np.max(spectrograms, axis=pre_aggregate_axis[0], keepdims=True)
            if lower_array is not None:
                lower_array = np.max(lower_array, axis=pre_aggregate_axis[0], keepdims=True)
                upper_array = np.max(upper_array, axis=pre_aggregate_axis[0], keepdims=True)
        elif pre_aggregate_axis[1] == 'mean':
            spectrograms = np.mean(spectrograms, axis=pre_aggregate_axis[0], keepdims=True)
            if lower_array is not None:
                lower_array = np.mean(lower_array, axis=pre_aggregate_axis[0], keepdims=True)
                upper_array = np.mean(upper_array, axis=pre_aggregate_axis[0], keepdims=True)
        else:
            raise ValueError(f"Unknown behavior for pre_aggregate_axis '{pre_aggregate_axis[1]}'")

    # Process each frequency band
    for band_label, (min_freq, max_freq) in frequency_bands.items():
        # Validate frequency range
        if min_freq < freqs.min() or max_freq > freqs.max():
            raise ValueError(
                f"Band '{band_label}' range ({min_freq}, {max_freq}) exceeds available "
                f"frequencies ({freqs.min():.2f}, {freqs.max():.2f})"
            )

        # Create frequency mask for this band
        frequency_mask = (freqs >= min_freq) & (freqs < max_freq)

        if not frequency_mask.any():
            print(f"No frequencies found for band '{band_label}' in range ({min_freq}, {max_freq})")

        # Extract spectrogram subset for this band
        spectrogram_subset = np.take(spectrograms, frequency_mask, axis=frequency_axis)

        # Apply log transform if requested (only to main spectrogram)
        if log_transform:
            spectrogram_subset = np.log10(spectrogram_subset + log_epsilon)

        # Aggregate across frequencies based on behaviour
        if behaviour == 'max':
            # Find max indices along frequency axis
            max_indices = np.argmax(spectrogram_subset, axis=frequency_axis, keepdims=True)

            # Apply max pooling to main array
            condensed = np.take_along_axis(
                spectrogram_subset,
                max_indices,
                axis=frequency_axis
            )

            # Apply same indices coherently to lower/upper bounds if provided
            if has_bounds:
                lower_subset = np.take(lower_array, frequency_mask, axis=frequency_axis)
                upper_subset = np.take(upper_array, frequency_mask, axis=frequency_axis)

                condensed_lower = np.take_along_axis(
                    lower_subset,
                    max_indices,
                    axis=frequency_axis
                )
                condensed_upper = np.take_along_axis(
                    upper_subset,
                    max_indices,
                    axis=frequency_axis
                )

        elif behaviour == 'mean':
            # Aggregate independently for mean (coherence doesn't matter)
            condensed = np.mean(spectrogram_subset, axis=frequency_axis, keepdims=True)

            if has_bounds:
                lower_subset = np.take(lower_array, frequency_mask, axis=frequency_axis)
                upper_subset = np.take(upper_array, frequency_mask, axis=frequency_axis)

                condensed_lower = np.mean(lower_subset, axis=frequency_axis, keepdims=True)
                condensed_upper = np.mean(upper_subset, axis=frequency_axis, keepdims=True)

        else:
            raise ValueError(f"Unknown behaviour '{behaviour}'")

        # Squeeze axes to remove keepdims
        if pre_aggregate_axis is not None:
            # Remove both axes
            condensed = np.squeeze(condensed, axis=(frequency_axis, pre_aggregate_axis[0]))
            if has_bounds:
                condensed_lower = np.squeeze(condensed_lower, axis=(frequency_axis, pre_aggregate_axis[0]))
                condensed_upper = np.squeeze(condensed_upper, axis=(frequency_axis, pre_aggregate_axis[0]))
        else:
            # Remove only frequency axis
            condensed = np.squeeze(condensed, axis=frequency_axis)
            if has_bounds:
                condensed_lower = np.squeeze(condensed_lower, axis=frequency_axis)
                condensed_upper = np.squeeze(condensed_upper, axis=frequency_axis)

        # Store in output dictionary
        if has_bounds:
            freq_aggregated_dict[band_label] = (condensed, condensed_lower, condensed_upper)
        else:
            freq_aggregated_dict[band_label] = condensed

    return freq_aggregated_dict


def aggregate_psd_spectrogram(
        psd_spectrograms: np.ndarray,
        psd_freqs: np.ndarray = None,
        normalize_mvc: bool = False,
        is_log_scaled: bool = False,
        freq_slice: tuple[float, float] | str = None,
        channel_indices: list[int] = None,
        aggregation_ops: list[tuple[str, int]] = None,
) -> np.ndarray:
    """
    Aggregate PSD spectrograms through multiple stages: normalization, slicing, and axis reduction.

    Processing order:
    1. MVC normalization (if requested)
    2. Frequency slicing (if freq_slice provided)
    3. Channel slicing (if channel_indices provided)
    4. Sequential aggregation operations in specified order

    Parameters
    ----------
    psd_spectrograms : np.ndarray
        PSD data with shape (n_times, n_frequencies, n_channels).
    psd_freqs : np.ndarray, optional
        Frequency values corresponding to the frequency axis. Required if freq_slice is used.
    normalize_mvc : bool, default=False
        Whether to apply MVC (Maximum Voluntary Contraction) normalization.
        Computes max over time and frequency per channel, then normalizes to percentage.
    is_log_scaled : bool, default=False
        Whether the data is already log-scaled. If True, skips MVC normalization.
    freq_slice : tuple[float, float] | str, optional
        Frequency range to slice. Can be:
        - Tuple (low, high): Custom frequency range in Hz
        - String: Predefined band name ('slow', 'fast', 'delta', 'theta', 'alpha', 'beta', 'gamma')
        Requires psd_freqs to be provided.
    channel_indices : list[int], optional
        List of channel indices to select. If None, uses all channels.
    aggregation_ops : list[tuple[str, int]], optional
        List of (operator, axis) tuples to apply sequentially.
        Operator can be 'mean' or 'max'.
        Axis refers to the current array shape after slicing.
        Example: [('mean', 1), ('max', 2)] means average axis 1 first, then max over axis 2.

    Returns
    -------
    np.ndarray
        Aggregated PSD array with reduced dimensions based on specified operations.

    Examples
    --------
    # EMG: Slice frequencies, mean over frequencies, then max over channels
    result = aggregate_psd_spectrogram(
        psd_spectrograms, psd_freqs,
        normalize_mvc=True, is_log_scaled=False,
        freq_slice='slow',
        aggregation_ops=[('mean', 1), ('max', 2)],  # mean freq axis, max channel axis
    )  # Output shape: (n_times,)

    # EEG: Select channels, mean over channels first, then frequencies
    result = aggregate_psd_spectrogram(
        psd_spectrograms, psd_freqs,
        channel_indices=[0, 1, 2, 5],
        freq_slice='alpha',
        aggregation_ops=[('mean', 2), ('mean', 1)],  # mean channels, then mean frequencies
    )  # Output shape: (n_times,)

    # Complex example: max over time, mean over channels, then max over frequencies
    result = aggregate_psd_spectrogram(
        psd_spectrograms, psd_freqs,
        aggregation_ops=[('max', 0), ('mean', 2), ('max', 1)],
    )  # Output shape: scalar
    """
    # Predefined frequency bands in Hz
    FREQUENCY_BANDS = {
        'all': (0, 250),
        'slow': (0, 40),
        'fast': (60, 250),
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (13, 30),
        'gamma': (30, 100),
    }

    # Create working copy
    result = psd_spectrograms.copy()

    # Stage 1: MVC Normalization
    if normalize_mvc and not is_log_scaled:
        # Maximum over time (axis 0) and frequencies (axis 1) per channel
        mvc = np.max(np.max(result, axis=0, keepdims=True), axis=1, keepdims=True)
        result = result / mvc * 100  # Convert to percentage

    # Stage 2: Frequency Slicing
    if freq_slice is not None:
        if psd_freqs is None:
            raise ValueError("psd_freqs must be provided when using freq_slice")

        # Convert string band name to tuple if needed
        if isinstance(freq_slice, str):
            if freq_slice not in FREQUENCY_BANDS:
                available_bands = ', '.join(FREQUENCY_BANDS.keys())
                raise ValueError(
                    f"Unknown frequency band '{freq_slice}'. "
                    f"Available bands: {available_bands}"
                )
            low_freq, high_freq = FREQUENCY_BANDS[freq_slice]
        else:
            low_freq, high_freq = freq_slice

        freq_mask = (psd_freqs >= low_freq) & (psd_freqs <= high_freq)
        result = result[:, freq_mask, :]

    # Stage 3: Channel Slicing
    if channel_indices is not None:
        result = result[:, :, channel_indices]

    # Stage 4: Sequential Aggregation Operations
    if aggregation_ops is not None:
        for operator, axis in aggregation_ops:
            if operator == 'mean':
                result = np.nanmean(result, axis=axis)
            elif operator == 'max':
                result = np.nanmax(result, axis=axis)
            else:
                raise ValueError(
                    f"Unknown operator '{operator}'. Supported operators: 'mean', 'max'"
                )

    return result


###################### OTHER SERIAL FEATURES ######################
def compute_heart_rate_and_variability(ecg_series: pd.Series, heart_beat_threshold_quantile: float = 0.8,
                       rolling_window: str = "15s",
                       refractory_period: str = "300ms",
                       output_smoothing_window_sec: float = 2.5,
                       min_bpm: float = 30.0,
                       max_bpm: float = 200.0,
                       max_hrv_seconds: float = 0.3,
                       verbose: bool = True):
    """
    Compute heart rate (BPM) and heart rate variability (HRV) from ECG signal.

    Uses adaptive rolling threshold with refractory period for beat detection.
    Applies physiological filtering to remove outlier intervals before computing BPM and HRV.
    HRV is computed as RMSSD (Root Mean Square of Successive Differences) from filtered intervals.

    Parameters
    ----------
    ecg_series : pd.Series
        ECG signal with DatetimeIndex.
    heart_beat_threshold_quantile : float, default 0.8
        Quantile for adaptive rolling threshold (0–1).
    rolling_window : str, default "15s"
        Window size for rolling threshold.
    refractory_period : str, default "300ms"
        Minimum time between detected beats.
    output_smoothing_window_sec : float, default 2.5
        Rolling mean window for output smoothing (seconds).
    min_bpm : float, default 30.0
        Minimum physiologically plausible heart rate (bpm).
    max_bpm : float, default 200.0
        Maximum physiologically plausible heart rate (bpm).
    max_hrv_seconds : float, default 0.3
        Maximum allowed successive difference for HRV (seconds).
    verbose : bool, default True
        Print statistics.

    Returns
    -------
    tuple of pd.Series
        (bpm_series, hrv_series) - both aligned to original ECG index and smoothed.
        Returns (None, None) if insufficient beats detected.
    """
    assert isinstance(ecg_series.index, pd.DatetimeIndex), "ecg_series index is not a datetime index!"

    # min-max scale:
    scaled_ecg = (ecg_series - ecg_series.min()) / (ecg_series.max() - ecg_series.min())

    # compute heart beat mask based on rolling quantile (adaptive threshold):
    rolling_threshold = scaled_ecg.rolling(window=rolling_window, min_periods=1).quantile(
        heart_beat_threshold_quantile)
    heart_beat_mask = scaled_ecg > rolling_threshold

    # compute onsets:
    heart_beat_onsets = ((heart_beat_mask != heart_beat_mask.shift()) &
                         heart_beat_mask)
    onset_timestamps = ecg_series.loc[heart_beat_onsets].index.tolist()

    # early exit: need at least 2 beats for any interval computation:
    if len(onset_timestamps) < 2:
        if verbose:
            print(f"ERROR: Only {len(onset_timestamps)} beat(s) detected. Need at least 2 for BPM calculation.")
        return None, None

    # apply refractory period: filter out onsets that are too close together:
    refractory_td = pd.Timedelta(refractory_period)
    filtered_onsets = []
    for t in onset_timestamps:
        if not filtered_onsets or (t - filtered_onsets[-1]) >= refractory_td:
            filtered_onsets.append(t)
    onset_timestamps = filtered_onsets

    # check again after refractory filtering:
    if len(onset_timestamps) < 2:
        if verbose:
            print(f"ERROR: Only {len(onset_timestamps)} beat(s) after refractory filtering. Need at least 2.")
        return None, None

    # compute onset differences (inter-beat intervals):
    beat_differences = np.diff(onset_timestamps)
    beat_differences_seconds = np.array([td.total_seconds() for td in beat_differences])

    # guard against zero intervals (shouldn't happen with refractory, but be safe):
    if np.any(beat_differences_seconds == 0):
        if verbose:
            print("WARNING: Zero-length intervals detected (timestamp precision issue). Removing...")
        non_zero_mask = beat_differences_seconds > 0
        beat_differences_seconds = beat_differences_seconds[non_zero_mask]
        # adjust onset_timestamps to match
        onset_timestamps = [onset_timestamps[0]] + [onset_timestamps[i + 1] for i in range(len(beat_differences)) if
                                                    non_zero_mask[i]]

    bpm_array = (1 / beat_differences_seconds) * 60

    # apply physiological outlier filtering:
    min_interval = 60.0 / max_bpm
    max_interval = 60.0 / min_bpm
    valid_interval_mask = (beat_differences_seconds >= min_interval) & (beat_differences_seconds <= max_interval)

    # check if any valid intervals remain:
    if np.sum(valid_interval_mask) == 0:
        if verbose:
            print(
                f"ERROR: All {len(beat_differences_seconds)} intervals filtered as outliers (outside [{min_bpm:.0f}-{max_bpm:.0f}] bpm).")
            print(f"Detected BPM range: [{bpm_array.min():.1f}-{bpm_array.max():.1f}]")
        return None, None

    # filter intervals and BPM:
    beat_differences_seconds_filtered = beat_differences_seconds[valid_interval_mask]
    bpm_array_filtered = bpm_array[valid_interval_mask]

    # get corresponding onset timestamps for valid intervals:
    valid_onset_pairs = [(onset_timestamps[i], onset_timestamps[i + 1])
                         for i in range(len(onset_timestamps) - 1)
                         if valid_interval_mask[i]]

    # compute HRV from filtered intervals:
    successive_diffs = np.diff(beat_differences_seconds_filtered)
    hrv_array_raw = np.abs(successive_diffs)

    # early exit for HRV if insufficient data:
    if len(hrv_array_raw) == 0:
        if verbose:
            print("WARNING: Insufficient intervals for HRV calculation (need at least 3 beats).")
        hrv_array_filtered = np.array([])
        rmssd = np.nan
    else:
        # filter HRV outliers:
        valid_hrv_mask = hrv_array_raw <= max_hrv_seconds
        hrv_array_filtered = hrv_array_raw[valid_hrv_mask]

        # compute RMSSD from filtered HRV values:
        if len(hrv_array_filtered) > 0:
            squared_diffs_filtered = hrv_array_filtered ** 2
            mean_squared_diffs = np.mean(squared_diffs_filtered)
            rmssd = np.sqrt(mean_squared_diffs)
        else:
            rmssd = np.nan

    if verbose:
        n_intervals_total = len(beat_differences_seconds)
        n_intervals_removed = np.sum(~valid_interval_mask)
        n_hrv_removed = np.sum(~valid_hrv_mask) if len(hrv_array_raw) > 0 else 0

        print(
            f"Heartbeat detection: rolling quantile threshold {heart_beat_threshold_quantile} (window: {rolling_window}, refractory: {refractory_period})")
        print(f"Detected {len(onset_timestamps)} beats, {n_intervals_total} intervals")
        print(
            f"Filtered {n_intervals_removed} outlier intervals outside range [{min_bpm:.0f}-{max_bpm:.0f}] bpm (i.e., [{min_interval:.3f}-{max_interval:.3f}] sec)")
        print(f"Kept {len(beat_differences_seconds_filtered)} valid intervals\n")

        print(f"Interval statistics (filtered):")
        print(f"\tmin.\t{np.min(beat_differences_seconds_filtered):.2f} sec")
        print(f"\t 5% \t{np.quantile(beat_differences_seconds_filtered, .05):.2f} sec")
        print(f"\t25% \t{np.quantile(beat_differences_seconds_filtered, .25):.2f} sec")
        print(f"\tavg.\t{np.average(beat_differences_seconds_filtered):.2f} sec")
        print(f"\t75% \t{np.quantile(beat_differences_seconds_filtered, .75):.2f} sec")
        print(f"\t95% \t{np.quantile(beat_differences_seconds_filtered, .95):.2f} sec")
        print(f"\tmax.\t{np.max(beat_differences_seconds_filtered):.2f} sec")

        print("\n+ BPM statistics (filtered):")
        print(f"\tmin.\t{np.min(bpm_array_filtered):.2f} bpm")
        print(f"\t 5% \t{np.quantile(bpm_array_filtered, .05):.2f} bpm")
        print(f"\t25% \t{np.quantile(bpm_array_filtered, .25):.2f} bpm")
        print(f"\tavg.\t{np.average(bpm_array_filtered):.2f} bpm")
        print(f"\t75% \t{np.quantile(bpm_array_filtered, .75):.2f} bpm")
        print(f"\t95% \t{np.quantile(bpm_array_filtered, .95):.2f} bpm")
        print(f"\tmax.\t{np.max(bpm_array_filtered):.2f} bpm")

        if len(hrv_array_filtered) > 0:
            print(f"\n+ HRV statistics (successive differences, filtered <{max_hrv_seconds}s):")
            print(f"\tRemoved {n_hrv_removed} outlier HRV values from {len(hrv_array_raw)} total")
            print(f"\tmin.\t{np.min(hrv_array_filtered):.4f} sec ({np.min(hrv_array_filtered) * 1000:.2f} ms)")
            print(
                f"\t 5% \t{np.quantile(hrv_array_filtered, .05):.4f} sec ({np.quantile(hrv_array_filtered, .05) * 1000:.2f} ms)")
            print(
                f"\t25% \t{np.quantile(hrv_array_filtered, .25):.4f} sec ({np.quantile(hrv_array_filtered, .25) * 1000:.2f} ms)")
            print(f"\tavg.\t{np.average(hrv_array_filtered):.4f} sec ({np.average(hrv_array_filtered) * 1000:.2f} ms)")
            print(
                f"\t75% \t{np.quantile(hrv_array_filtered, .75):.4f} sec ({np.quantile(hrv_array_filtered, .75) * 1000:.2f} ms)")
            print(
                f"\t95% \t{np.quantile(hrv_array_filtered, .95):.4f} sec ({np.quantile(hrv_array_filtered, .95) * 1000:.2f} ms)")
            print(f"\tmax.\t{np.max(hrv_array_filtered):.4f} sec ({np.max(hrv_array_filtered) * 1000:.2f} ms)")
            print(f"\tRMSSD\t{rmssd:.4f} sec ({rmssd * 1000:.2f} ms)")
        else:
            print("\n+ HRV: insufficient valid data after filtering")

    # construct BPM series: use ending timestamp of each valid interval:
    bpm_series = pd.Series(
        index=[pair[1] for pair in valid_onset_pairs],
        data=bpm_array_filtered
    )

    # construct HRV series: use ending timestamp of each successive interval pair:
    if len(valid_onset_pairs) >= 2 and len(hrv_array_raw) > 0:
        hrv_timestamps = [valid_onset_pairs[i + 1][1] for i in range(len(hrv_array_raw))]
        hrv_series_raw = pd.Series(index=hrv_timestamps, data=hrv_array_raw)
    else:
        hrv_series_raw = pd.Series(dtype=float)

    # merge with original timestamps:
    merged_df = ecg_series.to_frame(name='ecg').join(
        bpm_series.to_frame(name='bpm'), how='left'
    ).join(
        hrv_series_raw.to_frame(name='hrv'), how='left'
    )

    # forward-fill heart rate and HRV between onset timestamps:
    bpm_full_series = merged_df['bpm'].ffill()
    hrv_full_series = merged_df['hrv'].ffill()

    # apply output smoothing with rolling mean:
    bpm_smoothed = bpm_full_series.rolling(window=f"{output_smoothing_window_sec}s", min_periods=1).mean()
    hrv_smoothed = hrv_full_series.rolling(window=f"{output_smoothing_window_sec}s", min_periods=1).mean()

    return bpm_smoothed, hrv_smoothed


def compute_task_wise_scaled_force(fsr_series: pd.Series, enriched_log_df: pd.DataFrame,
                                   min_samples: int = 10,
                                   min_percentile: float = .01,
                                   max_percentile: float = .99,
                                   verbose: bool = True) -> pd.Series:
    """
    Returns identically indexed series with task-wise min-max scaled force values.

    Applies robust min-max scaling (using 1st and 99th percentiles) separately
    to each trial period. Returns NaNs outside trial periods and for invalid trials.

    Parameters
    ----------
    fsr_series : pd.Series
        Force sensor readings with DatetimeIndex.
    enriched_log_df : pd.DataFrame
        Trial metadata containing start/end times.
    min_samples : int, default 10
        Minimum number of valid samples required per trial for scaling.
        Trials with fewer samples are skipped with a warning.
    verbose : bool, default False
        Print warnings for skipped trials.

    Returns
    -------
    pd.Series
        Task-wise scaled force with same index as input. Values in [0, 1] during
        valid trials, NaN elsewhere.
    """
    assert isinstance(fsr_series.index, pd.DatetimeIndex), "fsr_series.index is not a datetime index!"
    fsr_series.index = data_analysis.make_timezone_aware(fsr_series.index,)

    # derive trial durations:
    trial_start_ends = data_integration.get_all_task_start_ends(enriched_log_df, output_type='list')

    # initialize output series with NaNs:
    adjusted_force_series = pd.Series(
        index=fsr_series.index,
        data=np.nan,  # explicit NaN instead of None
        dtype=float,
        name='Task-wise Scaled Force'
    )

    skipped_trials = 0

    for trial_idx, (start, end) in enumerate(trial_start_ends):
        # extract trial data:
        fsr_subset = fsr_series.loc[start:end]

        # check 1: empty trial (no data points in time range):
        if len(fsr_subset) == 0:
            if verbose:
                print(f"Trial {trial_idx} [{start} to {end}]: No data points, skipping")
            skipped_trials += 1
            continue

        # convert to numpy and remove NaNs:
        fsr_values = fsr_subset.dropna().to_numpy()

        # check 2: insufficient valid samples after dropping NaNs:
        if len(fsr_values) < min_samples:
            if verbose:
                print(
                    f"Trial {trial_idx} [{start} to {end}]: Only {len(fsr_values)} valid samples (< {min_samples}), skipping")
            skipped_trials += 1
            continue

        # compute robust scale bounds using quantiles:
        scale_min = np.quantile(fsr_values, q=min_percentile)
        scale_max = np.quantile(fsr_values, q=max_percentile)

        # check 3: constant or near-constant values (no variance):
        scale_range = scale_max - scale_min
        if scale_range < 1e-6:  # essentially zero (adjust threshold as needed)
            if verbose:
                print(
                    f"Trial {trial_idx} [{start} to {end}]: Constant values (range={scale_range:.2e}), setting to 0.5")
            # constant force → arbitrary scaling to mid-range:
            adjusted_force_series.loc[start:end] = 0.5
            continue

        # perform min-max scaling on original subset (including NaNs):
        scaled_fsr_subset = (fsr_subset - scale_min) / scale_range

        # clip to [0, 1] to handle edge cases from quantile estimation:
        scaled_fsr_subset = scaled_fsr_subset.clip(lower=0.0, upper=1.0)

        # assign back using .loc with same index (preserves alignment):
        adjusted_force_series.loc[scaled_fsr_subset.index] = scaled_fsr_subset.values

    if verbose and skipped_trials > 0:
        print(f"\nSkipped {skipped_trials}/{len(trial_start_ends)} trials due to insufficient data")

    return adjusted_force_series


###################### STATISTICAL FEATURES ######################
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











if __name__ == '__main__':
    from pathlib import Path
    from matplotlib import pyplot as plt
    import matplotlib as mpl

    mpl.use('MacOSX')
    """
    # global vars:
    ROOT = Path().resolve().parent.parent
    QTC_DATA = ROOT / "data" / "qtc_measurements" / "2025_06"
    subject_data_dir = QTC_DATA / "sub-10"

    # area to scrutinize:
    use_ch_subset = False
    ch_subset = EEG_CHANNELS_BY_AREA['Fronto-Central']
    ch_subset_inds = [EEG_CHANNEL_IND_DICT[ch]-1 for ch in ch_subset]  # -1 to convert to computer-indices (0 = start)

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


    # todo: ponder visualization of subset of EEG channels (e.g. np.zeros for the others)
    """
    # animation:
    #print(psd_sampling_freq)
    visualizations.initialise_electrode_heatmap(
        np.array([[i]*64 for i in range(1000)]).T,#freq_averaged_psd_dict['beta'].T,  # requires shape (n_timesteps, n_channels)
        positions=visualizations.EEG_POSITIONS, add_head_shape=True,
        #sampling_rate=10, animation_fps=10,
        value_label="Power [V^2/Hz]",# if not do_log_transform else "Power [V^2/Hz] [log10]",
        plot_title="EEG PSD (Beta-Band)"
    )

