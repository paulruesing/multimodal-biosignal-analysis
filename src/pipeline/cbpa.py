from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.stats import (
    combine_adjacency,
    permutation_cluster_1samp_test,
    spatio_temporal_cluster_1samp_test,
)
from scipy.stats import t as t_dist

from src.pipeline.signal_features import fetch_stored_spectrograms, aggregate_psd_spectrogram
import src.pipeline.data_integration as data_integration
import src.pipeline.data_analysis as data_analysis
from src.pipeline.channel_layout import EEG_CHANNEL_IND_DICT
import src.pipeline.visualizations as visualizations
import src.utils.file_management as filemgmt




# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS — mirror your existing channel definitions
# ══════════════════════════════════════════════════════════════════════════════

EEG_CHANNELS: list[str] = list(EEG_CHANNEL_IND_DICT.keys())   # full 64-ch set
EEG_SFREQ: float = 2048  # Hz — adjust to your actual EEG sampling rate

# CMC subset (11 left-hemisphere motor channels; will be mirrored for left-handers)
# todo: always needs to match subject_feature_extraction_workflow.py!
CMC_EEG_CHANNEL_SUBSET: list[str] = [
    "C5", "C3", "C1",
    "FC5", "FC3", "FC1", "F3",
    "CP5", "CP3", "CP1", "P3",
]


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CBPAConfig:
    """
    Fully specifies one CBPA run.

    Feature selection
    -----------------
    modality              : 'PSD' or 'CMC'
    modality_file_id      : file_identifier for fetch_stored_spectrograms
                            PSD → modality string (e.g. 'eeg')
                            CMC → muscle string  (e.g. 'Flexor', 'Extensor')
    freq_band             : band passed to aggregate_psd_spectrogram as
                            freq_slice (e.g. 'alpha', 'beta', 'gamma', 'theta')
    channels              : list of EEG channel names to include in CBPA.
                            None = all channels in the spectrogram.

    Contrast
    --------
    condition_column      : column name in the per-subject log_df that defines
                            the condition (e.g. 'Category or Silence',
                            'Perceived Category')
    condition_A           : value for condition A  (e.g. 'Happy')
    condition_B           : value for condition B  (e.g. 'Silence', 'Classic')

    Segmentation
    ------------
    n_within_trial_segs   : how many segments to split each ~45-sec trial into.
                            The *within-segment* PSD/CMC windows are used as the
                            CBPA time axis. All segments of the same condition
                            are averaged per subject before differencing.

    CBPA parameters
    ---------------
    alpha_cluster_forming : uncorrected threshold to form initial clusters
                            (default 0.05, two-tailed → t_crit from t-dist)
    n_permutations        : ≥1000 for exploration; ≥5000 for publication
    tail                  : -1 | 0 | 1  (0 = two-tailed, recommended default)
    use_spatio_temporal   : True  → spatio_temporal_cluster_1samp_test
                                    (needs standard_1020 sensor positions)
                            False → permutation_cluster_1samp_test on flattened
                                    space (no spatial adjacency; use for CMC
                                    or when channel count is very small)
    n_jobs                : parallelism for permutations (-1 = all cores)
    seed                  : random seed for reproducibility

    I/O
    ---
    data_root             : project root (parent of data/ and output/)
    psd_time_window_sec   : window size used during PSD computation (seconds)
    cmc_time_window_sec   : window size used during CMC computation (seconds)
    psd_is_log_scaled     : whether PSD was log-scaled during feature extraction
    output_dir            : where to save .npz results and plots
    hypothesis_label      : used in filenames and plot titles
    save_plots            : save PNG output
    show_plots            : display interactively
    """

    # Feature
    modality: Literal["PSD", "CMC"] = "PSD"
    modality_file_id: str = "eeg"
    freq_band: str = "alpha"
    channels: Optional[list[str]] = None

    # Contrast
    condition_column: str = "Category or Silence"
    condition_A: str = "Happy"
    condition_B: str = "Silence"

    # Segmentation
    n_within_trial_segs: int = 1

    # Subject Subset
    exclude_subjects: list[int] = None

    # CBPA
    alpha_cluster_forming: float = 0.05
    n_permutations: int = 1000
    tail: Literal[-1, 0, 1] = 0
    use_spatio_temporal: bool = True
    n_jobs: int = -1
    seed: int = 42

    # I/O
    data_root: Path = field(default_factory=lambda: Path().resolve().parent)
    psd_time_window_sec: float = 0.25
    cmc_time_window_sec: float = 2.0
    overlap_ratio: float = .5
    psd_is_log_scaled: bool = True
    output_dir: Path = field(
        default_factory=lambda: Path().resolve().parent / "output" / "statistics_post_hoc_testing"
    )
    hypothesis_label: str = "cbpa_run"
    save_plots: bool = True
    show_plots: bool = False

    # Phase normalisation (CMC only)
    use_phase_normalization: bool = False
    """If True, the time axis becomes force-cycle phase (0–360°) instead of
    clock time. Requires CMC modality and known Task Frequency per trial."""
    n_phase_bins: int = 36
    """Number of phase bins (default 36 → 10° resolution)."""
    min_samples_per_cycle: int = 2
    """Trials where a force cycle contains fewer CMC windows than this are
    skipped — phase interpolation is unreliable below 2 samples/cycle."""
    min_cycles_per_condition: int = 3
    """Minimum number of valid cycles a subject must contribute per condition
    to be included in the contrast."""

    # Optional target-sine subplot (below each main panel)
    show_target_sine: bool | None = None
    target_sine_min_pct_mvc: float = 7.5
    target_sine_max_pct_mvc: float = 22.5
    target_sine_frequency_hz: float = 0.1
    include_dynamometer_force: bool = True
    """If True, overlay the averaged per-cycle dynamometer force on target sine panels."""

    # Phase normalisation offsets
    phase_start_offset_sec: float | None = None
    """Seconds to skip at the start of each trial before counting cycles for
    CMC / EMG-PSD phase normalisation (``_band_power_per_phase``).
    ``None`` (default) uses the adaptive ``1 / task_freq`` heuristic, which
    skips exactly one cycle — cycle 0 is always incomplete because trial
    boundaries do not coincide with force-sine zero crossings.
    Set to ``TRIAL_ACCURACY_START_OFFSET_SEC`` (5.5) for CMC-accuracy plots
    so that CMC and accuracy exclude the same warm-up window.
    Set to ``0.0`` only when cycle-0 inclusion is intentionally desired."""

    force_phase_start_offset_sec: float | None = None
    """Seconds to skip at the start of each trial before counting cycles for
    the dynamometer-force phase normalisation overlay.
    ``None`` (default) uses the adaptive ``1 / task_freq`` heuristic, which
    skips exactly one cycle and is cycle-aligned by construction — appropriate
    for the task-onset force ramp.  Set an explicit float to override."""

    # Optional figure suptitle (useful to disable for publication figures)
    include_suptitle: bool = False

    # Time-axis reconstruction mode for stored spectrogram windows.
    use_stretched_window_timestamps: bool = False
    """If True, reconstruct absolute window-center timestamps by uniformly
    stretching ``n_windows`` centers between ``qtc_start + half_window`` and
    ``qtc_end - half_window``. This avoids cumulative drift when saved
    time-centers were derived with a slightly mismatched sampling frequency.
    If False, use stored second offsets directly (``qtc_start + sec``)."""


# ══════════════════════════════════════════════════════════════════════════════
#  MNE INFO & ADJACENCY
# ══════════════════════════════════════════════════════════════════════════════

def _build_mne_info(ch_names: list[str]) -> mne.Info:
    """
    Construct an mne.Info object for a subset of standard-1020 EEG channels.
    Mirrors the mne_raw_data() method in your codebase: create_info +
    set_montage('standard_1020').
    """
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=EEG_SFREQ,
        ch_types="eeg",
    )
    # Attach standard 1020 electrode positions so find_ch_adjacency can
    # build the Delaunay-triangulation adjacency from 2-D sensor layout.
    montage = mne.channels.make_standard_montage("standard_1020")
    with info._unlock():
        pass  # unlock not needed; set_montage on a plain Info works directly
    # We need a temporary RawArray just to call set_montage (Info is not enough
    # for set_montage in some MNE versions):
    dummy_data = np.zeros((len(ch_names), 10))
    raw_tmp = mne.io.RawArray(dummy_data, info, verbose=False)
    raw_tmp.set_montage(montage, on_missing="warn", verbose=False)
    return raw_tmp.info


def _build_adjacency(info: mne.Info, n_times: int) -> object:
    """
    Build the combined spatio-temporal adjacency matrix.

    MNE's spatio_temporal_cluster_1samp_test accepts either:
      (a) a spatial-only (n_ch × n_ch) adjacency  → it replicates across time
      (b) a combined (n_times*n_ch × n_times*n_ch) adjacency from combine_adjacency

    We use approach (b) via mne.stats.combine_adjacency so that the adjacency
    is explicit and auditable.
    """
    spatial_adj, _ = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    # Combine spatial adjacency with temporal adjacency (linear chain over time):
    combined_adj = combine_adjacency(n_times, spatial_adj)
    print(
        f"  [adjacency] spatial: {spatial_adj.shape}, "
        f"combined (time×space): {combined_adj.shape}, "
        f"nnz edges: {combined_adj.nnz}"
    )
    return combined_adj


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING — mirrors original pipeline exactly
# ══════════════════════════════════════════════════════════════════════════════

def _get_task_freq_for_trial(
    log_df: pd.DataFrame,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
) -> float | None:
    """
    Extract the Task Frequency (Hz) for a single trial from log_df.

    Reads the modal non-NaN value of 'Task Frequency' within the trial span.
    Returns None if the column is absent or entirely NaN in the span.

    Parameters
    ----------
    log_df  : enriched log DataFrame with DatetimeIndex
    t_start : trial start timestamp (inclusive)
    t_end   : trial end timestamp (exclusive)

    Returns
    -------
    float Hz value, or None if unresolvable
    """
    # Slice to trial span
    mask = (log_df.index >= t_start) & (log_df.index < t_end)
    col = log_df.loc[mask, "Task Frequency"].dropna()

    if col.empty:
        return None

    # Task Frequency is constant within a trial; mode is a safe guard
    return float(col.mode().iloc[0])


def _load_subject_data(
    cfg: CBPAConfig, subject_ind: int
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DataFrame]:
    """
    Load spectrogram + log for one subject.

    Returns
    -------
    spectrogram  : (n_windows, n_freqs, n_channels)
    freqs        : (n_freqs,)
    timestamps   : DatetimeIndex of length n_windows — absolute UTC timestamps
                   for each spectrogram window centre
    log_df       : enriched log dataframe with timezone-aware index
    """
    DATA = cfg.data_root / "data"
    EXPERIMENT_DATA = DATA / "experiment_results"
    FEATURE_DATA = DATA / "precomputed_features"

    subject_feat_dir = FEATURE_DATA / f"subject_{subject_ind:02}"
    subject_exp_dir = EXPERIMENT_DATA / f"subject_{subject_ind:02}"

    log_df = data_integration.fetch_enriched_log_frame(subject_exp_dir, verbose=False)
    log_df.index = data_analysis.make_timezone_aware(log_df.index)

    qtc_start, qtc_end = data_integration.get_qtc_measurement_start_end(log_df, False)
    spectrogram, times, freqs = fetch_stored_spectrograms(
        subject_feat_dir,
        modality=cfg.modality,
        file_identifier=cfg.modality_file_id,
    )
    # spectrogram shape: (n_windows, n_freqs, n_channels)

    times_arr = np.asarray(times, dtype=np.float64)
    if cfg.use_stretched_window_timestamps:
        half_window = 0.5 * (
            cfg.cmc_time_window_sec if cfg.modality == "CMC" else cfg.psd_time_window_sec
        )
        timestamps = data_analysis.add_time_index(
            start_timestamp=qtc_start + pd.Timedelta(seconds=half_window),
            end_timestamp=qtc_end - pd.Timedelta(seconds=half_window),
            n_timesteps=len(times_arr),
        )
    else:
        # Direct reconstruction from stored per-window second offsets.
        # Non-finite centres (outside-task slots) become NaT and are filtered downstream.
        timestamps = pd.DatetimeIndex([
            qtc_start + pd.Timedelta(seconds=float(sec)) if np.isfinite(sec) else pd.NaT
            for sec in times_arr
        ])
    timestamps = data_analysis.make_timezone_aware(timestamps)

    return spectrogram, freqs, timestamps, log_df



def _get_trial_spans(log_df: pd.DataFrame) -> dict:
    """
    Return trial start/end timestamps directly from log_df.
    Uses the same get_all_task_start_ends call as the LME pipeline,
    which already applies default transient cut-off seconds.
    Returns an ordered dict: {trial_id: (start_timestamp, end_timestamp)}
    """
    return data_integration.get_all_task_start_ends(log_df, "dict")


def _common_time_grid_from_spans(
    cfg: CBPAConfig,
    trial_spans: dict[int, tuple],
        overlap_ratio = .5,
) -> np.ndarray:
    """
    Derive the within-trial relative time grid in seconds, based on
    the native spectrogram step size (tw/2 for 50% overlap).
    Uses the first trial's duration as the reference.
    """
    tw = cfg.psd_time_window_sec if cfg.modality == "PSD" else cfg.cmc_time_window_sec
    first_start, first_end = next(iter(trial_spans.values()))
    trial_dur_sec = (pd.Timestamp(first_end) - pd.Timestamp(first_start)).total_seconds()
    n_times = max(1, int(trial_dur_sec / (tw * overlap_ratio)))
    return np.arange(n_times) * (tw * overlap_ratio)


def _band_power_per_trial(
    cfg: CBPAConfig,
    band_power: np.ndarray,        # (n_windows, n_channels)
    timestamps: pd.DatetimeIndex,
    trial_spans: dict[int, tuple], # {trial_id: (start, end)}
    target_n_times: int | None,
) -> tuple[np.ndarray, list[int]]:
    """
    Slice band_power into per-trial arrays at native time resolution.

    Returns
    -------
    trial_data    : (n_trials, n_times, n_channels)
    trial_ids_out : list[int] — trial IDs in row order of trial_data
                    (only trials with at least one window are included)
    """
    tw = cfg.psd_time_window_sec if cfg.modality == "PSD" else cfg.cmc_time_window_sec

    trial_slices: list[np.ndarray] = []
    trial_ids_out: list[int] = []
    trial_lengths: list[int] = []

    for trial_id, (t_start, t_end) in trial_spans.items():
        mask = (timestamps >= t_start) & (timestamps < t_end)
        slc = band_power[mask]
        if slc.shape[0] == 0:
            warnings.warn(f"  [WARN] Trial {trial_id}: no spectrogram windows found in span. Skipping.")
            continue
        trial_slices.append(slc)
        trial_ids_out.append(trial_id)
        trial_lengths.append(slc.shape[0])

    if len(trial_slices) == 0:
        raise RuntimeError("No trial windows found — check timestamp alignment.")

    if target_n_times is None:
        target_n_times = int(pd.Series(trial_lengths).mode().iloc[0])

    n_ch = trial_slices[0].shape[-1]
    result = np.full((len(trial_slices), target_n_times, n_ch), np.nan)

    for i, slc in enumerate(trial_slices):
        n_w = slc.shape[0]
        if n_w == target_n_times:
            result[i] = slc
        else:
            src_x = np.linspace(0, 1, n_w)
            dst_x = np.linspace(0, 1, target_n_times)
            for ch in range(n_ch):
                result[i, :, ch] = np.interp(dst_x, src_x, slc[:, ch])

    return result, trial_ids_out



# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICS FRAME — trial-level condition labels
#  Always loaded from the pre-computed 1seg Combined Statistics CSV.
#  This frame is the single source of truth for all condition assignments.
# ══════════════════════════════════════════════════════════════════════════════

STATS_FRAME_SEG_SUFFIX: str = "1seg"   # always use 1seg for trial labels


def load_stats_frame(data_root: Path) -> pd.DataFrame:
    """
    Load the pre-computed Combined Statistics frame (1seg version).
    This is the authoritative source for trial-level condition labels.
    Raises FileNotFoundError with a clear message if not found — this is
    a hard requirement; CBPA cannot run without it.

    Expected columns (relevant subset):
        Subject ID, Trial ID, Category or Silence, Perceived Category,
        Music Listening, Segment ID, ...

    Returns
    -------
    df : pd.DataFrame, rows = (subject × trial), indexed by default int index.
    """
    feature_dir = data_root / "data" / "precomputed_features"

    try:
        csv_path = filemgmt.most_recent_file(
            feature_dir, ".csv",
            [f"Combined Statistics {STATS_FRAME_SEG_SUFFIX}"]
        )
    except (ValueError, FileNotFoundError):
        raise FileNotFoundError(
            f"\n[CBPA] Required statistics frame not found in:\n"
            f"  {feature_dir}\n"
            f"Expected a file matching 'Combined Statistics {STATS_FRAME_SEG_SUFFIX}' "
            f"with extension '.csv'.\n"
            f"Please run the main statistical_workflow.py pipeline first "
            f"(with n_within_trial_segments=1) to generate it."
        )

    df = pd.read_csv(csv_path)

    # Validate required columns exist:
    required_cols = {"Subject ID", "Trial ID", "Category or Silence",
                     "Perceived Category", "Music Listening"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"[CBPA] Statistics frame is missing required columns: {missing}\n"
            f"  Loaded from: {csv_path}"
        )

    print(f"  [stats frame] Loaded: {csv_path.name}  "
          f"({len(df)} rows, {df['Subject ID'].nunique()} subjects, "
          f"{df['Trial ID'].nunique()} unique trial IDs)")
    return df


def get_trial_condition_map(
    stats_df: pd.DataFrame,
    subject_id: int,
    condition_column: str,
) -> dict[int, str | None]:
    """
    For a given subject, return a mapping of Trial ID → condition label.

    Silence trials have NaN in 'Perceived Category' but 'Silence' in
    'Category or Silence' — the stats frame already handles this correctly.

    Parameters
    ----------
    stats_df         : full Combined Statistics frame (all subjects)
    subject_id       : integer subject index
    condition_column : e.g. 'Category or Silence' or 'Perceived Category'

    Returns
    -------
    dict mapping int(trial_id) → str label (or None if column value is NaN)
    """
    subj_df = stats_df[stats_df["Subject ID"] == subject_id]
    if subj_df.empty:
        raise ValueError(
            f"[CBPA] Subject {subject_id} not found in statistics frame. "
            f"Available subjects: {sorted(stats_df['Subject ID'].unique())}"
        )

    trial_map: dict[int, str | None] = {}
    for _, row in subj_df.iterrows():
        trial_id = int(row["Trial ID"])
        val = row.get(condition_column, None)
        trial_map[trial_id] = None if pd.isna(val) else str(val)

    return trial_map


def _mode_label_for_span(
    log_df: pd.DataFrame,
    column: str,
    t_start,
    t_end,
) -> str | None:
    """Return the modal non-null label for a column within a time span.

    Parameters
    ----------
    log_df : pd.DataFrame
        Log frame with a DatetimeIndex.
    column : str
        Column name containing categorical labels.
    t_start, t_end : Any
        Inclusive/exclusive time bounds used for slicing ``log_df``.

    Returns
    -------
    str | None
        Most frequent non-null value in ``column`` within ``[t_start, t_end)``,
        or ``None`` if the span contains no valid labels.
    """
    mask = (log_df.index >= t_start) & (log_df.index < t_end)
    col = log_df.loc[mask, column].dropna()
    return col.mode().iloc[0] if not col.empty else None


# ══════════════════════════════════════════════════════════════════════════════
#  BAND POWER EXTRACTION — per window, per channel (no spatial averaging)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_band_power(
    cfg: CBPAConfig,
    spectrogram: np.ndarray,
    freqs: np.ndarray,
    channel_indices: list[int] | None,
    freq_pooling: Literal["max", "mean"] = "max",
    channel_pooling: Literal["max", "mean"] = "max",
) -> np.ndarray:
    """
    Reduce spectrogram to band-wise time x channel data.

    For CMC, this enforces the intended order:
      1) max/mean over EMG channels (if present as 4th axis),
      2) max/mean over frequencies within cfg.freq_band.

    For PSD, this computes the mean within cfg.freq_band.

    Output shape is always (n_windows, n_channels_selected).

    We call aggregate_psd_spectrogram with a single op on axis=1 (freq) so
    that time and channel axes are untouched.

    Pooling rule:
      - PSD: mean within frequency band
      - CMC: max/mean within frequency band (after optional EMG max/mean)

    Parameters
    ----------
    cfg : CBPAConfig
        Active run configuration that defines modality and frequency band.
    spectrogram : np.ndarray
        Input spectrogram. Expected shapes:
        - PSD: ``(n_windows, n_freqs, n_channels)``
        - CMC: ``(n_windows, n_freqs, n_eeg)`` or
          ``(n_windows, n_freqs, n_eeg, n_emg)``.
    freqs : np.ndarray
        Frequency vector matching the spectrogram frequency axis.
    channel_indices : list[int] | None
        Optional channel subset indices (used for PSD path).
    freq_pooling : {'max', 'mean'}, default 'max'
        Pooling operation for frequency dimension (default 'max' for CMC behavior).
    channel_pooling : {'max', 'mean'}, default 'max'
        Pooling operation for EMG channels (default 'max' for CMC behavior).

    Returns
    -------
    np.ndarray
        Band-reduced array of shape ``(n_windows, n_channels_selected)``.

    Raises
    ------
    ValueError
        If the spectrogram dimensionality does not match modality expectations.
    """
    spec = spectrogram

    if cfg.modality == "CMC":
        if spec.ndim == 4:
            # Expected raw CMC layout: (time, freq, eeg, emg)
            if channel_pooling == "mean":
                spec = np.nanmean(spec, axis=3)
            else:  # "max"
                spec = np.nanmax(spec, axis=3)
        elif spec.ndim != 3:
            raise ValueError(
                f"Unexpected CMC spectrogram shape {spec.shape}. "
                "Expected 3D (time,freq,eeg) or 4D (time,freq,eeg,emg)."
            )
    elif spec.ndim != 3:
        raise ValueError(
            f"Unexpected PSD spectrogram shape {spec.shape}. Expected 3D (time,freq,channel)."
        )

    band_op = freq_pooling if cfg.modality == "CMC" else "mean"

    band_power = aggregate_psd_spectrogram(
        spec,
        freqs,
        normalize_mvc=False,
        channel_indices=channel_indices,
        is_log_scaled=cfg.psd_is_log_scaled if cfg.modality == "PSD" else False,
        freq_slice=cfg.freq_band,
        aggregation_ops=[(band_op, 1)],  # reduce freq band only → (n_windows, n_channels)
    )
    # band_power: (n_windows, n_channels)
    return band_power

def _band_power_per_phase(
    cfg: CBPAConfig,
    band_power: np.ndarray,
    timestamps: pd.DatetimeIndex,
    trial_spans: dict[int, tuple],
    trial_cond_map: dict[int, str],
    log_df: pd.DataFrame,
        min_cycle_coverage_ratio: float = 0.8,
) -> dict[str, list[np.ndarray]]:
    phase_grid = np.linspace(0, 360, cfg.n_phase_bins, endpoint=False)
    cycles_by_condition: dict[str, list[np.ndarray]] = {}

    for trial_id, (t_start, t_end) in trial_spans.items():
        condition = trial_cond_map.get(int(trial_id))
        if condition is None:
            continue

        task_freq = _get_task_freq_for_trial(log_df, t_start, t_end)
        if task_freq is None or task_freq <= 0:
            warnings.warn(
                f"  [phase] Trial {trial_id}: Task Frequency missing or zero. Skipping."
            )
            continue

        cycle_dur_sec = 1.0 / task_freq
        tw_step = (cfg.cmc_time_window_sec if cfg.modality == "CMC"
                   else cfg.psd_time_window_sec) * (1.0 - cfg.overlap_ratio)
        samples_per_cycle = cycle_dur_sec / tw_step
        if samples_per_cycle < cfg.min_samples_per_cycle:
            warnings.warn(
                f"  [phase] Trial {trial_id}: only {samples_per_cycle:.1f} "
                f"CMC samples/cycle at {task_freq} Hz — skipping "
                f"(min={cfg.min_samples_per_cycle})."
            )
            continue

        mask     = (timestamps >= t_start) & (timestamps < t_end)
        trial_bp = band_power[mask]
        trial_ts = timestamps[mask]

        if len(trial_ts) == 0:
            continue

        t_rel = np.array(
            [(ts - t_start).total_seconds() for ts in trial_ts],
            dtype=float,
        )
        trial_dur_sec = (t_end - t_start).total_seconds()

        # Use cfg.phase_start_offset_sec when explicitly set;
        # otherwise fall back to 1/task_freq, which skips exactly one
        # cycle and is always cycle-aligned regardless of frequency.
        phase_offset = (
            float(cfg.phase_start_offset_sec)
            if cfg.phase_start_offset_sec is not None
            else float(1.0 / task_freq)
        )

        # print(f"Phase normalising: {cfg.freq_band} band of {cfg.modality} for {cfg.modality_file_id}")
        cycles = data_analysis.phase_normalize_cycles(
            signal=trial_bp,
            t_rel=t_rel,
            task_freq=task_freq,
            trial_dur_sec=trial_dur_sec,
            phase_grid=phase_grid,
            min_samples_per_cycle=cfg.min_samples_per_cycle,
            min_cycle_coverage_ratio=min_cycle_coverage_ratio,
            start_offset_sec=phase_offset,
            show_debug_cycle_wise_plots=False,
            show_debug_trial_wise_plots=False,
        )
        for cycle_profile in cycles:
            cycles_by_condition.setdefault(condition, []).append(cycle_profile)

    return cycles_by_condition



# ══════════════════════════════════════════════════════════════════════════════
#  BUILD CONTRAST ARRAY  X: (n_subjects, n_times, n_channels)
# ══════════════════════════════════════════════════════════════════════════════

def build_contrast_array(
    cfg: CBPAConfig,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Build X: (n_subjects, n_times, n_channels) — A minus B contrast array.

    Condition labels are resolved entirely from the pre-computed Combined
    Statistics frame (1seg), keyed on (Subject ID, Trial ID).

    Parameters
    ----------
    cfg : CBPAConfig
        Full analysis configuration (modality, band, conditions, segmentation,
        phase normalization options).

    Returns
    -------
    X : np.ndarray
        Subject-level contrast array of shape ``(n_subjects, n_times, n_channels)``.
    ch_names_out : list[str]
        Channel labels corresponding to axis 2 of ``X``.
    time_grid : np.ndarray
        Time axis in seconds (clock-time mode) or phase in degrees
        (phase-normalized mode).

    Raises
    ------
    RuntimeError
        If no subject yields a valid contrast after filtering and checks.
    """
    # ── Load stats frame once — hard failure if missing ──────────────────────
    stats_df = load_stats_frame(cfg.data_root)

    # Derive valid subject IDs directly from the stats frame.
    # This automatically excludes subjects 9/10 (not yet collected)
    # and any future gaps, without requiring manual n_subjects updates.
    valid_subject_ids = sorted(stats_df["Subject ID"].astype(int).unique())

    if cfg.exclude_subjects:
        print(f"  [Exclusions] Skipping subjects: {cfg.exclude_subjects}")
        valid_subject_ids = [s for s in valid_subject_ids if s not in cfg.exclude_subjects]

    print(f"  [subjects] Running on {len(valid_subject_ids)} subjects: {valid_subject_ids}")


    diffs: list[np.ndarray] = []
    ch_names_out: list[str] | None = None
    time_grid: np.ndarray | None = None
    n_times_ref: int | None = None

    # Resolve channel indices once — BEFORE the subject loop:
    if cfg.modality == "CMC":
        # CMC spectrograms are already stored as the 11-channel motor subset.
        # Never pass channel_indices here — it would index into an 11-element
        # axis using 64-channel EEG indices, causing an out-of-bounds crash.
        ch_indices = None
        ch_names_out = (
            cfg.channels if cfg.channels is not None else CMC_EEG_CHANNEL_SUBSET
        )
    else:
        # PSD: full 64-channel spectrogram; subset by EEG_CHANNEL_IND_DICT indices.
        if cfg.channels is not None:
            ch_indices = [EEG_CHANNEL_IND_DICT[ch] for ch in cfg.channels]
            ch_names_out = cfg.channels
        else:
            ch_indices = None
            ch_names_out = EEG_CHANNELS


    # Phase grid is fixed and subject-independent; set it immediately
    # when phase normalisation is requested so it is available for plotting.
    if cfg.use_phase_normalization:
        time_grid = np.linspace(0, 360, cfg.n_phase_bins, endpoint=False)
        n_times_ref = cfg.n_phase_bins


    for subj in valid_subject_ids:
        print(f"  Subject {subj:02} — loading...")
        try:
            spectrogram, freqs, timestamps, log_df = _load_subject_data(cfg, subj)
        except Exception as exc:
            warnings.warn(f"Subject {subj:02}: load failed ({exc}). Skipping.")
            continue

        # ── Condition labels from stats frame (not log_df) ───────────────────
        try:
            trial_cond_map = get_trial_condition_map(
                stats_df, subj, cfg.condition_column
            )
        except ValueError as exc:
            warnings.warn(str(exc) + " Skipping.")
            continue

        # ── Trial time spans from log_df (timestamps only, not labels) ───────
        trial_spans = _get_trial_spans(log_df)
        # trial_spans: {trial_id: (start_timestamp, end_timestamp)}
        # trial_id keys here are whatever get_all_task_start_ends returns;
        # cast to int to match the stats frame Trial ID dtype:
        trial_spans_int = {int(k): v for k, v in trial_spans.items()}

        # Validate that trial IDs in spans are present in stats frame:
        span_ids = set(trial_spans_int.keys())
        map_ids = set(trial_cond_map.keys())
        missing_in_map = span_ids - map_ids
        if missing_in_map:
            warnings.warn(
                f"Subject {subj:02}: trial IDs {missing_in_map} present in "
                f"log_df but missing from stats frame. These trials will be skipped."
            )

        # Derive time grid from first subject:
        if time_grid is None:
            time_grid = _common_time_grid_from_spans(cfg, trial_spans_int, overlap_ratio=cfg.overlap_ratio)
            n_times_ref = len(time_grid)

        # Band power at native resolution: (n_windows, n_channels)
        band_power = _extract_band_power(cfg, spectrogram, freqs, ch_indices)

        # ── Phase-normalised path ─────────────────────────────────────────
        if cfg.use_phase_normalization:
            # todo: replace with data analysis function
            cycles_by_cond = _band_power_per_phase(
                cfg, band_power, timestamps,
                trial_spans_int, trial_cond_map, log_df,
                min_cycle_coverage_ratio=.8,
            )

            cycles_A = cycles_by_cond.get(cfg.condition_A, [])
            cycles_B = cycles_by_cond.get(cfg.condition_B, [])

            if len(cycles_A) < cfg.min_cycles_per_condition:
                warnings.warn(
                    f"Subject {subj:02}: only {len(cycles_A)} valid cycles for "
                    f"'{cfg.condition_A}' (min={cfg.min_cycles_per_condition}). Skipping."
                )
                continue
            if len(cycles_B) < cfg.min_cycles_per_condition:
                warnings.warn(
                    f"Subject {subj:02}: only {len(cycles_B)} valid cycles for "
                    f"'{cfg.condition_B}' (min={cfg.min_cycles_per_condition}). Skipping."
                )
                continue

            # Per-subject mean phase profile across all valid cycles
            mean_A = np.nanmean(np.stack(cycles_A, axis=0), axis=0)  # (n_phase_bins, n_ch)
            mean_B = np.nanmean(np.stack(cycles_B, axis=0), axis=0)
            diffs.append(mean_A - mean_B)

            print(
                f"    → {len(cycles_A)} cycles '{cfg.condition_A}', "
                f"{len(cycles_B)} cycles '{cfg.condition_B}'"
            )
            continue  # skip clock-time path below

        # ── Clock-time path (unchanged) ───────────────────────────────────
        if time_grid is None:
            time_grid = _common_time_grid_from_spans(
                cfg, trial_spans_int, overlap_ratio=cfg.overlap_ratio
            )
            n_times_ref = len(time_grid)

        # Per-trial time series: (n_trials, n_times, n_channels)
        trial_data, trial_ids_used = _band_power_per_trial(
            cfg, band_power, timestamps, trial_spans_int, n_times_ref
        )
        # trial_ids_used: list[int] — trial IDs in the same order as trial_data rows

        # ── Assign conditions and average ────────────────────────────────────
        idx_A, idx_B = [], []
        for i, tid in enumerate(trial_ids_used):
            label = trial_cond_map.get(tid)
            if label == cfg.condition_A:
                idx_A.append(i)
            elif label == cfg.condition_B:
                idx_B.append(i)

        if len(idx_A) == 0:
            warnings.warn(
                f"Subject {subj:02}: no trials found for "
                f"'{cfg.condition_A}' in '{cfg.condition_column}'. Skipping."
            )
            continue
        if len(idx_B) == 0:
            warnings.warn(
                f"Subject {subj:02}: no trials found for "
                f"'{cfg.condition_B}' in '{cfg.condition_column}'. Skipping."
            )
            continue

        mean_A = np.nanmean(trial_data[idx_A], axis=0)   # (n_times, n_channels)
        mean_B = np.nanmean(trial_data[idx_B], axis=0)
        diffs.append(mean_A - mean_B)

        print(
            f"    → {len(idx_A)} trials '{cfg.condition_A}', "
            f"{len(idx_B)} trials '{cfg.condition_B}'"
        )

    if len(diffs) == 0:
        raise RuntimeError(
            "[CBPA] No valid subjects produced a contrast. "
            "Check data paths, subject IDs, and condition labels."
        )

    X = np.stack(diffs, axis=0)   # (n_subjects, n_times, n_channels)
    print(
        f"\n  Contrast array built: {X.shape}  "
        f"[{X.shape[0]} subjects × {X.shape[1]} time pts × {X.shape[2]} channels]"
    )
    return X, ch_names_out, time_grid


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CBPA RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def _add_phase_wraparound(
    adjacency: "sparse.spmatrix",
    n_times: int,
    n_ch: int,
    time_grid: np.ndarray,
) -> "sparse.spmatrix":
    """Add circular phase wrap-around edges to a (n_times*n_ch)² adjacency matrix.

    Parameters
    ----------
    adjacency : sparse matrix
        Combined (n_times*n_ch × n_times*n_ch) adjacency, index convention
        t*n_ch + ch  (C-order, matching combine_adjacency output).
    n_times : int
    n_ch : int
    time_grid : np.ndarray
        Used only for the log message (last bin label).

    Returns
    -------
    sparse.csr_matrix with bool dtype, wrap-around edges added.
    """
    from scipy import sparse

    wrap = sparse.lil_matrix(adjacency.shape, dtype=bool)
    for ch in range(n_ch):
        first = 0 * n_ch + ch
        last  = (n_times - 1) * n_ch + ch
        wrap[first, last] = True
        wrap[last, first] = True
    result = (adjacency.astype(bool) + wrap.tocsr()).astype(bool)
    print(f"  [adjacency] Phase wrap-around edges added "
          f"(0°↔{int(time_grid[-1])}° for {n_ch} channels)")
    return result


def run_cbpa(
    cfg: CBPAConfig,
    cluster_rows_accumulator: list[dict] | None = None,
) -> dict:
    """Full CBPA pipeline for one contrast configuration.

    Parameters
    ----------
    cfg : CBPAConfig
    cluster_rows_accumulator : list[dict] or None
        Passed through to _save_results. If provided, cluster rows are
        appended for later batch saving. None = save per-run CSV instead.

    Returns
    -------
    dict with keys: t_obs, clusters, cluster_pv, H0, good_cluster_inds,
                    ch_names, time_grid, cfg, n_valid_subjects
    """
    filemgmt.assert_dir(cfg.output_dir)
    _print_header(cfg)

    X, ch_names, time_grid = build_contrast_array(cfg)
    n_subj, n_times, n_ch = X.shape

    df_stat = n_subj - 1
    if cfg.tail == 0:
        t_thresh = t_dist.ppf(1.0 - cfg.alpha_cluster_forming / 2, df=df_stat)
    else:
        t_thresh = t_dist.ppf(1.0 - cfg.alpha_cluster_forming, df=df_stat)
    print(f"\n  Cluster-forming threshold  t({df_stat}) = ±{t_thresh:.4f}  "
          f"(α = {cfg.alpha_cluster_forming}, tail = {cfg.tail})")

    rng = np.random.default_rng(cfg.seed)

    # Anatomical (Delaunay) adjacency is used in both branches
    info = _build_mne_info(ch_names)
    adjacency = _build_adjacency(info, n_times)  # (n_times*n_ch)² combined matrix

    # Phase wrap-around is always applied when phase-normalised
    if cfg.use_phase_normalization:
        adjacency = _add_phase_wraparound(adjacency, n_times, n_ch, time_grid)

    if cfg.use_spatio_temporal:
        t_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(
            X, n_permutations=cfg.n_permutations, threshold=t_thresh,
            tail=cfg.tail, adjacency=adjacency, n_jobs=cfg.n_jobs,
            seed=rng, out_type="mask", verbose=True,
        )
    else:
        # Flatten to (n_subj, n_times*n_ch); adjacency index convention matches
        X_flat = X.reshape(n_subj, n_times * n_ch)
        print(f"  [adjacency] flat: {adjacency.shape}, nnz edges: {adjacency.nnz}")
        t_obs_flat, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
            X_flat, n_permutations=cfg.n_permutations, threshold=t_thresh,
            tail=cfg.tail, adjacency=adjacency, n_jobs=cfg.n_jobs,
            seed=rng, out_type="mask", verbose=True,
        )
        t_obs = t_obs_flat.reshape(n_times, n_ch)

    alpha_cbpa = 0.05
    good_cluster_inds = np.where(np.array(cluster_pv) < alpha_cbpa)[0]
    print(f"\n  Clusters found: {len(clusters)} total, "
          f"{len(good_cluster_inds)} significant (cluster p < {alpha_cbpa})")
    for idx in good_cluster_inds:
        print(f"    Cluster #{idx + 1:02d}:  p = {cluster_pv[idx]:.4f}")

    results = dict(
        t_obs=t_obs, t_thresh=t_thresh, clusters=clusters,
        cluster_pv=np.array(cluster_pv), H0=H0,
        good_cluster_inds=good_cluster_inds, ch_names=ch_names,
        time_grid=time_grid, cfg=cfg, n_valid_subjects=n_subj,
    )

    _save_results(
        results, cfg,
        cluster_rows_accumulator=cluster_rows_accumulator,
        save_per_run_cluster_csv=(cluster_rows_accumulator is None),
    )

    if cfg.save_plots or cfg.show_plots:
        visualizations.plot_cbpa_results(results, cfg)

    return results




# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _save_results(
    results: dict,
    cfg: CBPAConfig,
    cluster_rows_accumulator: list[dict] | None = None,
    save_per_run_cluster_csv: bool = False,
) -> None:
    """Save CBPA results for one run.

    Always saves the .npz archive and the per-run t_obs CSV.
    Appends cluster summary rows to cluster_rows_accumulator if provided —
    the caller (run_batch) is responsible for saving the combined frame.
    Optionally also saves a per-run cluster_summary CSV when
    save_per_run_cluster_csv=True (off by default once batch saving is used).

    Parameters
    ----------
    results : dict
        Output of run_cbpa.
    cfg : CBPAConfig
        Configuration for this run.
    cluster_rows_accumulator : list[dict] or None
        If provided, cluster summary rows are appended here instead of (or
        in addition to) being saved as a standalone CSV.
    save_per_run_cluster_csv : bool
        If True, also saves a per-run *_cluster_summary.csv alongside the NPZ.
        Defaults to False when called from run_batch (combined file suffices).
    """
    stem = filemgmt.file_title(cfg.hypothesis_label, "")

    # ── 1. NPZ archive (unchanged) ────────────────────────────────────────────
    npz_out = cfg.output_dir / (stem + ".npz")
    np.savez(
        npz_out,
        t_obs=results["t_obs"],
        cluster_pv=results["cluster_pv"],
        H0=results["H0"],
        ch_names=results["ch_names"],
        time_grid=results["time_grid"],
        good_cluster_inds=results["good_cluster_inds"],
    )
    print(f"  Results saved: {npz_out}")

    # ── 2. t_obs matrix CSV (per-run; shape varies across configs) ────────────
    t_obs = results["t_obs"]          # (n_times, n_channels)
    time_grid = results["time_grid"]  # (n_times,)
    ch_names = results["ch_names"]
    t_ax = time_grid if time_grid is not None else np.arange(t_obs.shape[0])

    t_obs_df = pd.DataFrame(
        t_obs,
        index=pd.Index(np.round(t_ax, 4), name="time_s"),
        columns=ch_names,
    )
    t_obs_csv = cfg.output_dir / (stem + "_t_obs.csv")
    t_obs_df.to_csv(t_obs_csv)
    print(f"  t_obs CSV saved: {t_obs_csv}")

    # ── 3. Cluster summary rows ───────────────────────────────────────────────
    clusters = results["clusters"]
    cluster_pv = results["cluster_pv"]
    good_inds = results["good_cluster_inds"]
    t_thresh = results["t_thresh"]
    n_times, n_ch = t_obs.shape

    rows: list[dict] = []
    axis_label = "phase_deg" if cfg.use_phase_normalization else "time_s"
    for idx, (cluster, pv) in enumerate(zip(clusters, cluster_pv)):
        # In _save_results, replace the mask resolution block with:
        if isinstance(cluster, np.ndarray) and cluster.dtype == bool:
            mask = cluster.reshape(n_times, n_ch) if cluster.ndim == 1 else cluster
        else:
            mask = np.zeros((n_times, n_ch), dtype=bool)
            mask[cluster] = True

        t_in = np.where(mask.any(axis=1))[0]
        ch_in = np.where(mask.any(axis=0))[0]

        rows.append({
            # Config identity — makes rows self-describing in the combined frame
            "hypothesis":           cfg.hypothesis_label,
            "modality":             cfg.modality,
            "freq_band":            cfg.freq_band,
            "condition_column":     cfg.condition_column,
            "condition_A":          cfg.condition_A,
            "condition_B":          cfg.condition_B,
            "n_within_trial_segs":  cfg.n_within_trial_segs,
            "n_permutations":       cfg.n_permutations,
            "alpha_cluster_forming": cfg.alpha_cluster_forming,
            "tail":                 cfg.tail,
            "n_valid_subjects":     results["n_valid_subjects"],
            # Cluster statistics
            "cluster_index":        idx + 1,
            "p_value":              round(float(pv), 6),
            "significant":          bool(idx in good_inds),
            "peak_t":               round(float(np.abs(t_obs[mask]).max()) if mask.any() else 0.0, 4),
            "t_thresh":             round(float(t_thresh), 4),
            "n_time_points":        int(len(t_in)),
            f"{axis_label}_start":         round(float(t_ax[t_in[0]]),  4) if len(t_in) > 0 else None,
            f"{axis_label}_end":           round(float(t_ax[t_in[-1]]), 4) if len(t_in) > 0 else None,
            "n_channels":           int(len(ch_in)),
            "channels":             "; ".join(ch_names[i] for i in ch_in),
        })

    if cluster_rows_accumulator is not None:
        cluster_rows_accumulator.extend(rows)

    if save_per_run_cluster_csv:
        cluster_csv = cfg.output_dir / (stem + "_cluster_summary.csv")
        pd.DataFrame(rows).to_csv(cluster_csv, index=False)
        print(f"  Cluster summary CSV saved: {cluster_csv}")



def _print_header(cfg: CBPAConfig) -> None:
    """Print a compact console summary of the active CBPA configuration.

    Parameters
    ----------
    cfg : CBPAConfig
        Configuration to display.
    """
    bar = "═" * 70
    print(f"\n{bar}")
    print(f"  CBPA: {cfg.hypothesis_label}")
    print(f"  Feature    : {cfg.modality} | {cfg.freq_band} band")
    print(f"  Channels   : {'all' if cfg.channels is None else cfg.channels}")
    print(f"  Contrast   : '{cfg.condition_A}'  −  '{cfg.condition_B}'")
    print(f"  Condition  : column = '{cfg.condition_column}'")
    print(f"  Segments   : {cfg.n_within_trial_segs} per trial")
    print(f"  Test       : {'spatio-temporal' if cfg.use_spatio_temporal else 'temporal-only'}")
    print(f"  Permutations: {cfg.n_permutations}  |  tail={cfg.tail}  |  α={cfg.alpha_cluster_forming}")
    print(f"{bar}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_batch(configs: list[CBPAConfig]) -> tuple[list[dict], pd.DataFrame]:
    """Run a list of CBPA configs sequentially and save a combined cluster summary.

    Parameters
    ----------
    configs : list[CBPAConfig]

    Returns
    -------
    all_results : list[dict]
        Per-run result dicts (unchanged from before).
    combined_cluster_df : pd.DataFrame
        All cluster rows across all configs in one frame, also saved to the
        output_dir of the first config as 'CBPA Combined Cluster Summary.csv'.
    """
    all_results: list[dict] = []
    cluster_rows_accumulator: list[dict] = []

    for i, cfg in enumerate(configs):
        print(f"\n[{i + 1}/{len(configs)}] Starting: {cfg.hypothesis_label}")
        result = run_cbpa(cfg, cluster_rows_accumulator=cluster_rows_accumulator)
        all_results.append(result)

    print(f"\n{'═' * 70}")
    print(f"  Batch complete: {len(all_results)} runs.")

    # Save combined cluster summary once, into the shared output_dir
    combined_cluster_df = pd.DataFrame(cluster_rows_accumulator)
    if not combined_cluster_df.empty:
        out_dir = configs[0].output_dir
        out_path = out_dir / filemgmt.file_title("CBPA Combined Cluster Summary", ".csv")
        combined_cluster_df.to_csv(out_path, index=False)
        n_sig = combined_cluster_df["significant"].sum()
        print(f"  Combined cluster summary → {out_path}")
        print(f"  ({len(combined_cluster_df)} total clusters, {n_sig} significant)")

    return all_results, combined_cluster_df
