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
from src.pipeline.visualizations import EEG_CHANNEL_IND_DICT, EEG_CHANNELS_BY_AREA
import src.utils.file_management as filemgmt




# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS — mirror your existing channel definitions
# ══════════════════════════════════════════════════════════════════════════════

EEG_CHANNELS: list[str] = list(EEG_CHANNEL_IND_DICT.keys())   # full 64-ch set
EEG_SFREQ: float = 2048  # Hz — adjust to your actual EEG sampling rate

# CMC subset (11 left-hemisphere motor channels; will be mirrored for left-handers)
# todo: always needs to match feature_extraction_workflow.py!
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

    log_df = data_integration.fetch_enriched_log_frame(subject_exp_dir)
    log_df.index = data_analysis.make_timezone_aware(log_df.index)

    qtc_start, qtc_end = data_integration.get_qtc_measurement_start_end(log_df, False)

    tw = cfg.psd_time_window_sec if cfg.modality == "PSD" else cfg.cmc_time_window_sec

    spectrogram, times, freqs = fetch_stored_spectrograms(
        subject_feat_dir,
        modality=cfg.modality,
        file_identifier=cfg.modality_file_id,
    )
    # spectrogram shape: (n_windows, n_freqs, n_channels)

    timestamps = data_analysis.add_time_index(
        start_timestamp=qtc_start + pd.Timedelta(seconds=tw / 2),
        end_timestamp=qtc_end - pd.Timedelta(seconds=tw / 2),
        n_timesteps=len(times),
    )
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
    mask = (log_df.index >= t_start) & (log_df.index < t_end)
    col = log_df.loc[mask, column].dropna()
    return col.mode().iloc[0] if not col.empty else None


# ══════════════════════════════════════════════════════════════════════════════
#  BAND POWER EXTRACTION — per window, per channel (no spatial averaging)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_band_power(
    cfg: CBPAConfig,
    spectrogram: np.ndarray,  # (n_windows, n_freqs, n_channels)
    freqs: np.ndarray,
    channel_indices: list[int] | None,
) -> np.ndarray:
    """
    Reduce spectrogram along frequency axis only; keep full time and channel
    resolution. Returns shape (n_windows, n_channels_selected).

    We call aggregate_psd_spectrogram with a single op on axis=1 (freq) so
    that the time and channel axes are untouched.
    """
    band_power = aggregate_psd_spectrogram(
        spectrogram,
        freqs,
        normalize_mvc=False,
        channel_indices=channel_indices,
        is_log_scaled=cfg.psd_is_log_scaled if cfg.modality == "PSD" else False,
        freq_slice=cfg.freq_band,
        aggregation_ops=[("mean", 1)],  # mean over freq band only → (n_windows, n_channels)
    )
    # band_power: (n_windows, n_channels)
    return band_power


def _band_power_per_phase(
    cfg: CBPAConfig,
    band_power: np.ndarray,          # (n_windows, n_channels)
    timestamps: pd.DatetimeIndex,
    trial_spans: dict[int, tuple],   # {trial_id: (start, end)}
    trial_cond_map: dict[int, str],  # {trial_id: condition_label}
    log_df: pd.DataFrame,
) -> dict[str, list[np.ndarray]]:
    """
    Convert band_power time series into phase-resolved profiles per condition.

    For each trial:
      1. Retrieve task frequency from log_df.
      2. Skip trial if samples-per-cycle < cfg.min_samples_per_cycle.
      3. For each complete force cycle, map CMC windows to phase (0–360°)
         and interpolate onto cfg.n_phase_bins.
      4. Accumulate cycles by condition label.

    Returns
    -------
    cycles_by_condition : dict mapping condition_label →
                          list of (n_phase_bins, n_channels) arrays,
                          one entry per valid cycle across all trials.
    """
    phase_grid = np.linspace(0, 360, cfg.n_phase_bins, endpoint=False)
    cycles_by_condition: dict[str, list[np.ndarray]] = {}

    for trial_id, (t_start, t_end) in trial_spans.items():
        condition = trial_cond_map.get(int(trial_id))
        if condition is None:
            continue  # trial not assigned to any condition

        # Retrieve task frequency for this trial
        task_freq = _get_task_freq_for_trial(log_df, t_start, t_end)
        if task_freq is None or task_freq <= 0:
            warnings.warn(
                f"  [phase] Trial {trial_id}: Task Frequency missing or zero. Skipping."
            )
            continue

        cycle_dur_sec = 1.0 / task_freq

        # Check temporal resolution: samples per cycle
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

        # Slice spectrogram windows belonging to this trial
        mask = (timestamps >= t_start) & (timestamps < t_end)
        trial_bp = band_power[mask]          # (n_windows_in_trial, n_channels)
        trial_ts = timestamps[mask]

        if len(trial_ts) == 0:
            continue

        # Convert absolute timestamps to seconds relative to trial start
        t_rel = np.array(
            [(ts - t_start).total_seconds() for ts in trial_ts],
            dtype=float,
        )

        # Phase of each CMC window: sine starts at 0 at trial onset
        # phase = (t_rel * task_freq * 360) % 360
        window_phases = (t_rel * task_freq * 360.0) % 360.0

        # Split into individual complete cycles
        trial_dur_sec = (t_end - t_start).total_seconds()
        n_complete_cycles = int(trial_dur_sec * task_freq)

        for cycle_idx in range(n_complete_cycles):
            # Identify windows belonging to this cycle via raw time offset
            t_cycle_start = cycle_idx * cycle_dur_sec
            t_cycle_end = (cycle_idx + 1) * cycle_dur_sec
            in_cycle = (t_rel >= t_cycle_start) & (t_rel < t_cycle_end)

            if in_cycle.sum() < cfg.min_samples_per_cycle:
                continue  # insufficient samples in this specific cycle

            # Phase within this cycle: 0–360°
            phases_in_cycle = window_phases[in_cycle]
            # Clamp to [0, 360) to handle floating point edge cases
            phases_in_cycle = np.clip(phases_in_cycle, 0.0, 360.0 - 1e-9)
            bp_in_cycle = trial_bp[in_cycle]  # (n_in_cycle, n_channels)

            # Interpolate each channel onto the shared phase grid
            cycle_profile = np.zeros((cfg.n_phase_bins, bp_in_cycle.shape[1]))
            for ch in range(bp_in_cycle.shape[1]):
                cycle_profile[:, ch] = np.interp(
                    phase_grid,
                    phases_in_cycle,
                    bp_in_cycle[:, ch],
                    left=bp_in_cycle[0, ch],   # extrapolate at edges with boundary value
                    right=bp_in_cycle[-1, ch],
                )

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
            cycles_by_cond = _band_power_per_phase(
                cfg, band_power, timestamps,
                trial_spans_int, trial_cond_map, log_df,
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
        _plot_results(results, cfg)

    return results




# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _plot_results(results: dict, cfg: CBPAConfig) -> None:
    t_obs        = results["t_obs"]
    t_thresh     = results["t_thresh"]
    clusters     = results["clusters"]
    cluster_pv   = results["cluster_pv"]
    good_inds    = results["good_cluster_inds"]
    ch_names     = results["ch_names"]
    time_grid    = results["time_grid"]
    n_valid_subjects = results["n_valid_subjects"]

    n_times, n_ch = t_obs.shape
    t_ax = time_grid if time_grid is not None else np.arange(n_times)

    # Shared mask resolver — handles all formats MNE may return:
    #   • bool ndarray shape (n_times, n_ch)       — spatio-temporal
    #   • bool ndarray shape (n_times * n_ch,)      — flat temporal-only
    #   • tuple of index arrays (rows, cols)        — out_type="indices"
    def _resolve_mask(cluster) -> np.ndarray:
        """
        Resolve any MNE cluster format to a (n_times, n_ch) boolean mask.

        MNE may return any of these formats depending on test type and version:
          - np.ndarray bool (n_times, n_ch)      spatio-temporal, out_type="mask"
          - np.ndarray bool (n_times * n_ch,)    flat test, out_type="mask"
          - (np.ndarray bool,)                   flat test, bool wrapped in tuple
          - (np.ndarray int,)                    flat test, int indices in tuple
          - (slice,)                             flat test, slice wrapped in tuple
          - (np.ndarray int, np.ndarray int)     2D index arrays (time, channel)
        """
        n_flat = n_times * n_ch
        flat_mask = np.zeros(n_flat, dtype=bool)

        # Unwrap single-element tuples to their inner value
        if isinstance(cluster, tuple) and len(cluster) == 1:
            cluster = cluster[0]

        # --- Bare boolean array ---
        if isinstance(cluster, np.ndarray) and cluster.dtype == bool:
            return cluster.reshape(n_times, n_ch)

        # --- Slice into flat space ---
        if isinstance(cluster, slice):
            flat_mask[cluster] = True
            return flat_mask.reshape(n_times, n_ch)

        # --- Tuple of two index arrays: (time_inds, ch_inds) ---
        if (isinstance(cluster, tuple) and len(cluster) == 2
                and isinstance(cluster[0], np.ndarray)):
            mask = np.zeros((n_times, n_ch), dtype=bool)
            mask[cluster[0], cluster[1]] = True
            return mask

        # --- Integer index array into flat space ---
        if isinstance(cluster, np.ndarray):
            idx = cluster.ravel().astype(int)
            idx = idx[(idx >= 0) & (idx < n_flat)]  # bounds guard
            flat_mask[idx] = True
            return flat_mask.reshape(n_times, n_ch)

        # --- Fallback: try coercing to integer indices ---
        try:
            idx = np.asarray(cluster).ravel().astype(int)
            idx = idx[(idx >= 0) & (idx < n_flat)]
            flat_mask[idx] = True
        except Exception as e:
            warnings.warn(f"[CBPA] _resolve_mask: unrecognised cluster format "
                          f"{type(cluster)} — mask will be all-False. Error: {e}")

        return flat_mask.reshape(n_times, n_ch)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"{cfg.hypothesis_label}\n"
        f"Contrast: '{cfg.condition_A}' − '{cfg.condition_B}'  |  "
        f"{cfg.modality} {cfg.freq_band}  |  "
        f"n = {n_valid_subjects} subjects, {cfg.n_permutations} permutations",
        fontsize=10,
    )

    # ── Panel A: heatmap + cluster contours ──────────────────────────────────
    ax = axes[0]
    # Use a fixed ±3 baseline for cross-plot comparability;
    # expand only if the observed t-values exceed it
    vlim = max(3.0, np.nanpercentile(np.abs(t_obs), 97))
    im = ax.imshow(
        t_obs.T, aspect="auto", origin="lower", cmap="RdBu_r",
        vmin=-vlim, vmax=vlim,
        extent=[t_ax[0], t_ax[-1], -0.5, n_ch - 0.5],
    )
    plt.colorbar(im, ax=ax, label="t-statistic", shrink=0.8)

    for idx, cluster in enumerate(clusters):
        mask = _resolve_mask(cluster)
        color = "black" if idx in good_inds else "silver"
        lw    = 1.8    if idx in good_inds else 0.8
        # contour needs at least one True and one False cell to draw anything
        if mask.any() and not mask.all():
            ax.contour(
                np.linspace(t_ax[0], t_ax[-1], n_times),
                np.arange(n_ch),
                mask.T.astype(float),
                levels=[0.5], colors=color, linewidths=lw,
            )

    x_label = "Force Cycle Phase (°)" if cfg.use_phase_normalization else "Time within trial (s)"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Channel index")
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels(ch_names, fontsize=6)
    ax.set_title("t-statistic map\n(black contour = significant cluster)")

    # ── Panel B: significant cluster time courses ─────────────────────────────
    ax2 = axes[1]
    if len(good_inds) == 0:
        ax2.text(0.5, 0.5, "No significant clusters", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12, color="grey")
    else:
        for idx in good_inds:
            # Use shared resolver — fixes the previously unreshaped 1D mask here
            mask = _resolve_mask(clusters[idx])

            ch_in_cluster = mask.any(axis=0)   # (n_ch,) bool
            t_in_cluster  = mask.any(axis=1)   # (n_times,) bool

            if not ch_in_cluster.any():
                continue

            t_course = t_obs[:, ch_in_cluster].mean(axis=1)  # (n_times,)
            ax2.plot(t_ax, t_course,
                     label=f"Cluster #{idx + 1}  p={cluster_pv[idx]:.3f}")
            ax2.fill_between(t_ax, 0, t_course, where=t_in_cluster, alpha=0.2)

        ax2.axhline(0,         color="k",   linewidth=0.8, linestyle="--")
        ax2.axhline( t_thresh, color="red", linewidth=0.8, linestyle=":",
                     label=f"±t_thresh ({t_thresh:.2f})")
        ax2.axhline(-t_thresh, color="red", linewidth=0.8, linestyle=":")
        ax2.set_xlabel(x_label)
        ax2.set_ylabel("Mean t-statistic over cluster channels")
        ax2.set_title("Significant cluster time courses")
        ax2.legend(fontsize=8)

    if cfg.use_phase_normalization:
        for a in [ax, ax2]:
            a.set_xticks([0, 90, 180, 270, 360])
            a.axvline(90,  color="grey", lw=0.5, ls=":")
            a.axvline(270, color="grey", lw=0.5, ls=":")

    plt.tight_layout()
    if cfg.save_plots:
        out = cfg.output_dir / filemgmt.file_title(cfg.hypothesis_label + "_clusters", ".png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {out}")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)



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