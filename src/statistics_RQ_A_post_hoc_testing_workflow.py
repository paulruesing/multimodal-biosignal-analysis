"""
cbpa_pipeline.py
────────────────────────────────────────────────────────────────────────────────
Cluster-Based Permutation Analysis (CBPA) — within-subject, multi-channel.

Compares 2 conditions, if categorical selects one as reference and one as treatment.
Averages trials for each condition per subject.

STATISTICAL RATIONALE
─────────────────────
LME (run on channel-ROI-averaged scalars) is the sole confirmatory test.
CBPA here is *exploratory*: it spatially/temporally decomposes effects that
LME already established. CBPA cluster p-values are used as cluster-selection
criteria for visualisation, NOT as primary inferential claims.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

import src.pipeline.cbpa as cbpa
import src.pipeline.data_integration as data_integration
import src.utils.file_management as filemgmt
from src.pipeline.visualizations import EEG_CHANNELS_BY_AREA, EEG_CHANNELS
from src.pipeline.cbpa import CBPAConfig, run_batch
from src.pipeline.heterogeneity_modelling import run_heterogeneity_modelling


def _phase_normalize_accuracy_cycles(
    accuracy: np.ndarray,
    phase_grid: np.ndarray,
    task_freq: float,
    trial_dur_sec: float,
    min_samples_per_cycle: int,
    start_offset_sec: float,
) -> list[np.ndarray]:
    """Map one trial's accuracy series to force-cycle phase and return valid cycles."""
    if accuracy is None or len(accuracy) < min_samples_per_cycle or task_freq <= 0:
        return []

    effective_dur = trial_dur_sec - start_offset_sec
    if effective_dur <= 0:
        return []

    t_rel = start_offset_sec + np.linspace(0.0, effective_dur, len(accuracy), endpoint=False)
    phases = (t_rel * task_freq * 360.0) % 360.0
    cycle_dur_sec = 1.0 / task_freq
    n_complete_cycles = int(trial_dur_sec * task_freq)

    out: list[np.ndarray] = []
    for cycle_idx in range(n_complete_cycles):
        t0 = cycle_idx * cycle_dur_sec
        t1 = (cycle_idx + 1) * cycle_dur_sec
        in_cycle = (t_rel >= t0) & (t_rel < t1)
        if int(in_cycle.sum()) < min_samples_per_cycle:
            continue

        phase_vals = np.clip(phases[in_cycle], 0.0, 360.0 - 1e-9)
        acc_vals = accuracy[in_cycle]

        order = np.argsort(phase_vals)
        phase_vals = phase_vals[order]
        acc_vals = acc_vals[order]
        phase_vals, unique_idx = np.unique(phase_vals, return_index=True)
        acc_vals = acc_vals[unique_idx]

        if len(phase_vals) < 2:
            continue

        out.append(
            np.interp(
                phase_grid,
                phase_vals,
                acc_vals,
                left=acc_vals[0],
                right=acc_vals[-1],
            )
        )

    return out


def _derive_cmc_accuracy_hypothesis_label(cfg: CBPAConfig) -> str:
    """Build a deterministic label from modality settings for this plot type."""
    return f"{cfg.modality}_{cfg.modality_file_id}_{cfg.freq_band}_phase_avg_vs_accuracy"


def plot_cmc_accuracy_phase_average(
    cfg: CBPAConfig,
    experiment_results_dir: Path,
    exclude_subjects: list[int] | None = None,
) -> None:
    """Create a CBPA-like 2-panel plot: mean CMC map + phase-normalized accuracy."""
    if cfg.modality != "CMC":
        raise ValueError("plot_cmc_accuracy_phase_average requires cfg.modality='CMC'.")
    if not cfg.use_phase_normalization:
        raise ValueError("plot_cmc_accuracy_phase_average requires phase normalization enabled.")

    filemgmt.assert_dir(cfg.output_dir)
    plot_label = _derive_cmc_accuracy_hypothesis_label(cfg)

    stats_df = cbpa.load_stats_frame(cfg.data_root)
    subject_ids = sorted(stats_df["Subject ID"].astype(int).unique())
    exclude = set(exclude_subjects or [])
    subject_ids = [sid for sid in subject_ids if sid not in exclude]

    phase_grid = np.linspace(0, 360, cfg.n_phase_bins, endpoint=False)
    ch_names = cfg.channels if cfg.channels is not None else cbpa.CMC_EEG_CHANNEL_SUBSET

    subject_cmc_profiles: list[np.ndarray] = []
    subject_acc_profiles: list[np.ndarray] = []
    valid_subjects: list[int] = []

    for subj in subject_ids:
        try:
            spectrogram, freqs, timestamps, log_df = cbpa._load_subject_data(cfg, subj)
        except Exception as exc:
            warnings.warn(f"Subject {subj:02}: failed to load data ({exc}). Skipping.")
            continue

        trial_spans = {int(k): v for k, v in cbpa._get_trial_spans(log_df).items()}
        if len(trial_spans) == 0:
            warnings.warn(f"Subject {subj:02}: no trial spans found. Skipping.")
            continue

        band_power = cbpa._extract_band_power(cfg, spectrogram, freqs, channel_indices=None)
        trial_cond_map = {trial_id: "ALL" for trial_id in trial_spans}
        cmc_cycles = cbpa._band_power_per_phase(
            cfg=cfg,
            band_power=band_power,
            timestamps=timestamps,
            trial_spans=trial_spans,
            trial_cond_map=trial_cond_map,
            log_df=log_df,
        ).get("ALL", [])

        if len(cmc_cycles) == 0:
            warnings.warn(f"Subject {subj:02}: no valid CMC cycles. Skipping.")
            continue

        accuracy_cycles: list[np.ndarray] = []
        subj_exp_dir = experiment_results_dir / f"subject_{subj:02}"
        for trial_id, (t_start, t_end) in trial_spans.items():
            task_freq = cbpa._get_task_freq_for_trial(log_df, t_start, t_end)
            if task_freq is None or task_freq <= 0:
                continue

            accuracy = data_integration.fetch_trial_accuracy(
                experiment_data_dir=subj_exp_dir,
                trial_id=int(trial_id),
                log_df=log_df,
                error_handling="continue",
            )
            if accuracy is None:
                continue

            trial_dur_sec = (t_end - t_start).total_seconds()
            accuracy_cycles.extend(
                _phase_normalize_accuracy_cycles(
                    accuracy=accuracy,
                    phase_grid=phase_grid,
                    task_freq=float(task_freq),
                    trial_dur_sec=float(trial_dur_sec),
                    min_samples_per_cycle=cfg.min_samples_per_cycle,
                    start_offset_sec=float(data_integration.TRIAL_ACCURACY_START_OFFSET_SEC),
                )
            )

        if len(accuracy_cycles) == 0:
            warnings.warn(f"Subject {subj:02}: no valid accuracy cycles. Skipping.")
            continue

        subject_cmc_profiles.append(np.nanmean(np.stack(cmc_cycles, axis=0), axis=0))
        subject_acc_profiles.append(np.nanmean(np.stack(accuracy_cycles, axis=0), axis=0))
        valid_subjects.append(subj)

    if len(subject_cmc_profiles) == 0 or len(subject_acc_profiles) == 0:
        warnings.warn("No valid subjects for CMC+accuracy phase plot. Nothing will be plotted.")
        return

    cmc_stack = np.stack(subject_cmc_profiles, axis=0)  # (n_subj, n_phase, n_ch)
    acc_stack = np.stack(subject_acc_profiles, axis=0)  # (n_subj, n_phase)

    cmc_mean = np.nanmean(cmc_stack, axis=0)
    acc_mean = np.nanmean(acc_stack, axis=0)
    acc_std = np.nanstd(acc_stack, axis=0)

    ax_tgt_left = None
    ax_tgt_right = None
    if cfg.show_target_sine:
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(
            2,
            4,
            width_ratios=[1.0, 0.05, 0.14, 1.0],
            height_ratios=[5.0, 1.0],
            wspace=0.12,
            hspace=0.28,
        )
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 3])
        ax_tgt_left = fig.add_subplot(gs[1, 0], sharex=ax)
        ax_tgt_right = fig.add_subplot(gs[1, 3], sharex=ax2)
        fig.add_subplot(gs[1, 1]).axis("off")
        fig.add_subplot(gs[0, 2]).axis("off")
        fig.add_subplot(gs[1, 2]).axis("off")
    else:
        fig = plt.figure(figsize=(16, 5))
        gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.05, 0.14, 1.0], wspace=0.12)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 3])
        fig.add_subplot(gs[0, 2]).axis("off")

    if cfg.include_suptitle:
        fig.suptitle(
            f"{plot_label}\n"
            f"Average {cfg.modality} ({cfg.modality_file_id}, {cfg.freq_band}) + Accuracy (RMSE)  |  "
            f"n = {len(valid_subjects)} subjects",
            fontsize=10,
        )

    cmc_vmin = float(np.nanpercentile(cmc_mean, 3))
    cmc_vmax = float(np.nanpercentile(cmc_mean, 97))
    if not np.isfinite(cmc_vmin) or not np.isfinite(cmc_vmax) or cmc_vmin == cmc_vmax:
        cmc_vmin, cmc_vmax = None, None
    im = ax.imshow(
        cmc_mean.T,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=cmc_vmin,
        vmax=cmc_vmax,
        extent=[phase_grid[0], 360.0, -0.5, len(ch_names) - 0.5],
    )
    plt.colorbar(im, cax=cax, label="CMC Value")
    if not cfg.show_target_sine:
        ax.set_xlabel("Force Cycle Phase (°)")
    ax.set_ylabel("Channel index")
    ax.set_yticks(range(len(ch_names)))
    ax.set_yticklabels(ch_names, fontsize=6)
    ax.set_title("Averaged phase-normalized CMC map")

    acc_band = 0.5 * acc_std
    ax2.plot(phase_grid, acc_mean, color="tab:blue", linewidth=1.8, label="Mean RMSE")
    ax2.fill_between(
        phase_grid,
        acc_mean - acc_band,
        acc_mean + acc_band,
        color="tab:blue",
        alpha=0.2,
        label="±0.5 × SD",
    )
    if not cfg.show_target_sine:
        ax2.set_xlabel("Force Cycle Phase (°)")
    ax2.set_ylabel("Accuracy (RMSE)")
    ax2.set_title("Phase-normalized average accuracy")
    ax2.legend(fontsize=8)

    if cfg.show_target_sine and ax_tgt_left is not None and ax_tgt_right is not None:
        cbpa._plot_target_sine_panel(
            ax_tgt_left,
            phase_grid,
            cfg,
            x_label="Force Cycle Phase (°)",
            show_ylabel=True,
        )
        cbpa._plot_target_sine_panel(
            ax_tgt_right,
            phase_grid,
            cfg,
            x_label="Force Cycle Phase (°)",
            show_ylabel=True,
        )

    phase_axes = [ax, ax2]
    if cfg.show_target_sine and ax_tgt_left is not None and ax_tgt_right is not None:
        phase_axes.extend([ax_tgt_left, ax_tgt_right])
    for a in phase_axes:
        a.set_xticks([0, 90, 180, 270, 360])
        a.axvline(90, color="grey", lw=0.5, ls=":")
        a.axvline(270, color="grey", lw=0.5, ls=":")

    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.10)
    if cfg.save_plots:
        out = cfg.output_dir / filemgmt.file_title(plot_label + "_cmc_accuracy_phase", ".png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {out}")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)



if __name__ == "__main__":

    ROOT = Path().resolve().parent
    OUTPUT = ROOT / "output" / "statistics_RQ_A" / "post_hoc_testing"
    EXPERIMENT_RESULTS = ROOT / "data" / "experiment_results"
    OMNIBUS_TESTING_RESULTS = ROOT / "output" / "statistics_RQ_A" / "omnibus_testing"

    # run on subset?
    exclude_subjects: list[int] = []

    # workflow control:
    analyse_cbp: bool = False
    analyse_cmc_accuracy_phase_plot: bool = True

    analyse_subject_heterogeneity: bool = False
    dep_vars_to_analyse: list[str] = [
        "CMC_Flexor_mean_beta", "CMC_Extensor_mean_beta",
        "CMC_Flexor_mean_gamma", "CMC_Extensor_mean_gamma",
    ]

    # Condition levels to scrutinise — level_key → (Condition_Variable, [non-reference conditions])
    conditions_to_evaluate: dict[str, tuple[str, list[str]]] = {
        "lvl_0": ("Music Listening", ["True"]),
        "lvl_1": ("Category or Silence", ["Happy", "Groovy", "Sad", "Classic"]),
    }

    # Mutual Information Analysis:
    analyse_mi_for_dfbetas: bool = True
    plot_mi_categories: list[Literal['dfbeta', 'cooks_d', 'contrast']] = []#'cooks_d', 'contrast']

    # Top-N MI-ranked moderators to carry into cluster→attribute scatter plots
    top_n_moderators: int = 3

    # Feature blocks to include in clustering (any subset of these three)
    clustering_measures: list[str] = ["contrast", "cooks_d"]

    # Minimum number of subjects per cluster — prevents trivial 1-subject clusters
    min_cluster_size: int = 2

    cmc_accuracy_plot_cfg = CBPAConfig(
        modality="CMC",
        modality_file_id="Flexor",
        freq_band="gamma",
        channels=None,
        use_phase_normalization=True,
        n_phase_bins=36,
        min_samples_per_cycle=2,
        overlap_ratio=0.5,
        data_root=ROOT,
        output_dir=OUTPUT,
        save_plots=True,
        show_plots=True,
    )

    # ══════════════════════════════════════════════════════════════════════════════
    #  Cluster-based Permutation Analysis (CBPA)
    # ══════════════════════════════════════════════════════════════════════════════
    if analyse_cbp:

        ### DEFINE CONTRASTS TO SCRUTINISE
        # Only run CBPA for effects already established by LME.
        # Add / remove / comment out entries here.

        CONTRASTS: list[CBPAConfig] = [



            # ── H1: CMC Extensor beta, Classic vs Silence ─────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Extensor",
                freq_band="beta",
                channels=None,
                condition_column="Category or Silence",
                condition_A="Classic",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                output_dir=OUTPUT, exclude_subjects = exclude_subjects,
                hypothesis_label="H1_CMC_Extensor_beta_Classic_vs_Silence",
            ),

            # ── H1: CMC Extensor beta, Happy vs Classic ───────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Extensor",
                freq_band="beta",
                channels=None,
                condition_column="Perceived Category",
                condition_A="Happy",
                condition_B="Classic",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                output_dir=OUTPUT, exclude_subjects = exclude_subjects,
                hypothesis_label="H1_CMC_Extensor_beta_Happy_vs_Classic",
            ),

            # ── H1: CMC Extensor beta, Groovy vs Classic ──────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Extensor",
                freq_band="beta",
                channels=None,
                condition_column="Perceived Category",
                condition_A="Groovy",
                condition_B="Classic",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                output_dir=OUTPUT, exclude_subjects = exclude_subjects,
                hypothesis_label="H1_CMC_Extensor_beta_Groovy_vs_Classic",
            ),

            #### HAPPY VS SILENCE
            # ── H1: CMC Flexor beta, Happy vs Silence ────────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Flexor",
                freq_band="beta",
                channels=None,
                condition_column="Category or Silence",
                condition_A="Happy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                output_dir=OUTPUT, exclude_subjects = exclude_subjects,
                hypothesis_label="H1_CMC_Flexor_beta_Happy_vs_Silence",
            ),

            # ── H1: CMC Extensor beta, Happy vs Silence ────────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Extensor",
                freq_band="beta",
                channels=None,
                condition_column="Category or Silence",
                condition_A="Happy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                output_dir=OUTPUT, exclude_subjects = exclude_subjects,
                hypothesis_label="H1_CMC_Extensor_beta_Happy_vs_Silence",
            ),


            # ── H1: CMC Extensor gamma, Happy vs Silence ──────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Extensor",
                freq_band="gamma",
                channels=None,
                condition_column="Category or Silence",
                condition_A="Happy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                output_dir=OUTPUT, exclude_subjects = exclude_subjects,
                hypothesis_label="H1_CMC_Extensor_gamma_Happy_vs_Silence",
            ),

            # ── H1: CMC Flexor gamma, Happy vs Silence ────────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Flexor",
                freq_band="gamma",
                channels=None,
                condition_column="Category or Silence",
                condition_A="Happy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                output_dir=OUTPUT, exclude_subjects = exclude_subjects,
                hypothesis_label="H1_CMC_Flexor_gamma_Happy_vs_Silence",
            ),

            #### Groovy VS SILENCE
            # ── H1: CMC Flexor beta, Groovy vs Silence ────────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Flexor",
                freq_band="beta",
                channels=None,
                condition_column="Category or Silence",
                condition_A="Groovy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                output_dir=OUTPUT, exclude_subjects = exclude_subjects,
                hypothesis_label="H1_CMC_Flexor_beta_Groovy_vs_Silence",
            ),

            # ── H1: CMC Extensor beta, Groovy vs Silence ────────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Extensor",
                freq_band="beta",
                channels=None,
                condition_column="Category or Silence",
                condition_A="Groovy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                output_dir=OUTPUT, exclude_subjects = exclude_subjects,
                hypothesis_label="H1_CMC_Extensor_beta_Groovy_vs_Silence",
            ),

            # ── H1: CMC Extensor gamma, Groovy vs Silence ──────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Extensor",
                freq_band="gamma",
                channels=None,
                condition_column="Category or Silence",
                condition_A="Groovy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                output_dir=OUTPUT, exclude_subjects = exclude_subjects,
                hypothesis_label="H1_CMC_Extensor_gamma_Groovy_vs_Silence",
            ),

            # ── H1: CMC Flexor gamma, Groovy vs Silence ────────────────────────────
            CBPAConfig(
                modality="CMC",
                modality_file_id="Flexor",
                freq_band="gamma",
                channels=None,
                condition_column="Category or Silence",
                condition_A="Groovy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=1,
                n_permutations=1000,
                use_spatio_temporal=True,
                use_phase_normalization=True,
                data_root=ROOT,
                output_dir=OUTPUT, exclude_subjects = exclude_subjects,
                hypothesis_label="H1_CMC_Flexor_gamma_Groovy_vs_Silence",
            ),

            # ── H2: Fronto-central–centro-parietal–temporal theta, Happy vs Silence
            # Covers PSD_eeg_FC_CP_T_theta (Level 1 Happy effect)
            CBPAConfig(
                modality="PSD",
                modality_file_id="eeg",
                freq_band="theta",
                channels=(
                        EEG_CHANNELS_BY_AREA["Frontal"]
                        + EEG_CHANNELS_BY_AREA["Central"]
                        + EEG_CHANNELS_BY_AREA["Centro-Parietal"]
                        + EEG_CHANNELS_BY_AREA["Temporal"]
                ),
                condition_column="Category or Silence",
                condition_A="Happy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=0,
                n_permutations=1000,
                use_spatio_temporal=True,
                data_root=ROOT,
                output_dir=OUTPUT, exclude_subjects = exclude_subjects,
                hypothesis_label="H2_theta_FC_CP_T_Happy_vs_Silence",
            ),

            # ── H5: Global gamma, Happy vs Silence ────────────────────────────────
            # Covers PSD_eeg_Global_gamma (Level 1 Happy effect)
            CBPAConfig(
                modality="PSD",
                modality_file_id="eeg",
                freq_band="gamma",
                channels=EEG_CHANNELS,
                condition_column="Category or Silence",
                condition_A="Happy",
                condition_B="Silence",
                n_within_trial_segs=1,
                tail=0,
                n_permutations=1000,
                use_spatio_temporal=True,
                data_root=ROOT,
                output_dir=OUTPUT, exclude_subjects = exclude_subjects,
                hypothesis_label="H5_gamma_Global_Happy_vs_Silence",
            ),
        ]



        ### RUN
        results = run_batch(CONTRASTS)



    # ══════════════════════════════════════════════════════════════════════════════
    #  CMC and Accuracy Spectrogram
    # ══════════════════════════════════════════════════════════════════════════════
    if analyse_cmc_accuracy_phase_plot:
        plot_cmc_accuracy_phase_average(
            cfg=cmc_accuracy_plot_cfg,
            experiment_results_dir=EXPERIMENT_RESULTS,
            exclude_subjects=exclude_subjects,
        )



    # ══════════════════════════════════════════════════════════════════════════════
    #  Subject-Heterogeneity Modelling
    # ══════════════════════════════════════════════════════════════════════════════
    if analyse_subject_heterogeneity:

        run_heterogeneity_modelling(
            dep_vars=dep_vars_to_analyse,
            conditions_to_evaluate=conditions_to_evaluate,
            clustering_measures=clustering_measures,
            plot_mi_categories=plot_mi_categories,
            top_n_moderators=top_n_moderators,
            min_cluster_size=min_cluster_size,
            output_dir=OUTPUT,
            omnibus_results_dir=OMNIBUS_TESTING_RESULTS,
            experiment_results_dir=EXPERIMENT_RESULTS,
            exclude_subjects=exclude_subjects,
        )
