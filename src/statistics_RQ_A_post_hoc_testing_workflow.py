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
from src.pipeline.cbpa import CBPAConfig, run_batch
from src.pipeline.channel_layout import EEG_CHANNELS_BY_AREA, EEG_CHANNELS
from src.pipeline.heterogeneity_modelling import run_heterogeneity_modelling
from src.pipeline.visualizations import (
    plot_cmc_accuracy_phase_average,
    plot_emg_psd_phase_average_plot,
)


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
    analyse_emg_psd_phase_plot: bool = True

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
        n_phase_bins=36,  # -> 5º buckets
        min_samples_per_cycle=2,
        overlap_ratio=0.5,
        data_root=ROOT,
        output_dir=OUTPUT,
        save_plots=True,
        show_plots=True,
        include_dynamometer_force=True,
    )

    emg_psd_plot_cfg = CBPAConfig(
        modality="CMC",
        modality_file_id="Flexor",
        freq_band="gamma",
        channels=None,
        use_phase_normalization=True,
        n_phase_bins=180,  # -> 5º buckets
        min_samples_per_cycle=2,
        overlap_ratio=0.5,
        data_root=ROOT,
        output_dir=OUTPUT,
        save_plots=True,
        show_plots=True,
        include_dynamometer_force=True,
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

    if analyse_emg_psd_phase_plot:
        plot_emg_psd_phase_average_plot(
            cfg=emg_psd_plot_cfg,
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
