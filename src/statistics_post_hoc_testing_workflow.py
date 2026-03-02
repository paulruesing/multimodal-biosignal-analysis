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

from pathlib import Path
from src.pipeline.visualizations import EEG_CHANNEL_IND_DICT, EEG_CHANNELS_BY_AREA, EEG_CHANNELS
from src.pipeline.cbpa import CBPAConfig, run_batch







# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — define your contrasts here
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    ROOT = Path().resolve().parent
    OUTPUT = ROOT / "output" / "statistics_post_hoc_testing"
    N_SUBJECTS = 11  # needs to match statistical summary frame in pre-computed features.

    # ─────────────────────────────────────────────────────────────────────────
    # DEFINE CONTRASTS TO SCRUTINISE
    # Only run CBPA for effects already established by LME.
    # Add / remove / comment out entries here.
    # ─────────────────────────────────────────────────────────────────────────

    # todo: renew with new findings:
    CONTRASTS: list[CBPAConfig] = [

        # ── H4: Parieto-occipital alpha, Happy vs Silence ─────────────────────
        CBPAConfig(
            modality="PSD",
            modality_file_id="eeg",
            freq_band="alpha",
            channels=EEG_CHANNELS_BY_AREA["Parietal"] + EEG_CHANNELS_BY_AREA["Parieto-Occipital"],
            condition_column="Category or Silence",
            condition_A="Happy",
            condition_B="Silence",
            n_within_trial_segs=1,
            tail=0,
            n_permutations=1000,
            use_spatio_temporal=True,
            data_root=ROOT,
            n_subjects=N_SUBJECTS,
            output_dir=OUTPUT,
            hypothesis_label="H4_alpha_PO_Happy_vs_Silence",
        ),

        # ── H3: Fronto-central beta, Happy vs Silence ─────────────────────────
        CBPAConfig(
            modality="PSD",
            modality_file_id="eeg",
            freq_band="beta",
            channels=EEG_CHANNELS_BY_AREA["Frontal"] + EEG_CHANNELS_BY_AREA["Central"],
            condition_column="Category or Silence",
            condition_A="Happy",
            condition_B="Silence",
            n_within_trial_segs=1,
            tail=0,
            n_permutations=1000,
            use_spatio_temporal=True,
            data_root=ROOT,
            n_subjects=N_SUBJECTS,
            output_dir=OUTPUT,
            hypothesis_label="H3_beta_FC_Happy_vs_Silence",
        ),

        # ── H3: Fronto-central beta, Happy vs Classic ─────────────────────────
        CBPAConfig(
            modality="PSD",
            modality_file_id="eeg",
            freq_band="beta",
            channels=EEG_CHANNELS_BY_AREA["Frontal"] + EEG_CHANNELS_BY_AREA["Central"],
            condition_column="Perceived Category",
            condition_A="Happy",
            condition_B="Classic",
            n_within_trial_segs=1,
            tail=0,
            n_permutations=1000,
            use_spatio_temporal=True,
            data_root=ROOT,
            n_subjects=N_SUBJECTS,
            output_dir=OUTPUT,
            hypothesis_label="H3_beta_FC_Happy_vs_Classic",
        ),

        # ── H3: Fronto-central beta, Groovy vs Classic ────────────────────────
        CBPAConfig(
            modality="PSD",
            modality_file_id="eeg",
            freq_band="beta",
            channels=EEG_CHANNELS_BY_AREA["Frontal"] + EEG_CHANNELS_BY_AREA["Central"],
            condition_column="Perceived Category",
            condition_A="Groovy",
            condition_B="Classic",
            n_within_trial_segs=1,
            tail=0,
            n_permutations=1000,
            use_spatio_temporal=True,
            data_root=ROOT,
            n_subjects=N_SUBJECTS,
            output_dir=OUTPUT,
            hypothesis_label="H3_beta_FC_Groovy_vs_Classic",
        ),

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
            data_root=ROOT,
            n_subjects=N_SUBJECTS,
            output_dir=OUTPUT,
            hypothesis_label="H1_CMC_Flexor_beta_Happy_vs_Silence",
        ),

        # ── H1: CMC Extensor beta, Classic vs Silence ─────────────────────────
        # Covers CMC_Extensor_max_beta + CMC_Extensor_mean_beta (Level 1 Classic effect)
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
            data_root=ROOT,
            n_subjects=N_SUBJECTS,
            output_dir=OUTPUT,
            hypothesis_label="H1_CMC_Extensor_beta_Classic_vs_Silence",
        ),

        # ── H1: CMC Extensor beta, Happy vs Classic ───────────────────────────
        # Perceived Category Happy significant at Levels 2+3
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
            data_root=ROOT,
            n_subjects=N_SUBJECTS,
            output_dir=OUTPUT,
            hypothesis_label="H1_CMC_Extensor_beta_Happy_vs_Classic",
        ),

        # ── H1: CMC Extensor beta, Groovy vs Classic ──────────────────────────
        # Perceived Category Groovy significant at Level 2
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
            data_root=ROOT,
            n_subjects=N_SUBJECTS,
            output_dir=OUTPUT,
            hypothesis_label="H1_CMC_Extensor_beta_Groovy_vs_Classic",
        ),

        # ── H1: CMC Extensor gamma, Happy vs Silence ──────────────────────────
        # Covers CMC_Extensor_max_gamma (Level 1 Happy effect)
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
            data_root=ROOT,
            n_subjects=N_SUBJECTS,
            output_dir=OUTPUT,
            hypothesis_label="H1_CMC_Extensor_gamma_Happy_vs_Silence",
        ),

        # ── H1: CMC Flexor gamma, Happy vs Silence ────────────────────────────
        # Covers CMC_Flexor_max_gamma + CMC_Flexor_mean_gamma (Level 1 Happy effect)
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
            data_root=ROOT,
            n_subjects=N_SUBJECTS,
            output_dir=OUTPUT,
            hypothesis_label="H1_CMC_Flexor_gamma_Happy_vs_Silence",
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
            n_subjects=N_SUBJECTS,
            output_dir=OUTPUT,
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
            n_subjects=N_SUBJECTS,
            output_dir=OUTPUT,
            hypothesis_label="H5_gamma_Global_Happy_vs_Silence",
        ),
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # RUN
    # ─────────────────────────────────────────────────────────────────────────
    results = run_batch(CONTRASTS)
