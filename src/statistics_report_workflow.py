"""
Generates a human-readable Markdown report from the six statistical output
frames produced by the omnibus + post-hoc pipeline.

Structure:
  0.   Overview table         — significant + powered effects per hypothesis
  Per Hypothesis × DV block:
  I.   What is the finding?   (significant effects, Cohen's d, level, model)
  I.b  Cross-resolution       (stability across 1/5/20-seg)
  II.  Can you trust it?      (power, Cook's D / DFBETA flags)
  III. Where and when?        (CBPA spatio-temporal clusters)
  IV.  Population heterog.    (responder rates, subject CVs)
  V.   Model diagnostics      (one row per model, concise table)
"""


from __future__ import annotations
from pathlib import Path

from src.pipeline.statistical_reporting import *
import src.utils.file_management as filemgmt

# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    def _load_csv(directory: Path, suffixes: list[str]) -> pd.DataFrame:
        """
        Attempt to load the most recent CSV matching suffixes from directory.

        Returns an empty DataFrame and prints a warning if no matching file
        is found or the read fails, so the report can still be generated with
        validate_frames() emitting the appropriate warnings.

        Parameters
        ----------
        directory : Path
            Directory passed to filemgmt.most_recent_file.
        suffixes : list[str]
            Filename keyword filters passed to filemgmt.most_recent_file.
        """
        try:
            path = filemgmt.most_recent_file(directory, ".csv", suffixes)
            return pd.read_csv(path)
        except (ValueError, FileNotFoundError, Exception) as e:
            print(f"  ⚠️  Could not load {suffixes} from {directory}: {type(e).__name__}: {e}")
            return pd.DataFrame()



    ########## RESEARCH QUESTION A ##########
    from src.statistics_RQ_A_omnibus_testing_workflow import fetch_level_definitions

    RQ_A_DIR             = Path().resolve().parent / "output" / "statistics_RQ_A"
    RQ_A_OMNIBUS_RESULTS = RQ_A_DIR / "omnibus_testing"
    RQ_A_POST_HOC_RESULTS = RQ_A_DIR / "post_hoc_testing"
    RQ_A_REPORT_DIR      = RQ_A_DIR / "report"

    generate_statistical_report(
        omnibus_results_frame        = _load_csv(RQ_A_OMNIBUS_RESULTS,  ["All Time Resolutions Results"]),
        omnibus_diagnostics_frame    = _load_csv(RQ_A_OMNIBUS_RESULTS,  ["All Time Resolutions Diagnostics"]),
        power_analysis_results_frame = _load_csv(RQ_A_OMNIBUS_RESULTS,  ["Power Analysis MDE Summary"]),
        influence_measures_frame     = _load_csv(RQ_A_OMNIBUS_RESULTS,  ["Influence Analysis Combined"]),
        subject_heterogeneity_frame  = _load_csv(RQ_A_OMNIBUS_RESULTS,  ["Subject Effect Summary Combined"]),
        cbpa_results_frame           = _load_csv(RQ_A_POST_HOC_RESULTS, ["CBPA Combined Cluster Summary"]),
        mi_summary_frame             = _load_csv(RQ_A_POST_HOC_RESULTS, ["MI Summary"]),
        subject_clusters_frame       = _load_csv(RQ_A_POST_HOC_RESULTS, ["Subject Clusters"]),
        output_dir           = RQ_A_REPORT_DIR,
        file_identifier_suffix='RQ_A',

        primary_n_segments   = 1,
        resolution_segments  = [1, 5, 10],

        alpha_adjusted       = 0.05,
        include_ols          = False,
        target_power=0.80,

        # FDR Correction for multiple comparisons:
        fdr_levels_to_correct= [2, 3],
        fdr_group_by_dv=True,

        level_definitions=fetch_level_definitions(multi_segments_per_trial=True),

        hypothesis_groups=[
            {
                "label": "H1 – CMC Music Effects (all DVs)",
                "hypotheses": [h for h in _load_csv(RQ_A_OMNIBUS_RESULTS,  ["All Time Resolutions Results"])["Hypothesis"].unique()
                               if h.startswith("H1")],
            },
            {
                "label": "H2-H5: EEG PSD Effects (all DVs)",
                "hypotheses": [h for h in _load_csv(RQ_A_OMNIBUS_RESULTS, ["All Time Resolutions Results"])["Hypothesis"].unique()
                               if (h.startswith("H2") or h.startswith("H3") or h.startswith("H4") or h.startswith("H5"))],
            },
            {
                "label": "MEDIATION VARS: Biomarkers (all DVs)",
                "hypotheses": [h for h in
                               _load_csv(RQ_A_OMNIBUS_RESULTS, ["All Time Resolutions Results"])["Hypothesis"].unique()
                               if
                               (h.startswith("MEDIATION"))],
            }
        ],
    )



    ########## RESEARCH QUESTION B ##########
    from src.statistics_RQ_B_omnibus_testing_workflow import fetch_accuracy_level_definitions

    RQ_B_DIR             = Path().resolve().parent / "output" / "statistics_RQ_B"
    RQ_B_OMNIBUS_RESULTS = RQ_B_DIR / "omnibus_testing"
    RQ_B_REPORT_DIR      = RQ_B_DIR / "report"

    # Subject heterogeneity: not applicable (no categorical condition contrasts in RQ2)
    # CBPA: not applicable (non-spatial pipeline only)
    generate_statistical_report(
        omnibus_results_frame        = _load_csv(RQ_B_OMNIBUS_RESULTS, ["RQ2 All Time Resolutions Results"]),
        omnibus_diagnostics_frame    = _load_csv(RQ_B_OMNIBUS_RESULTS, ["RQ2 All Time Resolutions Diagnostics"]),
        power_analysis_results_frame = _load_csv(RQ_B_OMNIBUS_RESULTS, ["Power Analysis MDE Summary"]),
        influence_measures_frame     = _load_csv(RQ_B_OMNIBUS_RESULTS, ["Influence Analysis Combined"]),
        subject_heterogeneity_frame  = pd.DataFrame(),
        cbpa_results_frame           = pd.DataFrame(),
        mi_summary_frame             = pd.DataFrame(),
        subject_clusters_frame       = pd.DataFrame(),

        output_dir           = RQ_B_REPORT_DIR,
        file_identifier_suffix='RQ_B',

        primary_n_segments   = 5,
        resolution_segments  = [2, 5, 10],

        alpha_adjusted       = 0.05,
        include_ols          = False,
        target_power         = 0.80,

        level_definitions    = fetch_accuracy_level_definitions(
            include_emg_psd=True, multi_segments_per_trial=True
        ),
    )
