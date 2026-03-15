"""
statistical_report.py
─────────────────────────────────────────────────────────────────────────────
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

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import src.utils.file_management as filemgmt


# ══════════════════════════════════════════════════════════════════════════════
#  PER-CALL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field

@dataclass(frozen=True)
class _ReportConfig:
    """
    Immutable settings bundle threaded through every section builder.

    Parameters
    ----------
    primary_n_segments : int
        Canonical time resolution for Sections I–IV.
    resolution_segments : list[int]
        All resolutions to check in cross-resolution stability (Section I.b).
    alpha_adjusted : float
        Family-wise significance threshold applied to p_value_adjusted.
    include_ols : bool
        Whether OLS rows appear alongside LME in Sections I, I.b, V.
    cfg.target_power : float
        Minimum Power_at_Observed_Effect to label an effect "well-powered".
    """
    primary_n_segments:  int        = 1
    resolution_segments: list[int]  = field(default_factory=lambda: [1, 5, 20])
    alpha_adjusted:      float      = 0.05
    include_ols:         bool       = False
    target_power:        float      = 0.80

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION VI — PREDICTOR QUALITY SUMMARY  (optional, hypothesis-group-level)
# ══════════════════════════════════════════════════════════════════════════════

def _tl(rating: str) -> str:
    """Map rating string to display symbol."""
    return {
        "green":  "🟢",
        "yellow": "🟡",
        "red":    "🔴",
        "na":     "—",
    }.get(rating, "—")



def _rate_power(
    raw_params: set[str],
    dvs: set[str],
    power: pd.DataFrame,
    cfg: _ReportConfig,
) -> str:
    """
    Rate power for a canonical predictor across a set of DVs.

    Parameters
    ----------
    raw_params : set[str]
        All raw Parameter strings that map to this canonical predictor.
    dvs : set[str]
        Dependent variables belonging to the hypothesis group.
    power : pd.DataFrame
        Full power analysis frame.
    cfg : _ReportConfig

    Returns
    -------
    str
        'green' | 'yellow' | 'red'
    """
    if power.empty:
        return "red"
    rows = power[
        power["Parameter"].isin(raw_params) & power["Dependent_Variable"].isin(dvs)
    ]
    if rows.empty:
        return "red"
    n_well  = int((rows["Power_at_Observed_Effect"] >= cfg.target_power).sum())
    n_total = len(rows)
    if n_well == n_total:
        return "green"
    if n_well > 0:
        return "yellow"
    return "red"


def _rate_relevance(
    raw_params: set[str],
    dvs: set[str],
    res: pd.DataFrame,
    cfg: _ReportConfig,
) -> str:
    """
    Rate relevance as max |Cohen's d| across significant primary-resolution rows.

    Returns
    -------
    str
        'green' (≥ 0.5) | 'yellow' (≥ 0.2) | 'red' (< 0.2 or no sig. row)
    """
    rows = res[
        (res["N. Segments"] == cfg.primary_n_segments)
        & (res["Model_Type"] == "LME")
        & (res["p_value_adjusted"] < cfg.alpha_adjusted)
        & res["Parameter"].isin(raw_params)
        & res["Dependent_Variable"].isin(dvs)
    ]
    if rows.empty:
        return "red"
    max_d = rows["Cohen_d"].abs().max()
    if max_d >= 0.5:
        return "green"
    if max_d >= 0.2:
        return "yellow"
    return "red"


def _rate_generalization(
    raw_params: set[str],
    dvs: set[str],
    influence: pd.DataFrame,
    cfg: _ReportConfig,
) -> str:
    """
    Rate generalization: driven by single subjects?

    Returns 'green' | 'yellow' | 'red' | 'na'.
    'na' is returned when no influence data is available at the primary
    resolution, so the caller can render '—' rather than a traffic light.
    """
    if influence.empty:
        return "na"
    rows = influence[
        (influence["N_Segments"] == cfg.primary_n_segments)
        & influence["Parameter"].isin(raw_params)
        & influence["Dependent_Variable"].isin(dvs)
    ]
    if rows.empty:
        return "na"   # ← was "yellow"
    if rows["DFBETA_Flagged"].any():
        return "red"
    if rows["CooksD_Flagged"].any():
        return "yellow"
    return "green"



def _rate_time_consistency(
    raw_params: set[str],
    dvs: set[str],
    res: pd.DataFrame,
    cfg: _ReportConfig,
) -> str:
    """
    Rate time consistency: significant at primary resolution → how stable
    across all resolution_segments?

    For each (param, level, dv) triple that is significant at primary_n_segments,
    count how many cfg.resolution_segments also show significance.

    🟢 all triples robust across all resolutions
    🟡 mixed (some robust, some partial or resolution-specific)
    🔴 all resolution-specific (only at primary)

    Returns
    -------
    str
        'green' | 'yellow' | 'red'
    """
    primary_sig = res[
        (res["N. Segments"] == cfg.primary_n_segments)
        & (res["Model_Type"] == "LME")
        & (res["p_value_adjusted"] < cfg.alpha_adjusted)
        & res["Parameter"].isin(raw_params)
        & res["Dependent_Variable"].isin(dvs)
    ][["Parameter", "Comparison_Level", "Dependent_Variable"]].drop_duplicates()

    if primary_sig.empty:
        return "red"

    verdicts: list[str] = []
    for _, pr in primary_sig.iterrows():
        level_idx = _level_int(pr["Comparison_Level"])
        sig_at = []
        for n_seg in cfg.resolution_segments:
            match = res[
                (res["N. Segments"] == n_seg)
                & (res["Model_Type"] == "LME")
                & (res["Parameter"] == pr["Parameter"])
                & (res["Comparison_Level"].apply(_level_int) == level_idx)
                & (res["Dependent_Variable"] == pr["Dependent_Variable"])
            ]
            if not match.empty and match.iloc[0]["p_value_adjusted"] < cfg.alpha_adjusted:
                sig_at.append(n_seg)
        n_avail = len([s for s in cfg.resolution_segments
                       if not res[(res["N. Segments"] == s)
                                  & (res["Dependent_Variable"] == pr["Dependent_Variable"])].empty])
        if len(sig_at) == n_avail:
            verdicts.append("green")
        elif len(sig_at) > 1:
            verdicts.append("yellow")
        else:
            verdicts.append("red")

    if all(v == "green" for v in verdicts):
        return "green"
    if all(v == "red" for v in verdicts):
        return "red"
    return "yellow"


def _section_predictor_quality(
    hypothesis_groups: list[dict],
    res: pd.DataFrame,
    power: pd.DataFrame,
    influence: pd.DataFrame,
    cfg: _ReportConfig,
) -> str:
    """
    Section VI — Predictor Quality Summary across hypothesis groups.

    Produces one Markdown table per group, with one row per canonical predictor
    (significant in ≥ 1 DV within the group at the primary resolution).
    Each row receives four traffic-light ratings:
    Power, Relevance, Generalization, Time Consistency.

    Parameters
    ----------
    hypothesis_groups : list[dict]
        Each dict must have:
          - "label"       : str  — section heading
          - "hypotheses"  : list[str] — values to match against res["Hypothesis"]
    res : pd.DataFrame
        Full omnibus results frame.
    power : pd.DataFrame
        Full power analysis frame.
    influence : pd.DataFrame
        Full influence measures frame.
    cfg : _ReportConfig

    Returns
    -------
    str
        Markdown block.
    """
    lines = ["## Predictor Quality Summary\n"]
    lines.append(
        "> Traffic-light ratings per canonical predictor aggregated across "
        "all DVs in each hypothesis group.  \n"
        "> **Power**: ≥ target across all instances · "
        "**Relevance**: max |d| ≥ 0.2 · "
        "**Generalization**: no single-subject DFBETA drive · "
        "**Time Consistency**: robust across resolutions  \n"
        "> 🟢 Pass · 🟡 Partial/Mixed · 🔴 Fail · — Not assessable (no influence data at primary resolution)\n"
    )

    for group in hypothesis_groups:
        label      = group.get("label", "Unnamed Group")
        hyp_filter = group.get("hypotheses", [])

        lines.append(f"### {label}\n")

        # DVs belonging to this group
        group_res = res[res["Hypothesis"].isin(hyp_filter)]
        dvs = set(group_res["Dependent_Variable"].unique())

        if group_res.empty:
            lines.append("> ⚠️  No results found for the specified hypotheses.\n")
            continue

        # Collect all canonical predictors significant at primary resolution
        sig_primary = group_res[
            (group_res["N. Segments"] == cfg.primary_n_segments)
            & (group_res["Model_Type"] == "LME")
            & (group_res["p_value_adjusted"] < cfg.alpha_adjusted)
            & group_res["Parameter"].apply(_is_real_param)
        ].copy()

        if sig_primary.empty:
            lines.append(
                f"> No significant effects in this group at "
                f"{cfg.primary_n_segments}-seg primary resolution.\n"
            )
            continue

        # Map cleaned → set of raw parameter strings
        sig_primary["_canonical"] = sig_primary["Parameter"].apply(_clean_param)
        canonical_groups = (
            sig_primary.groupby("_canonical")["Parameter"]
            .apply(set)
            .to_dict()
        )

        # Best Cohen's d per canonical predictor (for sorting)
        best_d = (
            sig_primary.groupby("_canonical")["Cohen_d"]
            .apply(lambda s: s.abs().max())
            .to_dict()
        )

        sorted_predictors = sorted(
            canonical_groups.keys(),
            key=lambda c: best_d.get(c, 0),
            reverse=True,
        )

        lines.append(
            "| Predictor | Best \\|d\\| | Power | Relevance | "
            "Generalization | Time Consistency |"
        )
        lines.append("|---|---|---|---|---|---|")

        for canonical in sorted_predictors:
            raw_params = canonical_groups[canonical]
            d_val      = best_d.get(canonical, float("nan"))

            pwr  = _tl(_rate_power(raw_params, dvs, power, cfg))
            rel  = _tl(_rate_relevance(raw_params, dvs, group_res, cfg))
            gen  = _tl(_rate_generalization(raw_params, dvs, influence, cfg))
            tc   = _tl(_rate_time_consistency(raw_params, dvs, group_res, cfg))

            lines.append(
                f"| {canonical} "
                f"| {_fmt_float(d_val, 3)} "
                f"| {pwr} | {rel} | {gen} | {tc} |"
            )

        lines.append("")

    return "\n".join(lines)



# ══════════════════════════════════════════════════════════════════════════════
#  SMALL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _active_models(cfg: _ReportConfig) -> list[str]:
    """Return the model types to show, respecting cfg.include_ols."""
    return ["LME", "OLS"] if cfg.include_ols else ["LME"]


def _cohens_d_label(d: float) -> str:
    """Return a textual magnitude label for an absolute Cohen's d value."""
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"


def _level_int(level_str: str | int) -> int | None:
    """
    Extract the integer level index from strings like
    'Level 1 (Category or Silence + ...)' or bare integers.
    """
    if isinstance(level_str, (int, float)):
        return int(level_str)
    m = re.search(r"Level\s+(\d+)", str(level_str), re.IGNORECASE)
    return int(m.group(1)) if m else None


def _short_level(level_str: str) -> str:
    """Shorten 'Level 1 (long description...)' → 'Level 1'."""
    return re.sub(r"\s*\(.*\)", "", str(level_str)).strip()


def _hypothesis_prefix(hyp_str: str) -> str:
    """Extract 'H1', 'H2', ... from omnibus or CBPA label formats."""
    m = re.match(r"(H\d+)", str(hyp_str), re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _fmt_p(p: float) -> str:
    if pd.isna(p):   return "—"
    if p < 0.001:    return f"{p:.2e}"
    return f"{p:.4f}"


def _fmt_float(x, decimals: int = 4) -> str:
    if pd.isna(x):   return "—"
    return f"{x:.{decimals}f}"


def _stars(p: float) -> str:
    if pd.isna(p):   return ""
    if p < 0.001:    return "***"
    if p < 0.01:     return "**"
    if p < 0.05:     return "*"
    return ""


def _clean_param(param: str) -> str:
    """Shorten statsmodels parameter name for readable tables."""
    param = re.sub(r"C\(Q\('(.+?)'\)\)\[T\.(.+?)\]", r"\1 = \2", param)
    param = re.sub(r"Q\('(.+?)'\)",                   r"\1",       param)
    param = re.sub(r":(\w)",                           r" × \1",    param)
    return param


_SENTINEL_PARAMS = {"Intercept", "__residual_std__", "__re_std__"}


def _is_real_param(p: str) -> bool:
    return p not in _SENTINEL_PARAMS and not p.startswith("Intercept")


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "omnibus_results": {
        "Hypothesis", "Dependent_Variable", "Model_Type", "Comparison_Level",
        "Parameter", "Coefficient", "SE_adjusted", "p_value_adjusted",
        "Cohen_d", "N. Segments",
    },
    "omnibus_diagnostics": {
        "Hypothesis", "Dependent_Variable", "Model_Type", "Comparison_Level",
        "N_Observations", "Shapiro_p", "Shapiro_Violated", "Lag1_Autocorr",
        "Design_Effect", "SE_Inflation", "R_squared", "R_squared_adj",
        "AIC", "BIC", "LogLik", "R_squared_marginal", "R_squared_conditional",
        "ICC", "N. Segments",
    },
    "power_analysis": {
        "Dependent_Variable", "Comparison_Level", "N_Segments",
        "Parameter", "Power_at_Observed_Effect", "MDE_at_80%_power", "Interpretation",
    },
    "influence_measures": {
        "Dependent_Variable", "Comparison_Level", "N_Segments",
        "Parameter", "Subject_ID", "DFBETA", "DFBETA_Flagged",
        "CooksD", "CooksD_Flagged",
    },
    "subject_heterogeneity": {
        "Hypothesis", "Dependent_Variable", "Subject ID", "Comparison_Level",
        "Condition_Variable", "Condition", "Condition_Mean",
        "Raw_Contrast", "Normalised_Contrast", "Subject_CV", "Responder_Flag",
    },
    "cbpa_results": {
        "hypothesis", "modality", "freq_band", "condition_column",
        "condition_A", "condition_B", "cluster_index", "p_value",
        "significant", "peak_t", "t_thresh", "n_channels", "channels",
    },
}

_CBPA_DOMAIN_ALTERNATIVES: tuple[str, str] = ("time_start_s", "phase_start_deg")


def validate_frames(frames: dict[str, pd.DataFrame]) -> list[str]:
    """
    Check required columns are present in each frame.

    Parameters
    ----------
    frames : dict
        Mapping of frame name → DataFrame.

    Returns
    -------
    list[str]
        Warning strings (empty list = all clear).
    """
    out: list[str] = []
    for name, required in _REQUIRED_COLUMNS.items():
        if name not in frames:
            out.append(f"[MISSING FRAME] '{name}' was not supplied.")
            continue
        df = frames[name]
        if df is None or df.empty:
            out.append(f"[EMPTY FRAME] '{name}' is empty.")
            continue
        missing = required - set(df.columns)
        if missing:
            out.append(f"[MISSING COLUMNS] '{name}': {sorted(missing)}")

    if "cbpa_results" in frames:
        df = frames["cbpa_results"]
        if df is not None and not df.empty:
            time_col, phase_col = _CBPA_DOMAIN_ALTERNATIVES
            has_time = time_col in df.columns
            has_phase = phase_col in df.columns
            if not has_time and not has_phase:
                out.append(
                    f"[MISSING COLUMNS] 'cbpa_results': neither temporal domain "
                    f"('{time_col}') nor phase domain ('{phase_col}') columns are present."
                )

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE COVERAGE DIAGNOSTICS  (printed to stdout, not in report)
# ══════════════════════════════════════════════════════════════════════════════

def _print_pipeline_recommendations(
    res: pd.DataFrame,
    power: pd.DataFrame,
    influence: pd.DataFrame,
    cbpa: pd.DataFrame,
    alpha: float,
    primary_n_segments: int,
    resolution_segments: list[int],
) -> None:
    sep = "─" * 72

    # Guard: nothing meaningful to check if the results frame itself is empty
    if res.empty:
        print(f"\n{sep}")
        print("⚠️  Omnibus results frame is empty — pipeline recommendations skipped.")
        print(sep)
        return

    sig = res[
        (res["N. Segments"] == primary_n_segments)
        & (res["Model_Type"] == "LME")
        & (res["p_value_adjusted"] < alpha)
        & res["Parameter"].apply(_is_real_param)
    ].copy()

    if sig.empty:
        print(f"\n{sep}")
        print("✅  No significant effects found — no pipeline gaps to report.")
        print(sep)
        return

    issues: list[str] = []

    # ── 1. Power analysis gaps ────────────────────────────────────────────────
    if not power.empty:
        power_covered = set(zip(power["Dependent_Variable"], power["Parameter"]))
        missing_power_dvs: dict[str, list[str]] = {}
        for _, r in sig.iterrows():
            if (r["Dependent_Variable"], r["Parameter"]) not in power_covered:
                missing_power_dvs.setdefault(r["Dependent_Variable"], []).append(
                    _clean_param(r["Parameter"])
                )
        if missing_power_dvs:
            issues.append(
                f"⚠️  POWER ANALYSIS missing for {len(missing_power_dvs)} DV(s) "
                f"({sum(len(v) for v in missing_power_dvs.values())} significant parameters):"
            )
            for dv, params in sorted(missing_power_dvs.items()):
                issues.append(f"   {dv}  →  {len(params)} parameter(s):")
                for p in params:
                    issues.append(f"      · {p}")
            issues.append(
                "   → Re-run the power analysis pipeline for all DVs and segment counts."
            )
    else:
        issues.append("⚠️  POWER ANALYSIS frame is empty — coverage check skipped.")

    # ── 2. Influence measure gaps ─────────────────────────────────────────────
    if not influence.empty:
        inf_at_primary = influence[influence["N_Segments"] == primary_n_segments]
        dvs_with_inf   = set(inf_at_primary["Dependent_Variable"].unique())
        dvs_need_inf   = set(sig["Dependent_Variable"].unique())
        missing_inf    = sorted(dvs_need_inf - dvs_with_inf)
        if missing_inf:
            issues.append(
                f"\n⚠️  INFLUENCE MEASURES missing at {primary_n_segments}-seg "
                f"for {len(missing_inf)} DV(s):"
            )
            for dv in missing_inf:
                issues.append(f"   {dv}")
            other_segs = sorted(influence["N_Segments"].dropna().unique().astype(int))
            other_segs = [s for s in other_segs if s != primary_n_segments]
            if other_segs:
                issues.append(
                    f"   → Influence data found at segment(s) {other_segs} but NOT at "
                    f"{primary_n_segments}-seg. Either re-run influence analysis at "
                    f"{primary_n_segments}-seg, or change primary_n_segments to match."
                )
            else:
                issues.append(
                    f"   → Re-run the influence analysis pipeline at {primary_n_segments}-seg."
                )
    else:
        issues.append("\n⚠️  INFLUENCE MEASURES frame is empty — coverage check skipped.")

    # ── 3. CBPA gaps ──────────────────────────────────────────────────────────
    if not cbpa.empty:
        missing_cbpa_dvs: dict[str, str] = {}
        for _, r in sig[["Hypothesis", "Dependent_Variable"]].drop_duplicates().iterrows():
            hyp_p = _hypothesis_prefix(r["Hypothesis"])
            dv    = r["Dependent_Variable"]
            cbpa_hyp = cbpa[cbpa["hypothesis"].apply(_hypothesis_prefix) == hyp_p]
            if cbpa_hyp.empty:
                missing_cbpa_dvs[dv] = f"no CBPA data at all for {hyp_p}"
                continue
            modality_key, band_key, muscle_key = _dv_to_cbpa_keys(dv)
            mask = pd.Series(True, index=cbpa_hyp.index)
            if modality_key is not None:
                mask &= cbpa_hyp["modality"].str.upper() == modality_key.upper()
            if band_key is not None:
                mask &= cbpa_hyp["freq_band"].str.lower() == band_key.lower()
            if muscle_key is not None:
                mask &= cbpa_hyp["hypothesis"].str.contains(muscle_key, case=False)
            if not mask.any():
                keys = f"modality={modality_key}, band={band_key}, muscle={muscle_key}"
                missing_cbpa_dvs[dv] = f"no rows match {keys} within {hyp_p}"
        if missing_cbpa_dvs:
            issues.append(
                f"\n⚠️  CBPA missing for {len(missing_cbpa_dvs)} DV(s) with significant effects:"
            )
            for dv, reason in sorted(missing_cbpa_dvs.items()):
                issues.append(f"   {dv}  — {reason}")
            issues.append("   → Run the CBPA pipeline for the listed DVs / hypotheses.")
    else:
        issues.append("\n⚠️  CBPA frame is empty — coverage check skipped.")

    # ── 4. Cross-resolution gap ───────────────────────────────────────────────
    target_segs = [s for s in resolution_segments if s != primary_n_segments]
    for target_seg in target_segs:
        hits = sum(
            1 for _, r in sig.iterrows()
            if not res[
                (res["N. Segments"] == target_seg)
                & (res["Model_Type"] == "LME")
                & (res["Parameter"] == r["Parameter"])
                & (res["Comparison_Level"].apply(_level_int) == _level_int(r["Comparison_Level"]))
                & (res["Dependent_Variable"] == r["Dependent_Variable"])
            ].empty
        )
        if hits == 0:
            issues.append(
                f"\n⚠️  CROSS-RESOLUTION ({target_seg}-seg): 0 / {len(sig)} significant "
                f"parameters have matching rows at {target_seg}-seg. "
                f"All cross-resolution verdicts will read 'Resolution-specific'."
            )
            issues.append(
                f"   → Verify that the omnibus results export includes all "
                f"N_Segments values, or check for parameter-name / Comparison_Level "
                f"format differences between segment counts."
            )

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    if issues:
        print("🔍  PIPELINE COVERAGE RECOMMENDATIONS")
        print(sep)
        for line in issues:
            print(line)
    else:
        print("✅  All significant effects have power, influence, and CBPA coverage.")
    print(sep + "\n")




# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 0 — OVERVIEW TABLE
# ══════════════════════════════════════════════════════════════════════════════

def _section_level_definitions(level_defs: list[dict]) -> str:
    """
    Overview subsection: human-readable table of every comparison level.

    Renders the list returned by fetch_level_definitions() into a Markdown
    table so the report is self-contained without the pipeline source code.

    Parameters
    ----------
    level_defs : list[dict]
        Each dict must have the keys produced by fetch_level_definitions():
        df_filter, condition_vars, reference_categories,
        explanatory_vars, moderation_pairs.

    Returns
    -------
    str
        Markdown block (no trailing newline).
    """
    lines = [
        "### Comparison Levels\n",
        "| Level | Data subset | Condition variable(s) | Type | Reference | "
        "Covariates | Moderation pairs |",
        "|---|---|---|---|---|---|---|",
    ]

    for idx, d in enumerate(level_defs):
        # Infer data-subset label from the filter callable
        subset = "All trials" if d.get("df_filter") is None else "Music trials only"

        # Condition variables — may be multiple
        cond_vars   = d.get("condition_vars", {})
        ref_cats    = d.get("reference_categories", {})
        cond_parts, type_parts, ref_parts = [], [], []
        for var, vtype in cond_vars.items():
            cond_parts.append(f"`{var}`")
            type_parts.append(vtype)
            ref_parts.append(f"`{ref_cats.get(var, '—')}`")

        cond_str = "<br>".join(cond_parts)
        type_str = "<br>".join(type_parts)
        ref_str  = "<br>".join(ref_parts)

        # Covariates — strip Trial/Segment ID bookkeeping vars for readability
        skip = {"Trial ID", "Segment ID"}
        covariates = [
            f"`{v}`" for v in d.get("explanatory_vars", []) if v not in skip
        ]
        cov_str = ", ".join(covariates) if covariates else "—"

        # Moderation pairs  →  "A × B"
        mod_pairs = d.get("moderation_pairs", [])
        mod_str   = "<br>".join(f"`{a}` × `{b}`" for a, b in mod_pairs) or "—"

        lines.append(
            f"| **Level {idx}** | {subset} | {cond_str} | {type_str} "
            f"| {ref_str} | {cov_str} | {mod_str} |"
        )

    lines.append(
        "\n> Trial ID and Segment ID are included as explanatory variables "
        "in all levels (exact set depends on `multi_segments_per_trial`). "
        "Ordinal variables are treated as continuous.\n"
    )
    return "\n".join(lines)


def _section_overview_table(
    res: pd.DataFrame,
    power: pd.DataFrame,
    cfg: _ReportConfig,
) -> str:
    lines = [
        "## Overview\n",
        f"*Counts at primary resolution ({cfg.primary_n_segments}-seg), "
        f"models: {', '.join(_active_models(cfg))}, "
        f"α = {cfg.alpha_adjusted}, target power = {cfg.target_power:.0%}*\n",
        "| Hypothesis | DV | Sig. effects | Well-powered | Underpowered | No power data |",
        "|---|---|---|---|---|---|",
    ]

    res_primary = res[
        (res["N. Segments"] == cfg.primary_n_segments)
        & (res["Model_Type"].isin(_active_models(cfg)))
        & res["Parameter"].apply(_is_real_param)
    ]

    hyp_dv_pairs = (
        res[["Hypothesis", "Dependent_Variable"]]
        .drop_duplicates()
        .sort_values(["Hypothesis", "Dependent_Variable"])
    )

    for _, row in hyp_dv_pairs.iterrows():
        hyp = row["Hypothesis"]
        dv  = row["Dependent_Variable"]

        sig_rows = res_primary[
            (res_primary["Hypothesis"] == hyp)
            & (res_primary["Dependent_Variable"] == dv)
            & (res_primary["p_value_adjusted"] < cfg.alpha_adjusted)
        ]
        n_sig = len(sig_rows)

        if n_sig == 0:
            lines.append(f"| {hyp} | `{dv}` | 0 | — | — | — |")
            continue

        # Power frame may be empty if the pipeline has not been run yet
        if power.empty:
            lines.append(
                f"| {hyp} | `{dv}` | **{n_sig}** | — | — | — {n_sig} |"
            )
            continue

        n_well_powered = 0
        n_underpowered = 0
        n_no_data      = 0

        for _, sr in sig_rows.iterrows():
            pwr_match = power[
                (power["Dependent_Variable"] == dv)
                & (power["Parameter"] == sr["Parameter"])
            ]
            if pwr_match.empty:
                n_no_data += 1
            else:
                pwr_val = pwr_match.iloc[0]["Power_at_Observed_Effect"]
                if pd.notna(pwr_val) and pwr_val >= cfg.target_power:
                    n_well_powered += 1
                else:
                    n_underpowered += 1

        lines.append(
            f"| {hyp} | `{dv}` "
            f"| **{n_sig}** "
            f"| ✅ {n_well_powered} "
            f"| ⚠️ {n_underpowered} "
            f"| — {n_no_data} |"
        )

    lines.append(
        f"\n> **Sig. effects** = adjusted p < α at primary resolution. "
        f"**Well-powered** = Power_at_Observed_Effect ≥ {cfg.target_power:.0%}. "
        "**No power data** = parameter not covered by power analysis.\n"
    )
    lines.append("---\n")
    return "\n".join(lines)



# ══════════════════════════════════════════════════════════════════════════════
#  SECTION I — FINDINGS
# ══════════════════════════════════════════════════════════════════════════════

def _section_findings(
    hyp: str,
    dv: str,
    res: pd.DataFrame,
    cfg: _ReportConfig,
) -> str:
    """
    Section I — What is the finding?

    Shows all significant fixed-effect parameters at cfg.primary_n_segments,
    for active model types, ranked by |Cohen's d|.
    Includes Comparison Level and Model Type columns to disambiguate
    duplicate parameter names across levels.

    Parameters
    ----------
    hyp : str
    dv : str
    res : pd.DataFrame
        Omnibus results, already filtered to this hyp × dv.
    alpha : float
    """
    alpha = cfg.alpha_adjusted

    lines = ["### I. Finding\n"]

    mask = (
        (res["N. Segments"] == cfg.primary_n_segments)
        & (res["Model_Type"].isin(_active_models(cfg)))
        & res["Parameter"].apply(_is_real_param)
    )
    sub = res[mask].copy()
    sig = sub[sub["p_value_adjusted"] < alpha].copy()

    model_label = "/".join(_active_models(cfg))

    if sig.empty:
        lines.append(
            f"> **No significant effects** found for `{dv}` at "
            f"adjusted α = {alpha} ({model_label}, {cfg.primary_n_segments}-seg).\n"
        )
        near = sub[sub["p_value_adjusted"] < 0.10].sort_values("p_value_adjusted")
        if not near.empty:
            lines.append("**Near-significant (0.05 < p < 0.10):**\n")
            lines.append("| Parameter | Level | Model | β | SE (adj) | p (adj) | Cohen's d |")
            lines.append("|---|---|---|---|---|---|---|")
            for _, r in near.iterrows():
                lines.append(
                    f"| {_clean_param(r['Parameter'])} "
                    f"| {_short_level(r['Comparison_Level'])} "
                    f"| {r['Model_Type']} "
                    f"| {_fmt_float(r['Coefficient'])} "
                    f"| {_fmt_float(r['SE_adjusted'])} "
                    f"| {_fmt_p(r['p_value_adjusted'])} "
                    f"| {_fmt_float(r.get('Cohen_d'))} |"
                )
            lines.append("")
        return "\n".join(lines)

    sig = sig.assign(abs_d=sig["Cohen_d"].abs()).sort_values("abs_d", ascending=False)

    lines.append(
        f"**{len(sig)} significant effect(s)** for `{dv}` "
        f"({model_label}, {cfg.primary_n_segments}-seg, adjusted α = {alpha}):\n"
    )
    lines.append(
        "| Parameter | Level | Model | β | SE (adj) | p (adj) | Cohen's d | Magnitude |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in sig.iterrows():
        d   = r.get("Cohen_d")
        mag = _cohens_d_label(d) if pd.notna(d) else "—"
        lines.append(
            f"| {_clean_param(r['Parameter'])}{_stars(r['p_value_adjusted'])} "
            f"| {_short_level(r['Comparison_Level'])} "
            f"| {r['Model_Type']} "
            f"| {_fmt_float(r['Coefficient'])} "
            f"| {_fmt_float(r['SE_adjusted'])} "
            f"| {_fmt_p(r['p_value_adjusted'])} "
            f"| {_fmt_float(d)} "
            f"| {mag} |"
        )
    lines.append("\n`*` p<0.05  `**` p<0.01  `***` p<0.001 (SE adjusted for autocorrelation)\n")

    all_segs = sorted(res["N. Segments"].dropna().unique())
    if len(all_segs) > 1:
        other = ", ".join(str(int(s)) for s in all_segs if s != cfg.primary_n_segments)
        lines.append(
            f"> **Time-resolution robustness:** primary = {cfg.primary_n_segments}-seg. "
            f"Also tested: {other}-seg — see cross-resolution table below.\n"
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION I.b — CROSS-RESOLUTION STABILITY
# ══════════════════════════════════════════════════════════════════════════════

def _section_cross_resolution(
    hyp: str,
    dv: str,
    res: pd.DataFrame,
    cfg: _ReportConfig,
) -> str:
    """
    Section I.b — Cross-resolution stability of significant parameters.

    For every parameter significant at cfg.primary_n_segments (LME, always),
    shows β / SE_adj / p_adj / Cohen_d at each resolution in
    cfg.resolution_segments.  One table per parameter.

    Always uses LME for the cross-resolution comparison regardless of
    INCLUDE_OLS, because the goal is temporal stability of the same model.

    Parameters
    ----------
    hyp : str
    dv : str
    res : pd.DataFrame
        Omnibus results, already filtered to this hyp × dv.
    alpha : float
    """
    alpha = cfg.alpha_adjusted

    lines = ["#### Cross-Resolution Stability (LME)\n"]

    # Significant parameters at the primary resolution — always LME
    primary_sig = res[
        (res["N. Segments"] == cfg.primary_n_segments)
        & (res["Model_Type"] == "LME")
        & (res["p_value_adjusted"] < alpha)
        & res["Parameter"].apply(_is_real_param)
    ][["Parameter", "Comparison_Level"]].drop_duplicates().values.tolist()

    if not primary_sig:
        lines.append(
            "> No significant LME parameters at the primary resolution — "
            "cross-resolution table omitted.\n"
        )
        return "\n".join(lines)

    available_segs = sorted(s for s in cfg.resolution_segments if s in res["N. Segments"].values)
    missing_segs   = [s for s in cfg.resolution_segments if s not in res["N. Segments"].values]
    if missing_segs:
        lines.append(
            f"> ⚠️  Resolutions absent from data: "
            f"{', '.join(str(s) for s in missing_segs)}-seg (omitted).\n"
        )
    if not available_segs:
        lines.append("> No resolution data available.\n")
        return "\n".join(lines)

    for param, comp_level in primary_sig:
        # Use level index for cross-segment matching — the full Comparison_Level
        # string changes between 1-seg and multi-seg due to Segment ID inclusion.
        level_idx = _level_int(comp_level)

        lines.append(
            f"**Parameter:** `{_clean_param(param)}` "
            f"| **Level:** {_short_level(comp_level)}\n"
        )
        lines.append("| Segs | β | SE (adj) | p (adj) | Cohen's d | Magnitude | Sig? |")
        lines.append("|---|---|---|---|---|---|---|")

        for n_seg in available_segs:
            row_match = res[
                (res["N. Segments"] == n_seg)
                & (res["Model_Type"] == "LME")
                & (res["Parameter"] == param)
                & (res["Comparison_Level"].apply(_level_int) == level_idx)
                ]
            if row_match.empty:
                lines.append(f"| {n_seg}-seg | — | — | — | — | — | — |")
                continue

            r              = row_match.iloc[0]
            p_adj          = r["p_value_adjusted"]
            cohen_d        = r.get("Cohen_d")
            mag            = _cohens_d_label(cohen_d) if pd.notna(cohen_d) else "—"
            sig_ico        = "✅" if (pd.notna(p_adj) and p_adj < alpha) else "⚠️"
            primary_marker = " ← primary" if n_seg == cfg.primary_n_segments else ""

            lines.append(
                f"| **{n_seg}-seg**{primary_marker} "
                f"| {_fmt_float(r['Coefficient'])} "
                f"| {_fmt_float(r['SE_adjusted'])} "
                f"| {_fmt_p(p_adj)}{_stars(p_adj)} "
                f"| {_fmt_float(cohen_d)} "
                f"| {mag} "
                f"| {sig_ico} |"
            )

        lines.append("")

        # sig_at builder:
        sig_at = []
        for n in available_segs:
            match = res[
                (res["N. Segments"] == n)
                & (res["Model_Type"] == "LME")
                & (res["Parameter"] == param)
                & (res["Comparison_Level"].apply(_level_int) == level_idx)
                ]
            if match.empty:
                continue
            p = match.iloc[0]["p_value_adjusted"]
            if pd.notna(p) and p < alpha:
                sig_at.append(n)

        n_avail = len(available_segs)
        n_sig   = len(sig_at)
        not_sig = [s for s in available_segs if s not in sig_at]

        # Exactly one branch executes — no fall-through
        if n_sig == n_avail:
            lines.append(
                f"> ✅ **Robust across all resolutions** "
                f"({', '.join(f'{s}-seg' for s in sig_at)}).\n"
            )
        elif n_sig > 1:
            lines.append(
                f"> ⚠️  **Partially robust** — significant at "
                f"{', '.join(f'{s}-seg' for s in sig_at)}, "
                f"not at {', '.join(f'{s}-seg' for s in not_sig)}. "
                f"Check sensitivity to temporal aggregation.\n"
            )
        elif n_sig == 1:
            lines.append(
                f"> ⚠️  **Resolution-specific** — significant only at "
                f"{sig_at[0]}-seg. Interpret with caution.\n"
            )
        else:
            lines.append(
                f"> ⚠️  **Temporal robustness unassessable** — p-values unavailable "
                f"at {', '.join(f'{s}-seg' for s in not_sig)}.\n"
            )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION II — TRUSTWORTHINESS
# ══════════════════════════════════════════════════════════════════════════════

def _section_trust(
    hyp: str,
    dv: str,
    res: pd.DataFrame,
    power: pd.DataFrame,
    influence: pd.DataFrame,
    cfg: _ReportConfig,
) -> str:
    """
    Section II — Can you trust the finding?

    Subsections:
    - 2a. Power analysis verdict per parameter at Level 1.
    - 2b. Influence analysis: per-subject summary table, with parameter-level
          detail collapsed behind a <details> toggle for flagged rows only.

    Parameters
    ----------
    hyp : str
    dv : str
    res : pd.DataFrame
        Omnibus results filtered to this hyp × dv.
    power : pd.DataFrame
        Full power analysis frame (may be empty if not yet run).
    influence : pd.DataFrame
        Full influence frame (may be empty if not yet run).
    cfg : _ReportConfig
    """
    lines = ["### II. Trustworthiness\n"]

    # ── 2a. Power ─────────────────────────────────────────────────────────────
    lines.append("#### Power Analysis\n")

    if power.empty:
        lines.append("> ⚠️  Power analysis frame is empty — not yet run for this pipeline.\n")
    else:
        pwr_sub = power[
            power["Dependent_Variable"] == dv
            ].copy()

        if pwr_sub.empty:
            lines.append("> ⚠️  No power analysis available for this DV.\n")
        else:
            # Sort by level index then parameter for readability
            pwr_sub["_lvl"] = pwr_sub["Comparison_Level"].apply(_level_int)
            pwr_sub = pwr_sub.sort_values(["_lvl", "Parameter"])

            lines.append(
                f"| Level | Parameter | Power @ effect | MDE ({cfg.target_power:.0%}) | Verdict |"
            )
            lines.append("|---|---|---|---|---|")
            for _, r in pwr_sub.iterrows():
                pwr_val = r.get("Power_at_Observed_Effect")
                mde = r.get("MDE_at_80%_power")
                well = pd.notna(pwr_val) and pwr_val >= cfg.target_power
                verdict = "✅ Well-powered" if well else "⚠️  Under-powered"
                lines.append(
                    f"| {_short_level(r['Comparison_Level'])} "
                    f"| {_clean_param(r['Parameter'])} "
                    f"| {_fmt_float(pwr_val, 3)} "
                    f"| {_fmt_float(mde, 6) if pd.notna(mde) else '—'} "
                    f"| {verdict} |"
                )
            lines.append("")

    # ── 2b. Influence ─────────────────────────────────────────────────────────
    lines.append("#### Influence Analysis (Cook's D & DFBETA)\n")

    if influence.empty:
        lines.append("> ⚠️  Influence measures frame is empty — not yet run for this pipeline.\n")
    else:
        inf_sub = influence[
            (influence["Dependent_Variable"] == dv)
            & (influence["N_Segments"] == cfg.primary_n_segments)
        ].copy()

        if inf_sub.empty:
            lines.append(
                f"> ⚠️  No influence measures for this DV at {cfg.primary_n_segments}-seg.\n"
            )
        else:
            n_subj         = inf_sub["Subject_ID"].nunique()
            n_params       = inf_sub["Parameter"].nunique()
            cooksd_flag    = inf_sub[inf_sub["CooksD_Flagged"] == True]
            dfbeta_flag    = inf_sub[inf_sub["DFBETA_Flagged"] == True]
            n_flagged_subj = len(
                set(cooksd_flag["Subject_ID"]) | set(dfbeta_flag["Subject_ID"])
            )

            # Overall verdict line
            if n_flagged_subj == 0:
                lines.append(
                    f"> ✅ **No influential observations** — "
                    f"{n_subj} subjects × {n_params} parameters checked, none flagged.\n"
                )
            else:
                lines.append(
                    f"> ⚠️  **{n_flagged_subj} / {n_subj} subject(s) flagged** across "
                    f"{n_params} parameters — see subject summary below.\n"
                )

            # Per-subject summary table (one row per subject)
            lines.append("**Per-subject influence summary:**\n")
            lines.append("| Subject | Cook's D (max) | Cook's D flags | DFBETA flags | Verdict |")
            lines.append("|---|---|---|---|---|")

            for subj_id in sorted(inf_sub["Subject_ID"].unique()):
                subj_rows   = inf_sub[inf_sub["Subject_ID"] == subj_id]
                max_cooksd  = subj_rows["CooksD"].max()
                n_cd_flags  = int(subj_rows["CooksD_Flagged"].sum())
                n_db_flags  = int(subj_rows["DFBETA_Flagged"].sum())
                any_flagged = n_cd_flags > 0 or n_db_flags > 0
                verdict     = "⚠️  Influential" if any_flagged else "✅ OK"
                lines.append(
                    f"| S{int(subj_id):02d} "
                    f"| {_fmt_float(max_cooksd, 4)} "
                    f"| {n_cd_flags} "
                    f"| {n_db_flags} "
                    f"| {verdict} |"
                )
            lines.append("")

            # Parameter-level detail — collapsed, shown only when flags exist
            flagged_all = pd.concat([cooksd_flag, dfbeta_flag]).drop_duplicates()
            if not flagged_all.empty:
                lines.append(
                    "<details><summary>Parameter-level detail (flagged rows only)</summary>\n"
                )
                lines.append("| Subject | Parameter | Cook's D | DFBETA | Flags |")
                lines.append("|---|---|---|---|---|")
                for _, r in flagged_all.sort_values(["Subject_ID", "Parameter"]).iterrows():
                    flags = []
                    if r.get("CooksD_Flagged"): flags.append("CooksD")
                    if r.get("DFBETA_Flagged"): flags.append("DFBETA")
                    lines.append(
                        f"| S{int(r['Subject_ID']):02d} "
                        f"| {_clean_param(r['Parameter'])} "
                        f"| {_fmt_float(r['CooksD'], 4)} "
                        f"| {_fmt_float(r['DFBETA'], 4)} "
                        f"| {', '.join(flags)} |"
                    )
                lines.append("\n</details>\n")

    return "\n".join(lines)




# ══════════════════════════════════════════════════════════════════════════════
#  SECTION III — CBPA
# ══════════════════════════════════════════════════════════════════════════════

def _dv_to_cbpa_keys(dv: str) -> tuple[str | None, str | None, str | None]:
    """
    Derive (modality, freq_band, muscle) filter keys for the CBPA frame from a DV name.

    modality  : "CMC" | "PSD"                 — matched against cbpa["modality"]
    freq_band : "beta" | "gamma" | "theta" …  — matched against cbpa["freq_band"]
    muscle    : "Flexor" | "Extensor" | None  — matched as a substring of the
                cbpa["hypothesis"] label (e.g. "H1_CMC_Flexor_beta_Happy_vs_Silence").
                Only set for H1 CMC DVs; None for EEG / EMG / unrecognised DVs.
                Max vs Mean is intentionally not distinguished: CBPA receives the
                full trial time-series, not a scalar summary.

    Returns (None, None, None) for unrecognised DV patterns; _section_cbpa falls
    back to hypothesis-prefix-only matching so nothing is silently lost.

    Parameters
    ----------
    dv : str
        Dependent variable name, e.g. "CMC_Extensor_mean_beta".
    """
    tokens    = dv.split("_")
    dv_lower  = dv.lower()

    # Modality
    if dv_lower.startswith("cmc"):
        modality = "CMC"
    elif dv_lower.startswith("psd"):
        modality = "PSD"
    else:
        modality = None

    # Muscle — present for H1 CMC DVs, encoded in the CBPA hypothesis label
    muscle = None
    for token in tokens:
        if token.lower() in {"flexor", "extensor"}:
            muscle = token.capitalize()   # "Flexor" or "Extensor"
            break

    # Frequency band — last token matching a known spectral band
    known_bands = {"alpha", "beta", "gamma", "theta", "delta", "all"}
    band = None
    for token in reversed(tokens):
        if token.lower() in known_bands:
            band = token.lower()
            break

    return modality, band, muscle


def _section_cbpa(hyp: str, dv: str, cbpa: pd.DataFrame) -> str:
    lines = ["### III. Spatio-Temporal Localisation (CBPA)\n"]

    if cbpa.empty:
        lines.append("> ⚠️  CBPA frame is empty — not yet run for this pipeline.\n")
        return "\n".join(lines)

    hyp_prefix = _hypothesis_prefix(hyp)
    cbpa_hyp = cbpa[cbpa["hypothesis"].apply(_hypothesis_prefix) == hyp_prefix].copy()

    if cbpa_hyp.empty:
        lines.append("> No CBPA results found for this hypothesis.\n")
        return "\n".join(lines)

    modality_key, band_key, muscle_key = _dv_to_cbpa_keys(dv)

    if modality_key is not None or band_key is not None or muscle_key is not None:
        mask = pd.Series(True, index=cbpa_hyp.index)
        if modality_key is not None:
            mask &= cbpa_hyp["modality"].str.upper() == modality_key.upper()
        if band_key is not None:
            mask &= cbpa_hyp["freq_band"].str.lower() == band_key.lower()
        if muscle_key is not None:
            mask &= cbpa_hyp["hypothesis"].str.contains(muscle_key, case=False)

        if mask.any():
            cbpa_sub = cbpa_hyp[mask].copy()
        else:
            other_combos = (
                cbpa_hyp[["modality", "freq_band"]]
                .drop_duplicates()
                .apply(lambda r: f"{r['modality']} {r['freq_band']}", axis=1)
                .tolist()
            )
            lines.append(
                f"> ⚠️  No CBPA rows match modality=`{modality_key}` / "
                f"band=`{band_key}` / muscle=`{muscle_key}` for {hyp_prefix}. "
                f"{len(cbpa_hyp)} row(s) exist for other combinations: "
                f"{', '.join(other_combos)}. "
                f"Check whether CBPA was run for this DV.\n"
            )
            return "\n".join(lines)
    else:
        cbpa_sub = cbpa_hyp.copy()

    # Detect axis column name and display label — phase vs clock time
    # Prefer phase_deg columns when present (phase-normalised CMC runs);
    # fall back to time_start_s / time_end_s for clock-time runs.
    if "phase_deg_start" in cbpa_sub.columns:
        axis_start_col = "phase_deg_start"
        axis_end_col   = "phase_deg_end"
        axis_label     = "Phase span (°)"
    else:
        axis_start_col = "time_start_s"
        axis_end_col   = "time_end_s"
        axis_label     = "Time span (s)"

    sig_clusters = cbpa_sub[cbpa_sub["significant"] == True]
    lines.append(
        f"Total clusters (modality={modality_key}, band={band_key}, muscle={muscle_key}): "
        f"**{len(cbpa_sub)}** | "
        f"Significant (p < 0.05): **{len(sig_clusters)}**\n"
    )

    if len(sig_clusters) == 0:
        lines.append(
            "> ⚠️  No significant clusters. Effect established by LME but not "
            "spatially/temporally localisable under cluster-level correction.\n"
        )
        top = cbpa_sub.nsmallest(3, "p_value")
        if not top.empty:
            lines.append("**Strongest non-significant clusters (exploratory):**\n")
            lines.append(f"| Contrast | Modality | Band | Cluster | p | peak-t | {axis_label} | Channels |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for _, r in top.iterrows():
                lines.append(
                    f"| {r['condition_A']} vs {r['condition_B']} "
                    f"| {r['modality']} | {r['freq_band']} "
                    f"| #{int(r['cluster_index'])} "
                    f"| {_fmt_p(r['p_value'])} "
                    f"| {_fmt_float(r['peak_t'], 2)} "
                    f"| {_fmt_float(r[axis_start_col], 2)}–{_fmt_float(r[axis_end_col], 2)} "
                    f"| {r['channels']} |"
                )
            lines.append("")
        return "\n".join(lines)

    for (cond_col, cA, cB, mod, band), grp in sig_clusters.groupby(
        ["condition_column", "condition_A", "condition_B", "modality", "freq_band"]
    ):
        lines.append(f"**{cA} vs {cB}  ({cond_col}) | {mod} {band}**\n")
        lines.append(f"| Cluster | p | peak-t | t-thresh | {axis_label} | N ch | Channels |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in grp.sort_values("p_value").iterrows():
            lines.append(
                f"| #{int(r['cluster_index'])} "
                f"| {_fmt_p(r['p_value'])} "
                f"| {_fmt_float(r['peak_t'], 3)} "
                f"| {_fmt_float(r['t_thresh'], 3)} "
                f"| {_fmt_float(r[axis_start_col], 3)}–{_fmt_float(r[axis_end_col], 3)} "
                f"| {int(r['n_channels'])} "
                f"| {r['channels']} |"
            )
        lines.append("")

    return "\n".join(lines)





# ══════════════════════════════════════════════════════════════════════════════
#  SECTION IV — POPULATION HETEROGENEITY
# ══════════════════════════════════════════════════════════════════════════════

def _section_heterogeneity(
    hyp: str,
    dv: str,
    subj: pd.DataFrame,
    influence: pd.DataFrame,
    cfg: _ReportConfig,
) -> str:
    lines = ["### IV. Population Heterogeneity\n"]

    # Guard: frame is empty when subject heterogeneity has not been computed
    if subj.empty:
        lines.append("> ⚠️  Subject heterogeneity frame is empty — not applicable for this pipeline.\n")
        return "\n".join(lines)

    sub = subj[subj["Comparison_Level"] == "lvl_1"].copy()
    if sub.empty:
        sub = subj.copy()

    if sub.empty:
        lines.append("> ⚠️  No subject-level data found.\n")
        return "\n".join(lines)

    cond_col = sub["Condition_Variable"].iloc[0] if "Condition_Variable" in sub.columns else "Condition"
    lines.append(
        f"**Condition variable:** `{cond_col}` | "
        f"Subjects: **{sub['Subject ID'].nunique()}**\n"
    )

    # Per-condition responder summary
    lines.append("| Condition | N | Responders ↑ | Non-responders ↓ | Mean contrast | Median CoeffOfVar |")
    lines.append("|---|---|---|---|---|---|")
    for cond, grp in sub.groupby("Condition"):
        n_total      = grp["Subject ID"].nunique()
        n_resp       = grp[grp["Responder_Flag"] == True]["Subject ID"].nunique()
        mean_c       = grp["Raw_Contrast"].mean()
        med_cv       = grp["Subject_CV"].median()
        lines.append(
            f"| {cond} | {n_total} "
            f"| {n_resp} ({100*n_resp/n_total:.0f}%) "
            f"| {n_total-n_resp} ({100*(n_total-n_resp)/n_total:.0f}%) "
            f"| {_fmt_float(mean_c, 4)} {'↑' if mean_c > 0 else '↓'} "
            f"| {_fmt_float(med_cv, 4)} |"
        )
    lines.append("")

    # Per-subject pivot of normalised contrasts
    pivot = sub.pivot_table(
        index="Subject ID", columns="Condition",
        values="Normalised_Contrast", aggfunc="mean",
    )
    if not pivot.empty:
        lines.append("**Per-subject normalised contrasts:**\n")
        lines.append("| Subject | " + " | ".join(str(c) for c in pivot.columns) + " |")
        lines.append("|" + "---|" * (len(pivot.columns) + 1))
        for sid, row in pivot.iterrows():
            vals = " | ".join(
                (f"**{_fmt_float(v, 3)}**" if pd.notna(v) and abs(v) > 0.05
                 else (_fmt_float(v, 3) if pd.notna(v) else "—"))
                for v in row
            )
            lines.append(f"| S{int(sid):02d} | {vals} |")
        lines.append(
            "\n> Bold = |normalised contrast| > 0.05. "
            "High-CV subjects may disproportionately drive group estimates.\n"
        )

    # Cross-reference influence flags
    inf_flagged = influence[
        (influence["Dependent_Variable"] == dv)
        & (influence["N_Segments"] == cfg.primary_n_segments)
        & ((influence["CooksD_Flagged"] == True) | (influence["DFBETA_Flagged"] == True))
    ]
    flagged_ids = set(inf_flagged["Subject_ID"].unique())
    if flagged_ids:
        lines.append(
            f"> ⚠️  Subjects with influence flags: "
            f"{', '.join(f'S{int(s):02d}' for s in sorted(flagged_ids))}. "
            f"Interpret their contrasts above with caution.\n"
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION V — DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def _section_diagnostics(hyp: str, dv: str, diag: pd.DataFrame,
    cfg: _ReportConfig,) -> str:
    """
    Section V — Model diagnostics.

    One row per model × comparison level × N_Segments combination.
    OLS shows R²/R²_adj; LME shows R²_marg/R²_cond/ICC.

    Parameters
    ----------
    hyp : str
    dv : str
    diag : pd.DataFrame
        Diagnostics frame, already filtered to hyp × dv.
    """
    lines = ["### V. Model Diagnostics\n"]

    if diag.empty:
        lines.append("> ⚠️  No diagnostics found.\n")
        return "\n".join(lines)

    sub = diag[diag["Model_Type"].isin(_active_models(cfg))].copy()
    sub["_Level"] = sub["Comparison_Level"].apply(_short_level)

    lines.append(
        "| Model | Level | Segs | N | Shapiro | ρ lag1 | DEFF | SE× | "
        "R² / R²_adj | R²_marg / R²_cond | ICC | AIC |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

    for _, r in sub.sort_values(["N. Segments", "Model_Type", "_Level"]).iterrows():
        model   = r["Model_Type"]
        shapiro = (f"{_fmt_p(r['Shapiro_p'])} "
                   f"{'⚠️' if r['Shapiro_Violated'] == 'Yes' else '✓'}")
        if model == "OLS":
            r2_str   = f"{_fmt_float(r['R_squared'],3)} / {_fmt_float(r['R_squared_adj'],3)}"
            r2mc_str = "— / —"
            icc_str  = "—"
        else:
            r2_str   = "— / —"
            r2mc_str = (f"{_fmt_float(r['R_squared_marginal'],3)} / "
                        f"{_fmt_float(r['R_squared_conditional'],3)}")
            icc_str  = _fmt_float(r["ICC"], 3)

        lines.append(
            f"| {model} | {r['_Level']} | {int(r['N. Segments'])} "
            f"| {int(r['N_Observations']) if pd.notna(r['N_Observations']) else '—'} "
            f"| {shapiro} "
            f"| {_fmt_float(r['Lag1_Autocorr'], 3)} "
            f"| {_fmt_float(r['Design_Effect'], 2)} "
            f"| {_fmt_float(r['SE_Inflation'], 2)}× "
            f"| {r2_str} | {r2mc_str} | {icc_str} "
            f"| {_fmt_float(r['AIC'], 1) if pd.notna(r.get('AIC')) else '—'} |"
        )

    lines.append(
        "\n> **DEFF** = design effect from lag-1 autocorrelation. "
        "**SE×** = inflation factor applied to all p-values. "
        "**ICC** = between-subject variance share (LME only). "
        "Shapiro ⚠️ = residual non-normality (p < 0.05).\n"
    )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def generate_statistical_report(
    omnibus_results_frame:        pd.DataFrame,
    omnibus_diagnostics_frame:    pd.DataFrame,
    power_analysis_results_frame: pd.DataFrame,
    influence_measures_frame:     pd.DataFrame,
    subject_heterogeneity_frame:  pd.DataFrame,
    cbpa_results_frame:           pd.DataFrame,
    output_dir:                   Path,
    primary_n_segments:           int        = 1,
    resolution_segments:          list[int]  = (1, 5, 20),
    alpha_adjusted:               float = 0.05,
        include_ols: bool = False,
        target_power: float = 0.80,
        level_definitions: list[dict] | None = None,
    hypothesis_groups:   list[dict] | None = None,
        file_identifier_suffix: str | None = None,
) -> Path:
    """
    Generate a Markdown report from the six statistical output frames.

    Parameters
    ----------
    omnibus_results_frame : pd.DataFrame
    omnibus_diagnostics_frame : pd.DataFrame
    power_analysis_results_frame : pd.DataFrame
    influence_measures_frame : pd.DataFrame
    subject_heterogeneity_frame : pd.DataFrame
    cbpa_results_frame : pd.DataFrame
    output_dir : Path
        Directory where the .md file is saved.
    primary_n_segments : int
        Canonical time resolution for Sections I–IV.
    resolution_segments : list[int]
        All resolutions checked in cross-resolution stability (Section I.b).
    alpha_adjusted : float
        Family-wise significance threshold applied to p_value_adjusted.
    include_ols : bool
        If True, OLS rows appear alongside LME in Sections I, I.b, and V.
        When False, only LME rows are shown throughout.
    target_power : float
        Minimum Power_at_Observed_Effect to label an effect "well-powered"
        in the overview table and Section II. Default 0.80.
    level_definitions : list[dict] or None
        Output of fetch_level_definitions(). When provided, a human-readable
        level-description table is prepended to the Overview section.
        When None the subsection is silently omitted.

    Returns
    -------
    Path
        Path to the saved report file.
    """
    # Build the immutable per-call config — no global mutation
    cfg = _ReportConfig(
        primary_n_segments  = primary_n_segments,
        resolution_segments = list(resolution_segments),
        alpha_adjusted      = alpha_adjusted,
        include_ols         = include_ols,
        target_power        = target_power,
    )

    # Print pipeline coverage gaps to stdout before writing the report
    _print_pipeline_recommendations(
        res                 = omnibus_results_frame,
        power               = power_analysis_results_frame,
        influence           = influence_measures_frame,
        cbpa                = cbpa_results_frame,
        alpha               = cfg.alpha_adjusted,
        primary_n_segments  = cfg.primary_n_segments,
        resolution_segments = cfg.resolution_segments,
    )

    frames = {
        "omnibus_results":       omnibus_results_frame,
        "omnibus_diagnostics":   omnibus_diagnostics_frame,
        "power_analysis":        power_analysis_results_frame,
        "influence_measures":    influence_measures_frame,
        "subject_heterogeneity": subject_heterogeneity_frame,
        "cbpa_results":          cbpa_results_frame,
    }
    validation_warnings = validate_frames(frames)

    lines: list[str] = [
        "# Statistical Analysis Report",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*  ",
        f"*Primary resolution: {cfg.primary_n_segments}-seg | "
        f"Models: {', '.join(_active_models(cfg))} | "
        f"α = {cfg.alpha_adjusted} | target power = {cfg.target_power:.0%}*\n",
        "---\n",
    ]

    if validation_warnings:
        lines.append("## ⚠️  Frame Validation Warnings\n")
        for w in validation_warnings:
            lines.append(f"- {w}")
        lines.append("\n---\n")
    else:
        lines.append("> ✅ All six frames validated — no missing columns.\n\n---\n")

    # Describe comparison levels
    if level_definitions is not None:
        lines.append(_section_level_definitions(level_definitions))

    # Section 0: overview table across all hypotheses
    lines.append(
        _section_overview_table(
            omnibus_results_frame,
            power_analysis_results_frame,
            cfg,
        )
    )

    if hypothesis_groups is not None:
        lines.append(
            _section_predictor_quality(
                hypothesis_groups=hypothesis_groups,
                res=omnibus_results_frame,
                power=power_analysis_results_frame,
                influence=influence_measures_frame,
                cfg=cfg,
            )
        )

    # Per-hypothesis × DV blocks
    hyp_dv_pairs = (
        omnibus_results_frame[["Hypothesis", "Dependent_Variable"]]
        .drop_duplicates()
        .sort_values(["Hypothesis", "Dependent_Variable"])
    )

    for _, row in hyp_dv_pairs.iterrows():
        hyp = row["Hypothesis"]
        dv  = row["Dependent_Variable"]

        lines += [f"---\n", f"## {hyp}", f"**Dependent variable:** `{dv}`\n"]

        # Pre-filter once per block for speed
        res_sub  = omnibus_results_frame[
            (omnibus_results_frame["Hypothesis"] == hyp)
            & (omnibus_results_frame["Dependent_Variable"] == dv)
        ]
        diag_sub = omnibus_diagnostics_frame[
            (omnibus_diagnostics_frame["Hypothesis"] == hyp)
            & (omnibus_diagnostics_frame["Dependent_Variable"] == dv)
        ]
        # Guard: subject_heterogeneity_frame may be an empty pd.DataFrame()
        # with no columns when not applicable for this pipeline
        subj_sub = (
            subject_heterogeneity_frame[
                (subject_heterogeneity_frame["Hypothesis"] == hyp)
                & (subject_heterogeneity_frame["Dependent_Variable"] == dv)
                ]
            if not subject_heterogeneity_frame.empty
            else pd.DataFrame()
        )

        lines.append(_section_findings(hyp, dv, res_sub, cfg))
        lines.append(_section_cross_resolution(hyp, dv, res_sub, cfg))
        lines.append(
            _section_trust(
                hyp, dv, res_sub,
                power_analysis_results_frame,
                influence_measures_frame,
                cfg,
            )
        )
        lines.append(_section_cbpa(hyp, dv, cbpa_results_frame))
        lines.append(_section_heterogeneity(hyp, dv, subj_sub, influence_measures_frame, cfg))
        lines.append(_section_diagnostics(hyp, dv, diag_sub, cfg))

    filemgmt.assert_dir(output_dir)
    out_path = output_dir / filemgmt.file_title(f"Statistical Report{f'_{file_identifier_suffix}' if file_identifier_suffix is not None else ''}", ".md")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ Report written → {out_path}")
    return out_path



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
        output_dir           = RQ_A_REPORT_DIR,
        file_identifier_suffix='RQ_A',

        primary_n_segments   = 1,
        resolution_segments  = [1, 5, 10],

        alpha_adjusted       = 0.05,
        include_ols          = False,
        target_power=0.80,

        level_definitions=fetch_level_definitions(multi_segments_per_trial=True),

        hypothesis_groups=[
            {
                "label": "H1 – CMC Music Effects (all DVs)",
                "hypotheses": [h for h in _load_csv(RQ_A_OMNIBUS_RESULTS,  ["All Time Resolutions Results"])["Hypothesis"].unique()
                               if h.startswith("H1")],
            },
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
