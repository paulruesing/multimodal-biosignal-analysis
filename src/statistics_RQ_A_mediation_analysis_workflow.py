"""
RQ-A mediation workflow focused on Level 1 (Category or Silence).

This workflow tests whether biological/emotional variables mediate the effect
of music condition (each category vs Silence) on CMC outcomes.

Scope in this file:
- X (predictor): Category or Silence (Level 1 contrasts vs Silence)
- M (mediators): Emotional_State, GSR, Median_HRV, Median_Heart_Rate
- Y (outcomes): CMC variables
"""

from pathlib import Path
import warnings
import re

import numpy as np
import pandas as pd

import src.utils.file_management as filemgmt
from statsmodels.formula.api import mixedlm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=FutureWarning)


LEVEL1_X_VAR = "Category or Silence"
GROUP_VAR = "Subject ID"
LEVEL1_CONTRASTS: list[tuple[str, str]] = [
    ("Happy", "Silence"),
    ("Groovy", "Silence"),
    ("Sad", "Silence"),
    ("Classic", "Silence"),
]

MEDIATOR_CANDIDATES: list[str] = [
    "Emotional_State",
    "GSR",
    "Median_HRV",
    "Median_Heart_Rate",
]

# All CMC dependent variables used in RQ-A omnibus testing.
RQA_CMC_DVS: list[str] = [
    "CMC_Flexor_max_beta",
    "CMC_Flexor_mean_beta",
    "CMC_Flexor_max_gamma",
    "CMC_Flexor_mean_gamma",
    "CMC_Extensor_max_beta",
    "CMC_Extensor_mean_beta",
    "CMC_Extensor_max_gamma",
    "CMC_Extensor_mean_gamma",
]

CMC_OUTCOMES: list[str] = RQA_CMC_DVS.copy()


def _fit_mixedlm_with_diagnostics(formula: str, data: pd.DataFrame, group_col: str) -> dict:
    """Fit one MixedLM and capture convergence/warning diagnostics."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mixedlm(formula, data, groups=data[group_col]).fit(reml=False, disp=0)

    warning_messages = [str(w.message) for w in caught]
    has_singular = any("Random effects covariance is singular" in msg for msg in warning_messages)
    has_convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    converged = bool(getattr(result, "converged", False))

    return {
        "result": result,
        "converged": converged,
        "n_warnings": len(caught),
        "has_singular_warning": has_singular,
        "has_convergence_warning": has_convergence_warning,
        "warning_messages": warning_messages,
    }


def _short_warning_signature(messages: list[str], max_items: int = 2) -> str:
    """Compact warning text for CSV/console readability."""
    if not messages:
        return ""
    cleaned: list[str] = []
    for msg in messages:
        short = re.sub(r"\s+", " ", msg).strip()
        if short not in cleaned:
            cleaned.append(short)
    return " | ".join(cleaned[:max_items])


def fetch_mediation_hypotheses() -> list[dict]:
    """Build Level-1 mediation configs (Category-or-Silence only)."""
    return [
        {
            "name": f"L1 Mediation: {mediator} mediates Category-or-Silence -> CMC",
            "x_var": LEVEL1_X_VAR,
            "x_contrasts": LEVEL1_CONTRASTS,
            "m_var": mediator,
            "y_vars": CMC_OUTCOMES,
            "description": (
                f"Level 1 only: does {mediator} explain category-vs-silence effects on CMC?"
            ),
        }
        for mediator in MEDIATOR_CANDIDATES
    ]


def fit_mediation_model(
    data: pd.DataFrame,
    x_var: str,
    x_contrast: tuple[str, str],
    m_var: str,
    y_var: str,
    group_var: str = GROUP_VAR,
    min_obs: int = 12,
    min_subjects: int = 6,
) -> dict:
    """Fit a/b/c/c' paths for one (X-contrast, mediator, outcome) configuration."""
    needed = {x_var, m_var, y_var, group_var}
    missing = sorted([c for c in needed if c not in data.columns])
    if missing:
        return {
            "status": "skipped_missing_columns",
            "x_var": x_var,
            "x_contrast": f"{x_contrast[0]} vs {x_contrast[1]}",
            "mediator": m_var,
            "outcome": y_var,
            "missing_columns": ", ".join(missing),
        }

    level_a, level_b = x_contrast
    df = data.loc[data[x_var].isin([level_a, level_b]), [x_var, m_var, y_var, group_var]].copy()
    df = df.dropna()

    if df.empty:
        return {
            "status": "insufficient_data",
            "x_var": x_var,
            "x_contrast": f"{level_a} vs {level_b}",
            "mediator": m_var,
            "outcome": y_var,
            "n_obs": 0,
            "n_subjects": 0,
            "reason": "no rows left after contrast filter and dropna",
        }

    present_levels = sorted(df[x_var].unique().tolist())
    if set(present_levels) != {level_a, level_b}:
        return {
            "status": "insufficient_data",
            "x_var": x_var,
            "x_contrast": f"{level_a} vs {level_b}",
            "mediator": m_var,
            "outcome": y_var,
            "n_obs": int(len(df)),
            "n_subjects": int(df[group_var].nunique()),
            "reason": f"contrast levels missing in data; found {present_levels}",
        }

    model_df = pd.DataFrame(
        {
            "x": (df[x_var] == level_a).astype(int),
            "m": pd.to_numeric(df[m_var], errors="coerce"),
            "y": pd.to_numeric(df[y_var], errors="coerce"),
            "group": df[group_var],
        }
    ).dropna()

    n_obs = int(len(model_df))
    n_subjects = int(model_df["group"].nunique())

    if n_obs < min_obs or n_subjects < min_subjects:
        return {
            "status": "insufficient_data",
            "x_var": x_var,
            "x_contrast": f"{level_a} vs {level_b}",
            "mediator": m_var,
            "outcome": y_var,
            "n_obs": n_obs,
            "n_subjects": n_subjects,
            "reason": f"needs at least {min_obs} obs and {min_subjects} subjects",
        }

    try:
        # Path a: X -> M
        fit_a = _fit_mixedlm_with_diagnostics("m ~ x", model_df, "group")
        res_a = fit_a["result"]
        coef_a = float(res_a.fe_params["x"])
        se_a = float(res_a.bse["x"])

        # Path c: X -> Y (total effect)
        fit_c = _fit_mixedlm_with_diagnostics("y ~ x", model_df, "group")
        res_c = fit_c["result"]
        coef_c = float(res_c.fe_params["x"])
        se_c = float(res_c.bse["x"])
        p_c = float(res_c.pvalues["x"])

        # Paths b and c': X + M -> Y
        fit_cprime = _fit_mixedlm_with_diagnostics("y ~ x + m", model_df, "group")
        res_cprime = fit_cprime["result"]
        coef_b = float(res_cprime.fe_params["m"])
        se_b = float(res_cprime.bse["m"])
        coef_cprime = float(res_cprime.fe_params["x"])

        path_converged = {
            "a": fit_a["converged"],
            "c": fit_c["converged"],
            "cprime": fit_cprime["converged"],
        }
        path_singular = {
            "a": fit_a["has_singular_warning"],
            "c": fit_c["has_singular_warning"],
            "cprime": fit_cprime["has_singular_warning"],
        }
        any_non_converged = not all(path_converged.values())
        any_singular = any(path_singular.values())

        indirect_effect = coef_a * coef_b

        fit_quality = (
            "strict_ok"
            if (not any_non_converged and not any_singular)
            else ("relaxed_ok" if not any_non_converged else "not_fittable")
        )

        return {
            "status": "fitted" if fit_quality != "not_fittable" else "non_converged",
            "x_var": x_var,
            "x_contrast": f"{level_a} vs {level_b}",
            "mediator": m_var,
            "outcome": y_var,
            "n_obs": n_obs,
            "n_subjects": n_subjects,
            "fit_quality": fit_quality,
            "path_a_converged": path_converged["a"],
            "path_c_converged": path_converged["c"],
            "path_cprime_converged": path_converged["cprime"],
            "path_a_singular_warning": path_singular["a"],
            "path_c_singular_warning": path_singular["c"],
            "path_cprime_singular_warning": path_singular["cprime"],
            "fit_warning_count": fit_a["n_warnings"] + fit_c["n_warnings"] + fit_cprime["n_warnings"],
            "fit_warning_signature": _short_warning_signature(
                fit_a["warning_messages"] + fit_c["warning_messages"] + fit_cprime["warning_messages"]
            ),
            "coef_a": coef_a,
            "se_a": se_a,
            "coef_b": coef_b,
            "se_b": se_b,
            "coef_c": coef_c,
            "se_c": se_c,
            "p_c": p_c,
            "coef_cprime": coef_cprime,
            "indirect_effect": indirect_effect,
            "mediation_prop": indirect_effect / coef_c if coef_c != 0 else np.nan,
            "model_df": model_df,
        }
    except Exception as exc:
        return {
            "status": "error",
            "x_var": x_var,
            "x_contrast": f"{level_a} vs {level_b}",
            "mediator": m_var,
            "outcome": y_var,
            "n_obs": n_obs,
            "n_subjects": n_subjects,
            "error": str(exc),
        }


def _cluster_bootstrap_sample(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Resample subject blocks with replacement and keep duplicates as separate groups."""
    subject_blocks = {subj: grp.copy() for subj, grp in df.groupby("group", sort=False)}
    subjects = list(subject_blocks.keys())
    sampled_indices = rng.integers(0, len(subjects), size=len(subjects))

    sampled_blocks: list[pd.DataFrame] = []
    for rep_idx, subj_ind in enumerate(sampled_indices):
        subj = subjects[int(subj_ind)]
        block = subject_blocks[subj].copy()
        block["boot_group"] = f"{subj}_rep{rep_idx}"
        sampled_blocks.append(block)

    return pd.concat(sampled_blocks, axis=0, ignore_index=True)


def bootstrap_indirect_effect(
    fit_result: dict,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict:
    """Estimate percentile CI for indirect effect (a*b) via clustered bootstrap."""
    if fit_result.get("status") != "fitted":
        return {
            "bootstrap_status": fit_result.get("status", "not_fitted"),
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "significant": False,
            "n_bootstrap": 0,
        }

    model_df = fit_result["model_df"]
    if model_df.empty:
        return {
            "bootstrap_status": "bootstrap_failed",
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "significant": False,
            "n_bootstrap": 0,
        }

    rng = np.random.default_rng(random_state)
    indirect_effects: list[float] = []
    bootstrap_success = 0
    bootstrap_non_converged = 0
    bootstrap_exceptions = 0

    for _ in range(n_bootstrap):
        boot_df = _cluster_bootstrap_sample(model_df, rng)
        try:
            fit_a = _fit_mixedlm_with_diagnostics("m ~ x", boot_df, "boot_group")
            fit_cprime = _fit_mixedlm_with_diagnostics("y ~ x + m", boot_df, "boot_group")
            if fit_a["converged"] and fit_cprime["converged"]:
                bootstrap_success += 1
                indirect_effects.append(float(fit_a["result"].fe_params["x"] * fit_cprime["result"].fe_params["m"]))
            else:
                bootstrap_non_converged += 1
        except Exception:
            bootstrap_exceptions += 1
            continue

    if len(indirect_effects) < 50:
        return {
            "bootstrap_status": "bootstrap_failed",
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "significant": False,
            "n_bootstrap": len(indirect_effects),
            "bootstrap_attempted": n_bootstrap,
            "bootstrap_success": bootstrap_success,
            "bootstrap_non_converged": bootstrap_non_converged,
            "bootstrap_exceptions": bootstrap_exceptions,
            "bootstrap_success_rate": bootstrap_success / n_bootstrap,
        }

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(indirect_effects, (alpha / 2) * 100))
    ci_upper = float(np.percentile(indirect_effects, (1 - alpha / 2) * 100))

    return {
        "bootstrap_status": "computed",
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": not (ci_lower <= 0 <= ci_upper),
        "n_bootstrap": len(indirect_effects),
        "bootstrap_attempted": n_bootstrap,
        "bootstrap_success": bootstrap_success,
        "bootstrap_non_converged": bootstrap_non_converged,
        "bootstrap_exceptions": bootstrap_exceptions,
        "bootstrap_success_rate": bootstrap_success / n_bootstrap,
    }


def extract_report_ready_mediation_table(
    results_frame: pd.DataFrame,
    include_relaxed_ok: bool = False,
    min_bootstrap_success_rate: float = 0.70,
    min_bootstrap_samples: int = 100,
) -> pd.DataFrame:
    """Extract a report-ready table from raw mediation results.

    Selection logic:
    - fitted rows only
    - bootstrap computed
    - fit quality strict_ok (or relaxed_ok as optional)
    - sufficient bootstrap success and sample size
    """
    if results_frame is None or results_frame.empty:
        return pd.DataFrame()

    needed = {
        "status", "bootstrap_status", "fit_quality", "bootstrap_success_rate",
        "n_bootstrap", "x_contrast", "mediator", "outcome", "n_obs", "n_subjects",
        "coef_a", "coef_b", "coef_c", "coef_cprime", "indirect_effect",
        "ci_lower", "ci_upper", "significant", "fit_warning_count",
        "bootstrap_success", "bootstrap_attempted",
    }
    missing = needed - set(results_frame.columns)
    if missing:
        raise ValueError(f"Missing required columns for report-ready table: {sorted(missing)}")

    allowed_fit_quality = ["strict_ok", "relaxed_ok"] if include_relaxed_ok else ["strict_ok"]

    table = results_frame.copy()
    table = table[
        (table["status"] == "fitted")
        & (table["bootstrap_status"] == "computed")
        & (table["fit_quality"].isin(allowed_fit_quality))
        & (pd.to_numeric(table["bootstrap_success_rate"], errors="coerce") >= min_bootstrap_success_rate)
        & (pd.to_numeric(table["n_bootstrap"], errors="coerce") >= min_bootstrap_samples)
    ].copy()

    if table.empty:
        return table

    table["Sign"] = np.where(pd.to_numeric(table["indirect_effect"], errors="coerce") >= 0, "+", "-")
    table["CI_Contains_Zero"] = (
        (pd.to_numeric(table["ci_lower"], errors="coerce") <= 0)
        & (pd.to_numeric(table["ci_upper"], errors="coerce") >= 0)
    )

    # Report-friendly naming and ordering.
    table = table[
        [
            "x_contrast", "mediator", "outcome",
            "n_obs", "n_subjects",
            "coef_a", "coef_b", "coef_c", "coef_cprime", "indirect_effect",
            "ci_lower", "ci_upper", "CI_Contains_Zero", "significant", "Sign",
            "fit_quality", "fit_warning_count",
            "bootstrap_success", "bootstrap_attempted", "bootstrap_success_rate",
        ]
    ].rename(
        columns={
            "x_contrast": "Contrast",
            "mediator": "Mediator",
            "outcome": "Outcome",
            "n_obs": "N_Obs",
            "n_subjects": "N_Subjects",
            "coef_a": "Path_a_X_to_M",
            "coef_b": "Path_b_M_to_Y_given_X",
            "coef_c": "Path_c_Total_X_to_Y",
            "coef_cprime": "Path_cprime_Direct_X_to_Y_given_M",
            "indirect_effect": "Indirect_a_times_b",
            "ci_lower": "CI95_Lower",
            "ci_upper": "CI95_Upper",
            "significant": "Indirect_Significant",
            "fit_quality": "Fit_Quality",
            "fit_warning_count": "Fit_Warning_Count",
            "bootstrap_success": "Bootstrap_Success",
            "bootstrap_attempted": "Bootstrap_Attempted",
            "bootstrap_success_rate": "Bootstrap_Success_Rate",
        }
    )

    return table.sort_values(["Contrast", "Mediator", "Outcome"]).reset_index(drop=True)





if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / "data"
    OUTPUT = ROOT / "output"
    FEATURE_OUTPUT_DATA = DATA / "precomputed_features"
    MEDIATION_OUTPUT = OUTPUT / "statistics_RQ_A" / "mediation_analysis"
    filemgmt.assert_dir(MEDIATION_OUTPUT)

    n_within_trial_segments = 1
    n_bootstrap = 300  # todo: drives runtime!

    input_path = filemgmt.most_recent_file(
        FEATURE_OUTPUT_DATA,
        ".csv",
        [f"Combined Statistics {n_within_trial_segments}seg"],
    )
    all_subject_data_frame = pd.read_csv(input_path)

    print("=" * 80)
    print("RQ-A MEDIATION ANALYSIS (Level 1: Category or Silence)")
    print("=" * 80)
    print(f"Input frame: {input_path}")
    print(
        f"Rows: {len(all_subject_data_frame)} | "
        f"Subjects: {all_subject_data_frame[GROUP_VAR].nunique()}"
    )
    print(f"Y outcomes (CMC DVs): {len(CMC_OUTCOMES)}")
    print(f"Mediators: {', '.join(MEDIATOR_CANDIDATES)}")
    print(f"Contrasts: {', '.join([f'{a} vs {b}' for a, b in LEVEL1_CONTRASTS])}")

    hypotheses = fetch_mediation_hypotheses()
    all_results: list[dict] = []

    for hyp in hypotheses:
        print(f"\n{hyp['name']}")
        for contrast in hyp["x_contrasts"]:
            for y_var in hyp["y_vars"]:
                result = fit_mediation_model(
                    data=all_subject_data_frame,
                    x_var=hyp["x_var"],
                    x_contrast=contrast,
                    m_var=hyp["m_var"],
                    y_var=y_var,
                    group_var=GROUP_VAR,
                )
                if result["status"] == "fitted":
                    result.update(
                        bootstrap_indirect_effect(
                            result,
                            n_bootstrap=n_bootstrap,
                            ci=0.95,
                            random_state=42,
                        )
                    )
                else:
                    result.update(
                        {
                            "bootstrap_status": result["status"],
                            "ci_lower": np.nan,
                            "ci_upper": np.nan,
                            "significant": False,
                            "n_bootstrap": 0,
                        }
                    )

                result["hypothesis"] = hyp["name"]
                result["description"] = hyp["description"]
                all_results.append(result)

                if result["bootstrap_status"] == "computed":
                    print(
                        f"  {result['x_contrast']} | {y_var}: "
                        f"indirect={result['indirect_effect']:.4f}, "
                        f"CI=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}], "
                        f"sig={result['significant']}, "
                        f"fit={result.get('fit_quality', 'n/a')}, "
                        f"boot_ok={result.get('bootstrap_success', 0)}/{result.get('bootstrap_attempted', 0)}"
                    )

    results_frame = pd.DataFrame(all_results)
    if "model_df" in results_frame.columns:
        del results_frame["model_df"]

    # Main export: all attempted tests with full diagnostics/status columns.
    out_path = MEDIATION_OUTPUT / filemgmt.file_title("Mediation Analysis Results", ".csv")
    results_frame.to_csv(out_path, index=False)

    summary_columns = [
        "hypothesis", "x_var", "x_contrast", "mediator", "outcome",
        "status", "bootstrap_status", "n_obs", "n_subjects",
        "fit_quality", "path_a_converged", "path_c_converged", "path_cprime_converged",
        "path_a_singular_warning", "path_c_singular_warning", "path_cprime_singular_warning",
        "fit_warning_count", "fit_warning_signature",
        "indirect_effect", "ci_lower", "ci_upper", "significant", "n_bootstrap",
        "bootstrap_attempted", "bootstrap_success", "bootstrap_non_converged",
        "bootstrap_exceptions", "bootstrap_success_rate",
        "reason", "missing_columns", "error",
    ]
    summary_columns = [col for col in summary_columns if col in results_frame.columns]
    summary_frame = results_frame[summary_columns].copy()
    summary_path = MEDIATION_OUTPUT / filemgmt.file_title("Mediation Analysis Summary", ".csv")
    summary_frame.to_csv(summary_path, index=False)

    if {"bootstrap_status", "significant"}.issubset(summary_frame.columns):
        significant_frame = summary_frame[
            (summary_frame["bootstrap_status"] == "computed")
            & (summary_frame["significant"] == True)
        ].copy()
    else:
        significant_frame = pd.DataFrame()
    sig_path = MEDIATION_OUTPUT / filemgmt.file_title("Mediation Analysis Significant", ".csv")
    significant_frame.to_csv(sig_path, index=False)

    report_ready_frame = extract_report_ready_mediation_table(
        results_frame,
        include_relaxed_ok=False,
        min_bootstrap_success_rate=0.70,
        min_bootstrap_samples=100,
    )
    report_ready_path = MEDIATION_OUTPUT / filemgmt.file_title("Mediation Analysis Report Ready", ".csv")
    report_ready_frame.to_csv(report_ready_path, index=False)

    print("\n" + "=" * 80)
    print(f"Saved all results: {out_path}")
    print(f"Saved summary:     {summary_path}")
    print(f"Saved significant: {sig_path}")
    print(f"Saved report-ready:{report_ready_path}")

    print("\nStatus counts:")
    status_counts = results_frame["status"].value_counts(dropna=False).to_dict()
    for key, value in status_counts.items():
        print(f"  {key}: {value}")

    if "bootstrap_status" in results_frame.columns:
        print("Bootstrap status counts:")
        boot_counts = results_frame["bootstrap_status"].value_counts(dropna=False).to_dict()
        for key, value in boot_counts.items():
            print(f"  {key}: {value}")

    if "fit_quality" in results_frame.columns:
        print("Fit quality counts:")
        fit_counts = results_frame["fit_quality"].value_counts(dropna=False).to_dict()
        for key, value in fit_counts.items():
            print(f"  {key}: {value}")

    if "bootstrap_success_rate" in results_frame.columns and len(results_frame) > 0:
        numeric_rates = pd.Series(pd.to_numeric(results_frame["bootstrap_success_rate"], errors="coerce"))
        mean_rate = numeric_rates.dropna().mean()
        if pd.notna(mean_rate):
            print(f"Mean bootstrap success rate: {mean_rate:.3f}")

    if "mediator" in results_frame.columns:
        print("\nComputed tests per mediator:")
        if "bootstrap_status" in results_frame.columns:
            mediator_counts = (
                results_frame[results_frame["bootstrap_status"] == "computed"]
                .groupby("mediator")
                .size()
                .to_dict()
            )
        else:
            mediator_counts = {}
        for key, value in mediator_counts.items():
            print(f"  {key}: {value}")

    computed = results_frame[results_frame["bootstrap_status"] == "computed"]
    significant = computed[computed["significant"] == True]
    print(f"Computed mediation tests: {len(computed)}")
    print(f"Significant indirect effects: {len(significant)}")
    print(f"Report-ready rows: {len(report_ready_frame)}")
    print("=" * 80)



