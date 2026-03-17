from __future__ import annotations  # with this, all annotations become strings during compilation, so forward references work

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from typing import Callable
import warnings
from dataclasses import dataclass, field

import src.utils.file_management as filemgmt


def _apply_reference_categories(
    df: pd.DataFrame,
    condition_vars: dict,
    reference_categories: dict | None,
) -> pd.DataFrame:
    """Reorder categorical column levels so the reference appears first.

    Must be called on a copy of the dataframe before formula construction.
    Patsy reads level order from the pandas Categorical dtype, so placing the
    reference first is sufficient — no Treatment() wrapper needed in the formula.

    Parameters
    ----------
    df : pd.DataFrame
        Working copy of the input dataframe.
    condition_vars : dict
        Mapping of variable name → 'categorical' | 'ordinal'.
    reference_categories : dict | None
        Mapping of categorical variable name → desired reference level string.
        Variables absent from this dict keep their default (alphabetical) order.

    Returns
    -------
    pd.DataFrame
        Dataframe with reordered categorical columns (all other columns unchanged).
    """
    if not reference_categories:
        return df

    print("\n[REFERENCE CATEGORY SETUP]")
    for var_name, var_type in condition_vars.items():
        if var_type != "categorical" or var_name not in reference_categories:
            continue

        ref = reference_categories[var_name]

        # Ensure categorical dtype
        if not pd.api.types.is_categorical_dtype(df[var_name]):
            df[var_name] = df[var_name].astype("category")

        current = df[var_name].cat.categories.tolist()

        if ref not in current:
            print(
                f"  ⚠️  '{ref}' not found in '{var_name}'. "
                f"Available: {current}. Using alphabetical default."
            )
        else:
            others = sorted(c for c in current if c != ref)
            df[var_name] = df[var_name].cat.reorder_categories([ref] + others)
            print(f"  ✓ '{var_name}': reference = '{ref}', others = {others}")

    print()
    return df


# ── OLS ───────────────────────────────────────────────────────────────────────

def fit_linear_regression_model(
        df: pd.DataFrame,
        response_var: str,
        condition_vars: dict,
        explanatory_vars: list,
        show_diagnostic_plots: bool = False,
        autocorr_threshold: float = 0.1,
        moderation_pairs: list = None,
        reference_categories: dict = None,
) -> dict:
    """Fit an OLS linear regression model with flexible condition and explanatory variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing all variables.
    response_var : str
        Name of the response variable (e.g. 'PSD').
    condition_vars : dict
        Mapping of variable names to treatment types:
        'categorical' | 'ordinal'.
    explanatory_vars : list
        Additional continuous explanatory variables.
    show_diagnostic_plots : bool
        Whether to display Q-Q and residual distribution plots.
    autocorr_threshold : float
        Minimum |ρ| required to apply SE inflation.
    moderation_pairs : list of tuple, optional
        (MODERATED_VAR, MODERATING_VAR) pairs for interaction terms.
    reference_categories : dict, optional
        Mapping of categorical variable name → reference level string.
        Example: {'Category or Silence': 'Silence'}.

    Returns
    -------
    dict
        Keys: 'model', 'results_df', 'diagnostics'.
    """
    df = df.copy()

    # Apply reference category reordering before any formula construction
    df = _apply_reference_categories(df, condition_vars, reference_categories)

    # Ensure response variable is numeric
    df[response_var] = pd.to_numeric(df[response_var], errors="coerce")

    # Ensure explanatory vars are numeric
    for var in explanatory_vars:
        if var not in condition_vars:
            df[var] = pd.to_numeric(df[var], errors="coerce")

    # Convert condition vars to correct dtypes
    for var_name, var_type in condition_vars.items():
        if var_type == "categorical":
            df[var_name] = df[var_name].astype("category")
        elif var_type == "ordinal":
            df[var_name] = pd.to_numeric(df[var_name], errors="coerce")

    # ── formula construction (unchanged from original) ────────────────────────
    condition_formula_parts = []
    for var_name, var_type in condition_vars.items():
        if var_type == "categorical":
            quoted_var = f"Q('{var_name}')" if " " in var_name else var_name
            condition_formula_parts.append(f"C({quoted_var})")
        elif var_type == "ordinal":
            condition_formula_parts.append(
                f"Q('{var_name}')" if " " in var_name else var_name
            )
        else:
            raise ValueError(f"Unknown variable type: {var_type}")

    explanatory_formula_parts = [
        f"Q('{var}')" if " " in var else var for var in explanatory_vars
    ]
    all_predictors = condition_formula_parts + explanatory_formula_parts

    if moderation_pairs:
        print("\n[MODERATION EFFECTS SPECIFIED]")
        for moderated_var, moderating_var in moderation_pairs:
            moderated_fmt = f"Q('{moderated_var}')" if " " in moderated_var else moderated_var
            moderating_fmt = f"Q('{moderating_var}')" if " " in moderating_var else moderating_var
            if moderated_var in condition_vars and condition_vars[moderated_var] == "categorical":
                moderated_fmt = f"C({moderated_fmt})"
            if moderating_var in condition_vars and condition_vars[moderating_var] == "categorical":
                moderating_fmt = f"C({moderating_fmt})"
            already_present = (
                moderating_var in condition_vars
                or moderating_var in explanatory_vars
                or moderating_fmt in all_predictors
            )
            if not already_present:
                all_predictors.append(moderating_fmt)
                print(f"  Added main effect: {moderating_var}")
            else:
                print(f"  Main effect already present: {moderating_var} (skipped)")
            all_predictors.append(f"{moderated_fmt}:{moderating_fmt}")
            print(f"  Added interaction: {moderated_var} × {moderating_var}")
        print()

    formula = response_var + " ~ " + " + ".join(all_predictors)
    # ── end formula construction ──────────────────────────────────────────────

    print("\n")
    print("-" * 100)
    print("---------------------     Linear Regression Analysis     --------------------- ")
    print("-" * 100, "\n")
    print(f"Formula: {formula}\n")

    print("Data Summary:")
    print(f"Total observations: {len(df)}")
    print(f"Unique participants: {df['Subject ID'].nunique()}")
    print(f"Observations per participant: {len(df) / df['Subject ID'].nunique():.1f}")

    print(f"\nCondition variables:")
    for var_name, var_type in condition_vars.items():
        if var_type == "categorical":
            print(f"  {var_name} (categorical): {df[var_name].nunique()} levels")
            print(f"    {df[var_name].value_counts().to_dict()}")
        elif var_type == "ordinal":
            print(f"  {var_name} (ordinal): range [{df[var_name].min()}, {df[var_name].max()}]")

    print(f"\nResponse variable ({response_var}):")
    print(f"  Range: [{df[response_var].min():.2f}, {df[response_var].max():.2f}]")
    print(f"\nExplanatory variables:")
    for var in explanatory_vars:
        print(f"  {var} range: [{df[var].min():.2f}, {df[var].max():.2f}]")

    # ── OLS fit and all downstream diagnostics (unchanged) ───────────────────
    print("\n" * 3 + "=" * 80)
    print("Linear Regression Model (OLS)")
    print("=" * 80)
    model = smf.ols(formula, data=df).fit()
    print(model.summary())

    residuals = model.resid
    if show_diagnostic_plots:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        stats.probplot(residuals, dist="norm", plot=axes[0])
        axes[0].set_title("Q-Q Plot (Residuals)")
        axes[0].grid(True, alpha=0.3)
        axes[1].hist(residuals, bins=30, edgecolor="black", density=True)
        axes[1].axvline(0, color="r", linestyle="--", label="Mean")
        axes[1].set_title("Residual Distribution")
        axes[1].legend()
        plt.tight_layout()
        plt.show()

    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"\nShapiro-Wilk p-value: {shapiro_p:.4f}")
    print(
        "  → Residuals significantly deviate from normality"
        if shapiro_p < 0.05
        else "  → Residuals approximately normal ✓"
    )

    lag1_autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    if np.isnan(lag1_autocorr):
        lag1_autocorr = 0.0


    ### Design Effect-based SE-Inflation
    # Compute rho and cluster size at the trial level — not the row level.
    # When n_within_trial_segments > 1, len(df) / n_subjects counts segment-rows,
    # making n_trials_per_subject far too large and the design effect enormous.
    # Instead we:
    #   (a) aggregate residuals to their trial mean (one value per trial),
    #   (b) compute lag-1 autocorrelation on those trial-level residuals,
    #   (c) use the mean number of trials per subject as the cluster size.
    # This ensures the design effect only captures between-trial dependence,
    # which is what the Kish formula requires.
    if "Trial ID" in df.columns:
        trial_resid = (
            pd.Series(residuals, index=df.index)
            .groupby(df["Trial ID"])
            .mean()
        )
        rho_raw = np.corrcoef(trial_resid.values[:-1], trial_resid.values[1:])[0, 1]
        rho_for_deff = 0.0 if np.isnan(rho_raw) else rho_raw
        n_trials_per_subject = df.groupby("Subject ID")["Trial ID"].nunique().mean()
    else:
        # Fallback when Trial ID is absent: use row-level autocorrelation.
        # This may overestimate the design effect at high segment counts.
        rho_for_deff = lag1_autocorr
        n_trials_per_subject = len(df) / df["Subject ID"].nunique()

    # Apply SE inflation only when between-trial autocorrelation exceeds threshold.
    # design_effect = 1 + (n - 1) * rho  (Kish 1965)
    # Inflation is capped at positive rho to prevent SE deflation.
    if abs(rho_for_deff) < autocorr_threshold:
        design_effect, se_inflation, inflation_applied = 1.0, 1.0, False
    else:
        design_effect = 1 + (n_trials_per_subject - 1) * max(0, rho_for_deff)
        se_inflation = np.sqrt(design_effect)
        inflation_applied = True


    adjusted_se = model.bse * se_inflation
    adjusted_z = model.params / adjusted_se
    adjusted_p = pd.Series(
        2 * (1 - stats.norm.cdf(np.abs(adjusted_z))), index=model.params.index
    )

    results_data = []
    for param in model.params.index:
        results_data.append({
            "Parameter": param,
            "Coefficient": model.params[param],
            "SE (unadjusted)": model.bse[param],
            "SE (adjusted)": adjusted_se[param],
            "p-value (unadjusted)": model.pvalues[param],
            "p-value (adjusted)": adjusted_p[param],
        })

    # Decompose total OLS residual variance into between- and within-subject components.
    # This mirrors LME's __residual_std__ / __re_std__ sentinels so that the power
    # analysis simulation receives realistic generative parameters even when run
    # from OLS estimates.
    subj_mean_resid = (
        pd.Series(residuals, index=df.index)
        .groupby(df["Subject ID"])
        .mean()
    )
    var_between = float(np.var(subj_mean_resid, ddof=1))
    var_within = max(float(model.mse_resid) - var_between, 0.0)

    results_data.append({
        "Parameter": "__residual_std__",
        "Coefficient": float(np.sqrt(var_within)),
        "SE (unadjusted)": np.nan,
        "SE (adjusted)": np.nan,
        "p-value (unadjusted)": np.nan,
        "p-value (adjusted)": np.nan,
    })
    results_data.append({
        "Parameter": "__re_std__",
        "Coefficient": float(np.sqrt(max(var_between, 0.0))),
        "SE (unadjusted)": np.nan,
        "SE (adjusted)": np.nan,
        "p-value (unadjusted)": np.nan,
        "p-value (adjusted)": np.nan,
    })

    # var_between / var_within already computed above for the sentinels.
    # Use the decomposed within-subject SD to match LME's result.scale semantics.
    # Store total residual SD separately for Cohen's d, which needs the full
    # unexplained SD regardless of between/within decomposition.
    residual_std_within = float(np.sqrt(var_within))
    residual_std_total = float(np.sqrt(model.mse_resid))

    diagnostics = {
        "n_observations": len(df),
        "n_trials_per_subject": n_trials_per_subject,
        "shapiro_stat": shapiro_stat,
        "shapiro_p": shapiro_p,
        "lag1_autocorr": lag1_autocorr,
        "rho_for_deff": rho_for_deff,
        "design_effect": design_effect,
        "se_inflation": se_inflation,
        "autocorr_threshold": autocorr_threshold,
        "inflation_applied": inflation_applied,
        "r_squared": model.rsquared,
        "r_squared_adj": model.rsquared_adj,
        # residual_std mirrors LME semantics: pure within-subject error SD.
        # Used for power simulation via __residual_std__ sentinel.
        "residual_std": residual_std_within,
        # total_residual_std = sqrt(MSE_resid): used for Cohen's d denominator.
        # In LME this would be sqrt(scale + var_random); in OLS it is sqrt(MSE_resid).
        "total_residual_std": residual_std_total,
        "icc": None,  # undefined for OLS
    }

    return {
        "model": model,
        "results_df": pd.DataFrame(results_data),
        "diagnostics": diagnostics,
    }


# ── LME ───────────────────────────────────────────────────────────────────────

def fit_mixed_effects_model(
        df: pd.DataFrame,
        response_var: str,
        condition_vars: dict,
        explanatory_vars: list,
        grouping_var: str = "Subject ID",
        show_diagnostic_plots: bool = False,
        autocorr_threshold: float = 0.1,
        moderation_pairs: list = None,
        reference_categories: dict = None,
) -> dict:
    """Fit a Linear Mixed-Effects Model with flexible condition and explanatory variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing all variables.
    response_var : str
        Name of the response variable.
    condition_vars : dict
        Mapping of variable names to 'categorical' | 'ordinal'.
    explanatory_vars : list
        Additional continuous explanatory variables.
    grouping_var : str
        Grouping variable for random intercepts (default: 'Subject ID').
    show_diagnostic_plots : bool
        Whether to display Q-Q and residual distribution plots.
    autocorr_threshold : float
        Minimum |ρ| required to apply SE inflation.
    moderation_pairs : list of tuple, optional
        (MODERATED_VAR, MODERATING_VAR) pairs for interaction terms.
    reference_categories : dict, optional
        Mapping of categorical variable name → reference level string.

    Returns
    -------
    dict
        Keys: 'model', 'result', 'results_df', 'random_effects_df', 'diagnostics'.
    """
    df = df.copy()

    # Apply reference category reordering before any formula construction
    df = _apply_reference_categories(df, condition_vars, reference_categories)

    # ── all remaining logic identical to original ─────────────────────────────
    df[response_var] = pd.to_numeric(df[response_var], errors="coerce")
    for var in explanatory_vars:
        if var not in condition_vars:
            df[var] = pd.to_numeric(df[var], errors="coerce")
    for var_name, var_type in condition_vars.items():
        if var_type == "categorical":
            df[var_name] = df[var_name].astype("category")
        elif var_type == "ordinal":
            df[var_name] = pd.to_numeric(df[var_name], errors="coerce")

    condition_formula_parts = []
    for var_name, var_type in condition_vars.items():
        if var_type == "categorical":
            quoted_var = f"Q('{var_name}')" if " " in var_name else var_name
            condition_formula_parts.append(f"C({quoted_var})")
        elif var_type == "ordinal":
            condition_formula_parts.append(
                f"Q('{var_name}')" if " " in var_name else var_name
            )
        else:
            raise ValueError(f"Unknown variable type: {var_type}")

    explanatory_formula_parts = [
        f"Q('{var}')" if " " in var else var for var in explanatory_vars
    ]
    all_predictors = condition_formula_parts + explanatory_formula_parts

    if moderation_pairs:
        print("\n[MODERATION EFFECTS SPECIFIED]")
        for moderated_var, moderating_var in moderation_pairs:
            moderated_fmt = f"Q('{moderated_var}')" if " " in moderated_var else moderated_var
            moderating_fmt = f"Q('{moderating_var}')" if " " in moderating_var else moderating_var
            if moderated_var in condition_vars and condition_vars[moderated_var] == "categorical":
                moderated_fmt = f"C({moderated_fmt})"
            if moderating_var in condition_vars and condition_vars[moderating_var] == "categorical":
                moderating_fmt = f"C({moderating_fmt})"
            already_present = (
                moderating_var in condition_vars
                or moderating_var in explanatory_vars
                or moderating_fmt in all_predictors
            )
            if not already_present:
                all_predictors.append(moderating_fmt)
                print(f"  Added main effect: {moderating_var}")
            else:
                print(f"  Main effect already present: {moderating_var} (skipped)")
            all_predictors.append(f"{moderated_fmt}:{moderating_fmt}")
            print(f"  Added interaction: {moderated_var} × {moderating_var}")
        print()

    formula = response_var + " ~ " + " + ".join(all_predictors)

    print("\n")
    print("-" * 100)
    print(f"---------------------     Linear Mixed-Effects Model Analysis     --------------------- ")
    print("-" * 100, "\n")
    print(f"Formula (fixed effects): {formula}")
    print(f"Random effects: Random intercept by {grouping_var}\n")

    ### DATA CHECK
    print("Data Summary:")
    print(f"Total observations: {len(df)}")
    print(f"Unique {grouping_var}: {df[grouping_var].nunique()}")
    print(
        f"Observations per {grouping_var}: {len(df) / df[grouping_var].nunique():.1f}")

    print(f"\nCondition variables:")
    for var_name, var_type in condition_vars.items():
        if var_type == 'categorical':
            print(f"  {var_name} (categorical): {df[var_name].nunique()} levels")
            print(f"    Levels: {df[var_name].cat.categories.tolist()}")
            print(f"    {df[var_name].value_counts().to_dict()}")
        elif var_type == 'ordinal':
            print(f"  {var_name} (ordinal): range [{df[var_name].min()}, {df[var_name].max()}]")
            print(f"    Distribution: {df[var_name].value_counts().sort_index().to_dict()}")

    print(f"\nResponse variable ({response_var}):")
    print(f"  dtype: {df[response_var].dtype}")
    print(f"  Range: [{df[response_var].min():.2f}, {df[response_var].max():.2f}]")
    print(f"  Non-null count: {df[response_var].notna().sum()} / {len(df)}")

    print(f"\nExplanatory variables:")
    for var in explanatory_vars:
        print(f"  {var} dtype: {df[var].dtype}, range: [{df[var].min():.2f}, {df[var].max():.2f}]")

    # Remove rows with NaN in key columns
    cols_to_check = [response_var, grouping_var] + list(condition_vars.keys()) + explanatory_vars
    df = df.dropna(subset=cols_to_check)
    print(f"\nAfter removing NaN: {len(df)} observations")

    ### LINEAR MIXED-EFFECTS MODEL
    print("\n" * 3 + "=" * 80)
    print("Linear Mixed-Effects Model (LME)")
    print("=" * 80)
    model = smf.mixedlm(formula, data=df, groups=df[grouping_var])


    ## Pre-fitting rank deficiency check
    X = model.exog
    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        print(
            f"  ⚠️  Rank-deficient design matrix: rank={rank}, n_params={X.shape[1]} "
            f"({X.shape[1] - rank} redundant columns). Skipping LME fit."
        )
        return None  # caller must handle None

    try:
        result = model.fit(reml=True, method=["lbfgs", "bfgs", "cg"])
    except (np.linalg.LinAlgError) as e:
        print(f"  ⚠️  LME singular matrix ({type(e).__name__}): {e}. Skipping.")
        return None


    ## Fitting
    print(result.summary())

    # ============ DIAGNOSTICS ============
    residuals = result.resid

    # Q-Q plot
    if show_diagnostic_plots:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        stats.probplot(residuals, dist="norm", plot=axes[0])
        axes[0].set_title("Q-Q Plot (Residuals)")
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(residuals, bins=30, edgecolor='black', density=True)
        axes[1].axvline(0, color='r', linestyle='--', label='Mean')
        axes[1].set_title("Residual Distribution")
        axes[1].legend()
        plt.tight_layout()
        plt.show()

    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"\nShapiro-Wilk p-value: {shapiro_p:.4f}")
    if shapiro_p < 0.05:
        print("  → Residuals significantly deviate from normality (consider transformation)")
    else:
        print("  → Residuals approximately normal ✓")

    # ============ AUTOCORRELATION CHECK ============
    print("\n" + "=" * 80)
    print("AUTOCORRELATION CHECK (temporal correlation between trials)")
    print("=" * 80)

    lag1_autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    if np.isnan(lag1_autocorr):
        print("Warning: Autocorrelation is NaN (constant residuals?)")
        lag1_autocorr = 0.0

    # Compute rho and cluster size at the trial level — not the row level.
    # When n_within_trial_segments > 1, len(df) / n_subjects counts segment-rows,
    # making n_trials_per_subject far too large and the design effect enormous.
    # Instead we:
    #   (a) aggregate residuals to their trial mean (one value per trial),
    #   (b) compute lag-1 autocorrelation on those trial-level residuals,
    #   (c) use the mean number of trials per subject as the cluster size.
    # This ensures the design effect only captures between-trial dependence
    # (temporal correlation across trials, NOT within-trial segments), which is
    # what the Kish formula is designed for. Within-trial structure is already
    # partially accounted for by the Segment ID covariate in the fixed effects.
    if "Trial ID" in df.columns:
        trial_resid = (
            pd.Series(residuals, index=df.index)
            .groupby(df["Trial ID"])
            .mean()
        )
        rho_raw = np.corrcoef(trial_resid.values[:-1], trial_resid.values[1:])[0, 1]
        rho_for_deff = 0.0 if np.isnan(rho_raw) else rho_raw
        n_trials_per_subject = df.groupby(grouping_var)["Trial ID"].nunique().mean()
    else:
        # Fallback when Trial ID is absent: use row-level autocorrelation.
        # This may overestimate the design effect at high segment counts.
        rho_for_deff = lag1_autocorr
        n_trials_per_subject = len(df) / df[grouping_var].nunique()

    # Apply SE inflation only when between-trial autocorrelation exceeds threshold.
    # design_effect = 1 + (n - 1) * rho  (Kish 1965)
    # Inflation is capped at positive rho only (prevents SE deflation for
    # negative autocorrelation, which is conservative but safe at n=11).
    if abs(rho_for_deff) < autocorr_threshold:
        design_effect = 1.0
        se_inflation = 1.0
        inflation_applied = False
    else:
        design_effect = 1 + (n_trials_per_subject - 1) * max(0, rho_for_deff)
        se_inflation = np.sqrt(design_effect)
        inflation_applied = True

    print(f"Lag-1 autocorrelation (ρ): {lag1_autocorr:.3f}")
    print(f"Average trials per subject: {n_trials_per_subject:.1f}")
    print(
        f"SE inflation threshold: |ρ| > {autocorr_threshold:.2f} (balancing power with n={df[grouping_var].nunique()} subjects)")
    print(f"Design effect: {design_effect:.2f}")
    print(f"SE inflation factor: {se_inflation:.2f}×")

    if not inflation_applied:
        if abs(lag1_autocorr) < autocorr_threshold:
            print(f"\n✓ Autocorrelation below threshold (|ρ| = {abs(lag1_autocorr):.3f} < {autocorr_threshold:.2f})")
            print(f"→ No SE inflation applied (preserving statistical power)")
        else:
            print(f"\n✓ Negative autocorrelation (ρ = {lag1_autocorr:.3f})")
            print(f"→ No inflation applied (SE inflation capped at 1.0×)")
    else:
        if lag1_autocorr > 0.2:
            print(f"\n⚠️  High autocorrelation detected (ρ = {lag1_autocorr:.3f})")
            print(f"→ Inflating SEs by {se_inflation:.2f}× to account for strong temporal dependence")
        else:
            print(f"\nℹ️  Moderate autocorrelation (ρ = {lag1_autocorr:.3f} > threshold)")
            print(f"→ Inflating SEs by {se_inflation:.2f}× (threshold balances Type I error control with power)")

    # Adjust SEs and p-values for FIXED EFFECTS ONLY
    adjusted_se = result.bse.loc[result.fe_params.index] * se_inflation
    adjusted_z = result.fe_params / adjusted_se
    adjusted_p = 2 * (1 - stats.norm.cdf(np.abs(adjusted_z)))

    # Ensure it's a Series
    if not isinstance(adjusted_p, pd.Series):
        adjusted_p = pd.Series(adjusted_p, index=result.fe_params.index)

    # ============ RESULTS TABLE ============
    print("\n" + "-" * 80)
    print("ADJUSTED FIXED EFFECTS (corrected for autocorrelation):")
    print("-" * 80)
    print(f"{'Parameter':<50s} {'β':>10s} {'SE (orig)':>12s} {'SE (adj)':>12s} {'p (orig)':>10s} {'p (adj)':>10s}")
    print("-" * 80)

    results_data = []
    for param in result.fe_params.index:
        orig_se = result.bse[param]
        adj_se = adjusted_se[param]
        orig_p = result.pvalues[param]
        adj_p = adjusted_p[param]
        print(
            f"{param:<50s} {result.fe_params[param]:>10.4f} {orig_se:>12.4f} {adj_se:>12.4f} {orig_p:>10.4f} {adj_p:>10.4f}")

        results_data.append({
            'Parameter': param,
            'Coefficient': result.fe_params[param],
            'SE (unadjusted)': orig_se,
            'SE (adjusted)': adj_se,
            'p-value (unadjusted)': orig_p,
            'p-value (adjusted)': adj_p
        })

    # Append variance component sentinels for downstream power analysis:
    re_var = result.cov_re.iloc[0, 0] if isinstance(result.cov_re, pd.DataFrame) else float(result.cov_re)
    results_data.append({
        "Parameter": "__residual_std__",
        "Coefficient": float(np.sqrt(result.scale)),
        "SE (unadjusted)": np.nan,
        "SE (adjusted)": np.nan,
        "p-value (unadjusted)": np.nan,
        "p-value (adjusted)": np.nan,
    })
    results_data.append({
        "Parameter": "__re_std__",
        "Coefficient": float(np.sqrt(max(re_var, 0.0))),
        "SE (unadjusted)": np.nan,
        "SE (adjusted)": np.nan,
        "p-value (unadjusted)": np.nan,
        "p-value (adjusted)": np.nan,
    })

    results_df = pd.DataFrame(results_data)

    # ============ RANDOM EFFECTS ============
    print("\n" + "-" * 80)
    print("RANDOM EFFECTS (Random Intercepts by Group):")
    print("-" * 80)

    random_effects = result.random_effects
    random_effects_df = pd.DataFrame([
        {grouping_var: group, 'Random Intercept': re['Group']}
        for group, re in random_effects.items()
    ])
    print(random_effects_df.to_string(index=False))

    print(f"\nRandom Intercept SD: {np.std(list(random_effects_df['Random Intercept'])):.4f}")

    # Calculate marginal and conditional R-squared for LME
    # Marginal R²: variance explained by fixed effects only
    # Conditional R²: variance explained by fixed + random effects
    print("\n[DEBUG] Computing R² metrics...")
    print(f"  result type: {type(result)}")
    print(f"  hasattr(result, 'cov_re'): {hasattr(result, 'cov_re')}")
    print(f"  type(result.cov_re): {type(result.cov_re) if hasattr(result, 'cov_re') else 'N/A'}")

    try:
        # Extract random intercept variance (cov_re is a pandas DataFrame in statsmodels)
        # It contains the random effects covariance matrix
        if hasattr(result, 'cov_re'):
            # cov_re is the random effects covariance matrix
            if isinstance(result.cov_re, pd.DataFrame):
                var_random = result.cov_re.iloc[0, 0]
            else:
                # If it's an array or scalar
                var_random = float(result.cov_re)
        else:
            var_random = 0.0

        # Calculate variance components
        # Fixed effects variance: variance of predictions using only fixed effects
        var_fixed = np.var(result.model.predict(result.fe_params, exog=result.model.exog))

        # Residual variance
        var_residual = result.scale
        print(f"  var_fixed={var_fixed:.4f}, var_random={var_random:.4f}, var_residual={var_residual:.4f}")

        r2_marginal = var_fixed / (var_fixed + var_random + var_residual)
        r2_conditional = (var_fixed + var_random) / (var_fixed + var_random + var_residual)
        print(f"✓ R² metrics computed successfully: R²_marginal={r2_marginal:.4f}, R²_conditional={r2_conditional:.4f}")

        # ICC: proportion of total variance attributable to between-subject
        # differences, ignoring fixed-effect variance.
        # Formula: var_random / (var_random + var_residual)
        # This is the standard random-intercept ICC (Nakagawa & Schielzeth 2013).
        # Note: NOT r2_conditional - r2_marginal — that difference is diluted by
        # var_fixed in the denominator and understates the clustering ratio.
        denom_icc = var_random + var_residual
        icc = float(var_random / denom_icc) if denom_icc > 0 else None
        print(f"  ICC = {icc:.4f}" if icc is not None else "  ICC: undefined (zero variance denominator)")


    except (AttributeError, KeyError, IndexError, TypeError) as e:
        # Fallback with detailed error message
        print(f"⚠️  Warning: Could not compute R² metrics: {type(e).__name__}: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        r2_marginal = r2_conditional = icc = None


    # Store diagnostics with defensive attribute access
    print("\n[DEBUG] Extracting model fit metrics...")
    print(f"  hasattr(result, 'llf'): {hasattr(result, 'llf')}")
    print(f"  hasattr(result, 'aic'): {hasattr(result, 'aic')}")
    print(f"  hasattr(result, 'bic'): {hasattr(result, 'bic')}")

    log_likelihood = getattr(result, 'llf', None)
    aic = getattr(result, 'aic', None)
    bic = getattr(result, 'bic', None)

    print(f"  Extracted values: llf={log_likelihood}, aic={aic}, bic={bic}")

    # If AIC/BIC are NaN, calculate them manually
    if log_likelihood is not None and (aic is None or np.isnan(aic) or bic is None or np.isnan(bic)):
        print("  ⚠️  AIC/BIC are NaN, calculating manually...")
        # Count parameters: fixed effects + random effect variance + residual variance
        n_fixed_params = len(result.fe_params)
        n_random_params = 1  # Random intercept variance
        n_residual_params = 1  # Residual variance
        k = n_fixed_params + n_random_params + n_residual_params
        n = len(df)

        # Manual calculation: AIC = -2*llf + 2*k, BIC = -2*llf + k*log(n)
        if aic is None or np.isnan(aic):
            aic = -2 * log_likelihood + 2 * k
            print(f"    Calculated AIC = -2*{log_likelihood:.2f} + 2*{k} = {aic:.2f}")

        if bic is None or np.isnan(bic):
            bic = -2 * log_likelihood + k * np.log(n)
            print(f"    Calculated BIC = -2*{log_likelihood:.2f} + {k}*log({n}) = {bic:.2f}")

    diagnostics = {
        "n_observations": len(df),
        "shapiro_stat": shapiro_stat,
        "shapiro_p": shapiro_p,
        "lag1_autocorr": lag1_autocorr,
        "rho_for_deff": rho_for_deff,
        "n_trials_per_subj": n_trials_per_subject,
        "design_effect": design_effect,
        "se_inflation": se_inflation,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
        "r_squared_marginal": r2_marginal,
        "r_squared_conditional": r2_conditional,
        # residual_std = sqrt(result.scale): pure within-subject error SD (REML). Matches OLS semantics.
        "residual_std": float(np.sqrt(result.scale)),
        # total_residual_std = sqrt(scale + var_random): full unexplained SD.
        # Used for Cohen's d denominator — comparable to OLS's sqrt(mse_resid).
        "total_residual_std": float(np.sqrt(result.scale + max(re_var, 0.0))),
        "icc": icc,
    }

    print("\n" + "=" * 80)
    print("MODEL FIT STATISTICS:")
    if log_likelihood is not None:
        print(f"  Log-Likelihood: {log_likelihood:.2f}")
    else:
        print("  Log-Likelihood: N/A")

    if aic is not None and not np.isnan(aic):
        print(f"  AIC: {aic:.2f}")
    else:
        print("  AIC: N/A (failed to calculate)")

    if bic is not None and not np.isnan(bic):
        print(f"  BIC: {bic:.2f}")
    else:
        print("  BIC: N/A (failed to calculate)")

    if r2_marginal is not None:
        print(f"R² (marginal - fixed only): {r2_marginal:.4f}")
        print(f"R² (conditional - fixed+random): {r2_conditional:.4f}")
    else:
        print("R² metrics: N/A (calculation failed)")
    print("=" * 80)

    return {
        'model': model,
        'result': result,
        'results_df': results_df,
        'random_effects_df': random_effects_df,
        'diagnostics': diagnostics
    }





# ============================================================================
# MODULAR MODEL FITTING FUNCTIONS
# ============================================================================
def fit_both_models(
        df: pd.DataFrame,
        response_var: str,
        condition_vars: dict,
        explanatory_vars: list,
        comparison_level_name: str,
        hypothesis_name: str,
        n_windows_per_trial: int = 9,
        show_diagnostic_plots: bool = False,
        reference_categories: dict = None,
        moderation_pairs: list = None,
) -> dict:
    """Fit both OLS and LME models, delegating reference-category setup to sub-functions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    response_var : str
        Dependent variable.
    condition_vars : dict
        Condition variables with types.
    explanatory_vars : list
        Explanatory variables.
    comparison_level_name : str
        Name for reporting.
    hypothesis_name : str
        Hypothesis name.
    n_windows_per_trial : int
        Segments per trial.
    show_diagnostic_plots : bool
        Show diagnostic plots.
    reference_categories : dict, optional
        Mapping of categorical variable name → reference level string.
    moderation_pairs : list of tuple, optional
        (MODERATED_VAR, MODERATING_VAR) pairs.

    Returns
    -------
    dict
        Keys: 'OLS', 'LME'.
    """
    print("\n" + "=" * 100)
    print(f"HYPOTHESIS: {hypothesis_name}")
    print(f"DEPENDENT VARIABLE: {response_var}")
    print(f"COMPARISON LEVEL: {comparison_level_name}")
    print("=" * 100)

    # Reference reordering now happens inside each sub-function —
    # no duplication needed here; just pass the dict through.

    return {
        "OLS": fit_linear_regression_model(
            df=df,
            response_var=response_var,
            condition_vars=condition_vars,
            explanatory_vars=explanatory_vars,
            show_diagnostic_plots=show_diagnostic_plots,
            moderation_pairs=moderation_pairs,
            reference_categories=reference_categories,
        ),
        "LME": fit_mixed_effects_model(
            df=df,
            response_var=response_var,
            condition_vars=condition_vars,
            explanatory_vars=explanatory_vars,
            grouping_var="Subject ID",
            show_diagnostic_plots=show_diagnostic_plots,
            moderation_pairs=moderation_pairs,
            reference_categories=reference_categories,
        ),
    }


def apply_fdr_correction(
    results_df: pd.DataFrame,
    levels_to_correct: list[int],
    alpha: float = 0.05,
    group_by_dv: bool = True,
) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction within each comparison level.

    Correction is applied separately per level, and optionally per DV within
    each level, to keep the test family conceptually coherent. Confirmatory
    levels (0, 1) are typically excluded; exploratory levels (2, 3) corrected.

    Sentinel rows (__residual_std__, __re_std__, Intercept) are excluded from
    the correction family and receive NaN in the FDR column.

    Parameters
    ----------
    results_df : pd.DataFrame
        Full results frame with p_value_adjusted column.
    levels_to_correct : list[int]
        Level indices to apply BH correction to, e.g. [2, 3].
    alpha : float
        FDR threshold. Default 0.05.
    group_by_dv : bool
        If True, correct within each (Level × DV) stratum — smaller, more
        conservative families. If False, correct across all DVs within a level
        — larger family, more power but mixes hypotheses. Default True.

    Returns
    -------
    pd.DataFrame
        Input frame with two new columns added:
        - p_value_fdr     : BH-corrected p-value (NaN for excluded rows)
        - significant_fdr : bool, p_value_fdr < alpha
    """
    from statsmodels.stats.multitest import multipletests

    df = results_df.copy()
    df["p_value_fdr"] = np.nan
    df["significant_fdr"] = False

    # Rows eligible for correction: real parameters only, LME only
    _SENTINEL = {"__residual_std__", "__re_std__"}
    eligible_mask = (
        (df["Model_Type"] == "LME")
        & df["Parameter"].apply(lambda p: p not in _SENTINEL and not p.startswith("Intercept"))
        & df["Comparison_Level"].apply(
            lambda lvl: any(lvl.startswith(f"Level {i} ") for i in levels_to_correct)
        )
    )

    if not eligible_mask.any():
        print("  [FDR] No eligible rows found for the specified levels.")
        return df

    eligible = df[eligible_mask].copy()

    # Define grouping keys
    group_cols = ["Comparison_Level", "N. Segments"]
    if group_by_dv:
        group_cols.append("Dependent_Variable")

    n_corrected = 0
    for group_keys, grp in eligible.groupby(group_cols):
        p_vals = grp["p_value_adjusted"].values
        valid = ~np.isnan(p_vals)

        if valid.sum() < 2:
            continue  # BH undefined for single test

        reject, p_fdr, _, _ = multipletests(
            p_vals[valid], alpha=alpha, method="fdr_bh"
        )

        # Write back to the full df using original index
        valid_idx = grp.index[valid]
        df.loc[valid_idx, "p_value_fdr"] = p_fdr
        df.loc[valid_idx, "significant_fdr"] = reject
        n_corrected += valid.sum()

    n_sig_before = eligible_mask.sum() and (df.loc[eligible_mask, "p_value_adjusted"] < alpha).sum()
    n_sig_after  = df.loc[eligible_mask, "significant_fdr"].sum()

    level_str = ", ".join(f"Level {i}" for i in levels_to_correct)
    group_str  = "per DV" if group_by_dv else "pooled across DVs"
    print(
        f"\n  [FDR] BH correction applied to {level_str} ({group_str}):\n"
        f"    {n_corrected} parameters corrected\n"
        f"    Significant before: {n_sig_before} → after: {n_sig_after} "
        f"(α_FDR = {alpha})"
    )

    # add fallback column:
    df["p_value_for_plot"] = df["p_value_fdr"].fillna(df["p_value_adjusted"])
    return df


def store_model_results(
        model_results: dict,
        hypothesis_name: str,
        dependent_variable: str,
        comparison_level_name: str,
        all_results_list: list,
        diagnostics_list: list = None,
) -> None:
    """Store results from both models into list.

    Appends one row per parameter to all_results_list (including Cohen_d),
    and one diagnostics row per model to diagnostics_list (including ICC).

    Parameters
    ----------
    model_results : dict
        Output from fit_both_models().
    hypothesis_name : str
        Hypothesis name.
    dependent_variable : str
        Dependent variable.
    comparison_level_name : str
        Comparison level.
    all_results_list : list
        Results accumulator (modified in place).
    diagnostics_list : list, optional
        Diagnostics accumulator (modified in place).
    """
    # Sentinel rows written by fit_mixed_effects_model — not real parameters,
    # so Cohen's d must be suppressed for them.
    _SENTINEL_PARAMS = {"__residual_std__", "__re_std__"}

    for model_type in ["OLS", "LME"]:
        if model_type not in model_results or model_results[model_type] is None:
            continue

        model_out = model_results[model_type]
        diag = model_out.get("diagnostics", {})

        # Residual SD is the Cohen's d denominator; None for both models if
        # the fit failed to compute it (graceful degradation).
        residual_std = diag.get("total_residual_std", None)

        # Store parameter results — one row per fixed-effect parameter
        for _, row in model_out["results_df"].iterrows():
            param = row["Parameter"]

            # Cohen's d = β / σ_residual.
            # Undefined for: sentinels, Intercept, or when residual_std unavailable.
            is_sentinel  = param in _SENTINEL_PARAMS
            is_intercept = param == "Intercept"
            if (
                residual_std is not None
                and residual_std > 0
                and not is_sentinel
                and not is_intercept
            ):
                cohens_d = float(row["Coefficient"]) / residual_std
            else:
                cohens_d = None

            all_results_list.append({
                "Hypothesis":           hypothesis_name,
                "Dependent_Variable":   dependent_variable,
                "Model_Type":           model_type,
                "Comparison_Level":     comparison_level_name,
                "Parameter":            param,
                "Coefficient":          row["Coefficient"],
                "SE_unadjusted":        row["SE (unadjusted)"],
                "SE_adjusted":          row["SE (adjusted)"],
                "p_value_unadjusted":   row["p-value (unadjusted)"],
                "p_value_adjusted":     row["p-value (adjusted)"],
                "p_value":              row["p-value (adjusted)"],  # backward compat
                "SE":                   row["SE (adjusted)"],       # backward compat
                "Cohen_d":              cohens_d,
            })

        # Store diagnostics
        if diagnostics_list is not None:
            try:
                if not diag:
                    continue

                diag_entry = {
                    "Hypothesis":           hypothesis_name,
                    "Dependent_Variable":   dependent_variable,
                    "Model_Type":           model_type,
                    "Comparison_Level":     comparison_level_name,
                    "N_Observations":       diag.get("n_observations",  None),
                    "Shapiro_p":            diag.get("shapiro_p",        None),
                    "Shapiro_Violated":     "Yes" if diag.get("shapiro_p", 1.0) < 0.05 else "No",
                    "Lag1_Autocorr":        diag.get("lag1_autocorr",    None),
                    "Design_Effect":        diag.get("design_effect",    None),
                    "SE_Inflation":         diag.get("se_inflation",     None),
                    # OLS-specific (None for LME)
                    "R_squared":            diag.get("r_squared",        None),
                    "R_squared_adj":        diag.get("r_squared_adj",    None),
                    # LME-specific fit indices (None for OLS)
                    "AIC":                  diag.get("aic",              None),
                    "BIC":                  diag.get("bic",              None),
                    "LogLik":               diag.get("log_likelihood",   None),
                    "R_squared_marginal":   diag.get("r_squared_marginal",   None),
                    "R_squared_conditional": diag.get("r_squared_conditional", None),
                    # ICC now populated from fit_mixed_effects_model;
                    # remains None for OLS (no random structure).
                    "ICC":                  diag.get("icc",              None),
                }
                diagnostics_list.append(diag_entry)

            except Exception as e:
                print(
                    f"⚠️  Warning: Could not store diagnostics for "
                    f"{hypothesis_name} - {model_type}: {e}"
                )



# ============================================================================
# SUBJECT-LEVEL ANALYSIS FUNCTIONS
# ============================================================================

def create_subject_effect_summary(
        all_model_results: list,
        original_data: pd.DataFrame,
        output_dir: Path,
        level_definitions: list[dict],
        subject_col: str = "Subject ID",
        save_pivot_tables: bool = False,
) -> pd.DataFrame:
    """Create subject-level marginal summaries and per-level condition contrasts.

    This function produces:
    1) A subject-level marginal summary (mean/std over all rows for a DV).
    2) A condition-contrast table computed separately per comparison level
       (honouring each level's df_filter).
    3) A combined table that joins (1) and (2) and computes reference-based
       raw and normalised contrasts.

    Pivot tables are optional and are saved into output_dir when enabled.

    Parameters
    ----------
    all_model_results : list
        Accumulated output of `store_model_results` across hypotheses and levels.
        Internal sentinel rows (Parameter starting with '__') are ignored.
    original_data : pd.DataFrame
        The combined statistics dataframe used for modelling (all subjects).
    output_dir : Path
        Directory to save CSV outputs to.
    level_definitions : list[dict]
        Output of `fetch_level_definitions(...)`. Each entry is a level definition
        dict containing at least 'condition_vars' and optionally 'df_filter'.
    subject_col : str
        Subject identifier column name.
    save_pivot_tables : bool
        If True, saves pivot CSVs into output_dir.

    Returns
    -------
    pd.DataFrame
        Combined subject × condition × comparison-level table (also saved).
    """
    # --- Ensure output directory exists ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Restrict to LME runs only (defines which Hypothesis × DV pairs are relevant) ---
    results_df = pd.DataFrame(all_model_results)
    if "Parameter" in results_df.columns:
        results_df = results_df[~results_df["Parameter"].astype(str).str.startswith("__")]
    lme_results = results_df[results_df["Model_Type"] == "LME"].copy()

    # --- Collect per-subject marginal summaries + per-level condition summaries ---
    join_keys = ["Hypothesis", "Dependent_Variable", subject_col]
    subject_summaries: list[dict] = []
    contrast_summaries: list[dict] = []

    for hypothesis in lme_results["Hypothesis"].dropna().unique():
        hyp_lme = lme_results[lme_results["Hypothesis"] == hypothesis]

        for dv in hyp_lme["Dependent_Variable"].dropna().unique():
            print(f"\n{'─' * 120}")
            print(f"Hypothesis: {hypothesis} | DV: {dv}")
            print(f"{'─' * 120}")

            # --- Per-subject baseline (all rows pooled) ---
            for subject_id in sorted(original_data[subject_col].dropna().unique()):
                subj_all = original_data[
                    (original_data[subject_col] == subject_id) & original_data[dv].notna()
                ]
                if subj_all.empty:
                    continue

                mean_val = float(subj_all[dv].mean())
                std_val = float(subj_all[dv].std())
                n_obs = int(len(subj_all))

                subject_summaries.append({
                    "Hypothesis": hypothesis,
                    "Dependent_Variable": dv,
                    subject_col: subject_id,
                    "Marginal_Mean": mean_val,
                    "Marginal_Std": std_val,
                    "N_Observations": n_obs,
                })
                print(f"  Subject {int(subject_id):02d}: Mean={mean_val:>8.4f}, Std={std_val:>8.4f}, N={n_obs:>4d}")

                # --- Per-level categorical condition means (keep all levels, filter later) ---
                for level_idx, level_def in enumerate(level_definitions):
                    comp_level = f"lvl_{level_idx}"

                    # Apply the level-specific filter (important: levels can differ conceptually)
                    if level_def.get("df_filter") is not None:
                        try:
                            subj_lvl = level_def["df_filter"](subj_all)
                        except Exception as e:
                            print(f"  ⚠️  df_filter failed for {comp_level}: {type(e).__name__}: {e}")
                            continue
                    else:
                        subj_lvl = subj_all

                    subj_lvl = subj_lvl[subj_lvl[dv].notna()]
                    if subj_lvl.empty:
                        continue

                    for var_name, var_type in level_def.get("condition_vars", {}).items():
                        if var_type != "categorical":
                            continue
                        if var_name not in subj_lvl.columns:
                            continue

                        for condition in subj_lvl[var_name].dropna().unique():
                            cond_data = subj_lvl[subj_lvl[var_name] == condition]
                            if cond_data.empty:
                                continue

                            contrast_summaries.append({
                                "Hypothesis": hypothesis,
                                "Dependent_Variable": dv,
                                subject_col: subject_id,
                                "Comparison_Level": comp_level,
                                "Condition_Variable": var_name,
                                "Condition": condition,
                                "Condition_Mean": float(cond_data[dv].mean()),
                                "Condition_Std": float(cond_data[dv].std()),
                                "N": int(len(cond_data)),
                            })

    if not subject_summaries or not contrast_summaries:
        print("⚠️  No summaries generated — returning empty frame.")
        return pd.DataFrame()

    marginal_df = pd.DataFrame(subject_summaries)
    contrasts_df = pd.DataFrame(contrast_summaries)

    # --- Combine: attach marginal baseline to every contrast row ---
    combined = contrasts_df.merge(
        marginal_df[join_keys + ["Marginal_Mean", "Marginal_Std", "N_Observations"]],
        on=join_keys,
        how="left",
    )

    # --- Define reference condition per variable (used for raw contrasts) ---
    # (Extend this dict as needed; unknown variables will yield NaN contrasts.)
    ref_map = {
        "Category or Silence": "Silence",
        "Music Listening": False,          # boolean in your frame
        "Perceived Category": "Classic",
    }
    combined["Reference_Condition"] = combined["Condition_Variable"].map(ref_map)

    # --- Compute reference mean per (Hypothesis, DV, Subject, Level, CondVar) ---
    ref_keys = join_keys + ["Comparison_Level", "Condition_Variable"]
    ref_mask = combined["Reference_Condition"].notna() & (combined["Condition"] == combined["Reference_Condition"])
    ref_means = (
        combined.loc[ref_mask, ref_keys + ["Condition_Mean"]]
        .rename(columns={"Condition_Mean": "Reference_Mean"})
        .drop_duplicates(subset=ref_keys)
    )
    combined = combined.merge(ref_means, on=ref_keys, how="left")

    # --- Derived metrics ---
    combined["Raw_Contrast"] = combined["Condition_Mean"] - combined["Reference_Mean"]

    denom = combined["Marginal_Mean"].abs().replace({0.0: np.nan})
    combined["Normalised_Contrast"] = combined["Raw_Contrast"] / denom
    combined["Subject_CV"] = combined["Marginal_Std"] / denom
    combined["Responder_Flag"] = combined["Raw_Contrast"] > 0

    # --- Save combined CSV with timestamp ---
    combined_path = output_dir / filemgmt.file_title("Subject Effect Summary Combined", ".csv")
    combined.to_csv(combined_path, index=False)
    print(f"\n✓ Saved combined subject summary → {combined_path}  ({len(combined)} rows)")

    # --- Optional pivot tables (stored in output_dir) ---
    if save_pivot_tables:
        for dv in combined["Dependent_Variable"].dropna().unique():
            dv_df = combined[combined["Dependent_Variable"] == dv]

            for comp_level in dv_df["Comparison_Level"].dropna().unique():
                lvl_df = dv_df[dv_df["Comparison_Level"] == comp_level]

                for cond_var in lvl_df["Condition_Variable"].dropna().unique():
                    sub = lvl_df[lvl_df["Condition_Variable"] == cond_var].copy()
                    sub = sub.drop_duplicates(subset=[subject_col, "Condition"])

                    pivot = sub.pivot_table(
                        index=subject_col,
                        columns="Condition",
                        values="Raw_Contrast",
                        aggfunc="first",
                    )

                    cond_slug = str(cond_var)[:25].replace(" ", "_").replace("[", "").replace("]", "")
                    dv_slug = str(dv)[:35].replace(" ", "_")
                    pivot_path = output_dir / filemgmt.file_title(
                        f"Pivot {comp_level} {dv_slug} {cond_slug}",
                        ".csv",
                    )
                    pivot.to_csv(pivot_path)
                    print(f"  ✓ Pivot → {pivot_path}")

    return combined




# ============================================================================
# SUMMARY TABLE FUNCTIONS
# ============================================================================

def add_significance_markers(df: pd.DataFrame, p_col_prefix: str = 'p_value') -> pd.DataFrame:
    """
    Add significance marker columns to dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with p-value columns
    p_col_prefix : str
        Prefix for p-value columns

    Returns
    -------
    pd.DataFrame
        Dataframe with added Sig_* columns
    """
    df = df.copy()

    # Find all p-value columns
    p_cols = [col for col in df.columns if p_col_prefix in col]

    for p_col in p_cols:
        sig_col = p_col.replace(p_col_prefix, 'Sig')
        df[sig_col] = df[p_col].apply(
            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        )

    return df


def create_summary_table(
        results_df: pd.DataFrame,
        filter_conditions: dict,
        index_cols: list,
        value_cols: list = None,
        output_file: str = None,
        output_dir: Path = None,
        table_name: str = "Summary Table"
) -> pd.DataFrame:
    """
    Create and save a summary table from results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Complete results dataframe
    filter_conditions : dict
        Conditions to filter results. Values can be:
        - str: exact match
        - callable: lambda function for filtering
        - list/tuple: value must be in list
    index_cols : list
        Columns for pivot table index
    value_cols : list, optional
        Columns to include as values
    output_file : str, optional
        Output filename
    output_dir : Path, optional
        Output directory
    table_name : str
        Name for display

    Returns
    -------
    pd.DataFrame
        Summary table
    """
    # Apply filters
    filtered_df = results_df.copy()
    for col, condition in filter_conditions.items():
        if isinstance(condition, str):
            filtered_df = filtered_df[filtered_df[col] == condition]
        elif callable(condition):
            try:
                filtered_df = filtered_df[filtered_df[col].apply(condition)]
            except Exception as e:
                print(f"⚠️  Filter error on column '{col}': {e}")
                continue
        elif isinstance(condition, (list, tuple)):
            filtered_df = filtered_df[filtered_df[col].isin(condition)]

    if len(filtered_df) == 0:
        print(f"\n⚠️  No data for {table_name}")
        print(f"    Applied filters: {filter_conditions}")
        return pd.DataFrame()

    # Create pivot table
    if value_cols is None:
        value_cols = ['Coefficient', 'p_value']

    summary = filtered_df.pivot_table(
        index=index_cols,
        columns='Model_Type',
        values=value_cols,
        aggfunc='first'
    )

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Add significance markers
    summary = add_significance_markers(summary)

    # Display
    print(f"\n{'=' * 120}")
    print(f"{table_name.upper()}")
    print(f"{'=' * 120}\n")
    print(summary.to_string(index=False))

    # Save
    if output_file and output_dir:
        filepath = output_dir / output_file
        summary.to_csv(filepath, index=False)
        print(f"\n✓ Saved to: {filepath}")

    return summary


def display_summary_statistics(results_df: pd.DataFrame) -> None:
    """
    Display comprehensive summary statistics.

    Parameters
    ----------
    results_df : pd.DataFrame
        Complete results dataframe with 'Significance' column
    """
    print(f"\n\n{'=' * 120}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 120}\n")

    # Strip internal sentinel rows before any counting:
    results_df = results_df[~results_df["Parameter"].str.startswith("__")]

    # Overall statistics
    total = len(results_df)
    sig_001 = len(results_df[results_df['p_value'] < 0.001])
    sig_01 = len(results_df[results_df['p_value'] < 0.01])
    sig_05 = len(results_df[results_df['p_value'] < 0.05])

    print(f"Total effects tested:              {total}")
    print(f"Significant at p < 0.001 (***):    {sig_001} ({sig_001 / total * 100:.1f}%)")
    print(f"Significant at p < 0.01 (**):      {sig_01} ({sig_01 / total * 100:.1f}%)")
    print(f"Significant at p < 0.05 (*):       {sig_05} ({sig_05 / total * 100:.1f}%)")
    print(f"Non-significant (ns):              {total - sig_05} ({(total - sig_05) / total * 100:.1f}%)")

    # By model type
    print(f"\n{'Breakdown by Model Type:':-<60}")
    for model_type in results_df['Model_Type'].unique():
        model_data = results_df[results_df['Model_Type'] == model_type]
        model_sig = len(model_data[model_data['p_value'] < 0.05])
        print(
            f"  {model_type:>5s}: {len(model_data)} effects, {model_sig} significant ({model_sig / len(model_data) * 100:.1f}%)")

    # By comparison level
    print(f"\n{'Breakdown by Comparison Level:':-<60}")
    for level in results_df['Comparison_Level'].unique():
        level_data = results_df[results_df['Comparison_Level'] == level]
        level_sig = len(level_data[level_data['p_value'] < 0.05])
        print(
            f"  {level}: {len(level_data)} effects, {level_sig} significant ({level_sig / len(level_data) * 100:.1f}%)")


def display_significant_effects(results_df: pd.DataFrame, significance_level: float = 0.05,
                                exclude_intercepts: bool = True) -> None:
    """
    Display all significant effects sorted by p-value.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with 'Significance' column
    significance_level : float
        Threshold for significance
    exclude_intercepts : bool
        If True, excludes intercept terms from display
    """
    significant = results_df[results_df['p_value'] < significance_level].copy()

    # Exclude intercepts and internal sentinels from display
    if exclude_intercepts:
        significant = significant[
            ~significant["Parameter"].str.contains("Intercept", case=False, na=False) &
            ~significant["Parameter"].str.startswith("__")
            ]

    if len(significant) == 0:
        print(f"\n\nNo significant effects found at p < {significance_level}")
        return

    print(f"\n\n{'=' * 120}")
    if exclude_intercepts:
        print(f"ALL SIGNIFICANT EFFECTS (p < {significance_level}, excluding intercepts): {len(significant)} effects")
    else:
        print(f"ALL SIGNIFICANT EFFECTS (p < {significance_level}): {len(significant)} effects")
    print(f"{'=' * 120}\n")

    # Sort by p-value
    significant = significant.sort_values('p_value')

    for _, row in significant.iterrows():
        hyp_short = row['Hypothesis'][:55] if len(row['Hypothesis']) > 55 else row['Hypothesis']
        print(
            f"{row['Parameter']:<45s} | {row['Model_Type']:<5s} | {hyp_short:<55s} | β={row['Coefficient']:>7.4f} | p={row['p_value']:>7.4f} {row['Significance']:>4s}")


def display_model_diagnostics(diagnostics_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Display and save model diagnostics and assumption tests.

    Parameters
    ----------
    diagnostics_df : pd.DataFrame
        DataFrame containing model diagnostics
    output_dir : Path
        Directory to save CSV file
    """
    if diagnostics_df is None or len(diagnostics_df) == 0:
        print("\n⚠️  No diagnostics data available")
        return

    print(f"\n\n{'=' * 140}")
    print("MODEL DIAGNOSTICS & ASSUMPTION TESTS")
    print(f"{'=' * 140}\n")

    # Format diagnostics for display
    diag_display = diagnostics_df.copy()

    # Round numeric columns for readability
    numeric_cols = ['Shapiro_p', 'Lag1_Autocorr', 'Design_Effect', 'SE_Inflation',
                    'R_squared', 'R_squared_adj', 'R_squared_marginal', 'R_squared_conditional',
                    'AIC', 'BIC', 'LogLik']
    for col in numeric_cols:
        if col in diag_display.columns:
            diag_display[col] = diag_display[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else '-')

    # Flag problematic models
    print("[LEGEND]")
    print("  Shapiro_Violated = Yes : Residuals deviate from normality (p < 0.05)")
    print("  Lag1_Autocorr > 0.3    : Moderate autocorrelation")
    print("  Lag1_Autocorr > 0.5    : High autocorrelation")
    print("  SE_Inflation > 1.5     : Substantial SE adjustment needed")
    print("  AIC/BIC               : Lower values indicate better model fit (LME only)")
    print("\n")

    # Display by model type
    for model_type in ['OLS', 'LME']:
        model_diag = diag_display[diag_display['Model_Type'] == model_type]
        if len(model_diag) > 0:
            print(f"\n{'-' * 140}")
            print(f"{model_type} MODELS ({len(model_diag)} models)")
            print(f"{'-' * 140}")

            # Select columns to display
            display_cols = ['Hypothesis', 'Dependent_Variable', 'Comparison_Level',
                            'N_Observations', 'Shapiro_Violated', 'Lag1_Autocorr', 'SE_Inflation']

            if model_type == 'OLS':
                display_cols.extend(['R_squared', 'R_squared_adj'])
            elif model_type == 'LME':
                display_cols.extend(['R_squared_marginal', 'R_squared_conditional', 'AIC', 'BIC'])

            # Filter to existing columns
            display_cols = [col for col in display_cols if col in model_diag.columns]

            print(model_diag[display_cols].to_string(index=False))

            # Summary statistics
            try:
                shapiro_violations = (diagnostics_df[
                    (diagnostics_df['Model_Type'] == model_type) &
                    (diagnostics_df['Shapiro_Violated'] == 'Yes')
                    ].shape[0])

                high_autocorr = (diagnostics_df[
                    (diagnostics_df['Model_Type'] == model_type) &
                    (diagnostics_df['Lag1_Autocorr'].abs() > 0.5)
                    ].shape[0])

                print(f"\n  Summary:")
                print(
                    f"    Shapiro-Wilk violations: {shapiro_violations}/{len(model_diag)} ({shapiro_violations / len(model_diag) * 100:.1f}%)")
                print(
                    f"    High autocorrelation (|ρ| > 0.5): {high_autocorr}/{len(model_diag)} ({high_autocorr / len(model_diag) * 100:.1f}%)")

                if model_type == 'LME':
                    mean_aic = diagnostics_df[diagnostics_df['Model_Type'] == model_type]['AIC'].mean()
                    print(f"    Mean AIC: {mean_aic:.2f}")
            except Exception as e:
                print(f"\n  ⚠️  Could not compute summary statistics: {e}")


def _save_subset(
    df: pd.DataFrame,
    mask: pd.Series,
    label: str,
    stem: str,
    output_dir: Path,
    file_suffix: str,
    table_num: int,
    display_cols: list[str],
) -> int:
    """Filter df by mask, save to CSV, print a summary line. Returns next table_num."""
    subset = df[mask].copy()
    if subset.empty:
        return table_num
    path = output_dir / filemgmt.file_title(f"{stem}{file_suffix}", ".csv")
    subset.to_csv(path, index=False)
    visible = [c for c in display_cols if c in subset.columns]
    print(f"\n{'=' * 120}\nTABLE {table_num}: {label}\n{'=' * 120}")
    print(f"✓ {path}  ({len(subset)} rows)")
    print(subset[visible].head(10).to_string(index=False))
    return table_num + 1


def generate_all_summary_tables(
        results_df: pd.DataFrame,
        output_dir: Path,
        diagnostics_df: pd.DataFrame = None,
        file_identifier: str = "",
        generate_per_level_tables: bool = False,
        generate_thematic_tables: bool = False,
) -> None:
    """Generate all summary tables from results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Complete results frame (sentinel rows are stripped internally).
    output_dir : Path
        Output directory.
    diagnostics_df : pd.DataFrame, optional
        Model diagnostics frame.
    file_identifier : str
        Suffix appended to all output filenames.
    generate_per_level_tables : bool
        Emit one CSV per comparison level.
    generate_thematic_tables : bool
        Emit music- and force-specific slices.
    """
    file_suffix = f"_{file_identifier}" if file_identifier else ""
    display_cols = [
        "Hypothesis", "Comparison_Level", "Model_Type",
        "Parameter", "Coefficient", "p_value_adjusted", "Significance_adjusted",
    ]

    # Strip sentinels and add significance columns
    df = results_df[~results_df["Parameter"].str.startswith("__")].copy()
    for p_col, sig_col in [
        ("p_value_unadjusted", "Significance_unadjusted"),
        ("p_value_adjusted",   "Significance_adjusted"),
    ]:
        if p_col in df.columns:
            df[sig_col] = df[p_col].apply(
                lambda p: "***" if pd.notna(p) and p < 0.001
                else ("**" if pd.notna(p) and p < 0.01
                else ("*"  if pd.notna(p) and p < 0.05 else "ns"))
            )
    if "Significance_adjusted" in df.columns:
        df["Significance"] = df["Significance_adjusted"]

    t = 1   # running table counter

    # Per-level tables
    if generate_per_level_tables:
        for level in sorted(df["Comparison_Level"].unique()):
            stem = "summary_level" + level.lower().split("(")[0].replace("level ", "").strip().replace(" ", "")
            t = _save_subset(df, df["Comparison_Level"] == level,
                             level, stem, output_dir, file_suffix, t, display_cols)

    # Thematic slices
    if generate_thematic_tables:
        music_mask = (
            df["Parameter"].str.contains("Music", case=False, na=False) &
            ~df["Parameter"].str.contains("Intercept", case=False, na=False)
        )
        t = _save_subset(df, music_mask, "MUSIC EFFECTS",
                         "summary_music_effects", output_dir, file_suffix, t, display_cols)

        force_mask = (
            df["Parameter"].str.contains("Force", case=False, na=False) &
            ~df["Parameter"].str.contains("Intercept", case=False, na=False)
        )
        t = _save_subset(df, force_mask, "FORCE EFFECTS",
                         "summary_force_effects", output_dir, file_suffix, t, display_cols)

    # Significant effects
    sig_mask = df["Significance_adjusted"].isin(["*", "**", "***"])
    t = _save_subset(df, sig_mask, "ALL SIGNIFICANT EFFECTS (p_adjusted < 0.05)",
                     "summary_significant_effects", output_dir, file_suffix, t, display_cols)

    # Master table
    path = output_dir / filemgmt.file_title(f"summary_all_results_master{file_suffix}", ".csv")
    df.to_csv(path, index=False)
    print(f"\nTABLE {t}: MASTER TABLE → {path}  ({len(df)} rows)")

    display_summary_statistics(df)
    display_significant_effects(df)

    # Diagnostics
    if diagnostics_df is not None and len(diagnostics_df) > 0:
        diag_path = output_dir / filemgmt.file_title(f"summary_model_diagnostics{file_suffix}", ".csv")
        diagnostics_df.to_csv(diag_path, index=False)
        display_model_diagnostics(diagnostics_df, output_dir)



def run_model_levels(
        base_df: pd.DataFrame,
        level_definitions: list[dict],
        response_var: str,
        hypothesis_name: str,
        n_windows_per_trial: int,
        all_results_list: list,
        diagnostics_list: list,
        levels_to_include: list[int] | None = None,
        show_diagnostic_plots: bool = False,
) -> None:
    """
    Iterate over a list of model level definitions, fit both models for each,
    and store the results.

    Each entry in `level_definitions` is a dict with keys:
        - df_filter   : callable(df) -> pd.DataFrame, or None for no filter
        - condition_vars       : dict
        - reference_categories : dict
        - explanatory_vars     : list[str]
        - moderation_pairs     : list[tuple[str, str]] | None

    The `comparison_level_name` is inferred automatically from the level index
    and the variables present in each definition.

    Parameters
    ----------
    base_df : pd.DataFrame
        The full dataframe; `df_filter` in each level definition is applied to it.
    level_definitions : list[dict]
        Ordered list of level definitions (index 0 = Level 0, etc.).
    levels_to_include : list[int], optional
        Which levels to actually run, e.g. [0, 2].
    response_var : str
        Dependent variable column name.
    hypothesis_name : str
        Hypothesis label for reporting.
    n_windows_per_trial : int
        Segments per trial.
    all_results_list : list
        Accumulated results list (mutated in place).
    diagnostics_list : list
        Accumulated diagnostics list (mutated in place).
    show_diagnostic_plots : bool
        Whether to show diagnostic plots.
    """
    if levels_to_include is None:  # base: include all levels
        levels_to_include = list(range(len(level_definitions)))

    for level_idx, level_def in enumerate(level_definitions):
        if level_idx not in levels_to_include: continue

        # --- resolve dataframe ---
        df_filter = level_def.get('df_filter', None)
        df = df_filter(base_df) if df_filter is not None else base_df

        condition_vars       = level_def['condition_vars']
        reference_categories = level_def.get('reference_categories', None)
        explanatory_vars     = level_def['explanatory_vars']
        moderation_pairs     = level_def.get('moderation_pairs', None)

        # --- auto-infer comparison_level_name ---
        comparison_level_name = _build_level_name(
            level_idx, condition_vars, explanatory_vars, moderation_pairs
        )

        results = fit_both_models(
            df=df,
            response_var=response_var,
            condition_vars=condition_vars,
            reference_categories=reference_categories,
            explanatory_vars=explanatory_vars,
            comparison_level_name=comparison_level_name,
            hypothesis_name=hypothesis_name,
            n_windows_per_trial=n_windows_per_trial,
            show_diagnostic_plots=show_diagnostic_plots,
            moderation_pairs=moderation_pairs,
        )

        store_model_results(
            model_results=results,
            hypothesis_name=hypothesis_name,
            dependent_variable=response_var,
            comparison_level_name=comparison_level_name,
            all_results_list=all_results_list,
            diagnostics_list=diagnostics_list,
        )


def _build_level_name(
        level_idx: int,
        condition_vars: dict,
        explanatory_vars: list[str],
        moderation_pairs: list[tuple] | None,
) -> str:
    """
    Auto-generate a human-readable comparison level name.

    Format:  Level N (VarA + VarB + ... [+ Interactions])
    Condition vars come first, then explanatory vars, with
    '+ Interactions' appended when moderation_pairs is non-empty.
    """
    # Friendly short labels: strip units/brackets for readability
    def _short(name: str) -> str:
        # e.g. 'Median Force Level [0-1]' -> 'Force'
        #      'Musical skill [0-7]_centered' -> 'Musical skill'
        name = name.replace('_centered', '')
        name = name.split('[')[0].strip()
        # shorten some known verbose names
        abbreviations = {
            'Median Force Level': 'Force',
            'Median Heart Rate': 'Heart Rate',
            'Median HRV': 'HRV',
        }
        return abbreviations.get(name, name)

    parts = [_short(v) for v in condition_vars] + [_short(v) for v in explanatory_vars]
    # deduplicate while preserving order
    seen = set()
    unique_parts = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique_parts.append(p)

    label = ' + '.join(unique_parts)
    if moderation_pairs:
        label += ' + Interactions'

    return f"Level {level_idx} ({label})"



# ============================================================================
# Fetch and Structure Result Functions
# ============================================================================

def load_recent_results_frame(frame_dir: str | Path) -> pd.DataFrame:
    if not isinstance(frame_dir, Path): frame_dir = Path(frame_dir)
    result_frame_path = filemgmt.most_recent_file(frame_dir, ".csv", ["All Time Resolutions Results"])
    return pd.read_csv(result_frame_path)

def load_recent_diagnostics_frame(frame_dir: str | Path) -> pd.DataFrame:
    if not isinstance(frame_dir, Path): frame_dir = Path(frame_dir)
    diagnostics_frame_path = filemgmt.most_recent_file(frame_dir, ".csv", ["All Time Resolutions Diagnostics"])
    return pd.read_csv(diagnostics_frame_path)


# ============================================================================
# LME Robustness Analysis + Subject Heterogeneity
# ============================================================================

# ── helper: LOSO fits ─────────────────────────────────────────────────────────

def _run_loso(
    all_subject_df: pd.DataFrame,
    dep_var: str,
    comp_lvl: int,
    n_segments: int,
    fetch_level_definitions: Callable[[bool], list[dict]],
    run_model_levels: Callable,
) -> pd.DataFrame:
    """Run leave-one-subject-out OLS/LME refits for one (dep_var, level, seg) config.

    Parameters
    ----------
    all_subject_df : pd.DataFrame
        Full combined statistics frame (all subjects).
    dep_var : str
        Dependent variable column name.
    comp_lvl : int
        Comparison level index.
    n_segments : int
        Number of within-trial segments.
    fetch_level_definitions : Callable
        Factory returning level_definitions list; called with bool (n_segments > 1).
    run_model_levels : Callable
        statistics.run_model_levels from the existing pipeline.

    Returns
    -------
    pd.DataFrame
        Concatenated LOSO result rows with an extra 'Dropped Subject ID' column.
    """
    frames: list[pd.DataFrame] = []

    for subject_id in all_subject_df["Subject ID"].dropna().unique():
        remaining = all_subject_df.loc[all_subject_df["Subject ID"] != subject_id]

        temp_results: list = []
        temp_diagnostics: list = []
        run_model_levels(
            base_df=remaining,
            level_definitions=fetch_level_definitions(n_segments > 1),
            levels_to_include=[comp_lvl],
            response_var=dep_var,
            hypothesis_name=f"LOSO {dep_var} drop_{subject_id:02}",
            n_windows_per_trial=n_segments,
            all_results_list=temp_results,
            diagnostics_list=temp_diagnostics,
        )

        frame = pd.DataFrame(temp_results)
        frame["Dropped Subject ID"] = subject_id
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


# ── helper: influence measures ────────────────────────────────────────────────

def _compute_influence(
    loso_df: pd.DataFrame,
    full_results_df: pd.DataFrame,
    dep_var: str,
    comp_lvl: int,
    n_segments: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute Cook's D approximation and DFBETA from a LOSO result frame.

    Parameters
    ----------
    loso_df : pd.DataFrame
        Concatenated LOSO results for one config; must contain
        'Dropped Subject ID', 'Model_Type', 'Parameter', 'Coefficient', 'SE'.
    full_results_df : pd.DataFrame
        All-subject results frame (all_time_resolutions_results_frame).
    dep_var : str
        Dependent variable column name.
    comp_lvl : int
        Comparison level index.
    n_segments : int
        Number of within-trial segments.

    Returns
    -------
    cooks_series : pd.Series
        Cook's D approximation indexed by Dropped Subject ID;
        name is set to dep_var for column-concatenation.
    dfbeta_df : pd.DataFrame
        Parameters × (dep_var, subject_id) MultiIndex DFBETA pivot.
    """
    level_names = [
        lvl for lvl in full_results_df["Comparison_Level"].unique()
        if lvl.startswith(f"Level {comp_lvl} ")
    ]

    full_ols = full_results_df.loc[
        (full_results_df["Model_Type"] == "OLS") &
        (full_results_df["Comparison_Level"].isin(level_names)) &
        (full_results_df["N. Segments"] == n_segments) &
        (full_results_df["Dependent_Variable"] == dep_var),
        ["Parameter", "Coefficient", "SE"],
    ].rename(columns={"Coefficient": "Coef_full", "SE": "SE_full"})

    loso_ols = loso_df[loso_df["Model_Type"] == "OLS"].copy()

    # DFBETA: positive → subject inflates coefficient; negative → suppresses it
    merged = loso_ols.merge(full_ols, on="Parameter", how="inner")
    merged["DFBETA"] = (merged["Coef_full"] - merged["Coefficient"]) / merged["SE_full"]

    # Cook's D approximation: mean squared DFBETA across all parameters per subject
    cooks_series = (
        merged.groupby("Dropped Subject ID")["DFBETA"]
        .apply(lambda x: np.mean(x ** 2))
        .rename(dep_var)
        .sort_values(ascending=False)
    )

    # DFBETA pivot with (dep_var, subject_id) MultiIndex columns
    dfbeta_pivot = merged.pivot_table(
        index="Parameter",
        columns="Dropped Subject ID",
        values="DFBETA",
    )
    dfbeta_pivot.columns = pd.MultiIndex.from_tuples(
        [(dep_var, subj) for subj in dfbeta_pivot.columns],
        names=["Dependent Variable", "Dropped Subject ID"],
    )

    return cooks_series, dfbeta_pivot


# ── workflow ──────────────────────────────────────────────────────────────────

def run_influence_analysis(
    configs: list[tuple[str, int, int]],
    full_results_df: pd.DataFrame,
    feature_output_data: Path,
    statistics_output_data: Path,
    fetch_level_definitions: Callable[[bool], list[dict]],
    run_model_levels: Callable,
    file_title: Callable[[str, str], str],
    dfbeta_flag_threshold: float = 1.0,
    cooks_flag_threshold: float | None = None,
) -> pd.DataFrame:
    """Run LOSO influence analysis for all (dep_var, comp_lvl, n_segments) configs.

    Produces a single long-format combined CSV with one row per
    (config × parameter × subject), containing DFBETA, Cook's D, and
    flag columns — directly joinable with condition contrast tables.

    Parameters
    ----------
    configs : list[tuple[str, int, int]]
        List of (dependent_variable, comparison_level, n_segments) tuples.
    full_results_df : pd.DataFrame
        All-subject LME/OLS results frame (all_time_resolutions_results_frame).
    feature_output_data : Path
        Directory containing precomputed 'Combined Statistics Xseg' CSV files.
    statistics_output_data : Path
        Output directory for the saved influence CSV.
    fetch_level_definitions : Callable
        Factory returning level_definitions; called with bool (n_segments > 1).
    run_model_levels : Callable
        statistics.run_model_levels from the existing pipeline.
    file_title : Callable
        filemgmt.file_title(stem, extension) → filename string with timestamp.
    dfbeta_flag_threshold : float
        |DFBETA| >= this value sets DFBETA_Flagged = True. Default 1.0.
    cooks_flag_threshold : float or None
        Cook's D >= this value sets CooksD_Flagged = True.
        If None, uses the conventional 4 / n_subjects per config. Default None.

    Returns
    -------
    pd.DataFrame
        Long-format combined influence table (also saved to statistics_output_data).
        Columns: Dependent_Variable, Comparison_Level, N_Segments, Parameter,
                 Subject_ID, DFBETA, DFBETA_Flagged, CooksD, CooksD_Flagged,
                 CooksD_Threshold.
    """
    # --- Validate config homogeneity before any computation ---
    unique_n_segments = {n for _, _, n in configs}
    unique_comp_lvls = {lvl for _, lvl, _ in configs}

    # Hard assertion: mixing segment counts produces a CSV that the report
    # only partially uses — the caller should restrict to primary_n_segments.
    if len(unique_n_segments) > 1:
        raise ValueError(
            f"run_influence_analysis received configs with multiple N_Segments values: "
            f"{sorted(unique_n_segments)}. "
            f"Restrict all configs to a single segment count (your primary resolution) "
            f"so the combined influence CSV is unambiguous."
        )

    # Warning: multiple comparison levels are valid per-model but are currently
    # merged without a Level column in _section_trust — review report output carefully.
    if len(unique_comp_lvls) > 1:
        print(
            f"  ⚠️  run_influence_analysis: configs span {len(unique_comp_lvls)} comparison "
            f"levels ({sorted(unique_comp_lvls)}). Results will be merged into one CSV. "
            f"Ensure _section_trust in the report distinguishes levels, or restrict to "
            f"one level per call."
        )

    all_rows: list[dict] = []

    for dep_var, comp_lvl, n_segments in configs:
        print("=" * 100)
        print(f"Influence analysis │ DV: {dep_var} │ Level: {comp_lvl} │ Segments: {n_segments}")
        print("=" * 100)

        all_subject_df = pd.read_csv(
            filemgmt.most_recent_file(
                feature_output_data, ".csv", [f"Combined Statistics {n_segments}seg"]
            )
        )
        n_subjects = all_subject_df["Subject ID"].nunique()
        effective_cooks_threshold = (
            cooks_flag_threshold if cooks_flag_threshold is not None
            else 4.0 / n_subjects
        )

        loso_df = _run_loso(
            all_subject_df=all_subject_df,
            dep_var=dep_var,
            comp_lvl=comp_lvl,
            n_segments=n_segments,
            fetch_level_definitions=fetch_level_definitions,
            run_model_levels=run_model_levels,
        )

        # cooks_series: index=subject_id (int), value=Cook's D
        # dfbeta_pivot: index=Parameter names, columns=MultiIndex(dep_var, subject_id)
        cooks_series, dfbeta_pivot = _compute_influence(
            loso_df=loso_df,
            full_results_df=full_results_df,
            dep_var=dep_var,
            comp_lvl=comp_lvl,
            n_segments=n_segments,
        )

        if dfbeta_pivot.empty or dfbeta_pivot.shape[1] == 0:
            print(f"  ⚠️  _compute_influence returned empty pivot — skipping config.")
            continue

        # Drop the dep_var level from the MultiIndex → plain subject_id columns
        dfbeta_flat = dfbeta_pivot.copy()
        if isinstance(dfbeta_flat.columns, pd.MultiIndex):
            dfbeta_flat.columns = dfbeta_flat.columns.droplevel(0)
        dfbeta_flat.columns = [int(c) for c in dfbeta_flat.columns]
        dfbeta_flat = dfbeta_flat.rename_axis("Parameter")

        # Melt to long format: one row per (Parameter × Subject)
        dfbeta_long = (
            dfbeta_flat
            .reset_index()
            .melt(id_vars="Parameter", var_name="Subject_ID", value_name="DFBETA")
        )
        dfbeta_long["Subject_ID"] = dfbeta_long["Subject_ID"].astype(int)

        # Merge Cook's D — normalise index dtype to int to guarantee key alignment
        cooks_map = (
            cooks_series
            .rename("CooksD")
            .rename_axis("Subject_ID")
            .reset_index()
        )
        cooks_map["Subject_ID"] = cooks_map["Subject_ID"].astype(int)
        dfbeta_long = dfbeta_long.merge(cooks_map, on="Subject_ID", how="left")

        # Flag columns
        dfbeta_long["DFBETA_Flagged"] = dfbeta_long["DFBETA"].abs() >= dfbeta_flag_threshold
        dfbeta_long["CooksD_Flagged"] = dfbeta_long["CooksD"] >= effective_cooks_threshold
        dfbeta_long["CooksD_Threshold"] = effective_cooks_threshold

        # Attach config metadata
        dfbeta_long.insert(0, "Dependent_Variable", dep_var)
        dfbeta_long.insert(1, "Comparison_Level", comp_lvl)
        dfbeta_long.insert(2, "N_Segments", n_segments)

        all_rows.append(dfbeta_long)

        # Console summary
        print(f"\n  Cook's D (threshold={effective_cooks_threshold:.4f}):")
        print(f"  {cooks_series.to_string()}")
        flagged_dfbeta = dfbeta_long[dfbeta_long["DFBETA_Flagged"]]
        if not flagged_dfbeta.empty:
            print(f"\n  Flagged |DFBETA| >= {dfbeta_flag_threshold}:")
            print(flagged_dfbeta[["Parameter", "Subject_ID", "DFBETA", "CooksD"]].to_string(index=False))
        else:
            print(f"\n  No |DFBETA| >= {dfbeta_flag_threshold} detected.")

    combined = pd.concat(all_rows, ignore_index=True)

    # Reorder columns for readability
    combined = combined[[
        "Dependent_Variable", "Comparison_Level", "N_Segments",
        "Parameter", "Subject_ID",
        "DFBETA", "DFBETA_Flagged",
        "CooksD", "CooksD_Flagged", "CooksD_Threshold",
    ]]

    out_path = statistics_output_data / file_title("Influence Analysis Combined", ".csv")
    combined.to_csv(out_path, index=False)
    print(f"\n✓ Saved combined influence table → {out_path}  ({len(combined)} rows)")

    return combined





# ============================================================================
# Statistical Power and Sensitivity Analysis
# ============================================================================

@dataclass
class PowerConfig:
    """Fully specifies one power analysis run.

    Attributes
    ----------
    dependent_var : str
        Dependent variable column name (e.g. 'CMC_Flexor_max_beta').
    comp_lvl : int
        Comparison level index (matches level_definitions).
    n_segments : int
        Number of within-trial segments to use.
    target_parameters : list[str]
        Subset of parameter names to evaluate power for.
        Must match statsmodels parameter names exactly (e.g.
        "C(Q('Category or Silence'))[T.Happy]").
    n_simulations : int
        Number of synthetic datasets to simulate per effect-size step.
    effect_multipliers : list[float]
        Multipliers applied to each target parameter's fitted coefficient
        to sweep the effect-size axis. [1.0] = power at observed effect only.
        Use e.g. [0.25, 0.5, 0.75, 1.0, 1.5, 2.0] for a full power curve.
    target_power : float
        Desired power level for MDE derivation (default 0.80).
    alpha : float
        Significance threshold (default 0.05).
    random_seed : int
        For reproducibility.
    """
    dependent_var: str
    comp_lvl: int
    n_segments: int
    target_parameters: list[str]
    n_simulations: int = 500
    effect_multipliers: list[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    )
    target_power: float = 0.80
    alpha: float = 0.05
    random_seed: int = 42


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _extract_lme_params(
    results_df: pd.DataFrame,
    dep_var: str,
    comp_lvl: int,
    n_segments: int,
) -> dict:
    """Extract all generative parameters from the pre-computed results frame.

    Requires that the results frame contains '__residual_std__' and '__re_std__'
    sentinel rows for the requested model configuration (added by run_model_levels
    after each LME fit).

    Parameters
    ----------
    results_df : pd.DataFrame
        Loaded 'All Time Resolutions Results' CSV.
    dep_var : str
        Dependent variable column name.
    comp_lvl : int
        Comparison level index.
    n_segments : int
        Number of within-trial segments.

    Returns
    -------
    dict with keys:
        fixed_effects  : dict  — {parameter_name: coefficient}
        residual_std   : float
        re_std         : float
    """
    level_names = [
        lvl for lvl in results_df["Comparison_Level"].unique()
        if lvl.startswith(f"Level {comp_lvl} ")
    ]

    # Select all LME rows for this config
    mask = (
        (results_df["Model_Type"] == "LME") &
        (results_df["Comparison_Level"].isin(level_names)) &
        (results_df["N. Segments"] == n_segments) &
        (results_df["Dependent_Variable"] == dep_var)
    )
    subset = results_df.loc[mask]

    if subset.empty:
        raise ValueError(
            f"No saved LME results for DV='{dep_var}', "
            f"Level {comp_lvl}, {n_segments} segments."
        )

    # Extract variance sentinels
    def _get_sentinel(key: str) -> float:
        row = subset.loc[subset["Parameter"] == key, "Coefficient"]
        if row.empty:
            raise KeyError(
                f"Sentinel '{key}' not found. "
                "Ensure run_model_levels stores variance components."
            )
        return float(row.iloc[0])

    residual_std = _get_sentinel("__residual_std__")
    re_std = _get_sentinel("__re_std__")

    # Fixed effects: all non-sentinel rows
    param_rows = subset[~subset["Parameter"].str.startswith("__")]
    fixed_effects = dict(zip(param_rows["Parameter"], param_rows["Coefficient"]))

    print(
        f"  Loaded {len(fixed_effects)} coefficients from results frame.\n"
        f"  Residual std: {residual_std:.4f} | Random-intercept std: {re_std:.4f}"
    )

    return {
        "fixed_effects": fixed_effects,
        "residual_std":  residual_std,
        "re_std":        re_std,
    }


def _simulate_and_fit(
    generative_params: dict,
    formula: str,                               # ← now passed explicitly
    data: pd.DataFrame,                         # ← now passed explicitly
    subject_col: str,                           # ← now passed explicitly
    target_parameter: str,
    effect_multiplier: float,
    n_simulations: int,
    alpha: float,
    rng: np.random.Generator,
) -> float:
    """Simulate datasets and return empirical power for one parameter × multiplier.

    Parameters
    ----------
    generative_params : dict
        Output of _extract_lme_params: keys 'fixed_effects', 'residual_std', 're_std'.
    formula : str
        Statsmodels formula string (dep_var ~ predictors).
    data : pd.DataFrame
        Filtered + NaN-dropped frame matching the formula structure.
    subject_col : str
        Grouping variable column name (e.g. 'Subject ID').
    target_parameter : str
        Parameter name to evaluate power for.
    effect_multiplier : float
        Multiplier applied to target parameter's fitted coefficient.
    n_simulations : int
        Number of synthetic datasets.
    alpha : float
        Significance threshold.
    rng : np.random.Generator
        Seeded RNG for reproducibility.

    Returns
    -------
    float
        Empirical power.
    """
    import patsy

    fixed: dict = generative_params["fixed_effects"].copy()
    residual_std: float = generative_params["residual_std"]
    re_std: float = generative_params["re_std"]

    if target_parameter not in fixed:
        raise KeyError(
            f"[Power] '{target_parameter}' not found in fitted parameters.\n"
            f"Available: {list(fixed.keys())}"
        )

    fixed[target_parameter] *= effect_multiplier

    subjects = data[subject_col].unique()
    n_subjects = len(subjects)

    # Dependent variable column name from LHS of formula
    dep_var_raw = formula.split("~")[0].strip()
    dep_var_col = dep_var_raw.strip("Q('\"")
    rhs = formula.split("~", 1)[1].strip()

    # Pre-compute design matrix once — structure is identical across all simulations
    X = patsy.dmatrix(rhs, data=data, return_type="dataframe")
    coef_vector = np.array([fixed.get(col, 0.0) for col in X.columns])
    mu = X.values @ coef_vector

    subj_idx = pd.Categorical(data[subject_col]).codes

    significant_count = 0
    for _ in range(n_simulations):
        re = rng.normal(0.0, re_std, size=n_subjects)
        eps = rng.normal(0.0, residual_std, size=len(data))
        y_sim = mu + re[subj_idx] + eps

        sim_data = data.copy()
        sim_data[dep_var_col] = y_sim

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit = smf.mixedlm(
                    formula, data=sim_data, groups=sim_data[subject_col]
                ).fit(reml=True, method="lbfgs", disp=False)

                p = fit.pvalues.get(target_parameter, np.nan)
                if not np.isnan(p) and p < alpha:
                    significant_count += 1
            except Exception:
                pass

    return significant_count / n_simulations





def _derive_mde(
    power_curve: pd.DataFrame,
    target_parameter: str,
    fitted_coefficient: float,
    target_power: float,
) -> float | None:
    """Find the minimum detectable effect magnitude at target_power.

    Interpolates linearly between the two bracketing multiplier points.

    Parameters
    ----------
    power_curve : pd.DataFrame
        Must have columns ['effect_multiplier', 'power'] for one parameter.
    target_parameter : str
        Used only for warning messages.
    fitted_coefficient : float
        Fitted coefficient value (used to convert multiplier → absolute MDE).
    target_power : float
        Desired power level.

    Returns
    -------
    float or None
        Minimum absolute coefficient value achieving target_power, or None
        if the power curve never reaches target_power.
    """
    curve = power_curve.sort_values("effect_multiplier")
    above = curve[curve["power"] >= target_power]
    if above.empty:
        warnings.warn(
            f"[Power] Power never reaches {target_power:.0%} for "
            f"'{target_parameter}' within the simulated multiplier range. "
            "Consider extending effect_multipliers."
        )
        return None

    first_above = above.iloc[0]
    idx = curve.index.get_loc(first_above.name)
    if idx == 0:
        return float(abs(fitted_coefficient * first_above["effect_multiplier"]))

    # Linear interpolation between the two bracketing rows
    row_lo = curve.iloc[idx - 1]
    row_hi = curve.iloc[idx]
    frac = (target_power - row_lo["power"]) / (row_hi["power"] - row_lo["power"] + 1e-12)
    mde_multiplier = row_lo["effect_multiplier"] + frac * (
        row_hi["effect_multiplier"] - row_lo["effect_multiplier"]
    )
    return float(abs(fitted_coefficient * mde_multiplier))


# ══════════════════════════════════════════════════════════════════════════════
#  WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════

def run_power_analysis(
    configs: list[PowerConfig],
    results_df: pd.DataFrame,
    feature_output_data: Path,
    statistics_output_data: Path,
    fetch_level_definitions: Callable[[bool], list[dict]],
    file_title: Callable[[str, str], str],
    save_full_power_curve: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run simulation-based power analysis for all PowerConfig entries.

    Always saves an MDE summary CSV (one row per config × parameter).
    The full power curve (one row per config × parameter × effect multiplier)
    is only saved when save_full_power_curve is True.

    Parameters
    ----------
    configs : list[PowerConfig]
        One entry per (dependent_var, comp_lvl, n_segments) configuration.
    results_df : pd.DataFrame
        Pre-computed results frame containing '__residual_std__' and '__re_std__'
        sentinel rows from fit_mixed_effects_model.
    feature_output_data : Path
        Directory containing 'Combined Statistics Xseg' CSVs.
    statistics_output_data : Path
        Output directory for saved result CSVs.
    fetch_level_definitions : Callable
        Factory returning level_definitions; called with bool (n_segments > 1).
    file_title : Callable
        filemgmt.file_title(stem, extension) → filename string with timestamp.
    save_full_power_curve : bool
        If True, saves a second CSV with all multiplier rows joined to MDE columns.
        Defaults to False — the MDE summary is sufficient for most reporting needs.

    Returns
    -------
    mde_df : pd.DataFrame
        One row per config × parameter — always saved.
    power_curve_df : pd.DataFrame
        One row per config × parameter × multiplier — empty frame if
        save_full_power_curve is False (still returned for in-memory use).
    """
    all_power_rows: list[dict] = []
    all_mde_rows: list[dict] = []
    join_keys = ["Dependent_Variable", "Comparison_Level", "N_Segments", "Parameter"]

    for cfg in configs:
        print("=" * 100)
        print(
            f"Power analysis │ DV: {cfg.dependent_var} "
            f"│ Level: {cfg.comp_lvl} │ Segments: {cfg.n_segments}"
        )
        print("=" * 100)

        rng = np.random.default_rng(cfg.random_seed)

        base_df = pd.read_csv(
            filemgmt.most_recent_file(
                feature_output_data, ".csv",
                [f"Combined Statistics {cfg.n_segments}seg"]
            )
        )

        print("  Loading generative parameters from results frame...")
        gen_params = _extract_lme_params(
            results_df=results_df,
            dep_var=cfg.dependent_var,
            comp_lvl=cfg.comp_lvl,
            n_segments=cfg.n_segments,
        )

        level_def = fetch_level_definitions(cfg.n_segments > 1)[cfg.comp_lvl]
        sim_data = base_df.copy()
        if level_def.get("df_filter") is not None:
            sim_data = level_def["df_filter"](sim_data)
        sim_data = _apply_reference_categories(
            sim_data,
            level_def["condition_vars"],
            level_def.get("reference_categories"),
        )

        condition_formula_parts = []
        for var_name, var_type in level_def["condition_vars"].items():
            if var_type == "categorical":
                quoted = f"Q('{var_name}')" if " " in var_name else var_name
                condition_formula_parts.append(f"C({quoted})")
            elif var_type == "ordinal":
                condition_formula_parts.append(
                    f"Q('{var_name}')" if " " in var_name else var_name
                )
        explanatory_parts = [
            f"Q('{v}')" if " " in v else v
            for v in level_def.get("explanatory_vars", [])
        ]
        formula = cfg.dependent_var + " ~ " + " + ".join(condition_formula_parts + explanatory_parts)

        cols_to_check = (
            [cfg.dependent_var, "Subject ID"]
            + list(level_def["condition_vars"].keys())
            + level_def.get("explanatory_vars", [])
        )
        sim_data = sim_data.dropna(subset=cols_to_check)
        print(f"  Formula: {formula}")
        print(f"  Simulation frame: {len(sim_data)} rows, {sim_data['Subject ID'].nunique()} subjects")

        target_params = cfg.target_parameters or [
            p for p in gen_params["fixed_effects"] if p != "Intercept"
        ]

        for param in target_params:
            fitted_coef = gen_params["fixed_effects"].get(param)
            if fitted_coef is None:
                warnings.warn(f"  [Power] Parameter '{param}' not in fitted model — skipping.")
                continue

            print(f"\n  Parameter: {param}  (fitted coef = {fitted_coef:.5f})")
            param_power_rows: list[dict] = []

            row_base = {
                "Dependent_Variable": cfg.dependent_var,
                "Comparison_Level":   cfg.comp_lvl,
                "N_Segments":         cfg.n_segments,
                "Parameter":          param,
                "Fitted_Coefficient": fitted_coef,
                "N_Simulations":      cfg.n_simulations,
                "Alpha":              cfg.alpha,
                "Target_Power":       cfg.target_power,
            }

            for multiplier in tqdm(cfg.effect_multipliers, desc="Effect multipliers"):
                power = _simulate_and_fit(
                    generative_params=gen_params,
                    formula=formula,
                    data=sim_data,
                    subject_col="Subject ID",
                    target_parameter=param,
                    effect_multiplier=multiplier,
                    n_simulations=cfg.n_simulations,
                    alpha=cfg.alpha,
                    rng=rng,
                )
                abs_effect = abs(fitted_coef * multiplier)
                print(
                    f"    multiplier={multiplier:.2f} | "
                    f"|coef|={abs_effect:.5f} | power={power:.3f}"
                )
                # Always accumulate for in-memory curve; only serialised if requested
                all_power_rows.append({
                    **row_base,
                    "Effect_Multiplier": multiplier,
                    "Absolute_Effect":   abs_effect,
                    "Power":             power,
                })
                param_power_rows.append({"effect_multiplier": multiplier, "power": power})

            param_curve = pd.DataFrame(param_power_rows)
            mde = _derive_mde(
                power_curve=param_curve,
                target_parameter=param,
                fitted_coefficient=fitted_coef,
                target_power=cfg.target_power,
            )
            observed_rows = param_curve.loc[param_curve["effect_multiplier"] == 1.0, "power"].values
            power_at_observed = float(observed_rows[0]) if len(observed_rows) else np.nan
            interpretation = (
                f"INFORMATIVE: well-powered at observed effect (power={power_at_observed:.2f})"
                if power_at_observed >= cfg.target_power
                else f"UNINFORMATIVE: under-powered (power={power_at_observed:.2f}) — "
                     f"null does not rule out this effect"
            )
            print(f"    → Power at 1×: {power_at_observed:.3f} | MDE: {'N/A' if mde is None else f'{mde:.5f}'}")
            print(f"    → {interpretation}")

            all_mde_rows.append({
                **row_base,
                "Power_at_Observed_Effect":          power_at_observed,
                f"MDE_at_{cfg.target_power:.0%}_power": mde,
                "Interpretation":                    interpretation,
            })

    # MDE summary — always saved
    mde_df = pd.DataFrame(all_mde_rows)
    mde_path = statistics_output_data / file_title("Power Analysis MDE Summary", ".csv")
    mde_df.to_csv(mde_path, index=False)
    print(f"\n✓ Saved MDE summary → {mde_path}  ({len(mde_df)} rows)")

    # Full power curve — built in memory regardless, only saved if requested
    power_curve_df = pd.DataFrame(all_power_rows)
    if save_full_power_curve:
        combined_df = power_curve_df.merge(
            mde_df[join_keys + ["Power_at_Observed_Effect",
                                f"MDE_at_{configs[0].target_power:.0%}_power",
                                "Interpretation"]],
            on=join_keys,
            how="left",
        )
        curve_path = statistics_output_data / file_title("Power Analysis Full Curve", ".csv")
        combined_df.to_csv(curve_path, index=False)
        print(f"✓ Saved full power curve → {curve_path}  ({len(combined_df)} rows)")

    print("\nPower analysis complete.")
    return mde_df, power_curve_df