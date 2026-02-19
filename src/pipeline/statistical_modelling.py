from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

import src.utils.file_management as filemgmt


def fit_linear_regression_model(
        df: pd.DataFrame,
        response_var: str,
        condition_vars: dict,
        explanatory_vars: list,
        show_diagnostic_plots: bool = False,
        autocorr_threshold: float = 0.1,
        moderation_pairs: list = None
) -> dict:
    """
    Fit an OLS linear regression model with flexible condition and explanatory variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing all variables
    response_var : str
        Name of the response variable (e.g., 'PSD')
    condition_vars : dict
        Dictionary mapping variable names to treatment types:
        - 'categorical': treated as factor with dummy variables
        - 'ordinal': treated as continuous scale (0-7)
        Example: {'Category': 'categorical', 'Familiarity': 'ordinal'}
    explanatory_vars : list
        List of additional explanatory variables (e.g., ['Force Level'])
    n_windows_per_trial : int
        Number of windows per trial (legacy parameter, not used for autocorrelation)
    show_diagnostic_plots : bool, default=False
        Whether to display Q-Q plot and residual distribution plot
    autocorr_threshold : float, default=0.1
        Minimum |ρ| required to apply SE inflation. Only autocorrelations exceeding this
        threshold trigger SE adjustment. Default of 0.1 balances Type I error control
        with statistical power for small samples (n=8 subjects).
    moderation_pairs : list of tuples, optional
        List of (MODERATED_VAR, MODERATING_VAR) tuples for interaction effects.
        Adds: MODERATING_VAR + MODERATED_VAR:MODERATING_VAR to the formula.
        Example: [('Force Level', 'Familiarity'), ('Category', 'Force Level')]

    Returns
    -------
    dict
        Dictionary containing:
        - 'model': fitted statsmodels OLS object
        - 'results_df': DataFrame with adjusted coefficients and p-values
        - 'diagnostics': Dictionary with Shapiro-Wilk and autocorrelation stats
    """

    df = df.copy()

    # Ensure response variable is numeric
    df[response_var] = pd.to_numeric(df[response_var], errors='coerce')

    # Ensure explanatory vars are numeric
    for var in explanatory_vars:
        if var not in condition_vars:  # Don't convert categorical/ordinal vars
            df[var] = pd.to_numeric(df[var], errors='coerce')

    # Convert categorical variables to category dtype
    for var_name, var_type in condition_vars.items():
        if var_type == 'categorical':
            df[var_name] = df[var_name].astype('category')
        elif var_type == 'ordinal':
            df[var_name] = pd.to_numeric(df[var_name], errors='coerce')

    # Build formula dynamically
    formula_parts = [response_var, '~']

    # Add condition variables with appropriate encoding
    condition_formula_parts = []
    for var_name, var_type in condition_vars.items():
        if var_type == 'categorical':
            # Quote variable names with spaces for C()
            quoted_var = f"Q('{var_name}')" if ' ' in var_name else var_name
            condition_formula_parts.append(f"C({quoted_var})")
        elif var_type == 'ordinal':
            # Quote variable names with spaces
            if ' ' in var_name:
                condition_formula_parts.append(f"Q('{var_name}')")
            else:
                condition_formula_parts.append(var_name)
        else:
            raise ValueError(f"Unknown variable type: {var_type}")

    # Add explanatory variables (assume continuous with Q() for spaces in names)
    explanatory_formula_parts = [
        f"Q('{var}')" if ' ' in var else var
        for var in explanatory_vars
    ]

    # Combine all parts
    all_predictors = condition_formula_parts + explanatory_formula_parts

    # Add moderation effects (interaction terms)
    if moderation_pairs:
        print("\n[MODERATION EFFECTS SPECIFIED]")
        for moderated_var, moderating_var in moderation_pairs:
            # Format variable names with Q() if they contain spaces
            moderated_formatted = f"Q('{moderated_var}')" if ' ' in moderated_var else moderated_var
            moderating_formatted = f"Q('{moderating_var}')" if ' ' in moderating_var else moderating_var

            # Check if moderated_var is categorical - if so, wrap in C()
            if moderated_var in condition_vars and condition_vars[moderated_var] == 'categorical':
                moderated_formatted = f"C({moderated_formatted})"

            # Check if moderating_var is categorical - if so, wrap in C()
            if moderating_var in condition_vars and condition_vars[moderating_var] == 'categorical':
                moderating_formatted = f"C({moderating_formatted})"

            # Check if moderating variable is already in the model
            # Need to check both the raw variable name AND the formatted version
            moderating_already_present = (
                    moderating_var in condition_vars.keys() or
                    moderating_var in explanatory_vars or
                    moderating_formatted in all_predictors
            )

            # Add moderating variable as main effect (if not already present)
            if not moderating_already_present:
                all_predictors.append(moderating_formatted)
                print(f"  Added main effect: {moderating_var}")
            else:
                print(f"  Main effect already present: {moderating_var} (skipped)")

            # Add interaction term
            interaction_term = f"{moderated_formatted}:{moderating_formatted}"
            all_predictors.append(interaction_term)
            print(f"  Added interaction: {moderated_var} × {moderating_var}")
        print()

    formula = formula_parts[0] + ' ~ ' + ' + '.join(all_predictors)

    print("\n")
    print("-" * 100)
    print(f"---------------------     Linear Regression Analysis     --------------------- ")
    print("-" * 100, "\n")
    print(f"Formula: {formula}\n")

    ### DATA CHECK
    print("Data Summary:")
    print(f"Total observations: {len(df)}")
    print(f"Unique participants: {df['Subject ID'].nunique()}")
    print(
        f"Observations per participant: {len(df) / df['Subject ID'].nunique():.1f}")

    print(f"\nCondition variables:")
    for var_name, var_type in condition_vars.items():
        if var_type == 'categorical':
            print(f"  {var_name} (categorical): {df[var_name].nunique()} levels")
            print(f"    {df[var_name].value_counts().to_dict()}")
        elif var_type == 'ordinal':
            print(f"  {var_name} (ordinal): range [{df[var_name].min()}, {df[var_name].max()}]")
            print(f"    Distribution: {df[var_name].value_counts().sort_index().to_dict()}")

    print(f"\nResponse variable ({response_var}):")
    print(f"  Range: [{df[response_var].min():.2f}, {df[response_var].max():.2f}]")

    print(f"\nExplanatory variables:")
    for var in explanatory_vars:
        print(f"  {var} range: [{df[var].min():.2f}, {df[var].max():.2f}]")

    ### LINEAR REGRESSION
    print("\n" * 3 + "=" * 80)
    print("Linear Regression Model (OLS)")
    print("=" * 80)

    model = smf.ols(formula, data=df).fit()
    print(model.summary())

    # ============ DIAGNOSTICS ============
    residuals = model.resid

    # Q-Q plot and residual distribution (optional)
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
        print("  → Residuals significantly deviate from normality")
    else:
        print("  → Residuals approximately normal ✓")

    # ============ AUTOCORRELATION CHECK ============
    print("\n" + "=" * 80)
    print("AUTOCORRELATION CHECK (temporal correlation between trials)")
    print("=" * 80)

    lag1_autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    if np.isnan(lag1_autocorr):
        print("Warning: Autocorrelation is NaN")
        lag1_autocorr = 0.0

    # Calculate design effect based on number of trials per subject
    # This accounts for temporal correlation across trials, NOT within-trial segments
    n_trials_per_subject = len(df) / df['Subject ID'].nunique()

    # Apply SE inflation only if autocorrelation exceeds threshold
    if abs(lag1_autocorr) < autocorr_threshold:
        design_effect = 1.0
        se_inflation = 1.0
        inflation_applied = False
    else:
        # Apply inflation for positive ρ only (prevents deflation)
        design_effect = 1 + (n_trials_per_subject - 1) * max(0, lag1_autocorr)
        se_inflation = np.sqrt(design_effect)
        inflation_applied = True

    print(f"Lag-1 autocorrelation (ρ): {lag1_autocorr:.3f}")
    print(f"Average trials per subject: {n_trials_per_subject:.1f}")
    print(
        f"SE inflation threshold: |ρ| > {autocorr_threshold:.2f} (balancing power with n={df['Subject ID'].nunique()} subjects)")
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

    # Adjust SEs for autocorrelation
    adjusted_se = model.bse * se_inflation
    adjusted_z = model.params / adjusted_se
    adjusted_p = 2 * (1 - stats.norm.cdf(np.abs(adjusted_z)))

    # Convert to Series for proper indexing
    if not isinstance(adjusted_p, pd.Series):
        adjusted_p = pd.Series(adjusted_p, index=model.params.index)

    # ============ RESULTS TABLE ============
    print("\n" + "-" * 80)
    print("ADJUSTED RESULTS (corrected for autocorrelation):")
    print("-" * 80)
    print(f"{'Parameter':<50s} {'β':>10s} {'SE (adj)':>12s} {'p (adj)':>10s}")
    print("-" * 80)

    results_data = []
    for param in model.params.index:
        adj_se_val = adjusted_se[param]
        adj_p_val = adjusted_p[param]
        print(f"{param:<50s} {model.params[param]:>10.4f} {adj_se_val:>12.4f} {adj_p_val:>10.4f}")

        results_data.append({
            'Parameter': param,
            'Coefficient': model.params[param],
            'SE (unadjusted)': model.bse[param],
            'SE (adjusted)': adj_se_val,
            'p-value (unadjusted)': model.pvalues[param],
            'p-value (adjusted)': adj_p_val
        })

    results_df = pd.DataFrame(results_data)

    # Store diagnostics
    diagnostics = {
        'n_observations': len(df),
        'n_trials_per_subject': n_trials_per_subject,
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'lag1_autocorr': lag1_autocorr,
        'design_effect': design_effect,
        'se_inflation': se_inflation,
        'autocorr_threshold': autocorr_threshold,
        'inflation_applied': inflation_applied,
        'r_squared': model.rsquared,
        'r_squared_adj': model.rsquared_adj
    }

    return {
        'model': model,
        'results_df': results_df,
        'diagnostics': diagnostics
    }


def non_interaction_fit_linear_regression_model(
        df: pd.DataFrame,
        response_var: str,
        condition_vars: dict,
        explanatory_vars: list,
        show_diagnostic_plots: bool = False,
        autocorr_threshold: float = 0.1
) -> dict:
    """
    Fit an OLS linear regression model with flexible condition and explanatory variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing all variables
    response_var : str
        Name of the response variable (e.g., 'PSD')
    condition_vars : dict
        Dictionary mapping variable names to treatment types:
        - 'categorical': treated as factor with dummy variables
        - 'ordinal': treated as continuous scale (0-7)
        Example: {'Category': 'categorical', 'Familiarity': 'ordinal'}
    explanatory_vars : list
        List of additional explanatory variables (e.g., ['Force Level'])
    n_windows_per_trial : int
        Number of windows per trial (legacy parameter, not used for autocorrelation)
    show_diagnostic_plots : bool, default=False
        Whether to display Q-Q plot and residual distribution plot
    autocorr_threshold : float, default=0.1
        Minimum |ρ| required to apply SE inflation. Only autocorrelations exceeding this
        threshold trigger SE adjustment. Default of 0.1 balances Type I error control
        with statistical power for small samples (n=8 subjects).

    Returns
    -------
    dict
        Dictionary containing:
        - 'model': fitted statsmodels OLS object
        - 'results_df': DataFrame with adjusted coefficients and p-values
        - 'diagnostics': Dictionary with Shapiro-Wilk and autocorrelation stats
    """

    df = df.copy()

    # Ensure response variable is numeric
    df[response_var] = pd.to_numeric(df[response_var], errors='coerce')

    # Ensure explanatory vars are numeric
    for var in explanatory_vars:
        if var not in condition_vars:  # Don't convert categorical/ordinal vars
            df[var] = pd.to_numeric(df[var], errors='coerce')

    # Convert categorical variables to category dtype
    for var_name, var_type in condition_vars.items():
        if var_type == 'categorical':
            df[var_name] = df[var_name].astype('category')
        elif var_type == 'ordinal':
            df[var_name] = pd.to_numeric(df[var_name], errors='coerce')

    # Build formula dynamically
    formula_parts = [response_var, '~']

    # Add condition variables with appropriate encoding
    condition_formula_parts = []
    for var_name, var_type in condition_vars.items():
        if var_type == 'categorical':
            # Quote variable names with spaces for C()
            quoted_var = f"Q('{var_name}')" if ' ' in var_name else var_name
            condition_formula_parts.append(f"C({quoted_var})")
        elif var_type == 'ordinal':
            # Quote variable names with spaces
            if ' ' in var_name:
                condition_formula_parts.append(f"Q('{var_name}')")
            else:
                condition_formula_parts.append(var_name)
        else:
            raise ValueError(f"Unknown variable type: {var_type}")

    # Add explanatory variables (assume continuous with Q() for spaces in names)
    explanatory_formula_parts = [
        f"Q('{var}')" if ' ' in var else var
        for var in explanatory_vars
    ]

    # Combine all parts
    all_predictors = condition_formula_parts + explanatory_formula_parts
    formula = formula_parts[0] + ' ~ ' + ' + '.join(all_predictors)

    print("\n")
    print("-" * 100)
    print(f"---------------------     Linear Regression Analysis     --------------------- ")
    print("-" * 100, "\n")
    print(f"Formula: {formula}\n")

    ### DATA CHECK
    print("Data Summary:")
    print(f"Total observations: {len(df)}")
    print(f"Unique participants: {df['Subject ID'].nunique()}")
    print(
        f"Observations per participant: {len(df) / df['Subject ID'].nunique():.1f}")

    print(f"\nCondition variables:")
    for var_name, var_type in condition_vars.items():
        if var_type == 'categorical':
            print(f"  {var_name} (categorical): {df[var_name].nunique()} levels")
            print(f"    {df[var_name].value_counts().to_dict()}")
        elif var_type == 'ordinal':
            print(f"  {var_name} (ordinal): range [{df[var_name].min()}, {df[var_name].max()}]")
            print(f"    Distribution: {df[var_name].value_counts().sort_index().to_dict()}")

    print(f"\nResponse variable ({response_var}):")
    print(f"  Range: [{df[response_var].min():.2f}, {df[response_var].max():.2f}]")

    print(f"\nExplanatory variables:")
    for var in explanatory_vars:
        print(f"  {var} range: [{df[var].min():.2f}, {df[var].max():.2f}]")

    ### LINEAR REGRESSION
    print("\n" * 3 + "=" * 80)
    print("Linear Regression Model (OLS)")
    print("=" * 80)

    model = smf.ols(formula, data=df).fit()
    print(model.summary())

    # ============ DIAGNOSTICS ============
    residuals = model.resid

    # Q-Q plot and residual distribution (optional)
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
        print("  → Residuals significantly deviate from normality")
    else:
        print("  → Residuals approximately normal ✓")

    # ============ AUTOCORRELATION CHECK ============
    print("\n" + "=" * 80)
    print("AUTOCORRELATION CHECK (temporal correlation between trials)")
    print("=" * 80)

    lag1_autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    if np.isnan(lag1_autocorr):
        print("Warning: Autocorrelation is NaN")
        lag1_autocorr = 0.0

    # Calculate design effect based on number of trials per subject
    # This accounts for temporal correlation across trials, NOT within-trial segments
    n_trials_per_subject = len(df) / df['Subject ID'].nunique()

    # Apply SE inflation only if autocorrelation exceeds threshold
    if abs(lag1_autocorr) < autocorr_threshold:
        design_effect = 1.0
        se_inflation = 1.0
        inflation_applied = False
    else:
        # Apply inflation for positive ρ only (prevents deflation)
        design_effect = 1 + (n_trials_per_subject - 1) * max(0, lag1_autocorr)
        se_inflation = np.sqrt(design_effect)
        inflation_applied = True

    print(f"Lag-1 autocorrelation (ρ): {lag1_autocorr:.3f}")
    print(f"Average trials per subject: {n_trials_per_subject:.1f}")
    print(
        f"SE inflation threshold: |ρ| > {autocorr_threshold:.2f} (balancing power with n={df['Subject ID'].nunique()} subjects)")
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

    # Adjust SEs for autocorrelation
    adjusted_se = model.bse * se_inflation
    adjusted_z = model.params / adjusted_se
    adjusted_p = 2 * (1 - stats.norm.cdf(np.abs(adjusted_z)))

    # Convert to Series for proper indexing
    if not isinstance(adjusted_p, pd.Series):
        adjusted_p = pd.Series(adjusted_p, index=model.params.index)

    # ============ RESULTS TABLE ============
    print("\n" + "-" * 80)
    print("ADJUSTED RESULTS (corrected for autocorrelation):")
    print("-" * 80)
    print(f"{'Parameter':<50s} {'β':>10s} {'SE (adj)':>12s} {'p (adj)':>10s}")
    print("-" * 80)

    results_data = []
    for param in model.params.index:
        adj_se_val = adjusted_se[param]
        adj_p_val = adjusted_p[param]
        print(f"{param:<50s} {model.params[param]:>10.4f} {adj_se_val:>12.4f} {adj_p_val:>10.4f}")

        results_data.append({
            'Parameter': param,
            'Coefficient': model.params[param],
            'SE (unadjusted)': model.bse[param],
            'SE (adjusted)': adj_se_val,
            'p-value (unadjusted)': model.pvalues[param],
            'p-value (adjusted)': adj_p_val
        })

    results_df = pd.DataFrame(results_data)

    # Store diagnostics
    diagnostics = {
        'n_observations': len(df),
        'n_trials_per_subject': n_trials_per_subject,
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'lag1_autocorr': lag1_autocorr,
        'design_effect': design_effect,
        'se_inflation': se_inflation,
        'autocorr_threshold': autocorr_threshold,
        'inflation_applied': inflation_applied,
        'r_squared': model.rsquared,
        'r_squared_adj': model.rsquared_adj
    }

    return {
        'model': model,
        'results_df': results_df,
        'diagnostics': diagnostics
    }


def fit_mixed_effects_model(
        df: pd.DataFrame,
        response_var: str,
        condition_vars: dict,
        explanatory_vars: list,
        grouping_var: str = 'Subject ID',
        show_diagnostic_plots: bool = False,
        autocorr_threshold: float = 0.1,
        moderation_pairs: list = None
) -> dict:
    """
    Fit a Linear Mixed-Effects Model with flexible condition and explanatory variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing all variables
    response_var : str
        Name of the response variable (e.g., 'PSD')
    condition_vars : dict
        Dictionary mapping variable names to treatment types:
        - 'categorical': treated as factor with dummy variables
        - 'ordinal': treated as continuous scale (0-7)
        Example: {'Category': 'categorical', 'Familiarity': 'ordinal'}
    explanatory_vars : list
        List of additional explanatory variables (e.g., ['Force Level'])
    grouping_var : str
        Name of the grouping variable for random intercepts (default: 'Subject ID')
    n_windows_per_trial : int
        Number of windows per trial (legacy parameter, not used for LME autocorrelation)
    show_diagnostic_plots : bool
        Whether to display diagnostic plots
    autocorr_threshold : float, default=0.1
        Minimum |ρ| required to apply SE inflation. Only autocorrelations exceeding this
        threshold trigger SE adjustment. Default of 0.1 balances Type I error control
        with statistical power for small samples (n=8 subjects).
    moderation_pairs : list of tuples, optional
        List of (MODERATED_VAR, MODERATING_VAR) tuples for interaction effects.
        Adds: MODERATING_VAR + MODERATED_VAR:MODERATING_VAR to the formula.
        Example: [('Force Level', 'Familiarity'), ('Category', 'Force Level')]

    Returns
    -------
    dict
        Dictionary containing:
        - 'model': fitted statsmodels MixedLM object
        - 'results_df': DataFrame with adjusted coefficients and p-values
        - 'diagnostics': Dictionary with Shapiro-Wilk and autocorrelation stats
        - 'random_effects': DataFrame of random intercepts per group
    """

    df = df.copy()

    # Ensure response variable is numeric
    df[response_var] = pd.to_numeric(df[response_var], errors='coerce')

    # Ensure explanatory vars are numeric
    for var in explanatory_vars:
        if var not in condition_vars:
            df[var] = pd.to_numeric(df[var], errors='coerce')

    # Convert categorical variables to category dtype
    for var_name, var_type in condition_vars.items():
        if var_type == 'categorical':
            df[var_name] = df[var_name].astype('category')
        elif var_type == 'ordinal':
            df[var_name] = pd.to_numeric(df[var_name], errors='coerce')

    # Build formula dynamically
    formula_parts = [response_var, '~']

    # Add condition variables with appropriate encoding
    condition_formula_parts = []
    for var_name, var_type in condition_vars.items():
        if var_type == 'categorical':
            quoted_var = f"Q('{var_name}')" if ' ' in var_name else var_name
            condition_formula_parts.append(f"C({quoted_var})")
        elif var_type == 'ordinal':
            if ' ' in var_name:
                condition_formula_parts.append(f"Q('{var_name}')")
            else:
                condition_formula_parts.append(var_name)
        else:
            raise ValueError(f"Unknown variable type: {var_type}")

    # Add explanatory variables
    explanatory_formula_parts = [
        f"Q('{var}')" if ' ' in var else var
        for var in explanatory_vars
    ]

    # Combine all parts
    all_predictors = condition_formula_parts + explanatory_formula_parts

    # Add moderation effects (interaction terms)
    if moderation_pairs:
        print("\n[MODERATION EFFECTS SPECIFIED]")
        for moderated_var, moderating_var in moderation_pairs:
            # Format variable names with Q() if they contain spaces
            moderated_formatted = f"Q('{moderated_var}')" if ' ' in moderated_var else moderated_var
            moderating_formatted = f"Q('{moderating_var}')" if ' ' in moderating_var else moderating_var

            # Check if moderated_var is categorical - if so, wrap in C()
            if moderated_var in condition_vars and condition_vars[moderated_var] == 'categorical':
                moderated_formatted = f"C({moderated_formatted})"

            # Check if moderating_var is categorical - if so, wrap in C()
            if moderating_var in condition_vars and condition_vars[moderating_var] == 'categorical':
                moderating_formatted = f"C({moderating_formatted})"

            # Check if moderating variable is already in the model
            # Need to check both the raw variable name AND the formatted version
            moderating_already_present = (
                    moderating_var in condition_vars.keys() or
                    moderating_var in explanatory_vars or
                    moderating_formatted in all_predictors
            )

            # Add moderating variable as main effect (if not already present)
            if not moderating_already_present:
                all_predictors.append(moderating_formatted)
                print(f"  Added main effect: {moderating_var}")
            else:
                print(f"  Main effect already present: {moderating_var} (skipped)")

            # Add interaction term
            interaction_term = f"{moderated_formatted}:{moderating_formatted}"
            all_predictors.append(interaction_term)
            print(f"  Added interaction: {moderated_var} × {moderating_var}")
        print()

    formula = formula_parts[0] + ' ~ ' + ' + '.join(all_predictors)

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
    result = model.fit()
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

    # Calculate design effect based on number of trials per subject
    # This accounts for temporal correlation across trials, NOT within-trial segments
    n_trials_per_subject = len(df) / df[grouping_var].nunique()

    # Apply SE inflation only if autocorrelation exceeds threshold
    if abs(lag1_autocorr) < autocorr_threshold:
        design_effect = 1.0
        se_inflation = 1.0
        inflation_applied = False
    else:
        # Apply inflation for positive ρ only (prevents deflation)
        design_effect = 1 + (n_trials_per_subject - 1) * max(0, lag1_autocorr)
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
        var_fixed = np.var(result.fittedvalues)
        # Residual variance
        var_residual = result.scale

        # R² calculations using Nakagawa & Schielzeth (2013) method
        r2_marginal = var_fixed / (var_fixed + var_random + var_residual)
        r2_conditional = (var_fixed + var_random) / (var_fixed + var_random + var_residual)

        print(f"  var_fixed={var_fixed:.4f}, var_random={var_random:.4f}, var_residual={var_residual:.4f}")
        print(f"✓ R² metrics computed successfully: R²_marginal={r2_marginal:.4f}, R²_conditional={r2_conditional:.4f}")
    except (AttributeError, KeyError, IndexError, TypeError) as e:
        # Fallback with detailed error message
        print(f"⚠️  Warning: Could not compute R² metrics: {type(e).__name__}: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        r2_marginal = None
        r2_conditional = None

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
        'n_observations': len(df),
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'lag1_autocorr': lag1_autocorr,
        'design_effect': design_effect,
        'se_inflation': se_inflation,
        'log_likelihood': log_likelihood,
        'aic': aic,
        'bic': bic,
        'r_squared_marginal': r2_marginal,
        'r_squared_conditional': r2_conditional
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



def non_interaction_fit_mixed_effects_model(
        df: pd.DataFrame,
        response_var: str,
        condition_vars: dict,
        explanatory_vars: list,
        grouping_var: str = 'Subject ID',
        show_diagnostic_plots: bool = False,
        autocorr_threshold: float = 0.1
) -> dict:
    """
    Fit a Linear Mixed-Effects Model with flexible condition and explanatory variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing all variables
    response_var : str
        Name of the response variable (e.g., 'PSD')
    condition_vars : dict
        Dictionary mapping variable names to treatment types:
        - 'categorical': treated as factor with dummy variables
        - 'ordinal': treated as continuous scale (0-7)
        Example: {'Category': 'categorical', 'Familiarity': 'ordinal'}
    explanatory_vars : list
        List of additional explanatory variables (e.g., ['Force Level'])
    grouping_var : str
        Name of the grouping variable for random intercepts (default: 'Subject ID')
    n_windows_per_trial : int
        Number of windows per trial (legacy parameter, not used for LME autocorrelation)
    show_diagnostic_plots : bool
        Whether to display diagnostic plots
    autocorr_threshold : float, default=0.1
        Minimum |ρ| required to apply SE inflation. Only autocorrelations exceeding this
        threshold trigger SE adjustment. Default of 0.1 balances Type I error control
        with statistical power for small samples (n=8 subjects).

    Returns
    -------
    dict
        Dictionary containing:
        - 'model': fitted statsmodels MixedLM object
        - 'results_df': DataFrame with adjusted coefficients and p-values
        - 'diagnostics': Dictionary with Shapiro-Wilk and autocorrelation stats
        - 'random_effects': DataFrame of random intercepts per group
    """

    df = df.copy()

    # Ensure response variable is numeric
    df[response_var] = pd.to_numeric(df[response_var], errors='coerce')

    # Ensure explanatory vars are numeric
    for var in explanatory_vars:
        if var not in condition_vars:
            df[var] = pd.to_numeric(df[var], errors='coerce')

    # Convert categorical variables to category dtype
    for var_name, var_type in condition_vars.items():
        if var_type == 'categorical':
            df[var_name] = df[var_name].astype('category')
        elif var_type == 'ordinal':
            df[var_name] = pd.to_numeric(df[var_name], errors='coerce')

    # Build formula dynamically
    formula_parts = [response_var, '~']

    # Add condition variables with appropriate encoding
    condition_formula_parts = []
    for var_name, var_type in condition_vars.items():
        if var_type == 'categorical':
            quoted_var = f"Q('{var_name}')" if ' ' in var_name else var_name
            condition_formula_parts.append(f"C({quoted_var})")
        elif var_type == 'ordinal':
            if ' ' in var_name:
                condition_formula_parts.append(f"Q('{var_name}')")
            else:
                condition_formula_parts.append(var_name)
        else:
            raise ValueError(f"Unknown variable type: {var_type}")

    # Add explanatory variables
    explanatory_formula_parts = [
        f"Q('{var}')" if ' ' in var else var
        for var in explanatory_vars
    ]

    # Combine all parts
    all_predictors = condition_formula_parts + explanatory_formula_parts
    formula = formula_parts[0] + ' ~ ' + ' + '.join(all_predictors)

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
    result = model.fit()
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

    # Calculate design effect based on number of trials per subject
    # This accounts for temporal correlation across trials, NOT within-trial segments
    n_trials_per_subject = len(df) / df[grouping_var].nunique()

    # Apply SE inflation only if autocorrelation exceeds threshold
    if abs(lag1_autocorr) < autocorr_threshold:
        design_effect = 1.0
        se_inflation = 1.0
        inflation_applied = False
    else:
        # Apply inflation for positive ρ only (prevents deflation)
        design_effect = 1 + (n_trials_per_subject - 1) * max(0, lag1_autocorr)
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
        var_fixed = np.var(result.fittedvalues)
        # Residual variance
        var_residual = result.scale

        # R² calculations using Nakagawa & Schielzeth (2013) method
        r2_marginal = var_fixed / (var_fixed + var_random + var_residual)
        r2_conditional = (var_fixed + var_random) / (var_fixed + var_random + var_residual)

        print(f"  var_fixed={var_fixed:.4f}, var_random={var_random:.4f}, var_residual={var_residual:.4f}")
        print(f"✓ R² metrics computed successfully: R²_marginal={r2_marginal:.4f}, R²_conditional={r2_conditional:.4f}")
    except (AttributeError, KeyError, IndexError, TypeError) as e:
        # Fallback with detailed error message
        print(f"⚠️  Warning: Could not compute R² metrics: {type(e).__name__}: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        r2_marginal = None
        r2_conditional = None

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
        'n_observations': len(df),
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'lag1_autocorr': lag1_autocorr,
        'design_effect': design_effect,
        'se_inflation': se_inflation,
        'log_likelihood': log_likelihood,
        'aic': aic,
        'bic': bic,
        'r_squared_marginal': r2_marginal,
        'r_squared_conditional': r2_conditional
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
        moderation_pairs: list = None
) -> dict:
    """
    Fit both OLS and LME models together.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    response_var : str
        Dependent variable
    condition_vars : dict
        Condition variables with types (e.g., {'Category': 'categorical', 'Familiarity': 'ordinal'})
    explanatory_vars : list
        Explanatory variables
    comparison_level_name : str
        Name for reporting
    hypothesis_name : str
        Hypothesis name
    n_windows_per_trial : int
        Segments per trial
    show_diagnostic_plots : bool
        Show plots
    reference_categories : dict, optional
        Dictionary mapping categorical variable names to their desired reference levels.
        Example: {'Category or Silence': 'Silence', 'Perceived Category': 'Classical'}
        If not provided, pandas default (alphabetical order) is used.
    moderation_pairs : list of tuples, optional
        List of (MODERATED_VAR, MODERATING_VAR) tuples for interaction effects.
        Adds: MODERATING_VAR + MODERATED_VAR:MODERATING_VAR to the formula.
        Example: [('Force Level', 'Familiarity'), ('Category', 'Force Level')]

    Returns
    -------
    dict
        {'OLS': ols_results, 'LME': lme_results}

    Examples
    --------
    >>> # Set Silence as reference for 'Category or Silence'
    >>> results = fit_both_models(
    ...     df=data,
    ...     response_var='CMC',
    ...     condition_vars={'Category or Silence': 'categorical'},
    ...     explanatory_vars=['Force'],
    ...     comparison_level_name='Level 2',
    ...     hypothesis_name='H1',
    ...     reference_categories={'Category or Silence': 'Silence'}
    ... )
    >>>
    >>> # With moderation: Does Familiarity moderate the effect of Force?
    >>> results = fit_both_models(
    ...     df=data,
    ...     response_var='CMC',
    ...     condition_vars={'Category': 'categorical', 'Familiarity': 'ordinal'},
    ...     explanatory_vars=['Force Level'],
    ...     comparison_level_name='Level 2',
    ...     hypothesis_name='H2: Force × Familiarity',
    ...     moderation_pairs=[('Force Level', 'Familiarity')]
    ... )
    """
    print("\n" + "=" * 100)
    print(f"HYPOTHESIS: {hypothesis_name}")
    print(f"DEPENDENT VARIABLE: {response_var}")
    print(f"COMPARISON LEVEL: {comparison_level_name}")
    print("=" * 100)

    # Set reference categories for categorical variables
    df = df.copy()  # Work on a copy to avoid modifying original

    if reference_categories is not None:
        print("\n[REFERENCE CATEGORY SETUP]")
        for var_name, var_type in condition_vars.items():
            if var_type == 'categorical' and var_name in reference_categories:
                reference_level = reference_categories[var_name]

                # Convert to categorical if not already
                if not pd.api.types.is_categorical_dtype(df[var_name]):
                    df[var_name] = df[var_name].astype('category')

                # Get current categories
                current_categories = df[var_name].cat.categories.tolist()

                # Check if reference level exists
                if reference_level not in current_categories:
                    print(f"  ⚠️  Warning: Reference level '{reference_level}' not found in '{var_name}'")
                    print(f"      Available categories: {current_categories}")
                    print(f"      Using default (alphabetical) reference instead.")
                else:
                    # Reorder categories to put reference first
                    other_categories = [c for c in current_categories if c != reference_level]
                    new_order = [reference_level] + sorted(other_categories)
                    df[var_name] = df[var_name].cat.reorder_categories(new_order)

                    print(f"  ✓ Variable: '{var_name}'")
                    print(f"      Reference level: '{reference_level}' (all coefficients compare TO this)")
                    print(f"      Other levels: {sorted(other_categories)}")
        print()

    results = {}

    # Fit OLS
    print("\n" + "-" * 100)
    print("OLS MODEL")
    print("-" * 100)
    results['OLS'] = fit_linear_regression_model(
        df=df,
        response_var=response_var,
        condition_vars=condition_vars,
        explanatory_vars=explanatory_vars,
        show_diagnostic_plots=show_diagnostic_plots,
        moderation_pairs=moderation_pairs
    )

    # Fit LME
    print("\n" + "-" * 100)
    print("LME MODEL")
    print("-" * 100)
    results['LME'] = fit_mixed_effects_model(
        df=df,
        response_var=response_var,
        condition_vars=condition_vars,
        explanatory_vars=explanatory_vars,
        grouping_var='Subject ID',
        show_diagnostic_plots=show_diagnostic_plots,
        moderation_pairs=moderation_pairs
    )

    return results


def non_interaction_fit_both_models(
        df: pd.DataFrame,
        response_var: str,
        condition_vars: dict,
        explanatory_vars: list,
        comparison_level_name: str,
        hypothesis_name: str,
        n_windows_per_trial: int = 9,
        show_diagnostic_plots: bool = False,
        reference_categories: dict = None
) -> dict:
    """
    Fit both OLS and LME models together.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    response_var : str
        Dependent variable
    condition_vars : dict
        Condition variables with types (e.g., {'Category': 'categorical', 'Familiarity': 'ordinal'})
    explanatory_vars : list
        Explanatory variables
    comparison_level_name : str
        Name for reporting
    hypothesis_name : str
        Hypothesis name
    n_windows_per_trial : int
        Segments per trial
    show_diagnostic_plots : bool
        Show plots
    reference_categories : dict, optional
        Dictionary mapping categorical variable names to their desired reference levels.
        Example: {'Category or Silence': 'Silence', 'Perceived Category': 'Classical'}
        If not provided, pandas default (alphabetical order) is used.

    Returns
    -------
    dict
        {'OLS': ols_results, 'LME': lme_results}

    Examples
    --------
    >>> # Set Silence as reference for 'Category or Silence'
    >>> results = fit_both_models(
    ...     df=data,
    ...     response_var='CMC',
    ...     condition_vars={'Category or Silence': 'categorical'},
    ...     explanatory_vars=['Force'],
    ...     comparison_level_name='Level 2',
    ...     hypothesis_name='H1',
    ...     reference_categories={'Category or Silence': 'Silence'}
    ... )
    """
    print("\n" + "=" * 100)
    print(f"HYPOTHESIS: {hypothesis_name}")
    print(f"DEPENDENT VARIABLE: {response_var}")
    print(f"COMPARISON LEVEL: {comparison_level_name}")
    print("=" * 100)

    # Set reference categories for categorical variables
    df = df.copy()  # Work on a copy to avoid modifying original

    if reference_categories is not None:
        print("\n[REFERENCE CATEGORY SETUP]")
        for var_name, var_type in condition_vars.items():
            if var_type == 'categorical' and var_name in reference_categories:
                reference_level = reference_categories[var_name]

                # Convert to categorical if not already
                if not pd.api.types.is_categorical_dtype(df[var_name]):
                    df[var_name] = df[var_name].astype('category')

                # Get current categories
                current_categories = df[var_name].cat.categories.tolist()

                # Check if reference level exists
                if reference_level not in current_categories:
                    print(f"  ⚠️  Warning: Reference level '{reference_level}' not found in '{var_name}'")
                    print(f"      Available categories: {current_categories}")
                    print(f"      Using default (alphabetical) reference instead.")
                else:
                    # Reorder categories to put reference first
                    other_categories = [c for c in current_categories if c != reference_level]
                    new_order = [reference_level] + sorted(other_categories)
                    df[var_name] = df[var_name].cat.reorder_categories(new_order)

                    print(f"  ✓ Variable: '{var_name}'")
                    print(f"      Reference level: '{reference_level}' (all coefficients compare TO this)")
                    print(f"      Other levels: {sorted(other_categories)}")
        print()

    results = {}

    # Fit OLS
    print("\n" + "-" * 100)
    print("OLS MODEL")
    print("-" * 100)
    results['OLS'] = fit_linear_regression_model(
        df=df,
        response_var=response_var,
        condition_vars=condition_vars,
        explanatory_vars=explanatory_vars,
        show_diagnostic_plots=show_diagnostic_plots
    )

    # Fit LME
    print("\n" + "-" * 100)
    print("LME MODEL")
    print("-" * 100)
    results['LME'] = fit_mixed_effects_model(
        df=df,
        response_var=response_var,
        condition_vars=condition_vars,
        explanatory_vars=explanatory_vars,
        grouping_var='Subject ID',
        show_diagnostic_plots=show_diagnostic_plots
    )

    return results


def store_model_results(
        model_results: dict,
        hypothesis_name: str,
        dependent_variable: str,
        comparison_level_name: str,
        all_results_list: list,
        diagnostics_list: list = None
) -> None:
    """
    Store results from both models into list.

    Parameters
    ----------
    model_results : dict
        Output from fit_both_models()
    hypothesis_name : str
        Hypothesis name
    dependent_variable : str
        Dependent variable
    comparison_level_name : str
        Comparison level
    all_results_list : list
        List to append to (modified in place)
    diagnostics_list : list, optional
        List to store diagnostics (modified in place)
    """
    for model_type in ['OLS', 'LME']:
        if model_type in model_results:
            # Store parameter results
            for _, row in model_results[model_type]['results_df'].iterrows():
                all_results_list.append({
                    'Hypothesis': hypothesis_name,
                    'Dependent_Variable': dependent_variable,
                    'Model_Type': model_type,
                    'Comparison_Level': comparison_level_name,
                    'Parameter': row['Parameter'],
                    'Coefficient': row['Coefficient'],
                    'SE_unadjusted': row['SE (unadjusted)'],
                    'SE_adjusted': row['SE (adjusted)'],
                    'p_value_unadjusted': row['p-value (unadjusted)'],
                    'p_value_adjusted': row['p-value (adjusted)'],
                    'p_value': row['p-value (adjusted)'],  # Keep for backward compatibility
                    'SE': row['SE (adjusted)']  # Keep for backward compatibility
                })

            # Store diagnostics if requested
            if diagnostics_list is not None:
                try:
                    diag = model_results[model_type].get('diagnostics', {})
                    if diag:  # Only append if diagnostics exist
                        diag_entry = {
                            'Hypothesis': hypothesis_name,
                            'Dependent_Variable': dependent_variable,
                            'Model_Type': model_type,
                            'Comparison_Level': comparison_level_name,
                            'N_Observations': diag.get('n_observations', None),
                            'Shapiro_p': diag.get('shapiro_p', None),
                            'Shapiro_Violated': 'Yes' if diag.get('shapiro_p', 1.0) < 0.05 else 'No',
                            'Lag1_Autocorr': diag.get('lag1_autocorr', None),
                            'Design_Effect': diag.get('design_effect', None),
                            'SE_Inflation': diag.get('se_inflation', None),
                        }

                        # Add model-specific metrics (both models have all metrics now)
                        diag_entry['R_squared'] = diag.get('r_squared', None)
                        diag_entry['R_squared_adj'] = diag.get('r_squared_adj', None)
                        diag_entry['AIC'] = diag.get('aic', None)
                        diag_entry['BIC'] = diag.get('bic', None)
                        diag_entry['LogLik'] = diag.get('log_likelihood', None)
                        diag_entry['R_squared_marginal'] = diag.get('r_squared_marginal', None)
                        diag_entry['R_squared_conditional'] = diag.get('r_squared_conditional', None)
                        diag_entry['ICC'] = diag.get('ICC', None)

                        diagnostics_list.append(diag_entry)
                except Exception as e:
                    print(f"⚠️  Warning: Could not store diagnostics for {hypothesis_name} - {model_type}: {e}")


# ============================================================================
# SUBJECT-LEVEL ANALYSIS FUNCTIONS
# ============================================================================

def create_subject_effect_summary(
        all_model_results: list,
        original_data: pd.DataFrame,
        output_dir: Path,
        subject_col: str = 'Subject ID'
) -> None:
    """
    Create comprehensive subject-level effect summaries for all hypotheses.

    Parameters
    ----------
    all_model_results : list
        All model results
    original_data : pd.DataFrame
        Original data
    output_dir : Path
        Output directory
    subject_col : str
        Subject identifier
    """
    print("\n" + "=" * 120)
    print("SUBJECT-LEVEL HETEROGENEITY ANALYSIS")
    print("=" * 120 + "\n")

    results_df = pd.DataFrame(all_model_results)

    # Get unique combinations
    lme_results = results_df[results_df['Model_Type'] == 'LME']

    subject_summaries = []

    for hypothesis in lme_results['Hypothesis'].unique():
        for dv in lme_results[lme_results['Hypothesis'] == hypothesis]['Dependent_Variable'].unique():

            print(f"\n{'-' * 120}")
            print(f"Hypothesis: {hypothesis}")
            print(f"Dependent Variable: {dv}")
            print(f"{'-' * 120}")

            # Per-subject statistics
            for subject_id in original_data[subject_col].unique():
                subject_data = original_data[
                    (original_data[subject_col] == subject_id) &
                    (original_data[dv].notna())
                    ]

                if len(subject_data) > 0:
                    mean_val = subject_data[dv].mean()
                    std_val = subject_data[dv].std()
                    n_obs = len(subject_data)

                    subject_summaries.append({
                        'Hypothesis': hypothesis,
                        'Dependent_Variable': dv,
                        subject_col: subject_id,
                        'Mean': mean_val,
                        'Std': std_val,
                        'N_Observations': n_obs
                    })

                    print(f"  Subject {subject_id:02d}: Mean={mean_val:>8.4f}, Std={std_val:>8.4f}, N={n_obs:>4d}")

    # Save to CSV
    if subject_summaries:
        subject_df = pd.DataFrame(subject_summaries)
        output_file = output_dir / 'subject_level_effects.csv'
        subject_df.to_csv(output_file, index=False)
        print(f"\n\n✓ Saved subject-level analysis to: {output_file}")

        # Create pivot table for easy comparison
        for hypothesis in subject_df['Hypothesis'].unique():
            hyp_data = subject_df[subject_df['Hypothesis'] == hypothesis]

            pivot = hyp_data.pivot_table(
                index=subject_col,
                columns='Dependent_Variable',
                values='Mean',
                aggfunc='first'
            )

            output_file = output_dir / f'subject_effects_{hypothesis[:30].replace(" ", "_").replace(":", "")}.csv'
            pivot.to_csv(output_file)
            print(f"✓ Saved pivot table: {output_file}")

    print("\n" + "=" * 120)
    print("SUBJECT-LEVEL ANALYSIS COMPLETE")
    print("=" * 120 + "\n")


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

    # Exclude intercepts if requested
    if exclude_intercepts:
        significant = significant[~significant['Parameter'].str.contains('Intercept', case=False, na=False)]

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

    # Save to CSV
    try:
        output_file = output_dir / 'model_diagnostics.csv'
        diagnostics_df.to_csv(output_file, index=False)
        print(f"\n\n{'=' * 140}")
        print(f"✓ Model diagnostics saved to: {output_file}")
        print(f"{'=' * 140}\n")
    except Exception as e:
        print(f"\n⚠️  Error saving diagnostics: {e}")

def generate_all_summary_tables(
        results_df: pd.DataFrame,
        output_dir: Path,
        diagnostics_df: pd.DataFrame = None,
        file_identifier: str = "",
        generate_per_level_tables: bool = False,
        generate_thematic_tables: bool = False,
) -> None:
    """
    Generate all summary tables from results.

    All output CSV files have the SAME structure as summary_all_results_master.csv:
    - One row per parameter per hypothesis/level
    - Both unadjusted and adjusted values
    - Significance ratings for both

    Parameters
    ----------
    results_df : pd.DataFrame
        Complete results dataframe with columns:
        - Hypothesis, Dependent_Variable, Comparison_Level, Model_Type, Parameter
        - Coefficient, SE_unadjusted, SE_adjusted
        - p_value_unadjusted, p_value_adjusted
    output_dir : Path
        Output directory for CSV files
    diagnostics_df : pd.DataFrame, optional
        Model diagnostics dataframe
    file_identifier : str, optional
        Suffix to append to all output filenames (e.g., "3seg", "relabeled")

    Notes
    -----
    All CSV outputs use the SAME format - just filtered subsets of the master table.
    No pivoting or restructuring applied to maintain consistency.
    Tables are automatically generated for each comparison level found in the data.
    """
    print("\n" * 3)
    print("=" * 120)
    print("=" * 120)
    print("COMPREHENSIVE STATISTICAL SUMMARY (Unified Format)")
    print("=" * 120)
    print("=" * 120)

    # Add file identifier suffix if provided
    file_suffix = f"_{file_identifier}" if file_identifier else ""

    # Debug info: Show what we have
    print("\n[DEBUG] Available comparison levels:")
    unique_levels = sorted(results_df['Comparison_Level'].unique())
    for level in unique_levels:
        count = len(results_df[results_df['Comparison_Level'] == level])
        print(f"  - {level}: {count} results")

    # Add significance columns for both unadjusted and adjusted p-values
    results_df_formatted = results_df.copy()

    # Ensure we have the required columns
    if 'p_value_unadjusted' in results_df_formatted.columns:
        results_df_formatted['Significance_unadjusted'] = results_df_formatted['p_value_unadjusted'].apply(
            lambda p: '***' if pd.notna(p) and p < 0.001 else (
                '**' if pd.notna(p) and p < 0.01 else ('*' if pd.notna(p) and p < 0.05 else 'ns'))
        )

    if 'p_value_adjusted' in results_df_formatted.columns:
        results_df_formatted['Significance_adjusted'] = results_df_formatted['p_value_adjusted'].apply(
            lambda p: '***' if pd.notna(p) and p < 0.001 else (
                '**' if pd.notna(p) and p < 0.01 else ('*' if pd.notna(p) and p < 0.05 else 'ns'))
        )
        # Keep 'Significance' pointing to adjusted for backward compatibility and primary reporting
        results_df_formatted['Significance'] = results_df_formatted['Significance_adjusted']

    # ============================================================
    # DYNAMIC TABLES: One table per comparison level
    # ============================================================
    table_num = 1
    if generate_per_level_tables:
        print(f"\n{'=' * 120}")
        print("TABLES BY COMPARISON LEVEL (Dynamically Generated)")
        print(f"{'=' * 120}")

        for level in unique_levels:
            level_data = results_df_formatted[
                results_df_formatted['Comparison_Level'] == level
                ].copy()

            if len(level_data) > 0:
                print(f"\n{'-' * 120}")
                print(f"TABLE {table_num}: {level}")
                print(f"{'-' * 120}\n")

                # Create clean filename from level name
                # E.g., "Level 1 (Music + Force + Trial ID)" -> "level1_music_force_trial"
                level_clean = level.lower()
                level_clean = level_clean.replace('level ', 'level')
                level_clean = level_clean.split('(')[0].strip()  # Take only "Level N" part
                level_clean = level_clean.replace(' ', '')

                # Save using file management utility
                filename = filemgmt.file_title(f"summary_{level_clean}{file_suffix}", ".csv")
                save_path = output_dir / filename
                level_data.to_csv(save_path, index=False)

                print(f"✓ Saved to: {save_path}")
                print(f"  Rows: {len(level_data)}")

                # Display summary statistics for this level
                n_hypotheses = level_data['Hypothesis'].nunique()
                n_parameters = level_data['Parameter'].nunique()
                n_significant = len(level_data[level_data['Significance_adjusted'].isin(['*', '**', '***'])])

                print(f"  Hypotheses: {n_hypotheses}")
                print(f"  Parameters: {n_parameters}")
                print(
                    f"  Significant effects: {n_significant}/{len(level_data)} ({100 * n_significant / len(level_data):.1f}%)")

                # Show sample rows
                print("\nSample rows:")
                display_cols = ['Hypothesis', 'Model_Type', 'Parameter', 'Coefficient',
                                'p_value_adjusted', 'Significance_adjusted']
                display_cols = [c for c in display_cols if c in level_data.columns]
                print(level_data[display_cols].head(5).to_string(index=False))

                table_num += 1

    # ============================================================
    # SPECIAL TABLES: Key Effects
    # ============================================================

    if generate_thematic_tables:

        # Music Listening Effects (across all levels with music parameter)
        music_effects = results_df_formatted[
            (results_df_formatted['Parameter'].str.contains('Music', case=False, na=False)) &
            (~results_df_formatted['Parameter'].str.contains('Intercept', case=False, na=False))
            ].copy()

        if len(music_effects) > 0:
            print(f"\n{'=' * 120}")
            print(f"TABLE {table_num}: MUSIC EFFECTS ACROSS ALL LEVELS")
            print(f"{'=' * 120}\n")

            filename = filemgmt.file_title(f"summary_music_effects{file_suffix}", ".csv")
            save_path = output_dir / filename
            music_effects.to_csv(save_path, index=False)

            print(f"✓ Saved to: {save_path}")
            print(f"  Rows: {len(music_effects)}")

            # Display summary
            print("\nSample rows:")
            display_cols = ['Hypothesis', 'Comparison_Level', 'Model_Type', 'Parameter',
                            'Coefficient', 'p_value_adjusted', 'Significance_adjusted']
            display_cols = [c for c in display_cols if c in music_effects.columns]
            print(music_effects[display_cols].head(10).to_string(index=False))

            table_num += 1

        # Force Level Effects (across all levels)
        force_results = results_df_formatted[
            (results_df_formatted['Parameter'].str.contains('Force', case=False, na=False)) &
            (~results_df_formatted['Parameter'].str.contains('Intercept', case=False, na=False))
            ].copy()

        if len(force_results) > 0:
            print(f"\n{'=' * 120}")
            print(f"TABLE {table_num}: FORCE EFFECTS ACROSS ALL LEVELS")
            print(f"{'=' * 120}\n")

            filename = filemgmt.file_title(f"summary_force_effects{file_suffix}", ".csv")
            save_path = output_dir / filename
            force_results.to_csv(save_path, index=False)

            print(f"✓ Saved to: {save_path}")
            print(f"  Rows: {len(force_results)}")

            # Display summary
            print("\nTop 10 force effects by significance:")
            force_sorted = force_results.sort_values('p_value_adjusted')
            display_cols = ['Hypothesis', 'Comparison_Level', 'Model_Type',
                            'Coefficient', 'p_value_adjusted', 'Significance_adjusted']
            display_cols = [c for c in display_cols if c in force_sorted.columns]
            print(force_sorted[display_cols].head(10).to_string(index=False))

            table_num += 1

    # ============================================================
    # TABLE: ALL SIGNIFICANT EFFECTS
    # ============================================================
    significant_effects = results_df_formatted[
        results_df_formatted['Significance_adjusted'].isin(['*', '**', '***'])
    ].copy()

    if len(significant_effects) > 0:
        print(f"\n{'=' * 120}")
        print(f"TABLE {table_num}: ALL SIGNIFICANT EFFECTS (p_adjusted < 0.05)")
        print(f"{'=' * 120}\n")

        filename = filemgmt.file_title(f"summary_significant_effects{file_suffix}", ".csv")
        save_path = output_dir / filename
        significant_effects.to_csv(save_path, index=False)

        print(f"✓ Saved to: {save_path}")
        print(f"  Rows: {len(significant_effects)}")
        print(f"  Total tests: {len(results_df_formatted)}")
        print(f"  Significant: {100 * len(significant_effects) / len(results_df_formatted):.1f}%")

        table_num += 1

    # ============================================================
    # TABLE: COMPLETE MASTER TABLE
    # ============================================================
    print(f"\n{'=' * 120}")
    print(f"TABLE {table_num}: COMPLETE RESULTS MASTER TABLE - ALL EFFECTS")
    print(f"{'=' * 120}\n")

    filename = filemgmt.file_title(f"summary_all_results_master{file_suffix}", ".csv")
    save_path = output_dir / filename
    results_df_formatted.to_csv(save_path, index=False)

    print(f"✓ Saved complete results to: {save_path}")
    print(f"  Rows: {len(results_df_formatted)}")
    print(f"  Columns: {list(results_df_formatted.columns)}")

    # Display sample
    print("\nSample (first 5 rows):")
    display_cols = ['Hypothesis', 'Comparison_Level', 'Model_Type', 'Parameter', 'Coefficient',
                    'p_value_unadjusted', 'p_value_adjusted', 'Significance_adjusted']
    display_cols = [c for c in display_cols if c in results_df_formatted.columns]
    if len(results_df_formatted) > 0:
        print(results_df_formatted[display_cols].head().to_string(index=False))

    # Display statistics
    display_summary_statistics(results_df_formatted)
    display_significant_effects(results_df_formatted)

    # ============================================================
    # TABLE: MODEL DIAGNOSTICS
    # ============================================================
    if diagnostics_df is not None and len(diagnostics_df) > 0:
        print(f"\n{'=' * 120}")
        print(f"TABLE {table_num + 1}: MODEL DIAGNOSTICS")
        print(f"{'=' * 120}\n")

        filename = filemgmt.file_title(f"summary_model_diagnostics{file_suffix}", ".csv")
        save_path = output_dir / filename
        diagnostics_df.to_csv(save_path, index=False)

        print(f"✓ Saved to: {save_path}")

        display_model_diagnostics(diagnostics_df, output_dir)

    print("\n" + "=" * 120)
    print("SUMMARY TABLES COMPLETE")
    print("=" * 120 + "\n")
    print(f"All CSV files have identical structure (same columns, just filtered rows)")
    if file_identifier:
        print(f"All files include identifier: '{file_identifier}'")



def run_model_levels(
        base_df: pd.DataFrame,
        level_definitions: list[dict],
        levels_to_include: list[int],
        response_var: str,
        hypothesis_name: str,
        n_windows_per_trial: int,
        all_results_list: list,
        diagnostics_list: list,
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
    levels_to_include : list[int]
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