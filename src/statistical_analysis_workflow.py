from pathlib import Path
import pandas as pd
import numpy as np
from typing import Literal
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

import src.pipeline.signal_features as features
import src.pipeline.data_integration as data_integration
import src.pipeline.data_analysis as data_analysis
from src.pipeline.visualizations import EEG_CHANNEL_IND_DICT, EEG_CHANNELS_BY_AREA
import src.utils.file_management as filemgmt


def aggregate_psd_spectrogram(
        psd_spectrograms: np.ndarray,
        psd_freqs: np.ndarray = None,
        normalize_mvc: bool = False,
        is_log_scaled: bool = False,
        freq_slice: tuple[float, float] | str = None,
        channel_indices: list[int] = None,
        aggregation_ops: list[tuple[str, int]] = None,
) -> np.ndarray:
    """
    Aggregate PSD spectrograms through multiple stages: normalization, slicing, and axis reduction.

    Processing order:
    1. MVC normalization (if requested)
    2. Frequency slicing (if freq_slice provided)
    3. Channel slicing (if channel_indices provided)
    4. Sequential aggregation operations in specified order

    Parameters
    ----------
    psd_spectrograms : np.ndarray
        PSD data with shape (n_times, n_frequencies, n_channels).
    psd_freqs : np.ndarray, optional
        Frequency values corresponding to the frequency axis. Required if freq_slice is used.
    normalize_mvc : bool, default=False
        Whether to apply MVC (Maximum Voluntary Contraction) normalization.
        Computes max over time and frequency per channel, then normalizes to percentage.
    is_log_scaled : bool, default=False
        Whether the data is already log-scaled. If True, skips MVC normalization.
    freq_slice : tuple[float, float] | str, optional
        Frequency range to slice. Can be:
        - Tuple (low, high): Custom frequency range in Hz
        - String: Predefined band name ('slow', 'fast', 'delta', 'theta', 'alpha', 'beta', 'gamma')
        Requires psd_freqs to be provided.
    channel_indices : list[int], optional
        List of channel indices to select. If None, uses all channels.
    aggregation_ops : list[tuple[str, int]], optional
        List of (operator, axis) tuples to apply sequentially.
        Operator can be 'mean' or 'max'.
        Axis refers to the current array shape after slicing.
        Example: [('mean', 1), ('max', 2)] means average axis 1 first, then max over axis 2.

    Returns
    -------
    np.ndarray
        Aggregated PSD array with reduced dimensions based on specified operations.

    Examples
    --------
    # EMG: Slice frequencies, mean over frequencies, then max over channels
    result = aggregate_psd_spectrogram(
        psd_spectrograms, psd_freqs,
        normalize_mvc=True, is_log_scaled=False,
        freq_slice='slow',
        aggregation_ops=[('mean', 1), ('max', 2)],  # mean freq axis, max channel axis
    )  # Output shape: (n_times,)

    # EEG: Select channels, mean over channels first, then frequencies
    result = aggregate_psd_spectrogram(
        psd_spectrograms, psd_freqs,
        channel_indices=[0, 1, 2, 5],
        freq_slice='alpha',
        aggregation_ops=[('mean', 2), ('mean', 1)],  # mean channels, then mean frequencies
    )  # Output shape: (n_times,)

    # Complex example: max over time, mean over channels, then max over frequencies
    result = aggregate_psd_spectrogram(
        psd_spectrograms, psd_freqs,
        aggregation_ops=[('max', 0), ('mean', 2), ('max', 1)],
    )  # Output shape: scalar
    """
    # Predefined frequency bands in Hz
    FREQUENCY_BANDS = {
        'all': (0, 250),
        'slow': (0, 40),
        'fast': (60, 250),
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (13, 30),
        'gamma': (30, 100),
    }

    # Create working copy
    result = psd_spectrograms.copy()

    # Stage 1: MVC Normalization
    if normalize_mvc and not is_log_scaled:
        # Maximum over time (axis 0) and frequencies (axis 1) per channel
        mvc = np.max(np.max(result, axis=0, keepdims=True), axis=1, keepdims=True)
        result = result / mvc * 100  # Convert to percentage

    # Stage 2: Frequency Slicing
    if freq_slice is not None:
        if psd_freqs is None:
            raise ValueError("psd_freqs must be provided when using freq_slice")

        # Convert string band name to tuple if needed
        if isinstance(freq_slice, str):
            if freq_slice not in FREQUENCY_BANDS:
                available_bands = ', '.join(FREQUENCY_BANDS.keys())
                raise ValueError(
                    f"Unknown frequency band '{freq_slice}'. "
                    f"Available bands: {available_bands}"
                )
            low_freq, high_freq = FREQUENCY_BANDS[freq_slice]
        else:
            low_freq, high_freq = freq_slice

        freq_mask = (psd_freqs >= low_freq) & (psd_freqs <= high_freq)
        result = result[:, freq_mask, :]

    # Stage 3: Channel Slicing
    if channel_indices is not None:
        result = result[:, :, channel_indices]

    # Stage 4: Sequential Aggregation Operations
    if aggregation_ops is not None:
        for operator, axis in aggregation_ops:
            if operator == 'mean':
                result = np.nanmean(result, axis=axis)
            elif operator == 'max':
                result = np.nanmax(result, axis=axis)
            else:
                raise ValueError(
                    f"Unknown operator '{operator}'. Supported operators: 'mean', 'max'"
                )

    return result


def fit_linear_regression_model(
        df: pd.DataFrame,
        response_var: str,
        condition_vars: dict,
        explanatory_vars: list,
        n_windows_per_trial: int = 9,
        show_diagnostic_plots: bool = False
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
        Number of windows per trial for autocorrelation adjustment
    show_diagnostic_plots : bool, default=True
        Whether to display Q-Q plot and residual distribution plot
    reference_level : str, optional
        Reference level for categorical variables (default: first alphabetical)

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
    print("AUTOCORRELATION CHECK (within segments per trial)")
    print("=" * 80)

    lag1_autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    if np.isnan(lag1_autocorr):
        print("Warning: Autocorrelation is NaN")
        lag1_autocorr = 0.0

    design_effect = 1 + (n_windows_per_trial - 1) * lag1_autocorr
    se_inflation = np.sqrt(design_effect)

    print(f"Lag-1 autocorrelation (ρ): {lag1_autocorr:.3f}")
    print(f"Design effect: {design_effect:.2f}")
    print(f"SE inflation factor: {se_inflation:.2f}×")

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
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'lag1_autocorr': lag1_autocorr,
        'design_effect': design_effect,
        'se_inflation': se_inflation
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
        n_windows_per_trial: int = 9,
        show_diagnostic_plots: bool = False
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
        Number of windows per trial for autocorrelation adjustment
    reference_level : str, optional
        Reference level for categorical variables (default: first alphabetical)

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
    print("AUTOCORRELATION CHECK (within segments per trial)")
    print("=" * 80)

    lag1_autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    if np.isnan(lag1_autocorr):
        print("Warning: Autocorrelation is NaN (constant residuals?)")
        lag1_autocorr = 0.0

    design_effect = 1 + (n_windows_per_trial - 1) * lag1_autocorr
    se_inflation = np.sqrt(design_effect)

    print(f"Lag-1 autocorrelation (ρ): {lag1_autocorr:.3f}")
    print(f"Design effect: {design_effect:.2f}")
    print(f"SE inflation factor: {se_inflation:.2f}×")
    print(f"→ True SEs are ~{se_inflation:.2f}× larger than reported")

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

    # Store diagnostics
    diagnostics = {
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'lag1_autocorr': lag1_autocorr,
        'design_effect': design_effect,
        'se_inflation': se_inflation,
        'log_likelihood': result.llf,
        'aic': result.aic,
        'bic': result.bic
    }

    print("\n" + "=" * 80)
    print(f"Log-Likelihood: {result.llf:.2f}")
    print(f"AIC: {result.aic:.2f}")
    print(f"BIC: {result.bic:.2f}")
    print("=" * 80)

    return {
        'model': model,
        'result': result,
        'results_df': results_df,
        'random_effects_df': random_effects_df,
        'diagnostics': diagnostics
    }



if __name__ == '__main__':
    ######## PREPARATION #########
    ROOT = Path().resolve().parent
    DATA = ROOT / "data"
    OUTPUT = ROOT / 'output'
    STUDY_PLOTS = OUTPUT / 'plots' / 'data_analysis_plots'
    EXPERIMENT_DATA = DATA / "experiment_results"
    # QTC_DATA = DATA / "qtc_data"  # not necessary, since we import pre-computed features (below)
    FEATURE_OUTPUT_DATA = DATA / "precomputed_features"


    #####################################################
    ################ FEATURE EXTRACTION #################
    #####################################################

    all_subject_data_frame = pd.read_csv(filemgmt.most_recent_file(FEATURE_OUTPUT_DATA, ".csv", ["Combined Statistics 15seg"]))  # pd.read_csv(FEATURE_OUTPUT_DATA / "statistics_temp_eeg_alpha.csv")  # set to None to run computation

    # add pre-trial and trial comparison? -> forego for now, since pre-trial isn't as strictly recorded as within
    if all_subject_data_frame is None:
        ### PSD PARAMETERS
        # average over below bands and channels (region should label channel group):
        modality_region_channels_band_psd_list: list[tuple[str, str, list[str], str]] = [
            # MODALITY, REGION_LABEL, REGION_CHANNELS, BAND
            ('eeg', 'FC_CP_T',
             EEG_CHANNELS_BY_AREA['Fronto-Central'] + EEG_CHANNELS_BY_AREA['Centro-Parietal'] + EEG_CHANNELS_BY_AREA['Temporal'],
            'theta'),  # H2
            ('eeg', 'F_C',
             EEG_CHANNELS_BY_AREA['Frontal'] + EEG_CHANNELS_BY_AREA['Central'], 'beta'),  # H3
            ('eeg', 'P_PO',
             EEG_CHANNELS_BY_AREA['Parietal'] + EEG_CHANNELS_BY_AREA['Parieto-Occipital'], 'alpha'),  # H4
            ('eeg', 'Global', None, 'gamma'),
            ('emg_1_flexor', 'Global', None, 'all'),
            ('emg_2_extensor', 'Global', None, 'all')
        ]
        # select target band from freq_band_psd_per_segment_dict from
        #   - EMG   -> 'slow', 'fast'
        #   - EEG
        #       -> 'delta' (not sufficient frequencies)
        #       -> 'theta' (beat perception, entrainment)
        #       -> 'alpha' (auditory attention)
        #       -> 'beta' (motor control)
        #       -> 'gamma'
            
        # window lenghts:
        psd_time_window_size_sec = .25
        psd_is_log_scaled: bool = True  # define whether PSD was log scaled during feature extraction




        ### CMC PARAMETERS
        muscle_operator_band_cmc_list: list[tuple[str, str, str]] = [
            # MUSCLE, OPERATOR (max / mean / median), BAND
            ('Flexor', 'max', 'beta'),
            ('Flexor', 'max', 'gamma'),
            ('Flexor', 'mean', 'beta'),
            ('Flexor', 'mean', 'gamma'),
            ('Extensor', 'max', 'beta'),
            ('Extensor', 'max', 'gamma'),
            ('Extensor', 'mean', 'beta'),
            ('Extensor', 'mean', 'gamma'),
        ]
        # select target band from freq_band_psd_per_segment_dict from
        #   - EMG   -> 'slow', 'fast'
        #   - EEG
        #       -> 'delta' (not sufficient frequencies)
        #       -> 'theta' (beat perception, entrainment)
        #       -> 'alpha' (auditory attention)
        #       -> 'beta' (motor control)
        #       -> 'gamma'

        # window lengths:
        cmc_time_window_size_sec = 2.0
        




        ### DATA AGGREGATION PARAMETERS
        # how many segments:
        n_within_trial_segments: int = 1  # slices per 15s trial
        print(f"Will split 45sec trial into {n_within_trial_segments} segments (each ~{45/n_within_trial_segments:.1f}sec)")
        # standardization?
        modalities_to_standardize: list[str] = []  #['PSD', 'Force']
        music_features_to_fetch = ('BPM_manual', 'Spectral Flux Mean', 'Spectral Centroid Mean', 'IOI Variance Coeff',
                                   'Syncopation Ratio')







        ########### ITERATE OVER ALL PARTICIPANTS ###########
        all_subject_data_frame = pd.DataFrame(columns=['Subject ID'])
        for subject_ind in range(8):
            print("\n")
            print("-" * 100)
            print(f"------------     Aggregating data for subject\t\t{subject_ind:02}     ------------- ")
            print("-" * 100)

            # dependent directories:
            subject_psd_save_dir = FEATURE_OUTPUT_DATA / f"subject_{subject_ind:02}"
            subject_cmc_save_dir = FEATURE_OUTPUT_DATA / f"subject_{subject_ind:02}"
            subject_experiment_data_dir = EXPERIMENT_DATA / f"subject_{subject_ind:02}"








            ### IMPORT LOG AND SERIAL DATAFRAMES
            log_df = data_integration.fetch_enriched_log_frame(subject_experiment_data_dir)
            serial_df = data_integration.fetch_enriched_serial_frame(subject_experiment_data_dir)
            # make time-zone aware:
            log_df.index = data_analysis.make_timezone_aware(log_df.index)
            serial_df.index = data_analysis.make_timezone_aware(serial_df.index)

            # slice towards qtc measurements:
            qtc_start, qtc_end = data_integration.get_qtc_measurement_start_end(log_df, False)
            sliced_log_df = log_df[qtc_start:qtc_end]
            sliced_serial_df = serial_df[qtc_start:qtc_end]











            ### DERIVE SEGMENT TIMESPANS
            # trial start end times:
            trial_start_end_dict = data_integration.get_all_task_start_ends(log_df, 'dict')
            # convert into segment start end times:
            seg_starts = []; seg_ends = []
            for start, end in trial_start_end_dict.values():
                seg_starts_range = pd.date_range(start, end, periods=n_within_trial_segments+1, inclusive='both')
                for ind, seg_start in enumerate(seg_starts_range.values[:-1]):
                    seg_starts.append(seg_start)
                    seg_ends.append(seg_starts_range.values[ind+1])






            ### PREPARE DATAFRAME
            single_subject_data_frame = pd.DataFrame(index=range(len(seg_starts)))







            ### IMPORT AND AGGREGATE PSD DATA PER HYPOTHESIS (modality_region_channels_band_psd_list)
            # loop over configurations:
            for modality, region_label, channels, band in modality_region_channels_band_psd_list:
                # import PSD:
                psd_spectrograms, psd_times, psd_freqs = features.fetch_stored_spectrograms(
                    subject_psd_save_dir, modality='PSD', file_identifier=modality)
                #   -> shape: (n_windows, n_freqs, n_channels), (n_windows), (n_freqs)
                # convert PSD second time-centers into timestamps:
                psd_timestamps = data_analysis.add_time_index(
                    start_timestamp=qtc_start + pd.Timedelta(seconds=psd_time_window_size_sec / 2),
                    end_timestamp=qtc_end - pd.Timedelta(seconds=psd_time_window_size_sec / 2),
                    n_timesteps=len(psd_times)
                )
                psd_timestamps = data_analysis.make_timezone_aware(psd_timestamps)

                # takes shapes (n_windows, n_freqs, n_channels)
                psd_aggregated = aggregate_psd_spectrogram(psd_spectrograms, psd_freqs, normalize_mvc=False,
                                                           channel_indices=[EEG_CHANNEL_IND_DICT[ch] for ch in channels] if channels is not None else None,
                                                           is_log_scaled=psd_is_log_scaled, freq_slice=band,
                                                           aggregation_ops=[('mean', 1),   # mean within freq band
                                                                            # mean over EEG channels, max over EMG ones:
                                                                            ('mean' if 'eeg' in modality else 'max', 1),
                                                                            ],)
                # returns shape (n_windows, )

                # split per segment:
                psd_per_segment = data_analysis.apply_window_operator(
                    window_timestamps=seg_starts,
                    window_timestamps_ends=seg_ends,

                    target_array=psd_aggregated,
                    target_timestamps=psd_timestamps,

                    operation='mean',
                    axis=0,  # time axis
                )  # (n_trials, )

                # save to dataframe:
                single_subject_data_frame[f"PSD_{modality}_{region_label}_{band}"] = psd_per_segment








                ### IMPORT AND AGGREGATE CMC DATA PER HYPOTHESIS (muscle_operator_band_cmc_list)
                # loop over configurations:
                for muscle, operator, band in muscle_operator_band_cmc_list:
                    # import CMC:
                    cmc_spectrograms, cmc_times, cmc_freqs = features.fetch_stored_spectrograms(
                        subject_cmc_save_dir, modality='CMC', file_identifier=muscle)
                    #   -> shape: (n_windows, n_freqs, n_channels), (n_windows), (n_freqs)
                    # convert CMC second time-centers into timestamps:
                    cmc_timestamps = data_analysis.add_time_index(
                        start_timestamp=qtc_start + pd.Timedelta(seconds=cmc_time_window_size_sec / 2),
                        end_timestamp=qtc_end - pd.Timedelta(seconds=cmc_time_window_size_sec / 2),
                        n_timesteps=len(cmc_times)
                    )
                    cmc_timestamps = data_analysis.make_timezone_aware(cmc_timestamps)

                    # takes shapes (n_windows, n_freqs, n_channels)
                    cmc_aggregated = aggregate_psd_spectrogram(cmc_spectrograms, cmc_freqs, normalize_mvc=False,
                                                               is_log_scaled=False, freq_slice=band,
                                                               aggregation_ops=[('max', 1),  # mean within freq band
                                                                                # either take peak or average over channels
                                                                                (operator, 1),
                                                                                ]
                                                               )
                    # returns shape (n_windows, )

                    # split per segment:
                    cmc_per_segment = data_analysis.apply_window_operator(
                        window_timestamps=seg_starts,
                        window_timestamps_ends=seg_ends,

                        target_array=cmc_aggregated,
                        target_timestamps=cmc_timestamps,

                        operation='mean',
                        axis=0,  # time axis
                    )  # (n_trials, )

                    # save to dataframe:
                    single_subject_data_frame[f"CMC_{muscle}_{operator}_{band}"] = cmc_per_segment











            ### INDEPENDENT VARIABLE AGGREGATION
            # force level:
            force_level_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=sliced_serial_df['Task-wise Scaled Force'],
                operation='median',
                axis=0,  # time axis
            )

            # trial category:
            song_id_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=log_df['Song ID'],
                operation='mode',  # most common string value
                axis=0,  # time axis
            )
            silence_id_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=log_df['Silence ID'],
                operation='mode',  # most common string value
                axis=0,  # time axis
            )
            is_music_trial = [not pd.isna(song_id) and pd.isna(silence_id) for song_id, silence_id in zip(song_id_per_segment, silence_id_per_segment)]

            # trial ID:
            trial_id_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=log_df['Trial ID'],
                operation='mode', axis=0,
            )

            # musical features:
            music_feature_tuples = [
                data_integration.fetch_music_features(log_df, trial_id=trial_id,
                                                      features_to_return=music_features_to_fetch) for trial_id in trial_id_per_segment
            ]


            # song category:
            song_category_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=log_df['Perceived Category'],
                operation='mode', axis=0,
            )
            song_familiarity_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=log_df['Familiarity'],
                operation='mode', axis=0,
            )
            emotional_state_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=log_df['Emotional State'],
                operation='mode', axis=0,
            )

            song_liking_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=log_df['Liking'],
                operation='mode', axis=0,
            )

            task_frequency_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=log_df['Task Frequency'],
                operation='mode', axis=0,
            )

            # heart rate::
            bpm_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=sliced_serial_df['bpm'],
                operation='median',
                axis=0,  # time axis
            )

            # heart rate variability:
            hrv_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=sliced_serial_df['hrv'],
                operation='median',
                axis=0,  # time axis
            )

            # galvanic skin response:
            gsr_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=sliced_serial_df['gsr'],
                operation='median',
                axis=0,
            )



            ### APPEND TO PER SUBJECT DATAFRAME
            # append to subject df:
            for column_name, data in [('Subject ID', [subject_ind] * len(seg_starts)),
                                      ('Trial ID', trial_id_per_segment),
                                      ('Music Listening', is_music_trial),
                                      ('Median Force Level [0-1]', force_level_per_segment),
                                      ('Task Frequency', task_frequency_per_segment),
                                      ('Emotional State [0-7]', emotional_state_per_segment),
                                      ('Median Heart Rate [bpm]', bpm_per_segment),
                                      ('Median HRV [sec]', hrv_per_segment),
                                      ('Galvanic Skin Response [0-3.3]', gsr_per_segment),
                                      # music features:
                                      ('Perceived Category', song_category_per_segment),
                                      ('Liking [0-7]', song_liking_per_segment),
                                      ('Familiarity [0-7]', song_familiarity_per_segment),
                                      (list(music_features_to_fetch), music_feature_tuples),
                                      ]:
                single_subject_data_frame[column_name] = data

            # standardize:
            for modality in modalities_to_standardize:
                for column in [c for c in single_subject_data_frame.columns if modality in c]:
                    # columns to standardize:
                    print("Standardizing statistics for: ", column)
                    single_subject_data_frame[column] = single_subject_data_frame[column].transform(lambda x: (x - x.mean()) / x.std())






            ### CONCAT WITH ALL SUBJECT DATAFRAME
            all_subject_data_frame = pd.concat([all_subject_data_frame, single_subject_data_frame], axis=0)






        ######### SAVE COMBINED STATISTICS #########
        all_subject_data_frame.to_csv(FEATURE_OUTPUT_DATA / filemgmt.file_title(f"Combined Statistics {int(n_within_trial_segments)}seg", ".csv"), index=False)







    ######## MODEL 1: MUSIC VS. SILENCE #########
    
    # Store all results for summary tables
    all_model_results = []
    
    for hypothesis, dependent_variable in [('H1: Flexor Beta Peak CMC Increases with Music', "CMC_Flexor_max_beta"),
                                           ('H1: Flexor Beta Avg. CMC Increases with Music', "CMC_Flexor_mean_beta"),
                                           ('H1: Flexor Gamma Peak CMC Increases with Music', "CMC_Flexor_max_gamma"),
                                           ('H1: Flexor Gamma Avg. CMC Increases with Music', "CMC_Flexor_mean_gamma"),
                                           ('H1: Extensor Beta Peak CMC Increases with Music', "CMC_Extensor_max_beta"),
                                           ('H1: Extensor Beta Avg. CMC Increases with Music', "CMC_Extensor_mean_beta"),
                                           ('H1: Extensor Gamma Peak CMC Increases with Music', "CMC_Extensor_max_gamma"),
                                           ('H1: Extensor Gamma Avg. CMC Increases with Music', "CMC_Extensor_mean_gamma"),

                                           ('H2: Temporal Prediction PSD Increases with Music', 'PSD_eeg_FC_CP_T_theta'),
                                           ('H3: Vigilance PSD Increases with Music', 'PSD_eeg_F_C_beta'),
                                           ('H4: Internal vs. External Attention PSD changes with Music', 'PSD_eeg_P_PO_alpha'),
                                           ('H5: Long Range Interactions PSD Increases with Music', 'PSD_eeg_Global_gamma'),
                                           ('VALIDATION: EMG Flexor PSD Increases with Force', 'PSD_emg_1_flexor_Global_all'),
                                           ('VALIDATION: EMG Extensor PSD Increases with Force', 'PSD_emg_2_extensor_Global_all'),]:
        print("\n")
        print("=" * 100)
        print("=" * 100)
        print(f"HYPOTHESIS:\t\t{hypothesis} ")
        print(f"DEPENDENT VARIABLE:\t{dependent_variable}")
        print("=" * 100)
        print("=" * 100, "\n"*3)

        print("\n")
        print("-" * 100)
        print(f"-------------------------     Comparison Level 1 (Music + Force Lvl + Trial ID)     --------------------------- ")
        print("-" * 100, "\n")
        
        # OLS Model - Level 1
        ols_level1_results = fit_linear_regression_model(
            all_subject_data_frame,
            response_var=dependent_variable,
            condition_vars={'Music Listening': 'categorical',},
            explanatory_vars=['Median Force Level [0-1]', 'Trial ID']
        )
        
        # Store results
        for _, row in ols_level1_results['results_df'].iterrows():
            all_model_results.append({
                'Hypothesis': hypothesis,
                'Dependent_Variable': dependent_variable,
                'Model_Type': 'OLS',
                'Comparison_Level': 'Level 1 (Music + Force + Trial ID)',
                'Parameter': row['Parameter'],
                'Coefficient': row['Coefficient'],
                'p_value': row['p-value (adjusted)'],
                'SE': row['SE (adjusted)']
            })
        
        # Mixed Effects Model - Level 1
        lme_level1_results = fit_mixed_effects_model(
            df=all_subject_data_frame,
            response_var=dependent_variable,
            condition_vars={'Music Listening': 'categorical'},
            explanatory_vars=['Median Force Level [0-1]', 'Trial ID'],
            grouping_var='Subject ID'
        )
        
        # Store results
        for _, row in lme_level1_results['results_df'].iterrows():
            all_model_results.append({
                'Hypothesis': hypothesis,
                'Dependent_Variable': dependent_variable,
                'Model_Type': 'LME',
                'Comparison_Level': 'Level 1 (Music + Force + Trial ID)',
                'Parameter': row['Parameter'],
                'Coefficient': row['Coefficient'],
                'p_value': row['p-value (adjusted)'],
                'SE': row['SE (adjusted)']
            })

        print("\n"*3)
        print("-" * 100)
        print(f"-------------------------     Comparison Level 2 (Category + Familiarity + Liking + Force Lvl + Trial ID)     --------------------------- ")
        print("-" * 100, "\n")
        
        # OLS Model - Level 2 (only music trials)
        ols_level2_results = fit_linear_regression_model(
            all_subject_data_frame.loc[all_subject_data_frame['Music Listening']],
            response_var=dependent_variable,
            condition_vars={'Perceived Category': 'categorical',
                            'Familiarity [0-7]': 'ordinal'},
            explanatory_vars=['Median Force Level [0-1]', 'Liking [0-7]', 'Trial ID']
        )
        
        # Store results
        for _, row in ols_level2_results['results_df'].iterrows():
            all_model_results.append({
                'Hypothesis': hypothesis,
                'Dependent_Variable': dependent_variable,
                'Model_Type': 'OLS',
                'Comparison_Level': 'Level 2 (Subjective Music Features)',
                'Parameter': row['Parameter'],
                'Coefficient': row['Coefficient'],
                'p_value': row['p-value (adjusted)'],
                'SE': row['SE (adjusted)']
            })

        # Mixed Effects Model - Level 2
        lme_level2_results = fit_mixed_effects_model(
            df=all_subject_data_frame.loc[all_subject_data_frame['Music Listening']],
            response_var=dependent_variable,
            condition_vars={'Perceived Category': 'categorical',
                            'Familiarity [0-7]': 'ordinal'},
            explanatory_vars=['Median Force Level [0-1]', 'Liking [0-7]', 'Trial ID'],
            grouping_var='Subject ID'
        )

        # Store results
        for _, row in lme_level2_results['results_df'].iterrows():
            all_model_results.append({
                'Hypothesis': hypothesis,
                'Dependent_Variable': dependent_variable,
                'Model_Type': 'LME',
                'Comparison_Level': 'Level 2 (Subjective Music Features)',
                'Parameter': row['Parameter'],
                'Coefficient': row['Coefficient'],
                'p_value': row['p-value (adjusted)'],
                'SE': row['SE (adjusted)']
            })

        """
        print("\n" * 3)
        print("-" * 100)
        print(
            f"-------------------------     Comparison Level 3 (Category + Familiarity + Liking + Force Lvl))     --------------------------- ")
        print("-" * 100, "\n")

        # OLS Model - Level 2 (only music trials)
        ols_level2_results = fit_linear_regression_model(
            all_subject_data_frame.loc[all_subject_data_frame['Music Listening']],
            response_var=dependent_variable,
            condition_vars={'Perceived Category': 'categorical',
                            'Familiarity [0-7]': 'ordinal'},
            explanatory_vars=['Median Force Level [0-1]', 'Liking [0-7]']
        )

        # Store results
        for _, row in ols_level2_results['results_df'].iterrows():
            all_model_results.append({
                'Hypothesis': hypothesis,
                'Dependent_Variable': dependent_variable,
                'Model_Type': 'OLS',
                'Comparison_Level': 'Level 2 (Subjective Music Features)',
                'Parameter': row['Parameter'],
                'Coefficient': row['Coefficient'],
                'p_value': row['p-value (adjusted)'],
                'SE': row['SE (adjusted)']
            })

        # Mixed Effects Model - Level 2
        lme_level2_results = fit_mixed_effects_model(
            df=all_subject_data_frame.loc[all_subject_data_frame['Music Listening']],
            response_var=dependent_variable,
            condition_vars={'Perceived Category': 'categorical',
                            'Familiarity [0-7]': 'ordinal'},
            explanatory_vars=['Median Force Level [0-1]', 'Liking [0-7]'],
            grouping_var='Subject ID'
        )

        # Store results
        for _, row in lme_level2_results['results_df'].iterrows():
            all_model_results.append({
                'Hypothesis': hypothesis,
                'Dependent_Variable': dependent_variable,
                'Model_Type': 'LME',
                'Comparison_Level': 'Level 2 (Subjective Music Features)',
                'Parameter': row['Parameter'],
                'Coefficient': row['Coefficient'],
                'p_value': row['p-value (adjusted)'],
                'SE': row['SE (adjusted)']
            })"""
    
    
    # ============================================================================
    # CREATE COMPREHENSIVE SUMMARY TABLES
    # ============================================================================
    
    print("\n" * 3)
    print("="*120)
    print("="*120)
    print("COMPREHENSIVE STATISTICAL SUMMARY")
    print("="*120)
    print("="*120)
    
    # Convert all results to DataFrame
    results_df = pd.DataFrame(all_model_results)
    
    # ============================================================================
    # TABLE 1: Music Listening Effect (All Hypotheses)
    # ============================================================================
    print("\n" + "="*120)
    print("TABLE 1: MUSIC LISTENING EFFECT ACROSS ALL HYPOTHESES (Level 1 Models)")
    print("="*120 + "\n")
    
    music_effects = results_df[
        (results_df['Comparison_Level'] == 'Level 1 (Music + Force)') &
        (results_df['Parameter'].str.contains('Music', case=False, na=False))
    ].copy()
    
    # Create pivot table
    music_summary = music_effects.pivot_table(
        index=['Hypothesis', 'Dependent_Variable'],
        columns='Model_Type',
        values=['Coefficient', 'p_value'],
        aggfunc='first'
    )
    
    # Flatten column names
    music_summary.columns = ['_'.join(col).strip() for col in music_summary.columns.values]
    music_summary = music_summary.reset_index()
    
    # Add significance markers
    if 'p_value_OLS' in music_summary.columns:
        music_summary['Sig_OLS'] = music_summary['p_value_OLS'].apply(
            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        )
    if 'p_value_LME' in music_summary.columns:
        music_summary['Sig_LME'] = music_summary['p_value_LME'].apply(
            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        )
    
    print(music_summary.to_string(index=False))
    
    # Save to CSV
    music_summary.to_csv(FEATURE_OUTPUT_DATA / "summary_music_effects.csv", index=False)
    print(f"\n✓ Saved to: {FEATURE_OUTPUT_DATA / 'summary_music_effects.csv'}")
    
    # Also save a simplified version for quick reference
    music_summary_simple = music_effects[['Hypothesis', 'Dependent_Variable', 'Model_Type', 'Coefficient', 'p_value']].copy()
    music_summary_simple['Significance'] = music_summary_simple['p_value'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    )
    music_summary_simple.to_csv(FEATURE_OUTPUT_DATA / "summary_music_effects_simple.csv", index=False)
    
    
    # ============================================================================
    # TABLE 2: Force Level Effect (All Hypotheses, Both Levels)
    # ============================================================================
    print("\n" + "="*120)
    print("TABLE 2: FORCE LEVEL EFFECT ACROSS ALL HYPOTHESES")
    print("="*120 + "\n")
    
    force_effects = results_df[
        results_df['Parameter'].str.contains('Force', case=False, na=False)
    ].copy()
    
    # Create summary table
    force_summary = force_effects.pivot_table(
        index=['Hypothesis', 'Dependent_Variable', 'Comparison_Level'],
        columns='Model_Type',
        values=['Coefficient', 'p_value'],
        aggfunc='first'
    )
    
    force_summary.columns = ['_'.join(col).strip() for col in force_summary.columns.values]
    force_summary = force_summary.reset_index()
    
    # Add significance markers
    if 'p_value_OLS' in force_summary.columns:
        force_summary['Sig_OLS'] = force_summary['p_value_OLS'].apply(
            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        )
    if 'p_value_LME' in force_summary.columns:
        force_summary['Sig_LME'] = force_summary['p_value_LME'].apply(
            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        )
    
    print(force_summary.to_string(index=False))
    
    # Save to CSV
    force_summary.to_csv(FEATURE_OUTPUT_DATA / "summary_force_effects.csv", index=False)
    print(f"\n✓ Saved to: {FEATURE_OUTPUT_DATA / 'summary_force_effects.csv'}")
    
    # Also save a simplified version
    force_summary_simple = force_effects[['Hypothesis', 'Dependent_Variable', 'Model_Type', 'Comparison_Level', 'Coefficient', 'p_value']].copy()
    force_summary_simple['Significance'] = force_summary_simple['p_value'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    )
    force_summary_simple.to_csv(FEATURE_OUTPUT_DATA / "summary_force_effects_simple.csv", index=False)
    
    
    # ============================================================================
    # TABLE 3: Music Features (Category, Familiarity, Liking) - Level 2 Only
    # ============================================================================
    print("\n" + "="*120)
    print("TABLE 3: MUSIC FEATURE EFFECTS (Level 2 Models - Music Trials Only)")
    print("="*120 + "\n")
    
    music_features = results_df[
        (results_df['Comparison_Level'] == 'Level 2 (Music Features)') &
        (~results_df['Parameter'].str.contains('Intercept', case=False, na=False)) &
        (~results_df['Parameter'].str.contains('Force', case=False, na=False))
    ].copy()
    
    # Group by hypothesis and parameter type
    for hypothesis_name in music_features['Hypothesis'].unique():
        print(f"\n{'-'*120}")
        print(f"{hypothesis_name}")
        print(f"{'-'*120}")
        
        hyp_data = music_features[music_features['Hypothesis'] == hypothesis_name]
        
        # Display each parameter
        for param in hyp_data['Parameter'].unique():
            param_data = hyp_data[hyp_data['Parameter'] == param]
            
            for _, row in param_data.iterrows():
                sig = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else ('*' if row['p_value'] < 0.05 else 'ns'))
                print(f"  {row['Parameter']:<50s}  β={row['Coefficient']:>8.4f}  p={row['p_value']:>8.4f} {sig:>4s}")
    
    # Save detailed table
    music_features_pivot = music_features.pivot_table(
        index=['Hypothesis', 'Parameter'],
        columns='Model_Type',
        values=['Coefficient', 'p_value'],
        aggfunc='first'
    )
    music_features_pivot.columns = ['_'.join(col).strip() for col in music_features_pivot.columns.values]
    music_features_pivot = music_features_pivot.reset_index()
    
    music_features_pivot.to_csv(FEATURE_OUTPUT_DATA / "summary_music_features.csv", index=False)
    print(f"\n✓ Saved to: {FEATURE_OUTPUT_DATA / 'summary_music_features.csv'}")
    
    
    # ============================================================================
    # TABLE 4: Master Summary - All Effects by Parameter Type
    # ============================================================================
    print("\n" + "="*120)
    print("TABLE 4: COMPLETE RESULTS MASTER TABLE - ALL EFFECTS")
    print("="*120 + "\n")
    
    # Save complete results
    results_df_formatted = results_df.copy()
    results_df_formatted['Significance'] = results_df_formatted['p_value'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    )
    
    results_df_formatted.to_csv(FEATURE_OUTPUT_DATA / "summary_all_results_master.csv", index=False)
    print(f"✓ Saved complete results to: {FEATURE_OUTPUT_DATA / 'summary_all_results_master.csv'}")
    
    # ============================================================================
    # Display organized by parameter type
    # ============================================================================
    
    # Group parameters by type
    parameter_groups = {
        'Intercept': results_df_formatted[results_df_formatted['Parameter'].str.contains('Intercept', case=False, na=False)],
        'Music Listening': results_df_formatted[results_df_formatted['Parameter'].str.contains('Music Listening', case=False, na=False)],
        'Perceived Category': results_df_formatted[results_df_formatted['Parameter'].str.contains('Perceived Category', case=False, na=False)],
        'Familiarity': results_df_formatted[results_df_formatted['Parameter'].str.contains('Familiarity', case=False, na=False)],
        'Liking': results_df_formatted[results_df_formatted['Parameter'].str.contains('Liking', case=False, na=False)],
        'Force Level': results_df_formatted[results_df_formatted['Parameter'].str.contains('Force', case=False, na=False)],
    }
    
    # Display each parameter group
    for group_name, group_data in parameter_groups.items():
        if len(group_data) > 0:
            print(f"\n{'='*120}")
            print(f"{group_name.upper()} - {len(group_data)} effects")
            print(f"{'='*120}")
            
            # Create pivot table for this group
            pivot = group_data.pivot_table(
                index=['Hypothesis', 'Parameter'],
                columns=['Model_Type', 'Comparison_Level'],
                values=['Coefficient', 'p_value'],
                aggfunc='first'
            )
            
            # Display in readable format
            for hypothesis in group_data['Hypothesis'].unique():
                hyp_data = group_data[group_data['Hypothesis'] == hypothesis]
                print(f"\n{hypothesis}")
                print("-" * 120)
                
                for param in hyp_data['Parameter'].unique():
                    param_data = hyp_data[hyp_data['Parameter'] == param]
                    print(f"\n  {param}:")
                    
                    for _, row in param_data.iterrows():
                        sig = row['Significance']
                        print(f"    {row['Model_Type']:>5s} | {row['Comparison_Level']:<30s} | β={row['Coefficient']:>8.4f} | SE={row['SE']:>8.4f} | p={row['p_value']:>8.4f} {sig:>4s}")
    
    # ============================================================================
    # Summary statistics
    # ============================================================================
    print(f"\n\n{'='*120}")
    print("SUMMARY STATISTICS")
    print(f"{'='*120}\n")
    
    total_effects = len(results_df_formatted)
    sig_001 = len(results_df_formatted[results_df_formatted['p_value'] < 0.001])
    sig_01 = len(results_df_formatted[results_df_formatted['p_value'] < 0.01])
    sig_05 = len(results_df_formatted[results_df_formatted['p_value'] < 0.05])
    
    print(f"Total effects tested:              {total_effects}")
    print(f"Significant at p < 0.001 (***):    {sig_001} ({sig_001/total_effects*100:.1f}%)")
    print(f"Significant at p < 0.01 (**):      {sig_01} ({sig_01/total_effects*100:.1f}%)")
    print(f"Significant at p < 0.05 (*):       {sig_05} ({sig_05/total_effects*100:.1f}%)")
    print(f"Non-significant (ns):              {total_effects - sig_05} ({(total_effects-sig_05)/total_effects*100:.1f}%)")
    
    # Breakdown by model type
    print(f"\n{'Breakdown by Model Type:':-<60}")
    for model_type in results_df_formatted['Model_Type'].unique():
        model_data = results_df_formatted[results_df_formatted['Model_Type'] == model_type]
        model_sig = len(model_data[model_data['p_value'] < 0.05])
        print(f"  {model_type:>5s}: {len(model_data)} effects, {model_sig} significant ({model_sig/len(model_data)*100:.1f}%)")
    
    # Breakdown by comparison level
    print(f"\n{'Breakdown by Comparison Level:':-<60}")
    for level in results_df_formatted['Comparison_Level'].unique():
        level_data = results_df_formatted[results_df_formatted['Comparison_Level'] == level]
        level_sig = len(level_data[level_data['p_value'] < 0.05])
        print(f"  {level}: {len(level_data)} effects, {level_sig} significant ({level_sig/len(level_data)*100:.1f}%)")
    
    # List all significant effects
    significant_results = results_df_formatted[results_df_formatted['p_value'] < 0.05]
    if len(significant_results) > 0:
        print(f"\n\n{'='*120}")
        print(f"ALL SIGNIFICANT EFFECTS (p < 0.05): {len(significant_results)} effects")
        print(f"{'='*120}\n")
        
        # Sort by p-value
        significant_results_sorted = significant_results.sort_values('p_value')
        
        for _, row in significant_results_sorted.iterrows():
            print(f"{row['Parameter']:<45s} | {row['Model_Type']:<5s} | {row['Hypothesis'][:55]:<55s} | β={row['Coefficient']:>7.4f} | p={row['p_value']:>7.4f} {row['Significance']:>4s}")
    else:
        print("\n\nNo significant effects found at p < 0.05")
    
    print("\n" + "="*120)
    print("SUMMARY TABLES COMPLETE")
    print("="*120 + "\n")














### MODALITY-DEPENDENT PSD AGGREGATION
"""if 'emg' in psd_modality:
    # 1) MVC Normalization per Channel if not log-scaled!
    if not psd_is_log_scaled:
        mvc = np.max(np.max(psd_spectrograms, axis=0, keepdims=True), axis=1, keepdims=True)  # maximum over time and frequencies
        psd_spectrograms = psd_spectrograms / mvc * 100  # * 100 for [%]

    # 2) Average over a: all frequencies (broad band) OR b: frequency bands
    freq_band_spec_dict: dict[str, np.ndarray] = features.aggregate_spectrogram_over_frequency_band(
        psd_spectrograms, psd_freqs, 'mean',
        frequency_bands={'slow': (0, 40), 'fast': (60, 250)},  # slow and fast twitching, deliberately omitting overlap zone (40-60)
    )  # value shape: (n_times, n_channels)

    # 3) Average over time (per trial)
    freq_band_psd_per_segment_dict: dict[str, np.ndarray] = {}
    for band, psd_arr in freq_band_spec_dict.items():
        # compute psd per segment:
        psd_per_segment = data_analysis.apply_window_operator(
            window_timestamps=seg_starts,
            window_timestamps_ends=seg_ends,

            target_array=psd_arr,
            target_timestamps=psd_timestamps,

            operation='mean',
            axis=0,  # time axis
        )  # (n_trials, n_channels)

        # 4) Max-Pool over Channels
        psd_per_segment = np.nanmax(psd_per_segment, axis=1)

        # 5) Save:
        freq_band_psd_per_segment_dict[band] = psd_per_segment


elif 'eeg' in psd_modality:
    # 1) Average over Region-of-Interest (RoI)
    eeg_channel_subset_inds = [EEG_CHANNEL_IND_DICT[ch] for ch in eeg_channel_subset]
    psd_subset = psd_spectrograms[:, :, eeg_channel_subset_inds]
    psd_roi_avg = np.nanmean(psd_subset, axis=2)
    # shape: (n_times, n_frequencies)

    # 2) Average over Frequency Bands
    freq_band_spec_dict: dict[str, np.ndarray] = features.aggregate_spectrogram_over_frequency_band(
        psd_roi_avg, psd_freqs, 'mean', frequency_axis=1,
    )  # value shape: (n_times)

    # 3) Average over time (per trial)
    freq_band_psd_per_segment_dict: dict[str, np.ndarray] = {}
    for band, psd_arr in freq_band_spec_dict.items():
        # compute psd per segment:
        psd_per_segment = data_analysis.apply_window_operator(
            window_timestamps=seg_starts,
            window_timestamps_ends=seg_ends,

            target_array=psd_arr,
            target_timestamps=psd_timestamps,

            operation='mean',
            axis=0,  # time axis
        )  # (n_trials)
        freq_band_psd_per_segment_dict[band] = psd_per_segment


"""