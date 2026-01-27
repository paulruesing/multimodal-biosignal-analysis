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


def fit_linear_regression_model(
        df: pd.DataFrame,
        response_var: str,
        condition_vars: dict,
        explanatory_vars: list,
        n_windows_per_trial: int = 9
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

    # Q-Q plot
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


    ################################################
    ################ PSD MODELLING #################
    ################################################

    all_subject_data_frame = pd.read_csv(FEATURE_OUTPUT_DATA / "statistics_temp_eeg_alpha.csv")  # set to None to run computation

    # todo: add pre-trial and trial comparison
    if all_subject_data_frame is None:
        ### PARAMETERS
        modality: Literal['eeg', 'emg_1_flexor', 'emg_2_extensor'] = 'eeg'
        is_log_scaled: bool = True  # define whether PSD was log scaled during feature extraction
        n_within_trial_segments: int = 9  # 45s trials
        print(f"Will split 45sec trial into {n_within_trial_segments} segments (each ~{45/n_within_trial_segments:.1f}sec)")

        # select target band from freq_band_psd_per_segment_dict from
        #   - EMG   -> 'slow', 'fast'
        #   - EEG
        #       -> 'delta' (not sufficient frequencies)
        #       -> 'theta' (beat perception, entrainment)
        #       -> 'alpha' (auditory attention)
        #       -> 'beta' (motor control)
        #       -> 'gamma'
        freq_band = 'slow' if 'emg' in modality else 'theta'
        eeg_channel_subset = EEG_CHANNELS_BY_AREA['Fronto-Central'] + EEG_CHANNELS_BY_AREA['Central'] + \
                             EEG_CHANNELS_BY_AREA['Centro-Parietal'] + EEG_CHANNELS_BY_AREA['Temporal'] + \
                             EEG_CHANNELS_BY_AREA['Temporo-Parietal'] + EEG_CHANNELS_BY_AREA['Parietal']

        # the below needs to match feature_extraction_workflow.py
        cmc_eeg_channel_subset = EEG_CHANNELS_BY_AREA['Fronto-Central'] + EEG_CHANNELS_BY_AREA['Central'] + \
                             EEG_CHANNELS_BY_AREA['Centro-Parietal'] + EEG_CHANNELS_BY_AREA['Temporal']






        ### ITERATE OVER ALL PARTICIPANTS
        all_subject_data_frame = pd.DataFrame(columns=['Subject ID', 'Music Listening', 'Force Level', 'PSD'])
        for subject_ind in range(6):
            print("\n")
            print("-" * 100)
            print(f"------------     Aggregating\t\t{modality:<20} data for subject\t\t{subject_ind:02}     ------------- ")
            print("-" * 100)


            # dependent directories:
            subject_psd_save_dir = FEATURE_OUTPUT_DATA / f"subject_{subject_ind:02}"
            subject_experiment_data_dir = EXPERIMENT_DATA / f"subject_{subject_ind:02}"






            ### IMPORT LOG AND SERIAL DATAFRAMES
            log_df = data_integration.fetch_enriched_log_frame(subject_experiment_data_dir)
            serial_df = data_integration.fetch_serial_measurements(subject_experiment_data_dir)
            # make time-zone aware:
            log_df.index = data_analysis.make_timezone_aware(log_df.index)
            serial_df.index = data_analysis.make_timezone_aware(serial_df.index)

            # slice towards qtc measurements:
            qtc_start, qtc_end = data_integration.get_qtc_measurement_start_end(log_df, False)
            sliced_log_df = log_df[qtc_start:qtc_end]
            sliced_serial_df = serial_df[qtc_start:qtc_end]






            ### IMPORT PSD
            psd_spectrograms, psd_times, psd_freqs = features.fetch_stored_spectrograms(subject_psd_save_dir,
                                                                                        modality='PSD',
                                                                                        file_identifier=modality)
            #   -> shape: (n_times, n_freqs, n_channels), (n_times), (n_freqs)
            # convert PSD second time-centers into timestamps:
            psd_time_window_size_sec = (psd_times[1] - psd_times[0])
            psd_timestamps = data_analysis.add_time_index(
                start_timestamp=qtc_start + pd.Timedelta(seconds=psd_time_window_size_sec / 2),
                end_timestamp=qtc_end - pd.Timedelta(seconds=psd_time_window_size_sec / 2),
                n_timesteps=len(psd_times)
            )
            psd_timestamps = data_analysis.make_timezone_aware(psd_timestamps)





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





            ### MODALITY-DEPENDENT PSD AGGREGATION
            if 'emg' in modality:
                # 1) MVC Normalization per Channel if not log-scaled!
                if not is_log_scaled:
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


            elif 'eeg' in modality:
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




            ### INDEPENDENT VARIABLE AGGREGATION
            # force level:
            force_level_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=sliced_serial_df['fsr'],
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




            ### PREPARE PER SUBJECT DATAFRAME
            # single subject df:
            single_subject_data_frame = pd.DataFrame(data={
                'Subject ID': [subject_ind] * len(seg_starts),
                'Music Listening': is_music_trial,
                'Force Level': force_level_per_segment,
                'PSD': freq_band_psd_per_segment_dict[freq_band],
                'Category': song_category_per_segment,
                'Familiarity': song_familiarity_per_segment,
            })

            # standardize:
            single_subject_data_frame['PSD'] = single_subject_data_frame['PSD'].transform(lambda x: (x - x.mean()) / x.std())
            single_subject_data_frame['Force Level'] = single_subject_data_frame['Force Level'].transform(
                lambda x: (x - x.mean()) / x.std())

            all_subject_data_frame = pd.concat([all_subject_data_frame, single_subject_data_frame], axis=0)


        all_subject_data_frame.to_csv(FEATURE_OUTPUT_DATA / f"statistics_temp_{modality}_{freq_band}.csv", index=False)







    ######## MODEL 1: MUSIC VS. SILENCE #########
    print("\n")
    print("-" * 100)
    print(f"-------------------------     Comparison Level 1 (Music vs. Silence)     --------------------------- ")
    print("-" * 100, "\n")
    fit_linear_regression_model(all_subject_data_frame, 'PSD',
                                condition_vars={'Music Listening': 'categorical',},
                                explanatory_vars=['Force Level'])
    """fit_mixed_effects_model(
        df=all_subject_data_frame,
        response_var='PSD',
        condition_vars={'Music Listening': 'categorical'},
        explanatory_vars=['Force Level'],
        grouping_var='Subject ID'
    )"""

    print("\n"*3)
    print("-" * 100)
    print(f"-------------------------     Comparison Level 2 (Music vs. Silence)     --------------------------- ")
    print("-" * 100, "\n")
    # only on music listening subset
    fit_linear_regression_model(all_subject_data_frame.loc[all_subject_data_frame['Music Listening']], 'PSD',
                                condition_vars={'Category': 'categorical',
                                                'Familiarity': 'ordinal',},
                                explanatory_vars=['Force Level'])
    quit()
    # Create condition variable from boolean 'Music Listening'
    all_subject_data_frame['condition'] = all_subject_data_frame['Music Listening'].apply(
        lambda x: 'music' if x else 'silence'
    )

    ### DATA CHECK
    print("Data Summary:")
    print(f"Total observations: {len(all_subject_data_frame)}")
    print(f"Unique participants: {all_subject_data_frame['Subject ID'].nunique()}")
    print(
        f"Observations per participant: {len(all_subject_data_frame) / all_subject_data_frame['Subject ID'].nunique():.1f}")
    print(f"\nCondition breakdown:")
    print(all_subject_data_frame['condition'].value_counts())
    print(f"\nPSD range: [{all_subject_data_frame['PSD'].min():.2f}, {all_subject_data_frame['PSD'].max():.2f}]")
    print(
        f"Force Level range: [{all_subject_data_frame['Force Level'].min():.2f}, {all_subject_data_frame['Force Level'].max():.2f}]")


    ### LINEAR REGRESSION
    print("\n"*3 + "=" * 80)
    print("Linear Regression Model (OLS)")
    print("=" * 80)

    model1 = smf.ols(
        "PSD ~ C(condition, Treatment('silence')) + Q('Force Level')",
        data=all_subject_data_frame
    ).fit()
    print(model1.summary())

    # ============ DIAGNOSTICS ============
    residuals1 = model1.resid

    # Q-Q plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    stats.probplot(residuals1, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q Plot (Residuals)")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals1, bins=30, edgecolor='black', density=True)
    axes[1].axvline(0, color='r', linestyle='--', label='Mean')
    axes[1].set_title("Residual Distribution")
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(residuals1)
    print(f"\nShapiro-Wilk p-value: {shapiro_p:.4f}")
    if shapiro_p < 0.05:
        print("  → Residuals significantly deviate from normality")
    else:
        print("  → Residuals approximately normal ✓")

    # ============ AUTOCORRELATION CHECK ============
    print("\n" + "=" * 80)
    print("AUTOCORRELATION CHECK (within segments per trial)")
    print("=" * 80)

    lag1_autocorr = np.corrcoef(residuals1[:-1], residuals1[1:])[0, 1]
    if np.isnan(lag1_autocorr):
        print("Warning: Autocorrelation is NaN")
        lag1_autocorr = 0.0

    n_windows_per_trial = 9
    design_effect = 1 + (n_windows_per_trial - 1) * lag1_autocorr
    se_inflation = np.sqrt(design_effect)

    print(f"Lag-1 autocorrelation (ρ): {lag1_autocorr:.3f}")
    print(f"Design effect: {design_effect:.2f}")
    print(f"SE inflation factor: {se_inflation:.2f}×")

    # Adjust SEs for autocorrelation
    result1_adjusted_se = model1.bse * se_inflation
    result1_adjusted_z = model1.params / result1_adjusted_se
    result1_adjusted_p = 2 * (1 - stats.norm.cdf(np.abs(result1_adjusted_z)))

    # Convert to Series for proper indexing
    if not isinstance(result1_adjusted_p, pd.Series):
        result1_adjusted_p = pd.Series(result1_adjusted_p, index=model1.params.index)

    print("\n" + "-" * 80)
    print("ADJUSTED RESULTS (corrected for autocorrelation):")
    print("-" * 80)
    print(f"{'Parameter':<50s} {'β':>10s} {'SE (adj)':>12s} {'p (adj)':>10s}")
    print("-" * 80)
    for param in model1.params.index:
        adj_se = result1_adjusted_se[param]
        adj_p = result1_adjusted_p[param]
        print(f"{param:<50s} {model1.params[param]:>10.4f} {adj_se:>12.4f} {adj_p:>10.4f}")






    ######## MODEL 2: CATEGORY TESTING #########
    print("\n")
    print("-" * 100)
    print(
        f"-------------------------     Comparison Level 1 (Category vs. Familiarity)     --------------------------- ")
    print("-" * 100, "\n")
    # Create condition variable from categorical 'Category' and ordinal 'Familiarity' (0-7)
    all_subject_data_frame['condition'] = all_subject_data_frame['Category'].astype(str) + '_Fam' + \
                                          all_subject_data_frame['Familiarity'].astype(str)

    ### DATA CHECK
    print("Data Summary:")
    print(f"Total observations: {len(all_subject_data_frame)}")
    print(f"Unique participants: {all_subject_data_frame['Subject ID'].nunique()}")
    print(
        f"Observations per participant: {len(all_subject_data_frame) / all_subject_data_frame['Subject ID'].nunique():.1f}")
    print(f"\nCondition breakdown:")
    print(all_subject_data_frame['condition'].value_counts())
    print(f"\nPSD range: [{all_subject_data_frame['PSD'].min():.2f}, {all_subject_data_frame['PSD'].max():.2f}]")
    print(
        f"Force Level range: [{all_subject_data_frame['Force Level'].min():.2f}, {all_subject_data_frame['Force Level'].max():.2f}]")
    print(
        f"\nFamiliarity range: [{all_subject_data_frame['Familiarity'].min()}, {all_subject_data_frame['Familiarity'].max()}]")

    ### LINEAR REGRESSION
    print("\n" * 3 + "=" * 80)
    print("Linear Regression Model (OLS)")
    print("=" * 80)

    model1 = smf.ols(
        "PSD ~ C(Category) + Familiarity + Q('Force Level')",
        data=all_subject_data_frame
    ).fit()
    print(model1.summary())

    # ============ DIAGNOSTICS ============
    residuals1 = model1.resid

    # Q-Q plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    stats.probplot(residuals1, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q Plot (Residuals)")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals1, bins=30, edgecolor='black', density=True)
    axes[1].axvline(0, color='r', linestyle='--', label='Mean')
    axes[1].set_title("Residual Distribution")
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(residuals1)
    print(f"\nShapiro-Wilk p-value: {shapiro_p:.4f}")
    if shapiro_p < 0.05:
        print("  → Residuals significantly deviate from normality")
    else:
        print("  → Residuals approximately normal ✓")

    # ============ AUTOCORRELATION CHECK ============
    print("\n" + "=" * 80)
    print("AUTOCORRELATION CHECK (within segments per trial)")
    print("=" * 80)

    lag1_autocorr = np.corrcoef(residuals1[:-1], residuals1[1:])[0, 1]
    if np.isnan(lag1_autocorr):
        print("Warning: Autocorrelation is NaN")
        lag1_autocorr = 0.0

    n_windows_per_trial = 9
    design_effect = 1 + (n_windows_per_trial - 1) * lag1_autocorr
    se_inflation = np.sqrt(design_effect)

    print(f"Lag-1 autocorrelation (ρ): {lag1_autocorr:.3f}")
    print(f"Design effect: {design_effect:.2f}")
    print(f"SE inflation factor: {se_inflation:.2f}×")

    # Adjust SEs for autocorrelation
    result1_adjusted_se = model1.bse * se_inflation
    result1_adjusted_z = model1.params / result1_adjusted_se
    result1_adjusted_p = 2 * (1 - stats.norm.cdf(np.abs(result1_adjusted_z)))

    # Convert to Series for proper indexing
    if not isinstance(result1_adjusted_p, pd.Series):
        result1_adjusted_p = pd.Series(result1_adjusted_p, index=model1.params.index)

    print("\n" + "-" * 80)
    print("ADJUSTED RESULTS (corrected for autocorrelation):")
    print("-" * 80)
    print(f"{'Parameter':<50s} {'β':>10s} {'SE (adj)':>12s} {'p (adj)':>10s}")
    print("-" * 80)
    for param in model1.params.index:
        adj_se = result1_adjusted_se[param]
        adj_p = result1_adjusted_p[param]
        print(f"{param:<50s} {model1.params[param]:>10.4f} {adj_se:>12.4f} {adj_p:>10.4f}")













    quit()
    ### LINEAR MIXED-EFFECTS MODEL
    print("\n"*3 + "=" * 80)
    print("LME Model")
    print("=" * 80)

    # LME model definition:
    model1 = smf.mixedlm(
        "PSD ~ C(condition, Treatment('silence')) + Q('Force Level')",
        data=all_subject_data_frame,
        groups=all_subject_data_frame['Subject ID']
    )
    result1 = model1.fit()
    print(result1.summary())

    # ============ DIAGNOSTICS: MODEL 1 ============
    print("\n" + "=" * 80)
    print("DIAGNOSTICS: Residual Normality (Model 1)")
    print("=" * 80)

    residuals1 = result1.resid

    # Q-Q plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    stats.probplot(residuals1, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q Plot (Residuals)")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals1, bins=30, edgecolor='black', density=True)
    axes[1].axvline(0, color='r', linestyle='--', label='Mean')
    axes[1].set_title("Residual Distribution")
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(residuals1)
    print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
    if shapiro_p < 0.05:
        print("  → Residuals significantly deviate from normality (consider transformation)")
    else:
        print("  → Residuals approximately normal ✓")

    # ============ AUTOCORRELATION CHECK ============
    print("\n" + "=" * 80)
    print("AUTOCORRELATION CHECK (within 5-sec segments per trial)")
    print("=" * 80)

    # Estimate lag-1 autocorrelation
    lag1_autocorr = np.corrcoef(residuals1[:-1], residuals1[1:])[0, 1]
    if np.isnan(lag1_autocorr):
        print("Warning: Autocorrelation is NaN (constant residuals?)")
        lag1_autocorr = 0.0
    n_windows_per_trial = 9  # 45 sec / 5 sec

    design_effect = 1 + (n_windows_per_trial - 1) * lag1_autocorr
    se_inflation = np.sqrt(design_effect)

    print(f"Lag-1 autocorrelation (ρ): {lag1_autocorr:.3f}")
    print(f"Design effect: {design_effect:.2f}")
    print(f"SE inflation factor: {se_inflation:.2f}×")
    print(f"→ True SEs are ~{se_inflation:.2f}× larger than reported")

    # Adjust SEs and p-values (ONLY for fixed effects, not random effects)
    result1_adjusted_se = result1.bse.loc[result1.fe_params.index] * se_inflation
    result1_adjusted_z = result1.fe_params / result1_adjusted_se
    result1_adjusted_p = 2 * (1 - stats.norm.cdf(np.abs(result1_adjusted_z)))

    # Ensure it's a Series
    if not isinstance(result1_adjusted_p, pd.Series):
        result1_adjusted_p = pd.Series(result1_adjusted_p, index=result1.fe_params.index)

    print("\n" + "-" * 80)
    print("ADJUSTED FIXED EFFECTS (corrected for autocorrelation):")
    print("-" * 80)
    print(f"{'Parameter':<50s} {'β':>10s} {'SE (orig)':>12s} {'SE (adj)':>12s} {'p (orig)':>10s} {'p (adj)':>10s}")
    print("-" * 80)
    for param in result1.fe_params.index:
        orig_se = result1.bse[param]
        adj_se = result1_adjusted_se[param]
        orig_p = result1.pvalues[param]
        adj_p = result1_adjusted_p[param]
        print(
            f"{param:<50s} {result1.fe_params[param]:>10.4f} {orig_se:>12.4f} {adj_se:>12.4f} {orig_p:>10.4f} {adj_p:>10.4f}")

    print("\n" + "=" * 80)