"""
Enhanced Alignment Validation Workflow: EMG-Force Synchronization Check

This script validates temporal alignment between precomputed EMG PSD and serial force measurements
by comparing peak timing across all tasks. Analyzes both flexor and extensor EMG to validate
which is which (flexor should show higher correlation with force and appropriate temporal lag).

©Paul Rüsing, INI ETH / UZH
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Tuple, Dict, List
from scipy import signal, stats
import matplotlib.pyplot as plt

import src.pipeline.signal_features as features
import src.pipeline.data_integration as data_integration
import src.utils.file_management as filemgmt


def load_emg_psd_both_muscles(
        subject_ind: int,
        feature_data_dir: Path,
        experiment_data_dir: Path,
        emg_frequency_band: Tuple[float, float] = (30, 250)
) -> Dict[str, Tuple[np.ndarray, pd.DatetimeIndex]]:
    """
    Load precomputed EMG PSD for both flexor and extensor muscles.

    Parameters
    ----------
    subject_ind : int
        Subject index
    feature_data_dir : Path
        Directory containing precomputed PSD files
    experiment_data_dir : Path
        Directory containing enriched serial frame
    emg_frequency_band : tuple
        Frequency range to extract from PSD (Hz)

    Returns
    -------
    dict
        Dictionary with keys 'flexor' and 'extensor', each containing:
        - emg_power: np.ndarray of shape (n_windows,)
        - psd_times: pd.DatetimeIndex
        - force_series: pd.Series (same for both)
    """
    print(f"\n{'=' * 100}")
    print(f"Loading EMG data (both muscles) for Subject {subject_ind:02d}")
    print(f"{'=' * 100}")

    subject_feature_dir = feature_data_dir / f"subject_{subject_ind:02}"
    subject_experiment_dir = experiment_data_dir / f"subject_{subject_ind:02}"
    
    # Load log frame to get QTC start time
    log_frame = data_integration.fetch_enriched_log_frame(
        subject_experiment_dir, set_time_index=True, verbose=False
    )
    qtc_start, qtc_end = data_integration.get_qtc_measurement_start_end(log_frame, verbose=False)
    
    # Load serial frame for force
    serial_frame = data_integration.fetch_enriched_serial_frame(
        subject_experiment_dir, set_time_index=True
    )
    
    # Make qtc_start/qtc_end timezone-aware to match serial_frame
    if serial_frame.index.tz is not None:
        if qtc_start.tz is None:
            qtc_start = qtc_start.tz_localize(serial_frame.index.tz)
        if qtc_end.tz is None:
            qtc_end = qtc_end.tz_localize(serial_frame.index.tz)
    
    # Find force column (try multiple possible names)
    force_col = None
    for possible_name in ['fsr', 'Force [N]', 'Force', 'force', 'Force Level [N]', 'Raw Force [N]']:
        if possible_name in serial_frame.columns:
            force_col = possible_name
            break
    
    if force_col is None:
        # If no direct match, look for columns containing 'force' or 'fsr'
        force_cols = [col for col in serial_frame.columns if 'force' in col.lower() or 'fsr' in col.lower()]
        if len(force_cols) > 0:
            force_col = force_cols[0]
            print(f"  Using force column: '{force_col}'")
        else:
            raise ValueError(f"No force column found in serial frame. Available columns: {serial_frame.columns.tolist()}")
    
    force_series = serial_frame[force_col]
    print(f"✓ Loaded force  '{force_col}' with {len(force_series)} samples")
    
    results = {'force_series': force_series}
    
    # Load both muscles
    for muscle_name, muscle_id in [('flexor', 'emg_1_flexor'), ('extensor', 'emg_2_extensor')]:
        try:
            emg_psd_path = filemgmt.most_recent_file(subject_feature_dir, ".npy", ["PSD Spectrograms", muscle_id])
            psd_times_path = filemgmt.most_recent_file(subject_feature_dir, ".npy", ["PSD Timecenters", muscle_id])
            psd_freqs_path = filemgmt.most_recent_file(subject_feature_dir, ".npy", ["PSD Frequencies", muscle_id])
            
            emg_psd = np.load(emg_psd_path)
            psd_times_sec = np.load(psd_times_path)
            psd_freqs = np.load(psd_freqs_path)
            
            # Convert to absolute timestamps
            psd_times_absolute = pd.to_datetime(
                qtc_start + pd.to_timedelta(psd_times_sec, unit='s')
            )
            
            # Extract power in frequency band
            freq_mask = (psd_freqs >= emg_frequency_band[0]) & (psd_freqs <= emg_frequency_band[1])
            emg_psd_band = emg_psd[:, freq_mask, :]
            emg_power = np.mean(emg_psd_band, axis=(1, 2))  # (n_windows,)
            
            results[muscle_name] = (emg_power, psd_times_absolute)
            print(f"✓ Loaded {muscle_name}: {emg_power.shape[0]} windows")
            
        except (FileNotFoundError, ValueError) as e:
            print(f"✗ Could not load {muscle_name}: {e}")
            results[muscle_name] = None
    
    return results


def analyze_task_alignment(
        emg_power: np.ndarray,
        psd_times: pd.DatetimeIndex,
        force_series: pd.Series,
        task_start: pd.Timestamp,
        task_end: pd.Timestamp,
        task_name: str,
        min_peak_distance: float = 2.0,
        max_matching_delay: float = 5.0
) -> Dict:
    """
    Analyze EMG-force alignment for a single task.
    
    Parameters
    ----------
    emg_power : np.ndarray
        EMG power time series
    psd_times : pd.DatetimeIndex
        PSD timestamps
    force_series : pd.Series
        Force measurements with DatetimeIndex
    task_start : pd.Timestamp
        Task start time
    task_end : pd.Timestamp
        Task end time
    task_name : str
        Name of the task
    min_peak_distance : float
        Minimum distance between peaks (seconds)
    max_matching_delay : float
        Maximum allowed delay for peak matching (seconds)
        
    Returns
    -------
    dict
        Alignment statistics for this task
    """
    # Slice data to task window
    psd_mask = (psd_times >= task_start) & (psd_times <= task_end)
    emg_power_task = emg_power[psd_mask]
    psd_times_task = psd_times[psd_mask]
    
    force_mask = (force_series.index >= task_start) & (force_series.index <= task_end)
    force_task = force_series[force_mask]
    
    if len(emg_power_task) < 10 or len(force_task) < 10:
        return {
            'task_name': task_name,
            'duration_sec': (task_end - task_start).total_seconds(),
            'n_emg_samples': len(emg_power_task),
            'n_force_samples': len(force_task),
            'error': 'Insufficient data'
        }
    
    # Calculate sampling rate properly (convert TimeDelta to seconds)
    time_diffs_sec = (psd_times_task[1:] - psd_times_task[:-1]).total_seconds()
    psd_sampling_rate = 1.0 / np.median(time_diffs_sec)
    
    # Detect peaks in EMG
    emg_peak_inds, _ = signal.find_peaks(
        emg_power_task,
        distance=int(min_peak_distance * psd_sampling_rate),
        prominence=np.percentile(emg_power_task, 80) - np.median(emg_power_task)
    )
    emg_peak_times = psd_times_task[emg_peak_inds]
    
    # Resample force to PSD times (convert timestamps to float seconds)
    psd_times_sec = (psd_times_task - psd_times_task[0]).total_seconds().values
    force_times_sec = (force_task.index - psd_times_task[0]).total_seconds().values
    
    force_at_psd_times = np.interp(
        psd_times_sec,
        force_times_sec,
        force_task.values
    )
    
    # Detect peaks in force
    force_peak_inds, _ = signal.find_peaks(
        force_at_psd_times,
        distance=int(min_peak_distance * psd_sampling_rate),
        prominence=np.percentile(force_at_psd_times, 75) - np.median(force_at_psd_times)
    )
    force_peak_times = psd_times_task[force_peak_inds]
    
    # Match peaks
    lags = []
    for emg_peak in emg_peak_times:
        delays = (force_peak_times - emg_peak).total_seconds().values
        within_window = np.abs(delays) <= max_matching_delay
        if within_window.any():
            closest_idx = np.argmin(np.abs(delays[within_window]))
            lags.append(delays[within_window][closest_idx])
    
    # Cross-correlation
    max_lag_samples = int(10.0 * psd_sampling_rate)
    
    # Normalize signals
    emg_norm = (emg_power_task - np.mean(emg_power_task)) / (np.std(emg_power_task) + 1e-10)
    force_norm = (force_at_psd_times - np.mean(force_at_psd_times)) / (np.std(force_at_psd_times) + 1e-10)
    
    correlation = signal.correlate(force_norm, emg_norm, mode='same')
    correlation = correlation / len(emg_power_task)
    
    # Extract relevant lag range
    center_idx = len(correlation) // 2
    if max_lag_samples < center_idx:
        start_idx = center_idx - max_lag_samples
        end_idx = center_idx + max_lag_samples + 1
        corr_slice = correlation[start_idx:end_idx]
        lags_sec = np.arange(-max_lag_samples, max_lag_samples + 1) / psd_sampling_rate
    else:
        corr_slice = correlation
        lags_sec = (np.arange(len(correlation)) - center_idx) / psd_sampling_rate
    
    optimal_idx = np.argmax(corr_slice)
    optimal_lag = lags_sec[optimal_idx]
    max_correlation = corr_slice[optimal_idx]
    
    return {
        'task_name': task_name,
        'duration_sec': (task_end - task_start).total_seconds(),
        'n_emg_peaks': len(emg_peak_times),
        'n_force_peaks': len(force_peak_times),
        'n_matched_peaks': len(lags),
        'match_rate': len(lags) / max(len(emg_peak_times), 1),
        'mean_lag_sec': np.mean(lags) if len(lags) > 0 else np.nan,
        'median_lag_sec': np.median(lags) if len(lags) > 0 else np.nan,
        'std_lag_sec': np.std(lags) if len(lags) > 0 else np.nan,
        'cross_corr_optimal_lag_sec': optimal_lag,
        'cross_corr_max': max_correlation,
        'error': None
    }


def validate_muscle_identity(
        subject_ind: int,
        feature_data_dir: Path,
        experiment_data_dir: Path,
        emg_freq_band: Tuple[float, float] = (30, 250),
        save_dir: Path = None
) -> Dict:
    """
    Validate muscle identity by comparing flexor vs extensor correlations with force.
    
    Expected results:
    - Flexor should have HIGHER correlation with grip force
    - Flexor should show temporal precedence or coincidence with force
    - Extensor may show weaker correlation or different timing pattern
    
    Parameters
    ----------
    subject_ind : int
        Subject index
    feature_data_dir : Path
        Feature data directory
    experiment_data_dir : Path
        Experiment data directory
    emg_freq_band : tuple
        EMG frequency band to analyze
    save_dir : Path
        Directory to save results
        
    Returns
    -------
    dict
        Validation results comparing flexor and extensor
    """
    print(f"\n{'#'*100}")
    print(f"# MUSCLE IDENTITY VALIDATION - SUBJECT {subject_ind:02d}")
    print(f"{'#'*100}\n")
    
    # Load both muscles
    data = load_emg_psd_both_muscles(subject_ind, feature_data_dir, experiment_data_dir, emg_freq_band)
    
    if data.get('flexor') is None or data.get('extensor') is None:
        return {'subject': subject_ind, 'error': 'Missing EMG data'}
    
    # Load task information
    subject_experiment_dir = experiment_data_dir / f"subject_{subject_ind:02}"
    log_frame = data_integration.fetch_enriched_log_frame(
        subject_experiment_dir, set_time_index=True, verbose=False
    )
    
    # Get all task timestamps (returns list of (start, end) tuples)
    task_periods = data_integration.get_all_task_start_ends(log_frame, output_type='list')
    
    print(f"\nFound {len(task_periods)} tasks to analyze...\n")
    
    if len(task_periods) == 0:
        print("ERROR: No tasks found in log frame!")
        return {'subject': subject_ind, 'error': 'No tasks found'}
    
    # Analyze each muscle
    results_by_muscle = {}
    
    for muscle_name in ['flexor', 'extensor']:
        emg_power, psd_times = data[muscle_name]
        force_series = data['force_series']
        
        print(f"\n{'='*100}")
        print(f"ANALYZING {muscle_name.upper()}")
        print(f"{'='*100}")
        
        task_results = []
        
        for task_idx, (task_start, task_end) in enumerate(task_periods):
            task_name = f"Task_{task_idx+1:02d}"
            
            result = analyze_task_alignment(
                emg_power, psd_times, force_series,
                task_start, task_end, task_name,
                min_peak_distance=2.0,
                max_matching_delay=5.0
            )
            
            task_results.append(result)
            
            if result.get('error') is None:
                print(f"{task_name}: {result['n_matched_peaks']}/{result['n_emg_peaks']} peaks matched, "
                      f"lag={result['mean_lag_sec']:.3f}±{result['std_lag_sec']:.3f}s, "
                      f"xcorr={result['cross_corr_max']:.3f} @ {result['cross_corr_optimal_lag_sec']:.3f}s")
        
        # Aggregate across tasks
        valid_results = [r for r in task_results if r.get('error') is None]
        
        if len(valid_results) > 0:
            # Calculate additional metrics: mean power and variability
            # Note: EMG power from PSD is in LOG SCALE (dB)
            # Convert to linear scale for meaningful power comparisons
            emg_power_linear = 10 ** emg_power  # Convert from log to linear
            
            mean_emg_power_linear = np.mean(emg_power_linear)
            std_emg_power_linear = np.std(emg_power_linear)
            max_emg_power_linear = np.max(emg_power_linear)
            
            # Also keep log-scale values for reference
            mean_emg_power_log = np.mean(emg_power)
            
            # Calculate coefficient of variation (normalized variability)
            cv_emg = std_emg_power_linear / (mean_emg_power_linear + 1e-20)
            
            aggregated = {
                'muscle': muscle_name,
                'n_tasks_analyzed': len(valid_results),
                'mean_match_rate': np.mean([r['match_rate'] for r in valid_results]),
                'mean_lag_sec': np.mean([r['mean_lag_sec'] for r in valid_results if not np.isnan(r['mean_lag_sec'])]),
                'std_lag_sec': np.std([r['mean_lag_sec'] for r in valid_results if not np.isnan(r['mean_lag_sec'])]),
                'mean_cross_corr': np.mean([r['cross_corr_max'] for r in valid_results]),
                'std_cross_corr': np.std([r['cross_corr_max'] for r in valid_results]),
                'mean_optimal_lag_sec': np.mean([r['cross_corr_optimal_lag_sec'] for r in valid_results]),
                'mean_emg_power_linear': mean_emg_power_linear,
                'std_emg_power_linear': std_emg_power_linear,
                'max_emg_power_linear': max_emg_power_linear,
                'mean_emg_power_log': mean_emg_power_log,
                'cv_emg': cv_emg,
                'task_results': task_results
            }
        else:
            aggregated = {'muscle': muscle_name, 'error': 'No valid tasks'}
        
        results_by_muscle[muscle_name] = aggregated
        
        # Print summary
        if 'error' not in aggregated:
            print(f"\n{'-'*100}")
            print(f"{muscle_name.upper()} SUMMARY:")
            print(f"  Tasks analyzed: {aggregated['n_tasks_analyzed']}")
            print(f"  Mean match rate: {aggregated['mean_match_rate']:.2f}")
            print(f"  Mean lag: {aggregated['mean_lag_sec']:.3f} ± {aggregated['std_lag_sec']:.3f} sec")
            print(f"  Mean cross-correlation: {aggregated['mean_cross_corr']:.3f} ± {aggregated['std_cross_corr']:.3f}")
            print(f"  Optimal lag: {aggregated['mean_optimal_lag_sec']:.3f} sec")
            print(f"\n  EMG Power Statistics (Linear Scale):")
            print(f"    Mean power: {aggregated['mean_emg_power_linear']:.3e}")
            print(f"    Max power:  {aggregated['max_emg_power_linear']:.3e}")
            print(f"    Std dev:    {aggregated['std_emg_power_linear']:.3e}")
            print(f"    CV (variability): {aggregated['cv_emg']:.3f}")
            print(f"  (Log scale mean: {aggregated['mean_emg_power_log']:.3f} dB)")
            print(f"{'-'*100}")
    
    # Compare muscles
    print(f"\n\n{'='*100}")
    print(f"MUSCLE COMPARISON")
    print(f"{'='*100}\n")
    
    flexor_res = results_by_muscle.get('flexor', {})
    extensor_res = results_by_muscle.get('extensor', {})
    
    if 'error' not in flexor_res and 'error' not in extensor_res:
        flexor_corr = flexor_res['mean_cross_corr']
        extensor_corr = extensor_res['mean_cross_corr']
        flexor_lag = flexor_res['mean_lag_sec']
        extensor_lag = extensor_res['mean_lag_sec']
        
        # Extract power metrics (LINEAR scale for proper comparison)
        flexor_mean_power = flexor_res['mean_emg_power_linear']
        extensor_mean_power = extensor_res['mean_emg_power_linear']
        flexor_max_power = flexor_res['max_emg_power_linear']
        extensor_max_power = extensor_res['max_emg_power_linear']
        flexor_cv = flexor_res['cv_emg']
        extensor_cv = extensor_res['cv_emg']
        
        # Also get log scale for display
        flexor_mean_power_log = flexor_res['mean_emg_power_log']
        extensor_mean_power_log = extensor_res['mean_emg_power_log']
        
        print(f"Cross-correlation with force:")
        print(f"  Flexor:   {flexor_corr:.3f} ± {flexor_res['std_cross_corr']:.3f}")
        print(f"  Extensor: {extensor_corr:.3f} ± {extensor_res['std_cross_corr']:.3f}")
        print(f"  Ratio:    {flexor_corr/extensor_corr if extensor_corr != 0 else np.inf:.2f}x")
        
        print(f"\nTemporal lag (EMG → Force):")
        print(f"  Flexor:   {flexor_lag:.3f} ± {flexor_res['std_lag_sec']:.3f} sec")
        print(f"  Extensor: {extensor_lag:.3f} ± {extensor_res['std_lag_sec']:.3f} sec")
        print(f"  Difference: {flexor_lag - extensor_lag:.3f} sec")
        
        print(f"\nEMG Power Levels (Mean, Linear Scale):")
        print(f"  Flexor:   {flexor_mean_power:.3e}  [{flexor_mean_power_log:.2f} dB]")
        print(f"  Extensor: {extensor_mean_power:.3e}  [{extensor_mean_power_log:.2f} dB]")
        power_ratio_mean = flexor_mean_power / extensor_mean_power if extensor_mean_power > 0 else np.inf
        print(f"  Ratio:    {power_ratio_mean:.2f}x")
        
        print(f"\nEMG Power Levels (Max, Linear Scale):")
        print(f"  Flexor:   {flexor_max_power:.3e}")
        print(f"  Extensor: {extensor_max_power:.3e}")
        power_ratio_max = flexor_max_power / extensor_max_power if extensor_max_power > 0 else np.inf
        print(f"  Ratio:    {power_ratio_max:.2f}x")
        
        print(f"\nEMG Variability (Coefficient of Variation):")
        print(f"  Flexor:   {flexor_cv:.3f}")
        print(f"  Extensor: {extensor_cv:.3f}")
        print(f"  Ratio:    {flexor_cv/extensor_cv if extensor_cv > 0 else np.inf:.2f}x")
        
        print(f"\n{'─'*100}")
        print(f"INTERPRETATION:")
        
        # Validation checks
        checks = []
        
        # Check 1: Correlation difference
        if flexor_corr > extensor_corr * 1.2:  # Flexor should be >20% higher
            checks.append("✓ Flexor shows HIGHER correlation with force (as expected)")
            muscle_identity_correct = True
        elif extensor_corr > flexor_corr * 1.2:
            checks.append("✗ WARNING: Extensor shows HIGHER correlation - possible mislabeling!")
            muscle_identity_correct = False
        else:
            checks.append("⚠ Flexor and extensor show SIMILAR correlations")
            muscle_identity_correct = None
            
            # Additional diagnostics for similar correlation case
            if flexor_mean_power > extensor_mean_power * 1.5:
                checks.append("  → BUT flexor has HIGHER power - likely CO-CONTRACTION during grip")
                muscle_identity_correct = True
            elif extensor_mean_power > flexor_mean_power * 1.5:
                checks.append("  → AND extensor has HIGHER power - CHECK ELECTRODE PLACEMENT!")
                muscle_identity_correct = False
            else:
                checks.append("  → AND similar power levels - likely CO-CONTRACTION or CROSS-TALK")
        
        # Check 2: Temporal lag
        if abs(flexor_lag) < 0.5:  # Should be within 500ms
            checks.append(f"✓ Flexor lag is appropriate ({flexor_lag:.3f}s)")
        else:
            checks.append(f"⚠ Flexor lag is large ({flexor_lag:.3f}s) - check alignment")
        
        # Check 3: Temporal difference
        if abs(flexor_lag - extensor_lag) > 0.2:  # Should differ by >200ms
            checks.append(f"✓ Muscles show different temporal patterns (Δ={abs(flexor_lag - extensor_lag):.3f}s)")
        else:
            checks.append(f"⚠ Muscles show similar temporal patterns (Δ={abs(flexor_lag - extensor_lag):.3f}s)")
            checks.append(f"  → This suggests CO-CONTRACTION (both active together)")
        
        # Check 4: Power ratio (use mean power)
        power_ratio = power_ratio_mean
        if power_ratio > 1.5:
            checks.append(f"✓ Flexor has higher power ({power_ratio:.2f}x) - consistent with grip task")
            if muscle_identity_correct is None:  # Override if power confirms
                muscle_identity_correct = True
        elif power_ratio < 0.67:
            checks.append(f"⚠ Extensor has higher power ({1/power_ratio:.2f}x) - unusual for grip!")
            if muscle_identity_correct is None:  # Override if power contradicts
                muscle_identity_correct = False
        else:
            checks.append(f"⚠ Similar power levels ({power_ratio:.2f}x) - CO-CONTRACTION likely")
        
        for check in checks:
            print(f"  {check}")
        
        print(f"{'─'*100}\n")
        
        comparison = {
            'flexor_correlation': flexor_corr,
            'extensor_correlation': extensor_corr,
            'correlation_ratio': flexor_corr / extensor_corr if extensor_corr != 0 else np.inf,
            'flexor_lag': flexor_lag,
            'extensor_lag': extensor_lag,
            'lag_difference': flexor_lag - extensor_lag,
            'flexor_mean_power': flexor_mean_power,
            'extensor_mean_power': extensor_mean_power,
            'power_ratio': power_ratio,
            'flexor_max_power': flexor_max_power,
            'extensor_max_power': extensor_max_power,
            'flexor_cv': flexor_cv,
            'extensor_cv': extensor_cv,
            'muscle_identity_correct': muscle_identity_correct,
            'validation_checks': checks
        }
    else:
        comparison = {'error': 'Missing data for comparison'}
    
    # Compile final results
    final_results = {
        'subject': subject_ind,
        'flexor': results_by_muscle.get('flexor'),
        'extensor': results_by_muscle.get('extensor'),
        'comparison': comparison
    }
    
    # Save detailed results
    if save_dir is not None:
        filemgmt.assert_dir(save_dir)
        
        # Save task-level results for each muscle
        for muscle_name in ['flexor', 'extensor']:
            if muscle_name in results_by_muscle and 'task_results' in results_by_muscle[muscle_name]:
                task_df = pd.DataFrame(results_by_muscle[muscle_name]['task_results'])
                task_csv_path = save_dir / f"task_alignment_{muscle_name}_subject_{subject_ind:02d}.csv"
                task_df.to_csv(task_csv_path, index=False)
                print(f"✓ Saved {muscle_name} task results to: {task_csv_path}")
        
        # Save comparison summary
        if 'error' not in comparison:
            summary_df = pd.DataFrame([{
                'subject': subject_ind,
                **comparison
            }])
            summary_path = save_dir / f"muscle_comparison_subject_{subject_ind:02d}.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"✓ Saved muscle comparison to: {summary_path}")
    
    return final_results


######## MAIN WORKFLOW ########
def create_multi_task_comparison_plot(
        flexor_power: np.ndarray,
        extensor_power: np.ndarray,
        psd_times: pd.DatetimeIndex,
        force_series: pd.Series,
        task_periods: List[Tuple[pd.Timestamp, pd.Timestamp]],
        subject_ind: int,
        n_tasks_to_display: int = 5,
        save_dir: Path = None
):
    """
    Create comparison plot showing multiple tasks concatenated (skipping between-task gaps).
    
    Parameters
    ----------
    flexor_power : np.ndarray
        Flexor EMG power time series
    extensor_power : np.ndarray
        Extensor EMG power time series
    psd_times : pd.DatetimeIndex
        PSD timestamps
    force_series : pd.Series
        Force measurements
    task_periods : List[Tuple[pd.Timestamp, pd.Timestamp]]
        List of (start, end) tuples for each task
    subject_ind : int
        Subject index
    n_tasks_to_display : int
        Number of tasks to display (default: 5)
    save_dir : Path
        Directory to save plot
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    
    # Select tasks to display (evenly spaced throughout recording)
    n_tasks_available = len(task_periods)
    if n_tasks_to_display > n_tasks_available:
        n_tasks_to_display = n_tasks_available
    
    # Select evenly spaced tasks
    task_indices = np.linspace(0, n_tasks_available - 1, n_tasks_to_display, dtype=int)
    selected_tasks = [task_periods[i] for i in task_indices]
    
    # Concatenate data from selected tasks
    flexor_concat = []
    extensor_concat = []
    force_concat = []
    ratio_concat = []
    x_concat = []
    task_boundaries = [0.0]  # Track where each task ends on x-axis
    
    cumulative_time = 0.0
    
    for task_idx, (task_start, task_end) in enumerate(selected_tasks):
        # Extract data for this task
        psd_mask = (psd_times >= task_start) & (psd_times <= task_end)
        force_mask = (force_series.index >= task_start) & (force_series.index <= task_end)
        
        flexor_task = flexor_power[psd_mask]
        extensor_task = extensor_power[psd_mask]
        
        # Interpolate force to PSD times for this task
        psd_times_task = psd_times[psd_mask]
        force_task = force_series[force_mask]
        
        if len(psd_times_task) > 0 and len(force_task) > 0:
            psd_times_sec = (psd_times_task - psd_times_task[0]).total_seconds().values
            force_times_sec = (force_task.index - psd_times_task[0]).total_seconds().values
            
            force_at_psd = np.interp(psd_times_sec, force_times_sec, force_task.values)
            
            # Compute ratio
            ratio_task = flexor_task / (extensor_task + 1e-10)
            
            # Create x-axis values (relative to concatenated timeline)
            task_duration = (task_end - task_start).total_seconds()
            x_task = np.linspace(cumulative_time, cumulative_time + task_duration, len(flexor_task))
            
            # Append to concatenated arrays
            flexor_concat.append(flexor_task)
            extensor_concat.append(extensor_task)
            force_concat.append(force_at_psd)
            ratio_concat.append(ratio_task)
            x_concat.append(x_task)
            
            cumulative_time += task_duration
            task_boundaries.append(cumulative_time)
    
    # Concatenate all arrays
    flexor_all = np.concatenate(flexor_concat)
    extensor_all = np.concatenate(extensor_concat)
    force_all = np.concatenate(force_concat)
    ratio_all = np.concatenate(ratio_concat)
    x_all = np.concatenate(x_concat)
    
    # Plot 1: Flexor EMG
    axes[0].plot(x_all, flexor_all, 'b-', linewidth=1, label='Flexor EMG')
    axes[0].set_ylabel('Flexor Power\n(log scale)', fontsize=11)
    axes[0].set_title(f'Subject {subject_ind:02d}: {n_tasks_to_display} Tasks Concatenated (Total: {cumulative_time:.1f}s)', 
                     fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Add task boundaries
    for boundary in task_boundaries[1:-1]:  # Skip first and last
        axes[0].axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot 2: Extensor EMG
    axes[1].plot(x_all, extensor_all, 'r-', linewidth=1, label='Extensor EMG')
    axes[1].set_ylabel('Extensor Power\n(log scale)', fontsize=11)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    for boundary in task_boundaries[1:-1]:
        axes[1].axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot 3: Force
    axes[2].plot(x_all, force_all, 'g-', linewidth=1, label='Grip Force')
    axes[2].set_ylabel('Force\n(Task-wise scaled, 0-1)', fontsize=11)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    for boundary in task_boundaries[1:-1]:
        axes[2].axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot 4: Ratio (Flexor/Extensor)
    axes[3].plot(x_all, ratio_all, 'purple', linewidth=1, label='Flexor/Extensor Ratio')
    axes[3].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Equal power')
    axes[3].set_ylabel('Power Ratio\n(Flexor/Extensor)', fontsize=11)
    axes[3].set_xlabel('Concatenated Time (seconds)', fontsize=11)
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    
    for boundary in task_boundaries[1:-1]:
        axes[3].axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add task labels at bottom
    for task_idx, (start_time, end_time) in enumerate(zip(task_boundaries[:-1], task_boundaries[1:])):
        mid_time = (start_time + end_time) / 2
        axes[3].text(mid_time, axes[3].get_ylim()[0] * 1.05, f'Task {task_indices[task_idx]+1}',
                    ha='center', va='top', fontsize=9, color='gray')
    
    plt.tight_layout()
    
    if save_dir is not None:
        filemgmt.assert_dir(save_dir)
        plot_path = save_dir / f"multi_task_comparison_subject_{subject_ind:02d}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved multi-task comparison plot to: {plot_path}")
    
    plt.close()


def create_power_comparison_plot(
        flexor_power: np.ndarray,
        extensor_power: np.ndarray,
        psd_times: pd.DatetimeIndex,
        force_series: pd.Series,
        subject_ind: int,
        save_dir: Path = None
):
    """
    Create a comparison plot showing flexor vs extensor power alongside force.
    
    Parameters
    ----------
    flexor_power : np.ndarray
        Flexor EMG power time series
    extensor_power : np.ndarray
        Extensor EMG power time series
    psd_times : pd.DatetimeIndex
        PSD timestamps
    force_series : pd.Series
        Force measurements
    subject_ind : int
        Subject index
    save_dir : Path
        Directory to save plot
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    
    # Use middle 5 minutes for visualization
    middle_time = psd_times[len(psd_times) // 2]
    window_start = middle_time - pd.Timedelta(minutes=2.5)
    window_end = middle_time + pd.Timedelta(minutes=2.5)
    
    psd_mask = (psd_times >= window_start) & (psd_times <= window_end)
    force_mask = (force_series.index >= window_start) & (force_series.index <= window_end)
    
    psd_times_rel = (psd_times[psd_mask] - window_start).total_seconds()
    force_times_rel = (force_series.index[force_mask] - window_start).total_seconds()
    
    # Determine common x-axis limits (use full 5-minute window)
    x_min = 0.0
    x_max = 300.0  # 5 minutes in seconds
    
    # Plot 1: Flexor EMG
    axes[0].plot(psd_times_rel, flexor_power[psd_mask], 'b-', linewidth=1, label='Flexor EMG')
    axes[0].set_ylabel('Flexor Power\n(log scale)', fontsize=11)
    axes[0].set_title(f'Subject {subject_ind:02d}: Flexor vs Extensor EMG Comparison (5-min window)', 
                     fontsize=13, fontweight='bold')
    axes[0].set_xlim(x_min, x_max)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Extensor EMG
    axes[1].plot(psd_times_rel, extensor_power[psd_mask], 'r-', linewidth=1, label='Extensor EMG')
    axes[1].set_ylabel('Extensor Power\n(log scale)', fontsize=11)
    axes[1].set_xlim(x_min, x_max)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Force
    axes[2].plot(force_times_rel, force_series[force_mask], 'g-', linewidth=1, label='Grip Force')
    axes[2].set_ylabel('Force\n(Task-wise scaled, 0-1)', fontsize=11)
    axes[2].set_xlim(x_min, x_max)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Ratio (Flexor/Extensor)
    ratio = flexor_power[psd_mask] / (extensor_power[psd_mask] + 1e-10)
    axes[3].plot(psd_times_rel, ratio, 'purple', linewidth=1, label='Flexor/Extensor Ratio')
    axes[3].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Equal power')
    axes[3].set_ylabel('Power Ratio\n(Flexor/Extensor)', fontsize=11)
    axes[3].set_xlabel('Time (seconds)', fontsize=11)
    axes[3].set_xlim(x_min, x_max)
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir is not None:
        filemgmt.assert_dir(save_dir)
        plot_path = save_dir / f"muscle_comparison_plot_subject_{subject_ind:02d}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison plot to: {plot_path}")
    
    plt.close()


if __name__ == "__main__":
    ROOT = Path().resolve().parent
    DATA = ROOT / "data"
    OUTPUT = ROOT / 'output'
    FEATURE_DATA = DATA / "precomputed_features"
    EXPERIMENT_DATA = DATA / "experiment_results"
    ALIGNMENT_REPORTS = OUTPUT / "alignment_validation_enhanced"

    filemgmt.assert_dir(ALIGNMENT_REPORTS)

    # Workflow control
    subjects_to_check = [0, 1, 2, 3, 4, 5, 6, 7]
    save_reports = True
    
    # NOTE: Subject 0 and 1 may have different labeling
    # From subject 2 onwards, labeling is confirmed correct

    # Analysis parameters
    emg_freq_band = (30, 250)  # Hz, typical EMG band

    # Store results across subjects
    all_muscle_comparisons = []

    for subject_ind in subjects_to_check:
        try:
            results = validate_muscle_identity(
                subject_ind=subject_ind,
                feature_data_dir=FEATURE_DATA,
                experiment_data_dir=EXPERIMENT_DATA,
                emg_freq_band=emg_freq_band,
                save_dir=ALIGNMENT_REPORTS if save_reports else None
            )
            
            # Create visualizations comparing muscles
            if 'flexor' in results and 'extensor' in results and results['flexor'] is not None and results['extensor'] is not None:
                if 'error' not in results['flexor'] and 'error' not in results['extensor']:
                    # Load data again for plotting
                    data = load_emg_psd_both_muscles(subject_ind, FEATURE_DATA, EXPERIMENT_DATA, emg_freq_band)
                    if data.get('flexor') is not None and data.get('extensor') is not None:
                        flexor_power, flexor_times = data['flexor']
                        extensor_power, extensor_times = data['extensor']
                        
                        # Get task periods for multi-task plot
                        subject_experiment_dir = EXPERIMENT_DATA / f"subject_{subject_ind:02}"
                        log_frame = data_integration.fetch_enriched_log_frame(
                            subject_experiment_dir, set_time_index=True, verbose=False
                        )
                        task_periods = data_integration.get_all_task_start_ends(log_frame, output_type='list')
                        
                        # Create single continuous window plot (5 minutes)
                        create_power_comparison_plot(
                            flexor_power, extensor_power, flexor_times,
                            data['force_series'], subject_ind,
                            save_dir=ALIGNMENT_REPORTS if save_reports else None
                        )
                        
                        # Create multi-task concatenated plot
                        create_multi_task_comparison_plot(
                            flexor_power, extensor_power, flexor_times,
                            data['force_series'], task_periods, subject_ind,
                            n_tasks_to_display=5,
                            save_dir=ALIGNMENT_REPORTS if save_reports else None
                        )
            
            if 'comparison' in results and 'error' not in results['comparison']:
                all_muscle_comparisons.append({
                    'subject': subject_ind,
                    **results['comparison']
                })

        except Exception as e:
            print(f"\n✗ ERROR processing subject {subject_ind:02d}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary across all subjects
    if len(all_muscle_comparisons) > 0:
        print(f"\n\n{'='*100}")
        print(f"SUMMARY ACROSS ALL SUBJECTS")
        print(f"{'='*100}\n")

        summary_df = pd.DataFrame(all_muscle_comparisons)
        
        print("\nMuscle Correlation with Force:")
        print(f"  Flexor (mean):   {summary_df['flexor_correlation'].mean():.3f} ± {summary_df['flexor_correlation'].std():.3f}")
        print(f"  Extensor (mean): {summary_df['extensor_correlation'].mean():.3f} ± {summary_df['extensor_correlation'].std():.3f}")
        print(f"  Ratio (mean):    {summary_df['correlation_ratio'].mean():.2f}x")
        
        print("\nEMG Power Levels (Mean, Linear Scale):")
        print(f"  Flexor (mean):   {summary_df['flexor_mean_power'].mean():.3e} ± {summary_df['flexor_mean_power'].std():.3e}")
        print(f"  Extensor (mean): {summary_df['extensor_mean_power'].mean():.3e} ± {summary_df['extensor_mean_power'].std():.3e}")
        # Calculate ratio properly: mean of individual ratios, excluding inf/nan
        valid_ratios = summary_df['power_ratio'][np.isfinite(summary_df['power_ratio'])]
        mean_power_ratio = valid_ratios.mean() if len(valid_ratios) > 0 else np.nan
        print(f"  Ratio (mean):    {mean_power_ratio:.2f}x")
        
        print("\nTemporal Lags:")
        print(f"  Flexor (mean):   {summary_df['flexor_lag'].mean():.3f} ± {summary_df['flexor_lag'].std():.3f} sec")
        print(f"  Extensor (mean): {summary_df['extensor_lag'].mean():.3f} ± {summary_df['extensor_lag'].std():.3f} sec")
        
        print("\nINTERPRETATION:")
        corr_ratio = summary_df['correlation_ratio'].mean()
        # Use properly calculated power ratio
        power_ratio_mean = mean_power_ratio
        
        if corr_ratio > 1.2:
            print(f"  ✓ Flexor clearly dominant in correlation ({corr_ratio:.2f}x)")
        elif corr_ratio < 0.83:
            print(f"  ⚠ Extensor shows higher correlation - CHECK LABELING!")
        else:
            print(f"  ⚠ Similar correlations ({corr_ratio:.2f}x) suggests:")
            if not np.isnan(power_ratio_mean):
                if power_ratio_mean > 1.3:
                    print(f"     → CO-CONTRACTION (both active, but flexor stronger: {power_ratio_mean:.2f}x power)")
                    print(f"     → This is NORMAL for precision grip tasks!")
                elif power_ratio_mean < 0.77:
                    print(f"     → Possible ELECTRODE ISSUE (extensor stronger: {1/power_ratio_mean:.2f}x power)")
                    print(f"     → Check electrode placement - unusual for flexor grip task!")
                else:
                    print(f"     → Either CO-CONTRACTION or CROSS-TALK (similar power: {power_ratio_mean:.2f}x)")
                    print(f"     → Check electrode impedance and placement")
            else:
                print(f"     → Unable to compute power ratio (check data)")
        
        print("\nMuscle Identity Validation:")
        if 'muscle_identity_correct' in summary_df.columns:
            correct_count = int(summary_df['muscle_identity_correct'].sum())
            incorrect_count = int((summary_df['muscle_identity_correct'] == False).sum())
            total_decisive = correct_count + incorrect_count
            unclear_count = int(summary_df['muscle_identity_correct'].isna().sum())
            print(f"  Clearly correct labeling: {correct_count}/{len(summary_df)}")
            if incorrect_count > 0:
                print(f"  Clearly INCORRECT labeling: {incorrect_count}/{len(summary_df)} ⚠️")
            if unclear_count > 0:
                print(f"  Unclear cases (likely co-contraction): {unclear_count}/{len(summary_df)}")
        
        if save_reports:
            summary_path = ALIGNMENT_REPORTS / "muscle_comparison_all_subjects.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"\n✓ Saved summary to: {summary_path}")

    print(f"\n{'='*100}")
    print("ENHANCED VALIDATION WORKFLOW COMPLETE")
    print(f"{'='*100}\n")
