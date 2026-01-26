import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Literal

############################## DATA MANIPULATION / INTEGRATION METHODS ##############################
def _convert_to_seconds(
    timestamps: np.ndarray | pd.DatetimeIndex,
    as_absolute: bool = True,
) -> np.ndarray:
    """
    Convert timestamps to seconds.

    Parameters
    ----------
    timestamps : np.ndarray | pd.DatetimeIndex
        Time values. If numeric, treated as seconds. If DatetimeIndex/datetime-like, converted to seconds.
    as_absolute : bool
        If True, returns absolute seconds from Unix epoch.
        If False, returns relative seconds from first timestamp (baseline = 0).

    Returns
    -------
    np.ndarray
        Time values in seconds (float dtype).

    Raises
    ------
    TypeError
        If timestamps cannot be interpreted as numeric or datetime.
    """
    if isinstance(timestamps, pd.DatetimeIndex):
        # Convert DatetimeIndex to Unix epoch seconds
        time_seconds = timestamps.astype(np.int64) // 10**9
    elif isinstance(timestamps, (list, tuple, np.ndarray)):
        arr = np.asarray(timestamps)
        # Check if it's numeric
        if np.issubdtype(arr.dtype, np.number):
            time_seconds = arr.astype(float)
        else:
            # Try to convert to DatetimeIndex
            dt_idx = pd.to_datetime(arr)
            time_seconds = dt_idx.astype(np.int64) // 10**9
    else:
        # Fallback: try pd.to_datetime
        dt_idx = pd.to_datetime(timestamps)
        time_seconds = dt_idx.astype(np.int64) // 10**9

    time_seconds = np.asarray(time_seconds, dtype=float)

    # Convert to relative if requested
    if not as_absolute:
        time_seconds = time_seconds - time_seconds[0]

    return time_seconds


def apply_window_operator(
        window_timestamps: np.ndarray,
        target_array: np.ndarray | pd.Series,
        target_timestamps: np.ndarray | pd.DatetimeIndex | None = None,
        window_size: float | None = None,
        is_time_center: bool | None = None,
        operation: Literal['min', 'max', 'mean', 'median', 'mode', 'std'] = 'mean',
        axis: int = 0,
        window_timestamps_ends: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply windowing operator to array along specified axis.

    Aggregates target_array into fixed or variable-sized time windows,
    applying a reduction operation along the specified axis.

    Parameters
    ----------
    window_timestamps : np.ndarray
        Time points defining window centers or starts.
        Can be numeric (seconds) or datetime-like (DatetimeIndex, timestamps, etc.).
    target_array : np.ndarray | pd.Series
        Data to aggregate. Can be 1D or multi-dimensional.
        If pd.Series with DatetimeIndex and target_timestamps is None,
        the Series index is used as target_timestamps.
    target_timestamps : np.ndarray | pd.DatetimeIndex | None
        Time index for target_array. If None, uses integer indices (0, 1, 2, ...).
        Can be numeric (seconds) or datetime-like.
    window_size : float | None
        Size of each window (in seconds).
        Required if window_timestamps_ends is None.
        Ignored if window_timestamps_ends is provided.
    is_time_center : bool | None
        If True, window_timestamps mark window centers.
        If False, window_timestamps mark window starts.
        Required when window_size is provided.
        Ignored if window_timestamps_ends is provided.
    operation : Literal['min', 'max', 'mean', 'median', 'mode', 'std']
        Aggregation operation to apply within each window.
    axis : int
        Axis along which to aggregate (0 = rows, 1 = columns, etc.).
    window_timestamps_ends : np.ndarray | None
        Explicit window end points. If provided, window_size and is_time_center are ignored.
        Must have same length as window_timestamps.
        Can be numeric (seconds) or datetime-like.
        Allows variable-sized windows.

    Returns
    -------
    np.ndarray
        Aggregated array with shape matching input except along specified axis,
        which becomes len(window_timestamps).

    Raises
    ------
    ValueError
        If neither window_size nor window_timestamps_ends is provided.
        If window_size is provided but is_time_center is None.
        If window_timestamps_ends length doesn't match window_timestamps.
        If time array length doesn't match target_array shape.
        If axis is out of bounds.
    """

    # ===== INPUT VALIDATION & EXTRACTION =====

    # Validate that either window_size or window_timestamps_ends is provided
    if window_size is None and window_timestamps_ends is None:
        raise ValueError(
            "Either 'window_size' or 'window_timestamps_ends' must be provided. "
            "Use window_size (with is_time_center) for fixed-size windows, "
            "or window_timestamps_ends for variable-size windows."
        )

    # Validate that is_time_center is provided when window_size is used
    if window_size is not None and is_time_center is None:
        raise ValueError(
            "When 'window_size' is provided, 'is_time_center' must also be specified. "
            "Set is_time_center=True if window_timestamps are window centers, "
            "or is_time_center=False if they are window starts."
        )

    # Convert pd.Series to ndarray; extract DatetimeIndex if present
    if isinstance(target_array, pd.Series):
        if target_timestamps is None and isinstance(target_array.index, pd.DatetimeIndex):
            target_timestamps = target_array.index
        target_array = target_array.values

    # Validate array dimensions
    if target_array.ndim == 0:
        raise ValueError("target_array must be at least 1D")

    if axis < 0 or axis >= target_array.ndim:
        raise ValueError(f"axis={axis} out of bounds for array with {target_array.ndim} dimensions")

    # Validate window_timestamps_ends if provided
    if window_timestamps_ends is not None:
        window_timestamps_ends = np.asarray(window_timestamps_ends)
        if len(window_timestamps_ends) != len(window_timestamps):
            raise ValueError(
                f"window_timestamps_ends length ({len(window_timestamps_ends)}) "
                f"must match window_timestamps length ({len(window_timestamps)})"
            )

    # ===== CONVERT ALL TIMESTAMPS TO ABSOLUTE SECONDS =====

    # Target timestamps: use absolute seconds (Unix epoch) for all conversions
    if target_timestamps is None:
        target_time_seconds = np.arange(target_array.shape[axis], dtype=float)
    else:
        target_time_seconds = _convert_to_seconds(target_timestamps, as_absolute=True)

    # Validate time array length
    if len(target_time_seconds) != target_array.shape[axis]:
        raise ValueError(
            f"Length of time axis ({len(target_time_seconds)}) does not match "
            f"target_array.shape[{axis}] ({target_array.shape[axis]})"
        )

    # Window timestamps: convert to absolute seconds
    window_time_seconds = _convert_to_seconds(window_timestamps, as_absolute=True)

    # Window end times: convert to absolute seconds (if provided)
    window_time_seconds_ends = None
    if window_timestamps_ends is not None:
        window_time_seconds_ends = _convert_to_seconds(window_timestamps_ends, as_absolute=True)

    # ===== CREATE WINDOW BOUNDARIES =====

    if window_time_seconds_ends is not None:
        # Variable window sizes: use provided endpoints
        starts = window_time_seconds
        ends = window_time_seconds_ends
    else:
        # Fixed window size: compute from is_time_center and window_size
        # (is_time_center is guaranteed to be non-None here by earlier validation)
        if is_time_center:
            starts = window_time_seconds - window_size / 2
            ends = window_time_seconds + window_size / 2
        else:
            starts = window_time_seconds
            ends = window_time_seconds + window_size

    # ===== ASSIGN EACH TIME POINT TO A WINDOW =====

    window_indices = np.full(len(target_time_seconds), np.nan, dtype=float)
    for i, (start, end) in enumerate(zip(starts, ends)):
        # Include boundaries: [start, end]
        mask = (target_time_seconds >= start) & (target_time_seconds <= end)
        window_indices[mask] = i

    # ===== RESHAPE FOR AGGREGATION =====

    # Move aggregation axis to position 0 for easier iteration
    target_array_moved = np.moveaxis(target_array, axis, 0)
    moved_shape = target_array_moved.shape

    # Reshape to (n_time, -1) for easier vectorized operations
    n_time = moved_shape[0]
    target_array_flat = target_array_moved.reshape(n_time, -1)

    # ===== AGGREGATE WITHIN WINDOWS =====

    n_windows = len(window_time_seconds)
    n_features = target_array_flat.shape[1]
    result = np.full((n_windows, n_features), np.nan, dtype=float)

    for window_idx in range(n_windows):
        mask = (window_indices == window_idx)

        if not np.any(mask):
            # No data in this window; leave as np.nan
            continue

        # Extract data for this window: shape (n_samples_in_window, n_features)
        window_data = target_array_flat[mask, :]

        # Apply aggregation operation
        if operation == 'mean':
            result[window_idx, :] = np.nanmean(window_data, axis=0)
        elif operation == 'median':
            result[window_idx, :] = np.nanmedian(window_data, axis=0)
        elif operation == 'min':
            result[window_idx, :] = np.nanmin(window_data, axis=0)
        elif operation == 'max':
            result[window_idx, :] = np.nanmax(window_data, axis=0)
        elif operation == 'std':
            result[window_idx, :] = np.nanstd(window_data, axis=0)
        elif operation == 'mode':
            # Mode requires element-wise computation (slower)
            for feat_idx in range(n_features):
                mode_result = pd.Series(window_data[:, feat_idx]).mode()
                result[window_idx, feat_idx] = mode_result[0] if len(mode_result) > 0 else np.nan
        else:
            raise ValueError(f"Unknown operation: {operation}")

    # ===== RESHAPE BACK TO ORIGINAL DIMENSIONS =====

    # Restore shape: (n_windows, feat1, feat2, ...)
    output_shape = list(moved_shape)
    output_shape[0] = n_windows
    result = result.reshape(output_shape)

    # Move aggregation axis back to original position
    result = np.moveaxis(result, 0, axis)

    return result


def old_only_pandas_target_apply_window_operator(
        window_time_steps: np.ndarray,
        is_time_center: bool,
        window_size: float,
        target_series: pd.Series,
        operation: Literal['min', 'max', 'mean', 'median', 'mode', 'std'] = 'mean',
) -> list:
    """ Target series needs to have time index (seconds or absolute). """

    # convert data to numeric only for non-mode operations:
    if operation != 'mode' and target_series.dtype == 'object':
        target_series = pd.to_numeric(target_series, errors='coerce')

    # derive time in seconds from time index:
    if isinstance(target_series.index, pd.DatetimeIndex):
        time_seconds = (target_series.index - target_series.index[0]).total_seconds().values
    else:
        time_seconds = target_series.index.values.astype(float)

    # create window boundaries:
    if is_time_center:
        starts = window_time_steps - window_size / 2
        ends = window_time_steps + window_size / 2
    else:
        starts = window_time_steps
        ends = window_time_steps + window_size

    # For each time point, find which window it belongs to
    window_indices = np.full(len(time_seconds), np.nan, dtype=float)

    for i, (start, end) in enumerate(zip(starts, ends)):
        mask = (time_seconds >= start) & (time_seconds < end)
        window_indices[mask] = i

    # Create dataframe with both the data and window indices
    df_with_windows = pd.DataFrame({
        'data': target_series.values,
        '_window': window_indices
    })

    # Filter out NaN groups BEFORE groupby
    df_with_windows_filtered = df_with_windows[df_with_windows['_window'].notna()]
    grouped = df_with_windows_filtered.groupby('_window', sort=False)['data']

    # Handle mode separately, use agg for others
    if operation == 'mode':
        result = grouped.apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan)
    else:
        result = grouped.agg(operation)

    all_windows = pd.RangeIndex(len(window_time_steps), name='_window')
    result = result.reindex(all_windows)  # ensures all windows present
    result = result.fillna(0)  # or ffill(), but now consistent
    return result.tolist()


def interpolate_per_window(
        window_time_steps: np.ndarray,
        target_series: pd.Series,
        method: Literal['linear', 'nearest', 'cubic', 'spline'] = 'linear',
        window_size: float = None,
        is_time_center: bool = False,
        extrapolate: bool = False,
        return_type: Literal['pandas', 'numpy'] = 'numpy'
) -> np.ndarray | pd.Series:
    """
    Interpolate target series values to match window_time_steps using temporal windows.

    Parameters
    ----------
    window_time_steps : np.ndarray
        Target timestamps (in seconds) where interpolation occurs.
    target_series : pd.Series
        Series with time index (sparse sampling) to interpolate from.
    method : Literal['linear', 'nearest', 'cubic', 'spline'], default 'linear'
        Interpolation method.
    window_size : float, optional
        Restricts interpolation to local window around each target.
        If None, uses full-range interpolation.
    is_time_center : bool, default False
        If True, window_size centers on window_time_steps.
        If False, window starts at window_time_steps.
    extrapolate : bool, default False
        Allow extrapolation beyond target_series time range.
    return_type : Literal['pandas', 'numpy'], default 'numpy'
        Return format: 'numpy' for array, 'pandas' for Series with window_time_steps as Index.

    Returns
    -------
    np.ndarray | pd.Series
        Interpolated values matching len(window_time_steps).
        If return_type='pandas', returns Series with window_time_steps as Index.

    Raises
    ------
    ValueError
        If target_series is empty or has fewer than 2 points for full-range interpolation.
    TypeError
        If window_time_steps is not array-like or target_series is not pd.Series.
    """

    # Input validation
    if not isinstance(target_series, pd.Series):
        raise TypeError(f"target_series must be pd.Series, got {type(target_series)}")

    if len(target_series) == 0:
        raise ValueError("target_series cannot be empty")

    window_time_steps = np.asarray(window_time_steps, dtype=float)

    # Normalize target_series time to seconds
    if isinstance(target_series.index, pd.DatetimeIndex):
        source_times = (target_series.index - target_series.index[0]).total_seconds().values
    else:
        source_times = target_series.index.values.astype(float)

    target_times = window_time_steps

    if window_size is None:
        # Full-range interpolation across entire dataset
        if len(target_series) < 2:
            raise ValueError("target_series must have at least 2 points for interpolation")

        interp_func = interp1d(
            source_times, target_series.values,
            kind=method, bounds_error=not extrapolate,
            fill_value='extrapolate' if extrapolate else np.nan
        )

        try:
            result = interp_func(target_times)
        except ValueError as e:
            raise ValueError(
                f"Interpolation failed. Ensure window_time_steps are within data range "
                f"[{source_times.min()}, {source_times.max()}] or set extrapolate=True"
            ) from e

    else:
        # Window-constrained interpolation
        result = np.full(len(target_times), np.nan)

        for i, t in enumerate(target_times):
            # Define window boundaries
            if is_time_center:
                start = t - window_size / 2
                end = t + window_size / 2
            else:
                start = t
                end = t + window_size

            # Find source data within window
            mask = (source_times >= start) & (source_times < end)
            num_points = np.sum(mask)

            if num_points < 2:
                # Insufficient points for interpolation within window
                continue

            window_source_times = source_times[mask]
            window_source_values = target_series.values[mask]

            # Interpolate within this window
            try:
                interp_func = interp1d(
                    window_source_times, window_source_values,
                    kind=method, bounds_error=False, fill_value=np.nan
                )
                result[i] = interp_func(t)
            except ValueError as e:
                # Skip this window if interpolation fails
                continue

        # Final fill for remaining NaNs if extrapolate enabled
        if extrapolate:
            remaining_mask = np.isnan(result)
            if np.any(remaining_mask):
                full_interp_func = interp1d(
                    source_times, target_series.values,
                    kind=method, bounds_error=False,
                    fill_value='extrapolate'
                )
                try:
                    result[remaining_mask] = full_interp_func(target_times[remaining_mask])
                except ValueError:
                    pass  # Leave as NaN if extrapolation fails

    # Format output
    if return_type == 'pandas':
        return pd.Series(result, index=window_time_steps, name=target_series.name)
    elif return_type == 'numpy':
        return result
    else:
        raise ValueError(f"return_type must be 'pandas' or 'numpy', got {return_type}")


def add_time_index(target_array: pd.Series | np.ndarray,
                   start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp,
                   ) -> pd.Series:
    """
    Add a time index to an array assuming constant sampling rate.

    Parameters
    ----------
    target_array : pd.Series | np.ndarray
        Data array to which time index will be added.
    start_timestamp : pd.Timestamp
        Start time of the time series.
    end_timestamp : pd.Timestamp
        End time of the time series.

    Returns
    -------
    pd.Series
        Series with DatetimeIndex calculated from start/end timestamps and array length.

    Raises
    ------
    ValueError
        If start_timestamp >= end_timestamp or array is empty.
    """

    if isinstance(target_array, pd.Series):
        target_array = target_array.to_numpy()

    if len(target_array) == 0:
        raise ValueError("target_array cannot be empty")

    if start_timestamp >= end_timestamp:
        raise ValueError(f"start_timestamp ({start_timestamp}) must be before end_timestamp ({end_timestamp})")

    # Create evenly-spaced time index based on array length
    time_index = pd.date_range(
        start=start_timestamp,
        end=end_timestamp,
        periods=len(target_array)
    )

    return pd.Series(target_array, index=time_index)