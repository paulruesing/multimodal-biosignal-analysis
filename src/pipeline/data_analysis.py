import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Literal, Union

############################## DATA MANIPULATION / INTEGRATION METHODS ##############################
def _normalize_to_datetimeindex(
    timestamps: np.ndarray | pd.DatetimeIndex | list | tuple,
    name: str = "timestamps",
) -> pd.DatetimeIndex:
    """
    Normalize explicit timestamp inputs to pd.DatetimeIndex.

    Parameters
    ----------
    timestamps : np.ndarray | pd.DatetimeIndex | list | tuple
        Time values to normalize. Must be datetime-like.
    name : str, default "timestamps"
        Parameter name for error messages.

    Returns
    -------
    pd.DatetimeIndex
        Normalized timestamps with nanosecond precision and UTC timezone.

    Raises
    ------
    TypeError
        If input is numeric (int/float array) or cannot be interpreted as datetime.
    """
    if isinstance(timestamps, pd.DatetimeIndex):
        return timestamps

    if isinstance(timestamps, (list, tuple)):
        arr = np.asarray(timestamps)
    else:
        arr = np.asarray(timestamps)

    # Reject numeric input - ambiguous time origin
    if np.issubdtype(arr.dtype, np.number):
        raise TypeError(
            f"'{name}' cannot be numeric (int/float). "
            f"Numeric timestamps are ambiguous—use explicit datetime conversion instead:\n"
            f"  For seconds from Unix epoch: pd.to_datetime(array, unit='s', utc=True)\n"
            f"  For days from Unix epoch: pd.to_datetime(array, unit='D', utc=True)\n"
            f"  Or provide a pd.DatetimeIndex directly."
        )

    try:
        dt_idx = pd.to_datetime(arr, utc=True)
    except Exception as e:
        raise TypeError(
            f"Could not interpret '{name}' as datetime. Ensure input is:\n"
            f"  - DatetimeIndex\n"
            f"  - datetime64 array\n"
            f"  - ISO 8601 strings (e.g., '2024-01-01T12:30:45')\n"
            f"  - Python datetime objects\n"
            f"Error: {e}"
        ) from e

    return pd.DatetimeIndex(dt_idx)


def apply_window_operator(
    window_timestamps: np.ndarray | pd.DatetimeIndex | list | tuple,
    target_array: np.ndarray | pd.Series,
    target_timestamps: np.ndarray | pd.DatetimeIndex | list | tuple | None = None,
    window_size: float | None = None,
    is_time_center: bool | None = None,
    operation: Literal['min', 'max', 'mean', 'median', 'mode', 'std'] = 'mean',
    axis: int = 0,
    window_timestamps_ends: np.ndarray | pd.DatetimeIndex | list | tuple | None = None,
) -> np.ndarray:
    """
    Apply windowing operator to array along specified axis using timestamp-based windows.

    Parameters
    ----------
    window_timestamps : np.ndarray | pd.DatetimeIndex | list | tuple
        Time points defining window centers or starts. Must be datetime-like.

    target_array : np.ndarray | pd.Series
        Data to aggregate. If pd.Series with DatetimeIndex and target_timestamps
        is not provided, the Series index is used.

    target_timestamps : np.ndarray | pd.DatetimeIndex | list | tuple | None, default None
        Time index for target_array. If None and target_array is Series with
        DatetimeIndex, uses Series index. If provided, overwrites Series index.
        Must have length equal to target_array.shape[axis].

    window_size : float | None, default None
        Size of each window in seconds. Required if window_timestamps_ends is None.

    is_time_center : bool | None, default None
        If True, window_timestamps mark centers. If False, mark starts.
        Required when window_size is provided.

    operation : Literal['min', 'max', 'mean', 'median', 'mode', 'std'], default 'mean'
        Aggregation operation to apply within each window.

    axis : int, default 0
        Axis along which to aggregate.

    window_timestamps_ends : np.ndarray | pd.DatetimeIndex | list | tuple | None, default None
        Explicit window end points for variable-sized windows.
        Must have same length as window_timestamps.

    Returns
    -------
    np.ndarray
        Aggregated array with windows along specified axis.

    Raises
    ------
    ValueError
        If parameters invalid or dimensions mismatched.
    TypeError
        If timestamp input is numeric or cannot be interpreted as datetime.
    """

    # ===== INPUT VALIDATION: WINDOW AND SIZE CONFIGURATION =====

    if window_size is None and window_timestamps_ends is None:
        raise ValueError(
            "Either 'window_size' or 'window_timestamps_ends' must be provided. "
            "Use window_size (with is_time_center) for fixed-size windows, "
            "or window_timestamps_ends for variable-size windows."
        )

    if window_size is not None and is_time_center is None:
        raise ValueError(
            "When 'window_size' is provided, 'is_time_center' must also be specified. "
            "Set is_time_center=True if window_timestamps are window centers, "
            "or is_time_center=False if they are window starts."
        )

    # ===== INPUT VALIDATION: TARGET ARRAY STRUCTURE AND TIMESTAMPS EXTRACTION =====

    target_timestamps_provided = target_timestamps is not None

    if isinstance(target_array, pd.Series):
        # If target_timestamps not provided, attempt to extract from Series index
        if not target_timestamps_provided:
            if isinstance(target_array.index, pd.DatetimeIndex):
                target_timestamps = target_array.index
            else:
                raise ValueError(
                    "target_timestamps is None and target_array (pd.Series) does not have "
                    "a DatetimeIndex as index. Provide target_timestamps explicitly, or "
                    "ensure target_array.index is a pd.DatetimeIndex."
                )
        target_array = target_array.values
    else:
        # target_array is ndarray: target_timestamps must be provided
        if target_timestamps is None:
            raise ValueError(
                "target_timestamps is None and target_array is ndarray (not pd.Series). "
                "Provide target_timestamps explicitly."
            )

    if target_array.ndim == 0:
        raise ValueError("target_array must be at least 1D")

    if axis < 0 or axis >= target_array.ndim:
        raise ValueError(
            f"axis={axis} out of bounds for array with {target_array.ndim} dimensions"
        )

    # ===== NORMALIZE ALL TIMESTAMPS TO pd.DatetimeIndex =====

    window_times_idx = _normalize_to_datetimeindex(window_timestamps, "window_timestamps")
    target_times_idx = _normalize_to_datetimeindex(target_timestamps, "target_timestamps")

    window_times_ends_idx = None
    if window_timestamps_ends is not None:
        window_times_ends_idx = _normalize_to_datetimeindex(
            window_timestamps_ends, "window_timestamps_ends"
        )
        if len(window_times_ends_idx) != len(window_times_idx):
            raise ValueError(
                f"window_timestamps_ends length ({len(window_times_ends_idx)}) "
                f"must match window_timestamps length ({len(window_times_idx)})"
            )

    # ===== VALIDATE TIME ARRAY LENGTHS =====

    if len(target_times_idx) != target_array.shape[axis]:
        raise ValueError(
            f"Length of target_timestamps ({len(target_times_idx)}) does not match "
            f"target_array.shape[{axis}] ({target_array.shape[axis]}). "
            f"Ensure target_timestamps has one element per time point in target_array."
        )

    # ===== CREATE WINDOW BOUNDARIES AS pd.Timestamp =====

    if window_times_ends_idx is not None:
        # Variable window sizes: use provided endpoints directly
        window_starts = window_times_idx
        window_ends = window_times_ends_idx
    else:
        # Fixed window size: compute boundaries from center/start logic
        window_delta = pd.Timedelta(seconds=float(window_size))

        if is_time_center:
            # Windows centered on window_timestamps: [center - delta/2, center + delta/2]
            half_delta = window_delta / 2
            window_starts = window_times_idx - half_delta
            window_ends = window_times_idx + half_delta
        else:
            # Windows start at window_timestamps: [start, start + delta]
            window_starts = window_times_idx
            window_ends = window_times_idx + window_delta

    # ===== VALIDATE WINDOW BOUNDARIES =====

    target_min = target_times_idx.min()
    target_max = target_times_idx.max()

    window_min = window_starts.min()
    window_max = window_ends.max()

    if window_min < target_min or window_max > target_max:
        raise ValueError(
            f"Window boundaries [{window_min}, {window_max}] exceed target timestamp "
            f"range [{target_min}, {target_max}]. Ensure all windows fall within "
            f"the target data range, or extend target_timestamps."
        )

    # ===== ASSIGN EACH TARGET TIME POINT TO A WINDOW =====

    window_indices = np.full(len(target_times_idx), -1, dtype=np.int64)

    for i, (start, end) in enumerate(zip(window_starts, window_ends)):
        # Inclusive boundaries: [start, end]
        mask = (target_times_idx >= start) & (target_times_idx <= end)
        window_indices[mask] = i

    # ===== RESHAPE ARRAY FOR AGGREGATION =====

    target_array_moved = np.moveaxis(target_array, axis, 0)
    moved_shape = target_array_moved.shape

    n_time = moved_shape[0]
    target_array_flat = target_array_moved.reshape(n_time, -1)

    # ===== AGGREGATE WITHIN WINDOWS =====

    n_windows = len(window_times_idx)
    n_features = target_array_flat.shape[1]
    result = np.full((n_windows, n_features), np.nan, dtype=object)

    for window_idx in range(n_windows):
        mask = (window_indices == window_idx)

        if not np.any(mask):
            continue

        window_data = target_array_flat[mask, :]

        # Apply aggregation operation using nan-aware functions
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
            for feat_idx in range(n_features):
                mode_result = pd.Series(window_data[:, feat_idx]).mode()
                result[window_idx, feat_idx] = (
                    mode_result.iloc[0] if len(mode_result) > 0 else np.nan
                )
        else:
            raise ValueError(f"Unknown operation: {operation}")

    # ===== RESHAPE BACK TO ORIGINAL DIMENSIONS =====

    output_shape = list(moved_shape)
    output_shape[0] = n_windows
    result = result.reshape(output_shape)

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


def add_time_index(
        start_timestamp: pd.Timestamp,
        end_timestamp: pd.Timestamp,
        target_array: Union[pd.Series, pd.DataFrame, np.ndarray, None] = None,
        n_timesteps: Union[int, None] = None,
) -> Union[pd.Series, pd.DataFrame, pd.DatetimeIndex]:
    """
    Add a time index to an array or DataFrame assuming constant sampling rate.

    Creates a DatetimeIndex spanning from start_timestamp to end_timestamp with
    evenly-spaced periods. If data is provided, returns a Series or DataFrame
    indexed by the timestamps. If no data is provided, returns just the DatetimeIndex.

    Parameters
    ----------
    start_timestamp : pd.Timestamp
        Start time of the time series. Must be timezone-aware if end_timestamp is
        timezone-aware, or both must be timezone-naive.
    end_timestamp : pd.Timestamp
        End time of the time series. Must be timezone-aware if start_timestamp is
        timezone-aware, or both must be timezone-naive.
    target_array : pd.Series | pd.DataFrame | np.ndarray, optional
        Data array or DataFrame to which time index will be added. If provided,
        n_timesteps is ignored and the length of target_array determines the number
        of timesteps. For arrays, only 1-D arrays are supported. For DataFrames,
        the number of rows determines n_timesteps. If None, n_timesteps must be provided.
    n_timesteps : int, optional
        Number of time steps for which to create timestamps. Only used if
        target_array is None. Must be a positive integer.

    Returns
    -------
    pd.Series | pd.DataFrame | pd.DatetimeIndex
        - If target_array is a Series or ndarray: pd.Series with data indexed by DatetimeIndex
        - If target_array is a DataFrame: pd.DataFrame with same columns indexed by DatetimeIndex
        - If target_array is None: pd.DatetimeIndex containing timestamps only

        The returned DatetimeIndex (or Series.index/DataFrame.index) is timezone-aware
        if input timestamps are timezone-aware, timezone-naive otherwise.

    Raises
    ------
    TypeError
        If start_timestamp or end_timestamp is not a pd.Timestamp.
        If n_timesteps is not an integer when provided.
        If target_array is not 1-D (for arrays) or of unsupported type.
    ValueError
        If start_timestamp >= end_timestamp.
        If target_array is empty.
        If n_timesteps is not provided when target_array is None.
        If n_timesteps <= 0.
        If start_timestamp and end_timestamp have mismatched timezone awareness.

    Notes
    -----
    - When both start_timestamp and end_timestamp are provided, pd.date_range
      distributes periods evenly across the time span. The actual time delta
      between consecutive timestamps may not be exactly equal due to rounding.
    - If target_array is provided, n_timesteps is ignored without warning.
    - This function assumes constant sampling rate; the time intervals between
      consecutive samples are approximately equal.
    - For DataFrames, all columns and dtypes are preserved; only the index is replaced.

    Examples
    --------
    Create a Series with 5 data points indexed by hourly timestamps:

    >>> start = pd.Timestamp("2024-01-01 00:00:00")
    >>> end = pd.Timestamp("2024-01-01 04:00:00")
    >>> data = np.array([10.5, 11.2, 12.1, 13.0, 14.5])
    >>> series = add_time_index(start, end, target_array=data)
    >>> series
    2024-01-01 00:00:00    10.5
    2024-01-01 01:00:00    11.2
    2024-01-01 02:00:00    12.1
    2024-01-01 03:00:00    13.0
    2024-01-01 04:00:00    14.5
    dtype: float64

    Create a DataFrame with time index:

    >>> df = pd.DataFrame({'temp': [20, 21, 22], 'humidity': [45, 50, 48]})
    >>> start = pd.Timestamp("2024-01-01 00:00:00")
    >>> end = pd.Timestamp("2024-01-01 02:00:00")
    >>> df_indexed = add_time_index(start, end, target_array=df)
    >>> df_indexed
                         temp  humidity
    2024-01-01 00:00:00    20        45
    2024-01-01 01:00:00    21        50
    2024-01-01 02:00:00    22        48

    Create a DatetimeIndex without data:

    >>> index = add_time_index(start, end, n_timesteps=5)
    >>> index
    DatetimeIndex(['2024-01-01 00:00:00', '2024-01-01 01:00:00',
                   '2024-01-01 02:00:00', '2024-01-01 03:00:00',
                   '2024-01-01 04:00:00'],
                  dtype='datetime64[ns]', freq=None)
    """

    # ============================================================================
    # INPUT VALIDATION: Type and Value Checking
    # ============================================================================

    # Validate timestamp types
    if not isinstance(start_timestamp, pd.Timestamp):
        raise TypeError(
            f"start_timestamp must be pd.Timestamp, got {type(start_timestamp)}"
        )
    if not isinstance(end_timestamp, pd.Timestamp):
        raise TypeError(
            f"end_timestamp must be pd.Timestamp, got {type(end_timestamp)}"
        )

    # Validate timestamp ordering
    if start_timestamp >= end_timestamp:
        raise ValueError(
            f"start_timestamp ({start_timestamp}) must be strictly before "
            f"end_timestamp ({end_timestamp})"
        )

    # Validate timezone consistency between start and end timestamps
    start_tz = start_timestamp.tz
    end_tz = end_timestamp.tz
    if (start_tz is None) != (end_tz is None):
        raise ValueError(
            "start_timestamp and end_timestamp must have matching timezone awareness: "
            f"start_timestamp.tz={start_tz}, end_timestamp.tz={end_tz}"
        )

    # ============================================================================
    # ARRAY AND TIMESTEP PROCESSING
    # ============================================================================

    # Determine n_timesteps and prepare data based on input type
    if target_array is not None:
        # Handle DataFrame input
        if isinstance(target_array, pd.DataFrame):
            # Validate DataFrame is not empty
            if len(target_array) == 0:
                raise ValueError("target_array DataFrame cannot be empty")

            # Use number of rows to determine timesteps
            n_timesteps = len(target_array)
            array_data = target_array
            data_type = 'dataframe'

        # Handle Series input
        elif isinstance(target_array, pd.Series):
            array_data = target_array.to_numpy()
            data_type = 'series'

            # Validate Series is not empty
            if len(array_data) == 0:
                raise ValueError("target_array Series cannot be empty")

            # Use Series length to determine timesteps
            n_timesteps = len(array_data)

        # Handle numpy array input
        elif isinstance(target_array, np.ndarray):
            array_data = target_array
            data_type = 'array'

            # Validate array is 1-D (prevents silent loss of multi-dimensional data)
            if array_data.ndim != 1:
                raise ValueError(
                    f"target_array must be 1-dimensional, got shape {array_data.shape}"
                )

            # Validate array is not empty
            if len(array_data) == 0:
                raise ValueError("target_array array cannot be empty")

            # Use array length to determine number of timesteps
            n_timesteps = len(array_data)

        else:
            raise TypeError(
                f"target_array must be pd.Series, pd.DataFrame, or np.ndarray, "
                f"got {type(target_array)}"
            )

    else:
        # target_array is None, so n_timesteps must be provided
        if n_timesteps is None:
            raise ValueError(
                "Either target_array or n_timesteps must be provided. "
                "If target_array is None, n_timesteps must be a positive integer."
            )

        # Validate n_timesteps is an integer type (rejects floats, strings, etc.)
        if not isinstance(n_timesteps, (int, np.integer)):
            raise TypeError(
                f"n_timesteps must be an integer, got {type(n_timesteps)} "
                f"with value {n_timesteps}"
            )

        # Validate n_timesteps is positive (prevents nonsensical date ranges)
        if n_timesteps <= 0:
            raise ValueError(
                f"n_timesteps must be a positive integer, got {n_timesteps}"
            )

        # Initialize variables since no data was provided
        array_data = None
        data_type = None

    # ============================================================================
    # TIME INDEX CREATION AND RETURN
    # ============================================================================

    # Create evenly-spaced time index based on number of timesteps
    time_index = pd.date_range(
        start=start_timestamp,
        end=end_timestamp,
        periods=n_timesteps,
    )

    # Return appropriate type based on input data type
    if array_data is not None:
        if data_type == 'dataframe':
            # Return DataFrame with new DatetimeIndex while preserving columns
            result = array_data.copy()
            result.index = time_index
            return result
        else:
            # Return Series for both array and series inputs
            return pd.Series(array_data, index=time_index)
    else:
        # Return DatetimeIndex when no data provided
        return time_index


def make_timezone_aware(
    dt_index: Union[pd.DatetimeIndex, pd.Series, pd.Timestamp],
    timezone: str = 'utc',
) -> Union[pd.DatetimeIndex, pd.Series, pd.Timestamp]:
    """
    Ensure a DatetimeIndex, Series with DatetimeIndex, or Timestamp is timezone-aware.

    If the input is already timezone-aware, returns it unchanged. If it is
    timezone-naive, localizes it to the specified timezone.

    Parameters
    ----------
    dt_index : pd.DatetimeIndex | pd.Series | pd.Timestamp
        A DatetimeIndex, a Series with DatetimeIndex, or a Timestamp to make
        timezone-aware.
    timezone : str, optional
        IANA timezone string (e.g., 'UTC', 'US/Eastern', 'Europe/London').
        Default is 'utc'. Must be a valid timezone recognized by pytz or zoneinfo.

    Returns
    -------
    pd.DatetimeIndex | pd.Series | pd.Timestamp
        Returns the same type as input, now timezone-aware. If already
        timezone-aware, returns input unchanged.

    Raises
    ------
    TypeError
        If dt_index is not a DatetimeIndex, Series with DatetimeIndex, or Timestamp.
    ValueError
        If timezone is not a valid IANA timezone string.
    Exception
        If the index/Series is already timezone-aware with a different timezone
        (ambiguity in intent: localize or convert?).

    Notes
    -----
    - This function assumes naive datetimes are in the target timezone (localization),
      not converting from another timezone.
    - If the input is already timezone-aware, it is returned unchanged to avoid
      unintended timezone conversions.
    - Valid timezone strings include 'UTC', 'US/Eastern', 'Europe/London',
      'Asia/Tokyo', etc. Use pytz.all_timezones or zoneinfo.available_timezones()
      to see all available timezones.

    Examples
    --------
    Make a naive DatetimeIndex timezone-aware:

    >>> idx = pd.date_range('2024-01-01', periods=3)
    >>> idx.tz
    None
    >>> aware_idx = make_timezone_aware(idx, timezone='UTC')
    >>> aware_idx.tz
    <UTC>

    Make a Series with naive DatetimeIndex timezone-aware:

    >>> s = pd.Series([1, 2, 3], index=pd.date_range('2024-01-01', periods=3))
    >>> s.index.tz
    None
    >>> aware_s = make_timezone_aware(s, timezone='US/Eastern')
    >>> aware_s.index.tz
    <DstTzInfo 'US/Eastern' EST-1 day, 19:00:00 STD>

    Already timezone-aware index is returned unchanged:

    >>> idx_aware = pd.date_range('2024-01-01', periods=3, tz='UTC')
    >>> result = make_timezone_aware(idx_aware, timezone='US/Eastern')
    >>> result.tz  # Still UTC, not converted
    <UTC>
    """

    # Normalize timezone string to lowercase for consistency
    timezone = timezone.lower()

    # Handle DatetimeIndex
    if isinstance(dt_index, pd.DatetimeIndex):
        if dt_index.tz is not None:
            # Already timezone-aware, return unchanged
            return dt_index
        else:
            # Naive DatetimeIndex: localize to specified timezone
            return dt_index.tz_localize(timezone)

    # Handle Series with DatetimeIndex
    elif isinstance(dt_index, pd.Series):
        if not isinstance(dt_index.index, pd.DatetimeIndex):
            raise TypeError(
                f"Series must have a DatetimeIndex, got {type(dt_index.index)}"
            )

        if dt_index.index.tz is not None:
            # Already timezone-aware, return unchanged
            return dt_index
        else:
            # Naive Series: localize index to specified timezone
            result = dt_index.copy()
            result.index = result.index.tz_localize(timezone)
            return result

    # Handle Timestamp
    elif isinstance(dt_index, pd.Timestamp):
        if dt_index.tz is not None:
            # Already timezone-aware, return unchanged
            return dt_index
        else:
            # Naive Timestamp: localize to specified timezone
            return dt_index.tz_localize(timezone)

    else:
        raise TypeError(
            f"dt_index must be pd.DatetimeIndex, pd.Series with DatetimeIndex, "
            f"or pd.Timestamp, got {type(dt_index)}"
        )


def create_trial_bins(df, columns_to_bin, n_bins_dict,
                      subject_col='Subject ID', trial_col='Trial ID'):
    """
    Convert continuous values to trial-wise bins based on intra-subject percentiles.

    This function averages continuous variables across time segments within each trial,
    then creates percentile-based bins separately for each subject. The binning is done
    intra-subject to account for individual differences.

    For columns with few unique values or uneven distributions, uses value-based binning
    instead of frequency-based binning to ensure all distinct values get separate bins.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with time-segmented trial data.
    columns_to_bin : list of str
        Column names to create bins for.
    n_bins_dict : dict
        Dictionary mapping column names to number of bins.
        Example: {'Median Force Level [0-1]': 5, 'Median Heart Rate [bpm]': 3}
    subject_col : str, default='Subject ID'
        Name of the subject identifier column.
    trial_col : str, default='Trial ID'
        Name of the trial identifier column.

    Returns
    -------
    pd.DataFrame
        Original dataframe with additional columns suffixed with '_bin' containing
        ordinal bin labels (1 to n_bins). NaN values in the original data will result
        in NaN bins.

    Examples
    --------
    >>> binned_df = create_trial_bins(
    ...     df=my_data,
    ...     columns_to_bin=['Trial ID', 'Median Force Level [0-1]'],
    ...     n_bins_dict={'Trial ID': 3, 'Median Force Level [0-1]': 5}
    ... )
    """
    # Create a copy to avoid modifying the original dataframe
    df_result = df.copy()

    # Determine which columns need aggregation (exclude grouping columns)
    grouping_cols = [subject_col, trial_col]
    columns_to_aggregate = [col for col in columns_to_bin if col not in grouping_cols]

    # Create trial-level data
    if columns_to_aggregate:
        trial_averages = df.groupby(grouping_cols, as_index=False)[columns_to_aggregate].mean()
    else:
        trial_averages = df[grouping_cols].drop_duplicates().reset_index(drop=True)

    # Iterate through each column to create bins
    for col in columns_to_bin:
        n_bins = n_bins_dict.get(col, 5)

        # Create bin column name
        bin_col_name = f"{col}_bin"

        # Initialize a list to store bin assignments
        bin_assignments = []

        # Bin separately for each subject (intra-subject binning)
        for subject_id in trial_averages[subject_col].unique():
            # Filter data for current subject
            subject_mask = trial_averages[subject_col] == subject_id
            subject_data = trial_averages[subject_mask].copy()
            subject_values = subject_data[col]

            # Handle all-NaN case
            if subject_values.isna().all():
                subject_data[bin_col_name] = np.nan
                bin_assignments.append(subject_data[[subject_col, trial_col, bin_col_name]])
                continue

            # Get non-NaN values for binning
            non_nan_values = subject_values.dropna()
            n_unique = non_nan_values.nunique()

            # If only one unique value, all get bin 1
            if n_unique == 1:
                subject_data[bin_col_name] = subject_values.notna().astype(int)
                subject_data.loc[subject_values.isna(), bin_col_name] = np.nan
                bin_assignments.append(subject_data[[subject_col, trial_col, bin_col_name]])
                continue

            # Determine effective number of bins
            effective_bins = min(n_bins, n_unique)

            # Initialize bin column with NaN
            subject_data[bin_col_name] = np.nan

            try:
                # For discrete-like data (few unique values relative to requested bins),
                # use rank-based binning to ensure different values get different bins
                if n_unique <= n_bins:
                    # Rank the values and map to bins
                    sorted_unique = sorted(non_nan_values.unique())
                    value_to_bin = {val: i + 1 for i, val in enumerate(sorted_unique)}
                    subject_data[bin_col_name] = subject_values.map(value_to_bin)
                else:
                    # For continuous data, use qcut for percentile-based binning
                    bins = pd.qcut(
                        subject_values,
                        q=effective_bins,
                        labels=False,
                        duplicates='drop'
                    )
                    # Convert to 1-indexed (bins is 0-indexed)
                    subject_data.loc[subject_values.notna(), bin_col_name] = bins.dropna() + 1

            except Exception as e:
                # Fallback: use cut with explicit boundaries
                try:
                    bins = pd.cut(
                        subject_values,
                        bins=effective_bins,
                        labels=False,
                        duplicates='drop'
                    )
                    subject_data.loc[subject_values.notna(), bin_col_name] = bins.dropna() + 1
                except:
                    # Last resort: all non-NaN values get bin 1
                    subject_data[bin_col_name] = subject_values.notna().astype(float)
                    subject_data.loc[subject_values.isna(), bin_col_name] = np.nan

            bin_assignments.append(subject_data[[subject_col, trial_col, bin_col_name]])

        # Combine all bin assignments for this column
        bin_df = pd.concat(bin_assignments, ignore_index=True)

        # Merge bin labels back to original dataframe
        df_result = df_result.merge(
            bin_df,
            on=[subject_col, trial_col],
            how='left'
        )

    # Convert bin columns to integer type (handling NaN values)
    bin_columns = [f"{col}_bin" for col in columns_to_bin]
    for bin_col in bin_columns:
        if bin_col in df_result.columns:
            df_result[bin_col] = df_result[bin_col].astype('Int64')

    return df_result


