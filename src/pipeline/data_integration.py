import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from typing import Literal

import src.utils.file_management as filemgmt
from src.pipeline.data_analysis import make_timezone_aware


############################## LOG FRAME HANDLING ##############################
def fetch_experiment_log(subject_data_dir: Path) -> pd.DataFrame:
    """
    Fetch and concatenate the most recent experiment log files from a subject directory.

    Prioritizes "Final Full Save" and concatenates "Working Memory Full Save" files.
    Falls back to "Interim Save" only if no "Final Full Save" exists. Converts 'Time'
    to datetime, sorts descending (newest first), and removes duplicates by timestamp.

    Parameters
    ----------
    subject_data_dir : Path
        Path to the subject data directory containing 'experiment_logs' subfolder.

    Returns
    -------
    pd.DataFrame
        Processed experiment log DataFrame, sorted descending by 'Time' (datetime),
        duplicates removed. Columns include 'Time' as datetime and experiment data.

    Raises
    ------
    ValueError
        If no log files found, incompatible columns, or missing 'Time' column.
    FileNotFoundError
        If log directory does not exist.

    Notes
    -----
    "Working Memory Full Save" files always concatenated with final save.
    "Interim Save" used only as fallback if no "Final Full Save" exists.
    Duplicates dropped keeping first occurrence (earliest load order).
    'Time' parsed via pd.to_datetime (inferred format).
    """
    log_dir = subject_data_dir / 'experiment_logs'
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    # Fetch Working Memory Full Save frames (always concatenate if present)
    wm_frames = []
    try:
        wm_data = filemgmt.most_recent_file(log_dir, ".csv",
                                            ["Working Memory Full Save"],
                                            return_type='dict')
        wm_frame_paths = wm_data['files']
        wm_frames = [pd.read_csv(path) for path in wm_frame_paths]
        print(f"Found {len(wm_frames)} Working Memory Full Save logs in {log_dir}.")
    except ValueError:
        print(f"No Working Memory Full Save logs found in {log_dir}.")

    # Fetch final frame: Final Full Save or fallback to Interim Save
    try:
        final_frame_path = filemgmt.most_recent_file(log_dir, ".csv", ["Final Full Save"])
    except ValueError:
        print(f"No 'Final Full Save' in {log_dir}. Using 'Interim Save' as fallback.")
        try:
            final_frame_path = filemgmt.most_recent_file(log_dir, ".csv", ["Interim Save"])
        except ValueError:
            raise ValueError(f"No log files found in {log_dir}")

    final_frame = pd.read_csv(final_frame_path)

    # Concat if Working Memory frames exist, validate columns
    frames = wm_frames + [final_frame] if wm_frames else [final_frame]
    if len(frames) > 1:
        if not all(frame.shape[1] == frames[0].shape[1] for frame in frames[1:]):
            raise ValueError("Incompatible columns across frames.")
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = frames[0]

    # Process: convert Time, sort desc, drop dups
    return _process_frame(combined)


def _process_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Helper: Convert 'Time' to datetime, sort desc by Time, drop timestamp dups."""
    if 'Time' not in df.columns:
        raise ValueError("DataFrame missing 'Time' column.")

    # Parse Time to datetime
    df['Time'] = pd.to_datetime(df['Time'])

    # Sort descending (newest first), drop duplicates keeping first, reset index
    df = df.sort_values('Time', ascending=True).drop_duplicates(subset=['Time'], keep='first').reset_index(drop=True)

    return df


def prepare_log_frame(log_frame: pd.DataFrame, set_time_index: bool = True) -> pd.DataFrame:
    ############### Derive Values from Status Strings ###############
    def derive_song_category_string(input: str) -> str:
        """ apply to music column"""
        elements = input.split(" | ")
        if len(elements) == 2:  # no category entry
            return "No category"
        elif len(elements) == 3:
            return elements[0]
        else: return "No song playing"
    category_string_series = log_frame['Music'].apply(derive_song_category_string)


    def derive_category(input: str) -> str:
        """ apply to series derived via derive_song_category_string """
        elements = input.split(" (")
        if len(elements) == 1: return "No category"
        else: return elements[0]
    log_frame['Music Category'] = category_string_series.apply(derive_category)


    def derive_category_index(input: str) -> int | None:
        """ apply to series derived via derive_song_category_string """
        elements = input.split(" (")
        if len(elements) == 1: return None
        else: return int(elements[1].split("/")[0])  # structure is eg CATEGORY (1/11) -> would yield 1
    log_frame['Within Category Song Index'] = category_string_series.apply(derive_category_index)

    def derive_song_info(input: str) -> str:
        """ apply to Music column """
        elements = input.split(" | ")
        if len(elements) == 2:  # no category entry
            return elements[0]
        elif len(elements) == 3: return elements[1]
        else: return "No song playing"

    log_frame['Song Info'] = log_frame['Music'].apply(derive_song_info)

    def add_song_info_title_and_artist(df: pd.DataFrame) -> pd.DataFrame:
        """ Based on Song Info """
        df = df.copy()
        # Split on the last occurrence of ' by '
        split_data = df['Song Info'].str.rsplit(' by ', n=1, expand=True)
        df['Song Title'] = split_data[0].str.strip()
        df['Song Artist'] = split_data[1].str.strip()
        return df
    log_frame = add_song_info_title_and_artist(log_frame)


    def derive_song_runtime(input: str) -> float | None:
        """ Based on Music (given) """
        elements = input.split(" | ")
        if len(elements) == 2:  # no category entry
            return float(elements[1].split(" / ")[0].split("s")[0])
        elif len(elements) == 3: return float(elements[2].split(" / ")[0].split("s")[0])
        else: return None
    log_frame['Song Runtime'] = log_frame['Music'].apply(derive_song_runtime)


    ############### Extract Values and Extend ###############
    # todo: pull task end 3s ahead, since there is on average a slight delay -> window closing, RMSE computation, RMSE documentation
    def add_task_freqs_and_average_rmse(df: pd.DataFrame, avg_end_delay_seconds: float = 3.0) -> pd.DataFrame:
        """ Based on Questionnaire (given) """
        # Step 1: Extract frequency and RMSE values
        df['Task Frequency'] = df['Questionnaire'].str.extract(
            r'target frequency ([\d.]+)Hz',
            expand=False
        )
        df['Task RMSE'] = df['Questionnaire'].str.extract(
            r'Achieved RMSE: ([\d.]+)',
            expand=False
        )

        # Step 1.5: Clear Task Frequency from test task rows
        # Identify rows with "Starting test motor task" and set their frequency to NaN
        is_test_task = df['Questionnaire'].str.contains(
            r'Starting\s+test\s+motor task',
            na=False,
            regex=True
        )
        df.loc[is_test_task, 'Task Frequency'] = np.nan

        # Step 2: Create task ID based on start markers
        # Use negative lookahead (?!test) to exclude "Starting test motor task"
        df['is_start'] = df['Questionnaire'].str.contains(
            r'Starting(?!\s+test)\s+motor task',
            na=False,
            regex=True
        )
        df['task_id'] = df['is_start'].cumsum()

        # Step 3: Forward fill frequency within task groups (unlimited)
        df['Task Frequency'] = df.groupby('task_id')['Task Frequency'].ffill()

        # Step 4: Backward fill RMSE within task groups
        df['Task RMSE'] = df.groupby('task_id')['Task RMSE'].bfill()

        # Step 5: Create mask to clear frequency values AFTER the RMSE row
        is_end = df['Questionnaire'].str.contains('Achieved RMSE', na=False)

        # adjust is_end by some seconds to account for RMSE computation delay:
        end_times = df.loc[is_end, 'Time'].values  # find the times of end markers
        adjusted_is_end = pd.Series(False, index=df.index)  # this will hold the adjusted values
        for end_time in end_times:  # find the closest row where Time <= (end_time - avg_end_delay_seconds) for each
            target_time = end_time - pd.Timedelta(seconds=avg_end_delay_seconds)
            task_of_end = df.loc[is_end & (df['Time'] == end_time), 'task_id'].iloc[0]

            # Find rows in this task with Time <= target_time, get the last one
            mask = (df['task_id'] == task_of_end) & (df['Time'] <= target_time)
            if mask.any():
                adjusted_is_end.iloc[df[mask].index[-1]] = True

        # rows_after_end computation:
        # 1) shift is_end row by 1 (default) and fill first with False
        # 2) groupby -> within a task ID, cum_sum (= forward fill) the is_end marker (somehow ffill() doesn't work correctly here)
        # 3) select the rows after end (the forward filled ones) by > 0
        rows_after_end = adjusted_is_end.shift(fill_value=False).groupby(df['task_id']).cumsum() > 0
        # set those frequencies (when task is over) to np.nan
        df.loc[rows_after_end, 'Task Frequency'] = np.nan
        df.loc[rows_after_end, 'Task RMSE'] = np.nan

        # Step 6: Clean up helper columns
        df = df.drop(columns=['is_start', 'task_id'])

        return df

    log_frame = add_task_freqs_and_average_rmse(log_frame)


    ############### Further Value Integration ###############
    def add_task_phase(df: pd.DataFrame) -> pd.DataFrame:
        """ Based on Music Category and Task Frequency """
        df['Phase'] = pd.Series(data=[None]*len(df), dtype=object)

        # Condition 1: Category exists, not "No category", and frequency exists
        mask1 = (df['Music Category'].notna()) & \
                (df['Music Category'] != 'No category') & \
                (df['Task Frequency'].notna())
        df.loc[mask1, 'Phase'] = df.loc[mask1, 'Music Category'] + ' Task'

        # Condition 2: Category is "No category" and frequency exists
        mask2 = (df['Music Category'] == 'No category') & (df['Task Frequency'].notna())
        df.loc[mask2, 'Phase'] = 'Silence Task'

        # Condition 3: Category exists, not "No category" but frequency missing (task not running)
        mask3 = (df['Music Category'].notna()) & \
                (df['Music Category'] != 'No category') & \
                (df['Task Frequency'].isna())
        df.loc[mask3, 'Phase'] = df.loc[mask3, 'Music Category'] + ' Listening'

        return df
    log_frame = add_task_phase(log_frame)
    
    
    def add_song_id(df: pd.DataFrame) -> pd.DataFrame:
        """ Based on Song Info and Music Category"""
        df = df.copy()
        df['Song ID'] = (
                (df['Song Info'] != df['Song Info'].shift()) &  # song changes compared to previous row
                (df['Song Info'] != "No song playing")  &  # doesn't change if no song is playing
                (df['Music Category'] != "No category")  # and if no category
        ).cumsum() - 1  # because starts at 1
        # cum sum integrates song differences (counter logic) and forwards values

        # now we need to remove the "No song playing" entries
        df.loc[df['Music Category'] == "No category", 'Song ID'] = np.nan
        df.loc[df['Song Info'] == "No song playing", 'Song ID'] = np.nan
        return df
    log_frame = add_song_id(log_frame)


    def add_song_skipped_bool(df: pd.DataFrame) -> pd.DataFrame:
        """ Based on Song ID and Task Frequency"""
        df = df.copy()
        task_freq_bfilled = df.groupby('Song ID')['Task Frequency'].bfill()
        task_freq_ffilled = df.groupby('Song ID')['Task Frequency'].ffill()
        df['Song Skipped'] = (
            (~df['Song ID'].isna()) &
            (task_freq_bfilled.isna()) &
            (task_freq_ffilled.isna())
        )
        # remove entry for where there is no Song
        df['Song Skipped'] = df['Song Skipped'].astype('boolean')  # Capital B
        df.loc[df['Song ID'].isna(), 'Song Skipped'] = pd.NA

        return df

    log_frame = add_song_skipped_bool(log_frame)


    def add_silence_id(df: pd.DataFrame) -> pd.DataFrame:
        """ Based on Phase """
        df = df.copy()
        # increment for each starting Silence Task
        df['Silence ID'] = (
            (df['Phase'] == 'Silence Task') &   # phase is Silence Task
            (df['Phase'] != df['Phase'].shift())  # phase changes
        ).cumsum() - 1  # because cumsum starts at 1

        # remove forward fills where there is no task
        df.loc[df['Phase'] != 'Silence Task', 'Silence ID'] = np.nan
        return df
    log_frame = add_silence_id(log_frame)


    def add_trial_id(df: pd.DataFrame) -> pd.DataFrame:
        """ Based on Song ID and Silence ID """
        df = df.copy()
        # increment for each change in Song or Silence ID:
        df['Trial ID'] = np.nan
        # set Song ID values where they aren't None
        df.loc[~df['Song ID'].isna(), 'Trial ID'] = df.loc[~df['Song ID'].isna(), 'Song ID']
        # set Silence ID values similarly:
        df.loc[~df['Silence ID'].isna(), 'Trial ID'] = df.loc[~df['Silence ID'].isna(), 'Silence ID']

        # now recompute increment:
        df['Trial ID'] = ((df['Trial ID'] != df['Trial ID'].shift()) &
                          (~df['Trial ID'].isna())).cumsum() - 1
        # and set nan where neither song nor silence ID is given:
        df.loc[(df['Song ID'].isna()) & (df['Silence ID'].isna()), 'Trial ID'] = np.nan
        return df
    log_frame = add_trial_id(log_frame)


    ############### Derive and Extend Questionnaire Results ###############
    # todo: propagates wrongly
    def add_questionnaire_results(df: pd.DataFrame) -> pd.DataFrame:
        """ Based on Questionnaire (given) and Trial ID"""
        df = df.copy()
        df['Questionnaire'] = df['Questionnaire'].astype(str)

        # Extract familiarity from questionnaire text
        def extract_familiarity(text):
            """
            Extract familiarity rating from questionnaire string.

            Parameters
            ----------
            text : str
                Questionnaire text containing "Familiarity check result: X"

            Returns
            -------
            float or np.nan
                Familiarity rating (1-5) or NaN if not found
            """
            match = re.search(r'Familiarity check result:\s*(\d)', text)
            return float(match.group(1)) if match else np.nan

        df['Familiarity'] = df['Questionnaire'].apply(extract_familiarity)

        # Extend familiarity values across equal Trial ID
        df['Familiarity'] = df.groupby('Trial ID')['Familiarity'].ffill()
        df['Familiarity'] = df.groupby('Trial ID')['Familiarity'].bfill()

        # Extract post-trial questionnaire data
        def extract_post_trial(text):
            """
            Extract post-trial rating dictionary from questionnaire string.

            Parses the dict representation from questionnaire text and ensures
            all expected keys are present with NaN as fallback.

            Parameters
            ----------
            text : str
                Questionnaire text containing post-trial ratings dict

            Returns
            -------
            dict
                Dictionary with keys: Liking, Fitting Category, Emotional State, Other category
            """
            import ast
            try:
                # Extract just the dict portion
                match = re.search(r"\{.*\}", text)
                if match:
                    dict_str = match.group(0)
                    # Convert from Python string representation to actual dict
                    data = ast.literal_eval(dict_str)

                    # Ensure all keys exist
                    if 'Liking' not in data.keys():
                        data['Liking'] = np.nan
                    if 'Fitting Category' not in data.keys():
                        data['Fitting Category'] = np.nan
                    if 'Other category' not in data.keys():
                        data['Other category'] = np.nan
                    # Emotional State is always asked
                    if 'Emotional State' not in data.keys():
                        data['Emotional State'] = np.nan

                    return data
            except:
                pass

            return {
                'Liking': np.nan,
                'Fitting Category': np.nan,
                'Emotional State': np.nan,
                'Other category': np.nan
            }

        # Extract post-trial data and create new columns
        post_trial_data = df['Questionnaire'].apply(extract_post_trial)
        df['Liking'] = post_trial_data.apply(lambda x: x['Liking'])
        df['Fitting Category'] = post_trial_data.apply(lambda x: x['Fitting Category'])
        df['Emotional State'] = post_trial_data.apply(lambda x: x['Emotional State'])
        df['Other category'] = post_trial_data.apply(lambda x: x['Other category'])

        # IMPORTANT: some post-trial questionnaires are received after the trial has finished. For that purpose:
        #       (1) forward-fill Trial ID temporarily
        #       (2) forward- and backward-fill values
        #       (3) reset Trial ID where there is neither song nor silence ID
        #       (4) reset values where there is no Trial ID
        df['Trial ID'] = df['Trial ID'].ffill()  # (1)

        # (2)
        for key in ['Liking', 'Fitting Category', 'Emotional State', 'Other category']:
            # Forward fill within Trial ID groups (propagate from first occurrence forward)
            df[key] = df.groupby('Trial ID')[key].ffill()
            # Backward fill within Trial ID groups (propagate from last occurrence backward)
            df[key] = df.groupby('Trial ID')[key].bfill()

        # (3)
        df.loc[(df['Song ID'].isna()) & (df['Silence ID'].isna()), 'Trial ID'] = np.nan

        # (4)
        for key in ['Liking', 'Fitting Category', 'Emotional State', 'Other category']:
            df.loc[df['Trial ID'].isna(), key] = np.nan

        return df

    log_frame = add_questionnaire_results(log_frame)


    def add_perceived_category(log_df: pd.DataFrame) -> pd.DataFrame:
        log_df = log_df.copy()

        # extract category after "Familliar"
        log_df['Perceived Category'] = log_df['Music Category'].str.extract(r'[Ff]amiliar\s+(\w+)', expand=False)

        # overwrite by other category (if not nan and if specified)
        log_df.loc[(~log_df['Other category'].isna()) & (log_df['Other category'] != 'None of them'), ['Perceived Category']] = log_df['Other category']

        return log_df

    log_frame = add_perceived_category(log_frame)



    ############## Add placeholder columns ##############
    log_frame['Trial Comment'] = [""] * len(log_frame)
    log_frame['Trial Exclusion Bool'] = [False] * len(log_frame)
    log_frame['Trial Exclusion Bool'] = log_frame['Trial Exclusion Bool'].astype('boolean')
    log_frame.loc[log_frame['Trial ID'].isna(), 'Trial Exclusion Bool'] = np.nan


    ############### Format and Return ###############
    if set_time_index:
        log_frame['Time'] = pd.to_datetime(log_frame['Time'])
        log_frame = log_frame.set_index('Time')

    return log_frame


def turn_trial_id_into_song_or_silence_id(log_df: pd.DataFrame,
                                          trial_id: int) -> tuple[int | None, int | None]:
    """ Returns song_id and silence_id as tuple, one is np.nan """
    subset = log_df.loc[log_df['Trial ID'] == trial_id]
    song_id = subset.iloc[0]['Song ID']
    silence_id = subset.iloc[0]['Silence ID']
    return int(song_id) if not np.isnan(song_id) else None, int(silence_id) if not np.isnan(silence_id) else None


def turn_song_or_silence_id_into_trial_id(log_df: pd.DataFrame,
                                          song_id: int | None = None,
                                          silence_id: int | None = None) -> int:
    """ Returns trial_id given either song_id or silence_id (one must be provided) """
    if song_id is not None:
        subset = log_df.loc[log_df['Song ID'] == song_id]
    elif silence_id is not None:
        subset = log_df.loc[log_df['Silence ID'] == silence_id]
    else:
        raise ValueError("Either song_id or silence_id must be provided")

    if len(subset) == 0:
        raise ValueError(f"No trial found with song_id={song_id} or silence_id={silence_id}")

    trial_id = subset.iloc[0]['Trial ID']
    return int(trial_id)



def get_song_start_end(df: pd.DataFrame,
                       song_id: int | None = None, song_title: str | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    if song_id is None and song_title is None: raise ValueError("Either song_id or song_title must be specified")

    # locate song
    if song_id is not None:
        subset_df = df.loc[df['Song ID'] == song_id]
    else:
        song_title_series = df['Song Title']
        subset_df = df.loc[song_title_series == song_title]

        # prevent catching multiple sequences:
        unique_song_ids = subset_df['Song ID'].dropna().unique().astype(int)
        if len(unique_song_ids) > 1:  # if matching song title has more than one ID
            raise ValueError(f"Song title appeared multiple times with Song IDs: {unique_song_ids.tolist()}\nChoose one and call this method with song_id!")

    if (subset_df['Song Skipped']).any():
        print(f"[INFO] Song {int(song_id)} got skipped, no corresponding task was executed.")

    if len(subset_df) == 0:
        raise ValueError("Specific song not found!")

    # retrieve timestamps:
    if isinstance(subset_df.index, pd.DatetimeIndex):
        times = subset_df.index
    elif 'Time' in subset_df.columns:
        times = pd.to_datetime(subset_df['Time'])
    else:
        raise ValueError('df must contain "Time" column or DatetimeIndex!')

    return times.min(), times.max()


def get_task_start_end(df: pd.DataFrame,
                       song_id: int | None = None, song_title: str | None = None,
                       trial_id: int | None = None,
                       silence_id: int | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    if song_id is None and song_title is None and silence_id is None and trial_id is None:
        raise ValueError("Either song_id, song_title, trial_id or silence_id must be specified")

    if trial_id is not None:
        song_id, silence_id = turn_trial_id_into_song_or_silence_id(df, trial_id)

    # locate song trial:
    if song_id is not None or song_title is not None:
        if song_id is not None:
            subset_df = df.loc[df['Song ID'] == song_id]
        else:
            song_title_series = df['Song Title']
            subset_df = df.loc[song_title_series == song_title]

            # prevent catching multiple sequences:
            unique_song_ids = subset_df['Song ID'].dropna().unique().astype(int)
            if len(unique_song_ids) > 1:  # if matching song title has more than one ID
                raise ValueError(f"Song title appeared multiple times with Song IDs: {unique_song_ids.tolist()}\nChoose one and call this method with song_id!")

        # check for trial exclusion or skipping:
        if (subset_df['Song Skipped']).any():
            print(f"[INFO] Song {int(song_id)} got skipped, no corresponding task was executed.")
        if subset_df['Trial Exclusion Bool'].any():
            print(f"[INFO] Song {int(song_id)} marked for exclusion!")

        # reduce to task time (where Task Frequency is not na):
        subset_df = subset_df.loc[~subset_df['Task Frequency'].isna()]

    else:
        subset_df = df.loc[df['Silence ID'] == silence_id]
        # check for trial exclusion:
        if subset_df['Trial Exclusion Bool'].any():
            print(f"[INFO] Silence trial {int(silence_id)} marked for exclusion!")


    # raise value errors for task omission (skipped or excluded)
    if len(subset_df) == 0:
        raise ValueError("Specific task not found!")
    if subset_df['Trial Exclusion Bool'].any():
        raise ValueError("Trial marked for exclusion!")


    # retrieve timestamps:
    if isinstance(subset_df.index, pd.DatetimeIndex):
        times = subset_df.index
    elif 'Time' in subset_df.columns:
        times = pd.to_datetime(subset_df['Time'])
    else:
        raise ValueError('df must contain "Time" column or DatetimeIndex!')

    return times.min(), times.max()


def get_all_task_start_ends(enriched_log_df: pd.DataFrame,
                            output_type: Literal['dict', 'list'] = 'dict',
                            ) -> dict[int, tuple[pd.Timestamp, pd.Timestamp]] | list[tuple[pd.Timestamp, pd.Timestamp]]:
    if output_type == 'dict': trial_start_end_dict: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    else: start_end_list: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for trial in enriched_log_df['Trial ID'].unique():
        if pd.isna(trial): continue
        try:
            start, end = get_task_start_end(enriched_log_df, trial_id=trial)
            start = make_timezone_aware(start)
            end = make_timezone_aware(end)
        except ValueError:
            continue

        if output_type == 'dict': trial_start_end_dict[int(trial)] = (start, end)
        else: start_end_list.append((start, end))

    return trial_start_end_dict if output_type == 'dict' else start_end_list



def get_qtc_measurement_start_end(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.Timestamp, pd.Timestamp]:
    """ Derive QTC measurement duration based on 'Start Trigger' and 'Stop Trigger' keywords in 'Event' column.
    If during data-integration, 'Actual Start Trigger' was inserted to indicate a measurement cut-off, this will be leveraged."""
    df = df.copy()  # copied here because we change the index below

    # convert to datetime index timestamps:
    if isinstance(df.index, pd.DatetimeIndex):
        pass
    elif 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)
    else:
        raise ValueError('df must contain "Time" column or DatetimeIndex!')

    try:  # retrieve start of QTC measurement
        qtc_start = df.loc[df['Event'] == "Start Trigger"].index.item()
    except ValueError:  # if 'Start Trigger' not found
        print("No 'Start Trigger' event found, assuming measurement started upon beginning")
        qtc_start = df.index.min()

    try:  # retrieve end
        qtc_end = df.loc[df['Event'] == "Stop Trigger"].index.item()
    except ValueError:  # if 'Stop Trigger' not found
        print("No 'Stop Trigger' event found, assuming measurement ran until end.")
        qtc_end = df.index.max()

    try:
        actual_qtc_start = df.loc[df['Event'] == "Actual Start Trigger"].index.item()
        print(f"Found 'Actual Start Trigger' event, indicating cut-off of initial measurements. Will return actual start timestamp: {actual_qtc_start}")
        qtc_start = actual_qtc_start
    except ValueError:
        pass

    # make timezone-aware:
    if qtc_start.tz is None:
        qtc_start = qtc_start.tz_localize('UTC')
    if qtc_end.tz is None:
        qtc_end = qtc_end.tz_localize('UTC')

    if verbose: print(f"EEG and EMG measurements last from {qtc_start} to {qtc_end}!\n")

    return qtc_start, qtc_end


def validate_force_measurements(log_df: pd.DataFrame, serial_df: pd.DataFrame,
                                   freeze_threshold_seconds: float = .2,
                                   ) -> pd.DataFrame:
    # ensure DatetimeIndex:
    if not isinstance(log_df.index, pd.DatetimeIndex): log_df = log_df.set_index('Time')
    if not isinstance(serial_df.index, pd.DatetimeIndex): serial_df = serial_df.set_index('Time')

    # check music trials:
    for trial_id in log_df['Trial ID'].unique():
        if pd.isna(trial_id):
            continue

        song_id, silence_id = turn_trial_id_into_song_or_silence_id(log_df, trial_id)

        try:
            start, end = get_task_start_end(log_df, song_id=song_id, silence_id=silence_id)
        except ValueError:  # song / task skipped
            continue

        # use consistent indexing (adjust if/else based on your actual index type)
        within_task_fsr_series = serial_df.loc[start:end, 'fsr']

        if len(within_task_fsr_series) == 0:
            continue

        # Calculate sampling rate (samples per second)
        serial_sampling_rate = len(within_task_fsr_series) / (end - start).total_seconds()

        # Create group identifier for consecutive runs
        group = within_task_fsr_series.ne(within_task_fsr_series.shift()).cumsum()
        consecutive_count = within_task_fsr_series.groupby(group).cumcount() + 1

        # Apply frozen threshold (in samples)
        freeze_threshold_samples = freeze_threshold_seconds * serial_sampling_rate
        is_frozen = consecutive_count >= freeze_threshold_samples

        trial_label = f'song_{int(song_id):03}' if song_id is not None else f'silence_{int(silence_id):03}'
        if is_frozen.any():
            print(f"[WARNING] Frozen force measurements (for more than {freeze_threshold_seconds}sec) found for {trial_label}.")
        else:
            print(f"Maximum duration of consecutive measurements for trial ID {int(trial_id)} ({trial_label}): {consecutive_count.max() / serial_sampling_rate:.2f}sec")


def validate_song_indices(df: pd.DataFrame,
                          experiment_data_dir: str | Path,
                          error_handling: Literal['raise', 'continue'] = 'continue',
                          verbose: bool = True) -> dict:
    """
    Validate that log frame entries match fetched song information from JSON files.
    Handles multiple entries per song ID by grouping and checking consistency.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'Song ID', 'Song Title', 'Song Artist'
        May have multiple rows per Song ID
    experiment_data_dir : str | Path
        Root directory containing trial subdirectories with song information
    error_handling : {'raise', 'continue'}
        How to handle missing song information files

    Returns
    -------
    dict
        Validation report with:
        - 'valid': bool, True if all validations pass
        - 'matches': list of valid song IDs with consistent metadata
        - 'mismatches': list of validation failures with details
        - 'duplicate_entries': Song IDs with conflicting Title/Artist across rows
        - 'missing_metadata': Song IDs unable to fetch metadata
        - 'summary': summary statistics
    """

    report = {
        'valid': True,
        'matches': [],
        'mismatches': [],
        'duplicate_entries': [],
        'missing_metadata': [],
        'summary': {}
    }

    experiment_data_dir = Path(experiment_data_dir)

    # Group by Song ID
    grouped = df.groupby('Song ID')

    for song_id, group in grouped:
        song_id = int(song_id)

        # Check consistency within the group
        titles = group['Song Title'].unique()
        artists = group['Song Artist'].unique()

        if len(titles) > 1 or len(artists) > 1:
            report['valid'] = False
            report['duplicate_entries'].append({
                'Song ID': song_id,
                'Unique Titles': list(titles),
                'Unique Artists': list(artists),
                'Number of Log Entries': len(group),
                'Issue': 'Multiple conflicting Title/Artist values for same Song ID'
            })
            continue

        # Use the first entry's values (now verified to be consistent)
        song_title = titles[0]
        song_artist = artists[0]

        # Fetch song information using provided method
        song_metadata = fetch_song_information(
            experiment_data_dir,
            song_id=song_id,
            error_handling=error_handling
        )

        # Handle missing metadata
        if song_metadata is None:
            report['valid'] = False
            report['missing_metadata'].append({
                'Song ID': song_id,
                'Song Title': song_title,
                'Song Artist': song_artist,
                'Number of Log Entries': len(group),
                'Status': 'Could not fetch metadata'
            })
            continue

        # Validate Title and Artist match
        try:
            metadata_title = song_metadata.get('Title', '')
            metadata_artist = song_metadata.get('Artist', '')

            title_match = metadata_title == song_title
            artist_match = metadata_artist == song_artist

            if title_match and artist_match:
                report['matches'].append({
                    'Song ID': song_id,
                    'Song Title': song_title,
                    'Song Artist': song_artist,
                    'Number of Log Entries': len(group),
                    'Status': 'Valid',
                    'Metadata': {
                        'Album': song_metadata.get('Album'),
                        'Genre': song_metadata.get('Genre'),
                        'Duration [ms]': song_metadata.get('Duration [ms]'),
                        'BPM': song_metadata.get('BPM')
                    }
                })
            else:
                report['valid'] = False
                report['mismatches'].append({
                    'Song ID': song_id,
                    'Number of Log Entries': len(group),
                    'Log Frame Title': song_title,
                    'Metadata Title': metadata_title,
                    'Title Match': title_match,
                    'Log Frame Artist': song_artist,
                    'Metadata Artist': metadata_artist,
                    'Artist Match': artist_match
                })

        except (KeyError, AttributeError, TypeError) as e:
            report['valid'] = False
            report['mismatches'].append({
                'Song ID': song_id,
                'Number of Log Entries': len(group),
                'Issue': f"Error extracting metadata fields: {str(e)}",
                'Metadata': song_metadata
            })

    # Summary statistics
    report['summary'] = {
        'total_log_entries': len(df),
        'unique_song_ids': len(grouped),
        'valid_matches': len(report['matches']),
        'mismatches': len(report['mismatches']),
        'duplicate_entries': len(report['duplicate_entries']),
        'missing_metadata': len(report['missing_metadata']),
        'validation_passed': report['valid']
    }

    # Output print:
    if verbose:
        if len(report['duplicate_entries']) > 0:
            print(f"[WARNING] Found {len(report['duplicate_entries'])} duplicate entries while matching song directories and log frame:")
            print(report['duplicate_entries'], "\n")
        if len(report['missing_metadata']) > 0:
            print(
                f"[WARNING] Couldn't find metadata (directories) for {len(report['missing_metadata'])} songs:")
            print(report['missing_metadata'], "\n")
        if len(report['mismatches']) > 0:
            print(
                f"[WARNING] Found no matching information for {len(report['mismatches'])} songs:")
            print(report['mismatches'], "\n")
        if report['valid']: print("Validation passed!\n")


    return report


def validate_trial_questionnaires(df: pd.DataFrame,
                                  experiment_data_dir: str | Path,
                                  error_handling: Literal['raise', 'continue'] = 'continue',
                                  verbose: bool = True) -> dict:
    """
    Validate that log frame entries match corresponding questionnaire data.
    Compares questionnaire dict values with log frame column values for each trial.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'Song ID', 'Silence ID', and questionnaire fields
        (Familiarity, Liking, Fitting Category, Emotional State, Other Category)
        May have multiple rows per Song ID or Silence ID
    experiment_data_dir : str | Path
        Root directory containing trial subdirectories with questionnaire data
    error_handling : {'raise', 'continue'}
        How to handle missing questionnaire files
    verbose : bool
        Whether to print validation report to console

    Returns
    -------
    dict
        Validation report with:
        - 'valid': bool, True if all validations pass
        - 'song_matches': list of valid song trials with matching data
        - 'silence_matches': list of valid silence trials with matching data
        - 'mismatches': questionnaire data not matching log frame
        - 'missing_questionnaires': IDs unable to fetch questionnaire data
        - 'summary': summary statistics
    """

    report = {
        'valid': True,
        'song_matches': [],
        'silence_matches': [],
        'mismatches': [],
        'missing_questionnaires': [],
        'summary': {}
    }

    experiment_data_dir = Path(experiment_data_dir)

    # Separate song and silence entries
    song_df = df[df['Song ID'].notna()].copy() if 'Song ID' in df.columns else pd.DataFrame()
    silence_df = df[df['Silence ID'].notna()].copy() if 'Silence ID' in df.columns else pd.DataFrame()

    # Process Song IDs
    if not song_df.empty:
        grouped_songs = song_df.groupby('Song ID')

        for song_id, group in grouped_songs:
            song_id = int(song_id)

            # Fetch questionnaire data
            questionnaire_data = fetch_trial_questionnaire(
                experiment_data_dir,
                song_id=song_id,
                silence_id=None,
                error_handling=error_handling
            )

            # Handle missing questionnaire data
            if not questionnaire_data:
                report['valid'] = False
                report['missing_questionnaires'].append({
                    'Trial ID': song_id,
                    'Trial Type': 'Song',
                    'Number of Log Entries': len(group),
                    'Status': 'Could not fetch questionnaire data'
                })
                continue

            # Compare questionnaire data with log frame entries
            mismatches_for_trial = []

            for field_name, questionnaire_value in questionnaire_data.items():
                # Check if column exists in dataframe
                if field_name not in group.columns:
                    continue

                # Get unique values from log frame for this field
                log_values = group[field_name].unique()

                # Check if all entries match the questionnaire value
                if len(log_values) != 1 or log_values[0] != questionnaire_value:
                    mismatches_for_trial.append({
                        'Field': field_name,
                        'Questionnaire Value': questionnaire_value,
                        'Log Frame Values': list(log_values),
                        'Match': log_values[0] == questionnaire_value if len(log_values) == 1 else False
                    })

            # Check if there are mismatches
            if mismatches_for_trial:
                report['valid'] = False
                report['mismatches'].append({
                    'Song ID': song_id,
                    'Trial Type': 'Song',
                    'Number of Log Entries': len(group),
                    'Mismatches': mismatches_for_trial,
                    'Questionnaire Data': questionnaire_data
                })
            else:
                # All validations passed
                report['song_matches'].append({
                    'Song ID': song_id,
                    'Number of Log Entries': len(group),
                    'Status': 'Valid',
                    'Questionnaire Data': questionnaire_data
                })

    # Process Silence IDs
    if not silence_df.empty:
        grouped_silences = silence_df.groupby('Silence ID')

        for silence_id, group in grouped_silences:
            silence_id = int(silence_id)

            # Fetch questionnaire data
            questionnaire_data = fetch_trial_questionnaire(
                experiment_data_dir,
                song_id=None,
                silence_id=silence_id,
                error_handling=error_handling
            )

            # Handle missing questionnaire data
            if not questionnaire_data:
                report['valid'] = False
                report['missing_questionnaires'].append({
                    'Trial ID': silence_id,
                    'Trial Type': 'Silence',
                    'Number of Log Entries': len(group),
                    'Status': 'Could not fetch questionnaire data'
                })
                continue

            # Compare questionnaire data with log frame entries
            mismatches_for_trial = []

            for field_name, questionnaire_value in questionnaire_data.items():
                # Check if column exists in dataframe
                if field_name not in group.columns:
                    continue

                # Get unique values from log frame for this field
                log_values = group[field_name].unique()

                # Check if all entries match the questionnaire value
                if len(log_values) != 1 or log_values[0] != questionnaire_value:
                    mismatches_for_trial.append({
                        'Field': field_name,
                        'Questionnaire Value': questionnaire_value,
                        'Log Frame Values': list(log_values),
                        'Match': log_values[0] == questionnaire_value if len(log_values) == 1 else False
                    })

            # Check if there are mismatches
            if mismatches_for_trial:
                report['valid'] = False
                report['mismatches'].append({
                    'Silence ID': silence_id,
                    'Trial Type': 'Silence',
                    'Number of Log Entries': len(group),
                    'Mismatches': mismatches_for_trial,
                    'Questionnaire Data': questionnaire_data
                })
            else:
                # All validations passed
                report['silence_matches'].append({
                    'Silence ID': silence_id,
                    'Number of Log Entries': len(group),
                    'Status': 'Valid',
                    'Questionnaire Data': questionnaire_data
                })

    # Summary statistics
    report['summary'] = {
        'total_log_entries': len(df),
        'unique_song_ids': len(song_df.groupby('Song ID')) if not song_df.empty else 0,
        'unique_silence_ids': len(silence_df.groupby('Silence ID')) if not silence_df.empty else 0,
        'valid_song_matches': len(report['song_matches']),
        'valid_silence_matches': len(report['silence_matches']),
        'mismatches': len(report['mismatches']),
        'missing_questionnaires': len(report['missing_questionnaires']),
        'validation_passed': report['valid']
    }

    # Verbose output
    if verbose:
        if len(report['missing_questionnaires']) > 0:
            print(f"[WARNING] Couldn't find any questionnaires for {len(report['missing_questionnaires'])} trial(s):")
            for missing in report['missing_questionnaires']:
                print(f"  - {missing['Trial Type']} {missing['Trial ID']}: {missing['Status']}\n")

        if len(report['mismatches']) > 0:
            print(
                f"[WARNING] Found {len(report['mismatches'])} trial(s) with mismatches between log frame and questionnaire:")
            for mismatch in report['mismatches']:
                trial_id = mismatch.get('Song ID') or mismatch.get('Silence ID')
                print(f"  - {mismatch['Trial Type']} {trial_id} ({mismatch['Number of Log Entries']} entries):")
                for field_mismatch in mismatch['Mismatches']:
                    print(f"    {field_mismatch['Field']}: Questionnaire={field_mismatch['Questionnaire Value']}, "
                          f"Log Frame={field_mismatch['Log Frame Values']}\n")

        print(f"\nValidation Summary:")
        print(f"  Total log entries: {report['summary']['total_log_entries']}")
        print(f"  Unique song IDs: {report['summary']['unique_song_ids']}")
        print(f"  Unique silence IDs: {report['summary']['unique_silence_ids']}")
        print(f"  Valid song matches: {report['summary']['valid_song_matches']}")
        print(f"  Valid silence matches: {report['summary']['valid_silence_matches']}")
        print(f"  Mismatches: {report['summary']['mismatches']}")
        print(f"  Missing questionnaires: {report['summary']['missing_questionnaires']}")

        if report['valid']:
            print(f"\n✓ Validation passed!\n")
        else:
            print(f"\n✗ Validation failed!\n")

    return report


def repair_trial_questionnaire_mismatches(df: pd.DataFrame, questionnaire_validation_report: dict):
    df = df.copy()
    for mismatch_dict in questionnaire_validation_report['mismatches']:
        # fetch trial type and ID:
        song_id = mismatch_dict['Song ID'] if mismatch_dict['Trial Type'] == 'Song' else None
        silence_id = mismatch_dict['Silence ID'] if mismatch_dict['Trial Type'] == 'Silence' else None
        print(f"Correcting {f'song_{song_id:03}' if song_id is not None else f'silence_{silence_id}'} mismatch:")

        for mismatch_entry in mismatch_dict['Mismatches']:
            field = mismatch_entry['Field']
            true_value = mismatch_entry['Questionnaire Value']
            log_value= mismatch_entry['Log Frame Values']
            print(f"\t-> Will replace logframe '{field}' value {int(log_value[0]) if isinstance(log_value, list) else int(log_value)} with {true_value} from stored questionnaire jsons.")

            if song_id is not None:
                df.loc[df['Song ID'] == song_id, field] = true_value
            else:
                df.loc[df['Silence ID'] == silence_id, field] = true_value

    return df



def remove_silence_trial(enriched_log: pd.DataFrame, log: pd.DataFrame, silence_ids: list[int]) -> pd.DataFrame:
    """ Remove 'target frequency' entry in 'Questionnaire' during (clear all entries),
    so that repeated prepare_log_frame doesn't recognize silence trial. """
    log = log.copy()
    if isinstance(enriched_log.index, pd.DatetimeIndex): enriched_log = enriched_log.reset_index()

    for silence_id in silence_ids:
        row_mask = (enriched_log['Silence ID'] == silence_id)
        if len(log.loc[row_mask, :]) == 0:
            print(f"No entries found for silence ID {silence_id}")
            continue

        log.loc[row_mask, 'Questionnaire'] = np.nan

        print(f"Removed task information for silence trial with ID {silence_id}")

    return log



def remove_song_entries(enriched_log: pd.DataFrame, log: pd.DataFrame, song_title_artist_id_tuples: list[tuple[str, str, int]],
                        include_questionnaire_entries: bool = True) -> pd.DataFrame:
    """ Remove songs from log frame that were wrongly executed.
    include_questionnaire_entries = True is recommended, if a motor task was started. """
    log = log.copy()
    if isinstance(enriched_log.index, pd.DatetimeIndex): enriched_log = enriched_log.reset_index()

    for title, artist, id in song_title_artist_id_tuples:
        row_mask = (enriched_log['Song Title'] == title) & (enriched_log['Song Artist'] == artist) & (enriched_log['Song ID'] == id)
        if len(log.loc[row_mask, :]) == 0:
            print(f"No entries found for '{title}' by '{artist}'.")
            continue

        log.loc[row_mask, "Music"] = "No track playing currently."

        print(f"Removed music information for {title} and {artist}.")

        if include_questionnaire_entries:
            log.loc[row_mask, "Questionnaire"] = np.nan
            print(f"Removed also all questionnaire information for {title}.")

    return log

def remove_single_row_by_timestamp(log_frame: pd.DataFrame, timestamp: pd.Timestamp | str) -> pd.DataFrame:
    """ Remove a single flawed row from log frame. """
    log_frame = log_frame.copy()
    if len(log_frame[log_frame['Time'] == timestamp]) != 0:
        print(f"Removing row with timestamp '{timestamp}' from log frame.\n")

    log_frame = log_frame.drop(log_frame[log_frame['Time'] == timestamp].index)
    return log_frame



def annotate_trial(log_df, comment: str, exclude: bool,
                   song_id: int | None = None, silence_id: int | None = None, trial_id: int | None = None,
                   ):
    log_df = log_df.copy()
    if trial_id is None:
        trial_id = turn_song_or_silence_id_into_trial_id(log_df, song_id, silence_id)

    log_df.loc[log_df['Trial ID'] == trial_id, 'Trial Comment'] = comment
    log_df.loc[log_df['Trial ID'] == trial_id, 'Trial Exclusion Bool'] = exclude

    if exclude: print(f"Marked trial {trial_id} for exclusion due to '{comment}'.")
    else: print(f"Commented trial {trial_id} with '{comment}'.")

    return log_df




############################## DATABASE MGMT. METHODS ##############################
def fetch_serial_measurements(subject_data_dir: Path, load_only_first_n_seconds: int | None = None,
                              set_time_index: bool = True) -> pd.DataFrame:
    """
    Fetch most recent serial measurements from a subject directory.

    Concatenates "Interim Save WorkMem Full" files in order, then appends
    the latest "Final Save". Falls back to latest "Redundant Save" if no
    "Final Save" exists.

    Args:
        subject_data_dir: Path to subject data directory
        load_only_first_n_seconds: If provided, stops loading data after first n seconds
            based on first column's timestamps. Occurs before concatenation, so interim
            saves may suffice without loading final save.

    Returns:
        pd.DataFrame: Concatenated serial measurements with datetime-converted timestamp column,
            sorted by time ascending, duplicates removed

    Raises:
        ValueError: If no measurement files found matching criteria
    """
    measurements_dir = subject_data_dir / 'serial_measurements'

    # Helper function to load CSV, handle index column, and convert first real column to datetime
    def _load_and_convert_timestamps(path: Path) -> pd.DataFrame:
        """
        Load CSV and convert time column to datetime format.
        Assumes time column is the last 'Unnamed' column before real data.
        """
        df = pd.read_csv(path)

        # Find the last Unnamed column (time column)
        unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed')]
        if unnamed_cols:
            time_col = unnamed_cols[-1]
            df[time_col] = pd.to_datetime(df[time_col])
            # Drop other Unnamed columns (index artifacts)
            df = df.drop(columns=[col for col in unnamed_cols if col != time_col])
            df.rename(columns={time_col: 'Time'}, inplace=True)

        return df

    # Helper function to filter dataframe by time window
    def _filter_by_time_window(df: pd.DataFrame, n_seconds: int) -> pd.DataFrame:
        """Filter dataframe to only include first n seconds from start timestamp."""
        first_col = df.columns[0]
        start_time = df[first_col].min()
        cutoff_time = start_time + pd.Timedelta(seconds=n_seconds)
        return df[df[first_col] <= cutoff_time]

    # Helper function to sort by time and remove duplicates
    def _sort_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort rows by first (timestamp) column ascending and remove duplicates
        with equal timestamps, keeping first occurrence.
        """
        first_col = df.columns[0]
        df = df.sort_values(by=first_col, ascending=True)
        df = df.drop_duplicates(subset=[first_col], keep='first')
        return df.reset_index(drop=True)

    # Fetch interim saves: concatenate all "Interim Save WorkMem Full" in sorted order
    try:
        interim_frame_paths = filemgmt.most_recent_file(
            measurements_dir,
            ".csv",
            ["Interim Save WorkMem Full"],
            return_type='dict'
        )['files']
        interim_frames = [_load_and_convert_timestamps(path) for path in interim_frame_paths]
        print(
            f"Found {len(interim_frames)} working-memory-full measurements in {measurements_dir}. Will concatenate with final save.")
    except ValueError:
        interim_frames = []  # no interim frames found

    # Apply time window filter to interim frames if specified
    if load_only_first_n_seconds is not None and len(interim_frames) > 0:
        interim_frames = [_filter_by_time_window(df, load_only_first_n_seconds) for df in interim_frames]

        # Check if we have sufficient data from interim saves alone
        total_interim_duration = (interim_frames[-1][interim_frames[-1].columns[0]].max() -
                                  interim_frames[0][interim_frames[0].columns[0]].min()).total_seconds()
        if total_interim_duration >= load_only_first_n_seconds:
            print(
                f"Interim saves cover {total_interim_duration:.1f}s (requested: {load_only_first_n_seconds}s). Skipping final save.")
            final_frame = pd.DataFrame()  # empty frame to skip final save loading
        else:
            final_frame = None  # signal to load final save
    else:
        final_frame = None

    # Fetch final frame only if needed
    if final_frame is None:
        try:
            final_frame_path = filemgmt.most_recent_file(
                measurements_dir,
                ".csv",
                ["Final Save"]
            )
            final_frame = _load_and_convert_timestamps(final_frame_path)

            # Apply time window filter to final frame if specified
            if load_only_first_n_seconds is not None:
                final_frame = _filter_by_time_window(final_frame, load_only_first_n_seconds)

        except ValueError:  # if "Final Save" not found, use latest "Redundant Save"
            print(
                f"No 'Final Save' measurement file found in {measurements_dir}\nWill utilize last 'Redundant Save', leading to potential data loss...")
            final_frame_path = filemgmt.most_recent_file(
                measurements_dir,
                ".csv",
                ["Redundant Save"]
            )
            final_frame = _load_and_convert_timestamps(final_frame_path)

            # Apply time window filter to fallback frame if specified
            if load_only_first_n_seconds is not None:
                final_frame = _filter_by_time_window(final_frame, load_only_first_n_seconds)

    # Concatenate interim frames with final frame
    frames_to_concat = interim_frames + ([final_frame] if len(final_frame) > 0 else [])

    if len(frames_to_concat) > 0:
        result = pd.concat(frames_to_concat, ignore_index=True)
        # Sort by timestamp ascending and remove duplicates
        result = _sort_and_deduplicate(result)
    else:
        raise ValueError("No data loaded after applying filters!")

    if set_time_index:  # set time index if desired
        result = result.set_index("Time")

    return result


def fetch_trial_dir(experiment_data_dir: str | Path,
                              song_id: int | None = None, silence_id: int | None = None,
                    trial_id: int | None = None,
                    log_df: pd.DataFrame | None = None,
                    ) -> Path:
    """ Raises a ValueError if directory is missing. """
    if song_id is None and silence_id is None and trial_id is None:
        raise ValueError("Either song_id, silence_id or trial_id must be specified to derive respective trial!")

    if trial_id is not None:
        if log_df is None: raise ValueError("log_df must be specified if trial_id is not None")
        song_id, silence_id = turn_trial_id_into_song_or_silence_id(log_df, trial_id)

    # fetch respective directory
    dir_name = f"song_{song_id:03}" if song_id is not None else f"silence_{silence_id:03}"
    trial_dir = Path(experiment_data_dir) / dir_name

    # check whether present:
    if trial_dir.is_dir: return trial_dir
    else: raise FileNotFoundError(f"Trial directory {trial_dir} not found.")


def fetch_trial_questionnaire(experiment_data_dir: str | Path,
                              song_id: int | None = None, silence_id: int | None = None,
                              error_handling: Literal['raise', 'continue'] = 'continue',
                              verbose: bool = False,
                              ) -> dict:
    """ Familiarity check, liking, fitting category, alternative category and emotional state. """
    trial_dir = fetch_trial_dir(experiment_data_dir, song_id, silence_id, )

    output_dict = {}

    # try fetching familiarity:
    if song_id is not None:  # only for song id
        try:
            familiarity_json_path = filemgmt.most_recent_file(trial_dir, ".json", ["Familiarity Check"])
            with open(familiarity_json_path, "r") as f:
                output_dict.update(json.load(f))

        except ValueError:  # if first questionnaire wasn't found
            msg = f"Couldn't find familiarity questionnaire for {f'song_{song_id:03}'}. Perhaps trial was skipped even before that."
            if error_handling == 'raise':
                raise ValueError(msg)
            else:
                if verbose: print(msg)
                return output_dict

    # fetch post trial data:
    try:
        post_questionnaire_path = filemgmt.most_recent_file(trial_dir, ".json", ["Post-Trial Rating"])
        with open(post_questionnaire_path, "r") as f:
            output_dict.update(json.load(f))
    except ValueError:  # if post questionnaire wasn't found
        msg = f"Couldn't find post-trial questionnaire for {f'song_{song_id:03}' if song_id is not None else f'silence_{silence_id:003}'}. Perhaps trial was skipped."
        if error_handling == 'raise':
            raise ValueError(msg)
        else:
            if verbose: print(msg)

    return output_dict


def fetch_trial_accuracy(experiment_data_dir: str | Path,
                         song_id: int | None = None, silence_id: int | None = None,
                         log_df: pd.DataFrame | None = None,
                         trial_id : int | None = None,
                         error_handling: Literal['raise', 'continue'] = 'continue',
                         verbose: bool = False,
                         ) -> np.ndarray | None:
    trial_dir = fetch_trial_dir(experiment_data_dir, song_id, silence_id, trial_id, log_df)

    # try fetching accuracy:
    try:
        accuracy_path = filemgmt.most_recent_file(trial_dir, ".csv", ["Trial Accuracy Results"])
        return pd.read_csv(accuracy_path).iloc[:, -1].to_numpy()

    except ValueError:  # if file not found
        msg = f"Couldn't find accuracy results for {f'song_{song_id:03}' if song_id is not None else f'silence{silence_id:003}'}. Perhaps trial was skipped."
        if error_handling == 'raise':
            raise ValueError(msg)
        else:
            if verbose: print(msg)
            return None


def fetch_all_accuracies_and_questionnaires(experiment_data_dir: str | Path,
                                            max_song_ind: int, max_silence_ind: int,
                                            verbose: bool = False,
                                            ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """ Returns tuple accuracy_per_trial_dict and questionnaire_per_trial_dict. Leverages above two methods. """
    # fetch all accuracies:
    accuracy_per_trial_dict: dict[str, np.ndarray] = {
        f"song_{song_id:03}": fetch_trial_accuracy(experiment_data_dir, song_id=song_id, error_handling='continue') for
        song_id in range(max_song_ind)}
    accuracy_per_trial_dict.update({f"silence_{silence_id:03}": fetch_trial_accuracy(experiment_data_dir,
                                                                                     silence_id=silence_id,
                                                                                     error_handling='continue',
                                                                                     verbose=verbose) for
                                    silence_id in range(max_silence_ind)})

    # fetch all questionnaires:
    questionnaire_per_trial_dict: dict[str, dict] = {
        f"song_{song_id:03}": fetch_trial_questionnaire(experiment_data_dir, song_id=song_id, error_handling='continue')
        for song_id in range(max_song_ind)}
    questionnaire_per_trial_dict.update({f"silence_{silence_id:03}": fetch_trial_questionnaire(experiment_data_dir,
                                                                                               silence_id=silence_id,
                                                                                               verbose=verbose,
                                                                                               error_handling='continue')
                                         for silence_id in range(max_silence_ind)})

    return accuracy_per_trial_dict, questionnaire_per_trial_dict


def fetch_song_information(experiment_data_dir: str | Path,
                         song_id: int | None = None,
                         error_handling: Literal['raise', 'continue'] = 'continue',
                         ) -> np.ndarray | None:
    trial_dir = fetch_trial_dir(experiment_data_dir, song_id, silence_id=None, )

    # try fetching song info:
    try:
        song_json_path = filemgmt.most_recent_file(trial_dir, ".json", ["song", "information"])
        with open(song_json_path, "r") as f:
            song_dict = json.load(f)
        return song_dict

    except ValueError:  # if file not found
        msg = f"Couldn't find song information for {f'song_{song_id:03}'}!"
        if error_handling == 'raise':
            raise ValueError(msg)
        else:
            print(msg)
            return None


def fetch_onboarding_questionnaire(experiment_data_dir: str | Path,) -> dict:
    onboarding_filepath = filemgmt.most_recent_file(experiment_data_dir, ".json", ["Subject", "Data"])

    with open(onboarding_filepath, "r") as f:
        onboarding_dict = json.load(f)

    return onboarding_dict


def fetch_offboarding_questionnaire(experiment_data_dir: str | Path,) -> dict:
    offboarding_filepath = filemgmt.most_recent_file(experiment_data_dir, ".json", ["Post-Study Feedback Data"])

    with open(offboarding_filepath, "r") as f:
        offboarding_dict = json.load(f)

    return offboarding_dict


def fetch_excluded_trials(enriched_log_df: pd.DataFrame) -> list[int]:
    excluded_trials: list[int] = []
    if enriched_log_df['Trial Exclusion Bool'].any():
        for trial_id in range(int(enriched_log_df['Trial ID'].max()) + 1):
            if enriched_log_df.loc[enriched_log_df['Trial ID'] == trial_id, 'Trial Exclusion Bool'].any():
                excluded_trials.append(int(trial_id))

    return excluded_trials


def fetch_skipped_trials(enriched_log_df: pd.DataFrame) -> list[int]:
    skipped_trials: list[int] = []
    if enriched_log_df['Song Skipped'].any():
        for trial_id in range(int(enriched_log_df['Trial ID'].max()) + 1):
            if enriched_log_df.loc[enriched_log_df['Trial ID'] == trial_id, 'Song Skipped'].any():
                skipped_trials.append(trial_id)

    return skipped_trials


def fetch_enriched_serial_frame(experiment_data_dir: str | Path, set_time_index: bool = True,
                                #verbose: bool = True,
                                ) -> pd.DataFrame:
    serial_dir = experiment_data_dir / "serial_measurements"

    try:
        serial_path = filemgmt.most_recent_file(serial_dir, ".csv", ["Enriched Serial Frame"])
        serial_frame = pd.read_csv(serial_path)
        if set_time_index:
            serial_frame['Time'] = pd.to_datetime(serial_frame['Time'], format='ISO8601')
            serial_frame = serial_frame.set_index("Time")
    except ValueError:
        raise ValueError(
            f"Couldn't find enriched (integrated) serial frame with signature 'Enriched Serial Frame' in file title within {serial_dir}...\nPlease ensure to run feature_extraction_workflow.py on subject data beforehand.")

    return serial_frame


def fetch_enriched_log_frame(experiment_data_dir: str | Path, set_time_index: bool = True,
                             verbose: bool = True) -> pd.DataFrame:
    log_dir = experiment_data_dir / "experiment_logs"
    try:
        log_path = filemgmt.most_recent_file(log_dir, ".csv", ["Enriched Experiment Log"])
        log_frame = pd.read_csv(log_path)
        if set_time_index:
            log_frame['Time'] = pd.to_datetime(log_frame['Time'])
            log_frame = log_frame.set_index("Time")
    except ValueError:
        raise ValueError(f"Couldn't find enriched (integrated) experiment log frame with signature 'Enriched Experiment Log' in file title within {log_dir}...\nPlease ensure to run data_integration_workflow.py on subject data beforehand.")


    if verbose:
        print(f"Imported enriched log frame from {experiment_data_dir}:\n")
        qtc_start, qtc_end = get_qtc_measurement_start_end(log_frame, False)
        print(f"- Duration of EEG/EMG measurements: {(qtc_end - qtc_start).total_seconds():.2f} seconds ({qtc_start} to {qtc_end})")
        # trial info:
        print(f"- Number of trials {int(log_frame['Trial ID'].max()+1)} ({int(log_frame['Song ID'].max()+1)} music, {int(log_frame['Silence ID'].max()+1)} silence)")
        excluded_trials = fetch_excluded_trials(log_frame)
        if len(excluded_trials) > 0:
            print(f"- Thereof {len(excluded_trials)} trial(s) marked for exclusion: {excluded_trials}")
        skipped_trials = fetch_skipped_trials(log_frame)
        if len(skipped_trials) > 0:
            print(f"- Thereof {len(skipped_trials)} trial(s) skipped: {skipped_trials}")
        if len(skipped_trials) > 0 or len(excluded_trials) > 0:
            print(f"- Remaining valid trials: {int(log_frame['Trial ID'].max()+1) - len(skipped_trials) - len(excluded_trials)}")

        # music info:
        print(f"- Valid trial list:")
        for trial_id in range(int(log_frame['Trial ID'].max()+1)):
            subset = log_frame.loc[log_frame['Trial ID'] == trial_id, :]
            if subset['Trial Exclusion Bool'].any(): continue
            if subset['Song Skipped'].any(): continue
            else:
                if subset['Silence ID'].notna().any():
                    category = 'Silence'
                    familiarity = None
                else:
                    category = subset['Music Category'].iloc[0]
                    familiarity = subset['Familiarity'].dropna().iloc[0]
                rmse = subset['Task RMSE'].max()

                print(f"\t-> Trial {trial_id:<3}:\t{category:<20}\t\twith RMSE: {rmse:.2f}{f'\t\t\tand familarity: {int(familiarity)}' if familiarity is not None else ''}")
        print("\n")


    return log_frame


def fetch_music_features(log_df: pd.DataFrame, music_lookup_table_path: str | Path | None = None,
                         song_id: int | None = None, trial_id: int | None = None,
                         features_to_return: tuple[str, ...] = ('BPM_manual', 'Spectral Flux Mean',
                                                          'Spectral Centroid Mean', 'IOI Variance Coeff',
                                                           'Syncopation Ratio'),
                         ) -> list[float]:
    if music_lookup_table_path is None:  # use hardcoded definition if not provided
        music_lookup_dir = Path().resolve().parent / "data" / "song_characteristics"
        music_lookup_table_path = filemgmt.most_recent_file(music_lookup_dir, ".csv", ["Lookup Table"])

    # read lookup frame:
    lookup_frame = pd.read_csv(music_lookup_table_path)

    # fetch song id:
    if song_id is None and trial_id is None: raise ValueError("Must provide either song or trial ID")
    elif song_id is None:
        song_id, silence_id = turn_trial_id_into_song_or_silence_id(log_df, trial_id)
    # if Song ID is None still, it's no music trial -> return nans
    if song_id is None: return [np.nan] * len(features_to_return)

    # fetch song data: (Song Title and Song Artist)
    subset_log_df = log_df.loc[log_df['Song ID'] == song_id, ['Song Title', 'Song Artist']]
    if len(subset_log_df) == 0: raise ValueError(f"Couldn't find song_id {song_id} in log_frame table...")
    title = subset_log_df['Song Title'].iloc[0]
    artist = subset_log_df['Song Artist'].iloc[0]

    # fetch features from lookup table:
    song_row = lookup_frame.loc[(lookup_frame['Artist'] == artist) & (lookup_frame['Title'] == title), :]
    if len(song_row) == 0: raise ValueError(f"Song {title} not found in lookup table")
    elif len(song_row) > 1: raise ValueError(f"Song {title} found multiple times in lookup table. Needs to be unique.")

    # read out and return
    output_list = []
    for feature in features_to_return:
        output_list.append(song_row[feature].item())
    return output_list