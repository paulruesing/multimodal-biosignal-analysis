import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import re
import json
from pathlib import Path
from typing import Literal

import src.utils.file_management as filemgmt


############################## LOG FRAME HANDLING ##############################
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
    def add_task_freqs_and_average_rmse(df: pd.DataFrame) -> pd.DataFrame:
        """ Based on Questionnaire (given) """
        # Step 1: Extract frequency and RMSE values
        df['Task Frequency'] = df['Questionnaire'].str.extract(
            r'target frequency ([\d.]+)Hz',
            expand=False
        )
        df['Task Avg. RMSE'] = df['Questionnaire'].str.extract(
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
        df['Task Avg. RMSE'] = df.groupby('task_id')['Task Avg. RMSE'].bfill()

        # Step 5: Create mask to clear frequency values AFTER the RMSE row
        is_end = df['Questionnaire'].str.contains('Achieved RMSE', na=False)
        # rows_after_end computation:
        # 1) shift is_end row by 1 (default) and fill first with false
        # 2) groupby -> within a task ID, cum_sum (= forward fill) the is_end marker (somehow ffill() doesn't work correctly here)
        # 3) select the rows after end (the forward filled ones) by > 0
        rows_after_end = is_end.shift(fill_value=False).groupby(df['task_id']).cumsum() > 0
        # set those frequencies (when task is over) to np.nan
        df.loc[rows_after_end, 'Task Frequency'] = np.nan

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



    ############## Add placeholder columns ##############
    log_frame['Trial Comment'] = [""] * len(log_frame)
    log_frame['Trial Exclusion Bool'] = [False] * len(log_frame)
    log_frame['Trial Exclusion Bool'] = log_frame['Trial Exclusion Bool'].astype('boolean')
    log_frame.loc[log_frame['Trial ID'].isna(), 'Trial Exclusion Bool'] = np.nan


    ############### Format and Return ###############
    if set_time_index:
        log_frame = log_frame.set_index('Time')

    return log_frame


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

        if (subset_df['Song Skipped']).any():
            print(f"[INFO] Song {int(song_id)} got skipped, no corresponding task was executed.")

        # reduce to task time (where Task Frequency is not na):
        subset_df = subset_df.loc[~subset_df['Task Frequency'].isna()]

    else:
        subset_df = df.loc[df['Silence ID'] == silence_id]

    if len(subset_df) == 0:
        raise ValueError("Specific task not found!")

    # retrieve timestamps:
    if isinstance(subset_df.index, pd.DatetimeIndex):
        times = subset_df.index
    elif 'Time' in subset_df.columns:
        times = pd.to_datetime(subset_df['Time'])
    else:
        raise ValueError('df must contain "Time" column or DatetimeIndex!')

    return times.min(), times.max()


def get_qtc_measurement_start_end(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.Timestamp, pd.Timestamp]:
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
                f"[WARNING] Couldn't find matching directories for {len(report['missing_metadata'])} songs:")
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


############################## DATA MANIPULATION / INTEGRATION METHODS ##############################
def apply_window_operator(
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





############################## DATABASE MGMT. METHODS ##############################
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


# todo: use this, to validate / overrule log frame entries
def fetch_trial_questionnaire(experiment_data_dir: str | Path,
                              song_id: int | None = None, silence_id: int | None = None,
                              error_handling: Literal['raise', 'continue'] = 'continue',
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
                print(msg)
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
        else: print(msg)

    return output_dict


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


def fetch_trial_accuracy(experiment_data_dir: str | Path,
                         song_id: int | None = None, silence_id: int | None = None,
                         log_df: pd.DataFrame | None = None,
                         trial_id : int | None = None,
                         error_handling: Literal['raise', 'continue'] = 'continue',
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
            print(msg)
            return None


def fetch_all_accuracies_and_questionnaires(experiment_data_dir: str | Path,
                                            max_song_ind: int, max_silence_ind: int,
                                            ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """ Returns tuple accuracy_per_trial_dict and questionnaire_per_trial_dict. Leverages above two methods. """
    # fetch all accuracies:
    accuracy_per_trial_dict: dict[str, np.ndarray] = {
        f"song_{song_id:03}": fetch_trial_accuracy(experiment_data_dir, song_id=song_id, error_handling='continue') for
        song_id in range(max_song_ind)}
    accuracy_per_trial_dict.update({f"silence_{silence_id:03}": fetch_trial_accuracy(experiment_data_dir,
                                                                                     silence_id=silence_id,
                                                                                     error_handling='continue') for
                                    silence_id in range(max_silence_ind)})

    # fetch all questionnaires:
    questionnaire_per_trial_dict: dict[str, dict] = {
        f"song_{song_id:03}": fetch_trial_questionnaire(experiment_data_dir, song_id=song_id, error_handling='continue')
        for song_id in range(max_song_ind)}
    questionnaire_per_trial_dict.update({f"silence_{silence_id:03}": fetch_trial_questionnaire(experiment_data_dir,
                                                                                               silence_id=silence_id,
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



# todo: implement
def fetch_onboarding_questionnaire(experiment_data_dir: str | Path,) -> dict:
    pass


# todo: implement
def fetch_offboarding_questionnaire(experiment_data_dir: str | Path,) -> dict:
    pass






""" FUNCTION GRAVEYARD

        def slow_info_per_window_sec(
                window_time_centers: np.ndarray,
                window_size: float,
                target_series: pd.DataFrame,
                operation: Literal['min', 'max', 'mean', 'median', 'mode', 'std'] = 'mean',
        ) -> list:
            ''' Target series needs to have time index (seconds or absolute). '''
            starts = window_time_centers - window_size / 2
            ends = window_time_centers + window_size / 2

            # derive time in seconds from time index:
            if isinstance(target_series.index, pd.DatetimeIndex):
                time_seconds = (target_series.index - target_series.index[0]).total_seconds().to_series().reset_index(drop=True)
            else:
                time_seconds = target_series.index.to_series().reset_index(drop=True)

            # compute target per window:
            target_list = []
            for ind, (start, end) in enumerate(zip(starts, ends)):
                relevant_slice = target_series.reset_index(drop=True).loc[(time_seconds >= start) & (time_seconds < end)]

                # simple:
                if len(relevant_slice) == 0: target = target_list[ind - 1]  # if no relevant slice within window, take previous
                elif len(relevant_slice) == 1: target = relevant_slice.item()

                #
                elif operation == 'min': target = relevant_slice.min()
                elif operation == 'max': target = relevant_slice.max()
                elif operation == 'mean': target = relevant_slice.mean()
                elif operation == 'median': target = relevant_slice.median()
                elif operation == 'mode': target = relevant_slice.mode().item()
                elif operation == 'std': target = relevant_slice.std()
                else: raise ValueError('Unknown operation {}'.format(operation))

                target_list.append(target)

            return target_list

"""