import pandas as pd
import numpy as np
import re


############################## LOG FRAME HANDLING ##############################
def prepare_log_frame(log_frame: pd.DataFrame, set_time_index: bool = True) -> pd.DataFrame:
    ############### Derive Values from Status Strings ###############
    def derive_song_category_string(input: str) -> str:
        elements = input.split(" | ")
        if len(elements) == 2:  # no category entry
            return "No category"
        elif len(elements) == 3:
            return elements[0]
        else:
            return "No song playing"

    category_string_series = log_frame['Music'].apply(derive_song_category_string)


    def derive_category(input: str) -> str:
        elements = input.split(" (")
        if len(elements) == 1:
            return "No category"
        else:
            return elements[0]

    log_frame['Music Category'] = category_string_series.apply(derive_category)


    def derive_category_index(input: str) -> int | None:

        elements = input.split(" (")
        if len(elements) == 1:
            return None
        else:
            return int(elements[1].split("/")[0])  # structure is eg CATEGORY (1/11) -> would yield 1

    log_frame['Music Category Index'] = category_string_series.apply(derive_category_index)


    def derive_song_title_string(input: str) -> str:
        elements = input.split(" | ")
        if len(elements) == 2:  # no category entry
            return elements[0]
        elif len(elements) == 3:
            return elements[1]
        else:
            return "No song playing"

    log_frame['Song Info'] = log_frame['Music'].apply(derive_song_title_string)


    def derive_song_runtime(input: str) -> float | None:
        elements = input.split(" | ")
        if len(elements) == 2:  # no category entry
            return float(elements[1].split(" / ")[0].split("s")[0])
        elif len(elements) == 3:
            return float(elements[2].split(" / ")[0].split("s")[0])
        else:
            return None

    log_frame['Song Runtime'] = log_frame['Music'].apply(derive_song_runtime)


    ############### Extract Values and Extend ###############
    def add_task_freqs_and_average_rmse(df: pd.DataFrame) -> pd.DataFrame:
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


    ############### Derive and Extend Questionnaire Results ###############
    # happens here, because leverages song IDs
    def add_questionnaire_results(df: pd.DataFrame) -> pd.DataFrame:
        df['Questionnaire'] = df['Questionnaire'].astype(str)

        # derive familiarity:
        def extract_familiarity(text):
            match = re.search(r'Familiarity check result:\s*(\d)', text)
            return float(match.group(1)) if match else np.nan
        df['Song Familiarity'] = df['Questionnaire'].apply(extract_familiarity)

        # extend values across equal Song ID:
        df['Song Familiarity'] = df.groupby('Song ID')['Song Familiarity'].ffill()
        df['Song Familiarity'] = df.groupby('Song ID')['Song Familiarity'].bfill()

        # derive post trial questionnaire:
        def extract_post_trial(text):
            import ast
            try:
                # Extract just the dict portion
                match = re.search(r"\{.*\}", text)
                if match:
                    dict_str = match.group(0)
                    # Convert from Python string representation to actual dict
                    data = ast.literal_eval(dict_str)

                    if 'Liking' not in data.keys(): data['Liking'] = np.nan
                    if 'Fitting Category' not in data.keys(): data['Fitting Category'] = np.nan
                    if 'Other category' not in data.keys(): data['Other category'] = np.nan
                    # no need to check for Emotional State, always asked
                    return data
            except:
                pass

            return {'Liking': np.nan, 'Fitting Category': np.nan, 'Emotional State': np.nan, 'Other category': np.nan}

        post_trial_data = df['Questionnaire'].apply(extract_post_trial)
        df['Liking'] = post_trial_data.apply(lambda x: x['Liking'])
        df['Fitting Category'] = post_trial_data.apply(lambda x: x['Fitting Category'])
        df['Emotional State'] = post_trial_data.apply(lambda x: x['Emotional State'])
        df['Other category'] = post_trial_data.apply(lambda x: x['Other category'])


        # extend values
        for key in ['Liking', 'Fitting Category', 'Emotional State', 'Other category']:
            reverse_filled_result = df[key].bfill()
            reverse_filled_result.loc[df['Song ID'].isna()] = np.nan
            df[key] = reverse_filled_result

        return df

    log_frame = add_questionnaire_results(log_frame)


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
        song_title_series = df['Song Info'].str.extract(r'(.*?)\s+by\s+', expand=False)
        subset_df = df.loc[song_title_series == song_title]

        # prevent catching multiple sequences:
        unique_song_ids = subset_df['Song ID'].dropna().unique().astype(int)
        if len(unique_song_ids) > 1:  # if matching song title has more than one ID
            raise ValueError(f"Song title appeared multiple times with Song IDs: {unique_song_ids.tolist()}\nChoose one and call this method with song_id!")

    if (subset_df['Song Skipped']).any():
        print(f"[WARNING] This song got skipped, no corresponding task was executed.")

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
                       silence_id: int | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    if song_id is None and song_title is None and silence_id is None:
        raise ValueError("Either song_id, song_title or silence_id must be specified")

    # locate song trial:
    if song_id is not None or song_title is not None:
        if song_id is not None:
            subset_df = df.loc[df['Song ID'] == song_id]
        else:
            song_title_series = df['Song Info'].str.extract(r'(.*?)\s+by\s+', expand=False)
            subset_df = df.loc[song_title_series == song_title]

            # prevent catching multiple sequences:
            unique_song_ids = subset_df['Song ID'].dropna().unique().astype(int)
            if len(unique_song_ids) > 1:  # if matching song title has more than one ID
                raise ValueError(f"Song title appeared multiple times with Song IDs: {unique_song_ids.tolist()}\nChoose one and call this method with song_id!")

        if (subset_df['Song Skipped']).any():
            print(f"[WARNING] This song got skipped, no corresponding task was executed.")

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
    subset_df = df.copy()  # copied here because we change the index below

    # convert to datetime index timestamps:
    if isinstance(subset_df.index, pd.DatetimeIndex):
        pass
    elif 'Time' in subset_df.columns:
        subset_df['Time'] = pd.to_datetime(subset_df['Time'])
        subset_df.set_index('Time', inplace=True)
    else:
        raise ValueError('df must contain "Time" column or DatetimeIndex!')

    try:  # retrieve start of QTC measurement
        qtc_start = subset_df.loc[df['Event'] == "Start Trigger"].index.item()
    except ValueError:  # if 'Start Trigger' not found
        print("No 'Start Trigger' event found, assuming measurement started upon beginning")
        qtc_start = subset_df.index.min()

    try:  # retrieve end
        qtc_end = subset_df.loc[df['Event'] == "Stop Trigger"].index.item()
    except ValueError:  # if 'Stop Trigger' not found
        print("No 'Stop Trigger' event found, assuming measurement ran until end.")
        qtc_end = subset_df.index.max()

    if verbose: print(f"EEG and EMG measurements last from {qtc_start} to {qtc_end}!")

    return qtc_start, qtc_end
