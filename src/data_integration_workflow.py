import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Callable, Literal, Tuple

from pandas import DatetimeIndex
from torch.backends.mkl import verbose

import src.pipeline.signal_features as features
import src.pipeline.preprocessing as preprocessing
import src.pipeline.visualizations as visualizations
import src.pipeline.data_integration as data_integration
import src.utils.file_management as filemgmt

from src.pipeline.preprocessing import BiosignalPreprocessor
from src.pipeline.visualizations import EEG_CHANNEL_IND_DICT, EEG_CHANNELS_BY_AREA


if __name__=="__main__":
    ######## PREPARATION #########
    ROOT = Path().resolve().parent
    OUTPUT = ROOT / 'output'
    STUDY_PLOTS = OUTPUT / 'plots' / 'data_analysis_plots'
    EXPERIMENT_DATA = ROOT / "data" / "experiment_results"

    ### WORKFLOW CONTROL
    subject_ind = 7
    save_result: bool = True  # only set to True if manual adjustments finalized

    # experiment results import behaviour:
    subject_plot_dir = STUDY_PLOTS / f"subject_{subject_ind:02}"
    subject_experiment_data = EXPERIMENT_DATA / f"subject_{subject_ind:02}"  # should have folders experiment_logs/, serial_measurements/, song_000/, ...



    ################## IMPORT ##################
    ### Import:
    # load experiment files:
    log_frame = data_integration.fetch_experiment_log(subject_experiment_data)
    data_integration.get_qtc_measurement_start_end(log_frame, verbose=True)
    # log columns: ['Time', 'Music', 'Event', 'Questionnaire']
    serial_frame = data_integration.fetch_serial_measurements(subject_experiment_data)
    # serial columns: ['Time', 'fsr', 'ecg', 'gsr']

    # enrich log frame columns based on 'Event' and 'Questionnaire' data:
    enriched_log_frame = data_integration.prepare_log_frame(log_frame, set_time_index=True)



    ################## SUBJECT-SPECIFIC AMENDMENTS ##################
    if subject_ind == 0:
        pass


    elif subject_ind == 1:
        log_frame = data_integration.remove_song_entries(
            enriched_log_frame, log_frame,
            song_title_artist_id_tuples=[("Ain't No Sunshine", "Bill Withers", 17),
                                         ("Merry-Go-Round of Life - from 'Howl's Moving Castle'", "Joe Hisaishi", 21),
                                         ("As", "George Michael", 24),
                                         ("Dancing In the Dark", "Bruce Springsteen", 28)
                                         ])
        enriched_log_frame = data_integration.prepare_log_frame(log_frame, set_time_index=True)
        enriched_log_frame = data_integration.annotate_trial(enriched_log_frame,
                                                          "Talking and frustration because of briefly stuck measurement",
                                                          True, trial_id=11)

        enriched_log_frame = data_integration.annotate_trial(enriched_log_frame,
                                                          "Talking and frustration because of briefly stuck measurement",
                                                          True, trial_id=15)


    elif subject_ind == 2:
        log_frame = data_integration.remove_song_entries(enriched_log_frame, log_frame,
                                                      [("I Say a Little Prayer", "Aretha Franklin", 0),
                                                       ("Celebration", "Kool & The Gang", 1),
                                                       ("Uptown Funk (feat. Bruno Mars)", "Mark Ronson", 2)])
        #log_frame = data_integration.remove_silence_trial(enriched_log_frame, log_frame, silence_ids=[0, 1, 2])
        enriched_log_frame = data_integration.prepare_log_frame(log_frame, set_time_index=True)

        # mark some trials:
        enriched_log_frame = data_integration.annotate_trial(enriched_log_frame,
                                                          "Flawed Dynamometer Measurement and Corresponding Talking",
                                                          True, trial_id=0)
        enriched_log_frame = data_integration.annotate_trial(enriched_log_frame,
                                                          "Flawed Dynamometer Measurement and Corresponding Talking",
                                                          True,
                                                          trial_id=1)
        enriched_log_frame = data_integration.annotate_trial(enriched_log_frame,
                                                          "Flawed Dynamometer Measurement and Corresponding Talking",
                                                          True,
                                                          trial_id=2)
        enriched_log_frame = data_integration.annotate_trial(enriched_log_frame,
                                                          "Flawed Dynamometer Measurement and Corresponding Talking",
                                                          True,
                                                          trial_id=4)
        enriched_log_frame = data_integration.annotate_trial(enriched_log_frame,
                                                          "Flawed Dynamometer Measurement and Corresponding Talking",
                                                          True,
                                                          trial_id=5)

        # mark idle state recording:
        enriched_log_frame.loc[pd.Timestamp("2026-01-17 21:05:20"):, 'Phase'] = 'Idle State'


        # mark delayed EEG/EMG recordings start: "Actual Start Trigger" after 20 minutes (since before was much talking, also within task)
        qtc_start, _ = data_integration.get_qtc_measurement_start_end(log_frame)

        first_idx = enriched_log_frame.loc[qtc_start + pd.Timedelta(minutes=15):].index[0]
        enriched_log_frame.loc[first_idx, 'Event'] = 'Actual Start Trigger'

        data_integration.get_qtc_measurement_start_end(enriched_log_frame, verbose=True)


    elif subject_ind == 3:
        log_frame = data_integration.remove_song_entries(enriched_log_frame, log_frame,
                                                      [("Merry-Go-Round of Life - from 'Howl's Moving Castle'",
                                                        "Joe Hisaishi", 2),
                                                       ("Never Too Much", "Luther Vandross", 14)])
        log_frame = data_integration.remove_single_row_by_timestamp(log_frame, timestamp = '2026-01-22 18:59:30.676946')
        enriched_log_frame = data_integration.prepare_log_frame(log_frame, set_time_index=True)

        # mark idle state recording:
        enriched_log_frame.loc[pd.Timestamp("2026-01-22 19:08:00"):, 'Phase'] = 'Idle State'


    elif subject_ind == 4:
        log_frame = data_integration.remove_song_entries(
            enriched_log_frame, log_frame,
            song_title_artist_id_tuples=[("Can't Get Enough! - Vocal Club Mix", "Soulsearcher", 8),])
        enriched_log_frame = data_integration.prepare_log_frame(log_frame, set_time_index=True)

        enriched_log_frame = data_integration.annotate_trial(enriched_log_frame, "Talking", False, song_id=8)
        # dont exclude the above, since we do not have THAT many measurements and it was only a brief comment
        enriched_log_frame = data_integration.annotate_trial(enriched_log_frame, "Talking and then was repeated anyways", True, silence_id=1)

        # mark idle state recording:
        enriched_log_frame.loc[pd.Timestamp("2026-01-23 17:56:00"):, 'Phase'] = 'Idle State'


    elif subject_ind == 5:

        # song start wasn't immediately registered:
        log_frame = data_integration.remove_single_row_by_timestamp(log_frame, "2026-01-27 16:22:35.172122")

        log_frame = data_integration.remove_song_entries(
            enriched_log_frame, log_frame,
            song_title_artist_id_tuples=[
                ("Comptine d\'un autre été, l\'après-midi", "Yann Tiersen", 4),  # jumped somehow...
                ("Guilty - 2001 Remastered Version", "George Shearing", 6),  # because previous wrong song ends too early
                ("For You - Original Radio Edit", "The Disco Boys", 12),
                ("Crying at the Discoteque - Radio Edit", "Alcazar", 15),
                ("Mas Que Nada", "Sérgio Mendes", 19),
                ("Can\'t Get You out of My Head", "Kylie Minogue", 22)
            ])

        enriched_log_frame = data_integration.prepare_log_frame(log_frame,)

        # mark some trials: (dynamometer jumped somehow)
        enriched_log_frame = data_integration.annotate_trial(enriched_log_frame,
                                                             "Flawed Dynamometer Measurement and Corresponding Talking",
                                                             True, trial_id=1)
        enriched_log_frame = data_integration.annotate_trial(enriched_log_frame,
                                                             "Flawed Dynamometer Measurement and Corresponding Talking",
                                                             True, trial_id=2)
        enriched_log_frame = data_integration.annotate_trial(enriched_log_frame,
                                                             "Wrong Song Playing",
                                                             False,  # maybe still keep
                                                             trial_id=5)
        enriched_log_frame = data_integration.annotate_trial(enriched_log_frame,
                                                             "Flawed Dynamometer Measurement and Corresponding Talking",
                                                             True, trial_id=11)

        enriched_log_frame.loc[pd.Timestamp("2026-01-27 16:54:00"):, 'Phase'] = 'Idle State'


    elif subject_ind == 6:
        # log_frame = data_integration.remove_single_row_by_timestamp(log_frame, "2026-01-27 16:22:35.172122")

        log_frame = data_integration.remove_song_entries(
            enriched_log_frame, log_frame,
            song_title_artist_id_tuples=[
                # spotify sometimes skips to the next song, I however catched all occurences and remove them below:
                ("Merry-Go-Round of Life - from 'Howl's Moving Castle'", "Joe Hisaishi", 0),  # jumped somehow...
                ("Mas Que Nada", "Sérgio Mendes", 11),
                #("Lamento (No Morro)", "Sérgio Mendes", 12),
                ("Can't Get Enough! - Vocal Club Mix", "Soulsearcher", 14),
                #("Can't Get Enough (Robbie's Filtered Monster Anthem Mix)", "Soulsearcher", 15),
                ("Something Got Me Started - 2008 Remaster", "Simply Red", 17),
                #("Thrill Me - Live in Hamburg, 1992", "Simply Red", 18),
                ("I Was Made For Lovin' You", "KISS", 20),
                ("Waiting For Godard - Full Mix", "Marco Andrea Pes", 25),
            ])

        enriched_log_frame = data_integration.prepare_log_frame(log_frame, )

        # some dynamometer freezes and corresponding talking:
        enriched_log_frame = data_integration.annotate_trial(enriched_log_frame,
                                                             "Flawed Dynamometer Measurement and Corresponding Talking",
                                                             True, trial_id=18)

        enriched_log_frame = data_integration.annotate_trial(enriched_log_frame,
                                                             "Flawed Dynamometer Measurement and Corresponding Talking",
                                                             True, trial_id=28)

        # idle state:
        enriched_log_frame.loc[pd.Timestamp("2026-01-28 19:35:10"):, 'Phase'] = 'Idle State'


    elif subject_ind == 7:

        # song start wasn't immediately registered for
        log_frame = data_integration.remove_single_row_by_timestamp(log_frame, "2026-02-08 15:10:33.846501")
        log_frame = data_integration.remove_single_row_by_timestamp(log_frame, "2026-02-08 15:19:42.174379")

        enriched_log_frame = data_integration.prepare_log_frame(log_frame, )

        # mark some trials: (dynamometer jumped somehow)
        enriched_log_frame = data_integration.annotate_trial(enriched_log_frame,
                                                             "Flawed Dynamometer Measurement and Corresponding Talking",
                                                             True, trial_id=10)
        enriched_log_frame = data_integration.annotate_trial(enriched_log_frame,
                                                             "Flawed Dynamometer Measurement and Corresponding Talking",
                                                             True, trial_id=15)
        enriched_log_frame = data_integration.annotate_trial(enriched_log_frame,
                                                             "Flawed Dynamometer Measurement and Corresponding Talking",
                                                             True, trial_id=17)

        enriched_log_frame.loc[pd.Timestamp("2026-02-08 15:50:00"):, 'Phase'] = 'Idle State'



    ################## FINAL SONG + QUESTIONNAIRE VALIDATION ##################
    print("\n\n")
    print("-" * 80)
    print("Song Data Validation")
    print("-" * 80)
    song_data_consistency_report = data_integration.validate_song_indices(enriched_log_frame, subject_experiment_data,
                                                                       verbose=True)

    print("\n\n")
    print("-" * 80)
    print("Questionnaire Data Validation")
    print("-" * 80)
    questionnaire_data_consistency_report = data_integration.validate_trial_questionnaires(enriched_log_frame,
                                                                                        subject_experiment_data,
                                                                                        verbose=True)
    enriched_log_frame = data_integration.repair_trial_questionnaire_mismatches(enriched_log_frame, questionnaire_data_consistency_report)




    ################## INSPECT FLAWED ACCURACY MEASUREMENTS ##################
    print("\n\n")
    print("-" * 80)
    print("Dynamometer Data Validation")
    print("-" * 80)
    data_integration.validate_force_measurements(enriched_log_frame, serial_frame)

    print("\n\n")
    print("-" * 80)
    print("Avg. Accuracy (RMSE) per Trial")
    print("-" * 80)
    for trial_id in enriched_log_frame['Trial ID'].unique():
        if np.isnan(trial_id): continue
        excluded = enriched_log_frame.loc[enriched_log_frame['Trial ID'] == trial_id]['Trial Exclusion Bool'].any()
        rmse = enriched_log_frame.loc[enriched_log_frame['Trial ID'] == trial_id]['Task RMSE'].astype(float).mean()
        print(f"Trial {int(trial_id)} RMSE: {rmse:.2f}{' (Excluded!)' if excluded else ''}")




    ################## EXPORT ##################
    if save_result: enriched_log_frame.to_csv(subject_experiment_data / "experiment_logs" / filemgmt.file_title("Enriched Experiment Log", ".csv"))
