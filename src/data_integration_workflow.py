import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Callable, Literal, Tuple

from pandas import DatetimeIndex

import src.pipeline.signal_features as features
import src.pipeline.preprocessing as preprocessing
import src.pipeline.visualizations as visualizations
import src.pipeline.data_analysis as data_analysis
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
    subject_ind = 4
    save_result: bool = True  # only set to True if manual adjustments finalized

    # experiment results import behaviour:
    subject_plot_dir = STUDY_PLOTS / f"subject_{subject_ind:02}"
    subject_experiment_data = EXPERIMENT_DATA / f"subject_{subject_ind:02}"  # should have folders experiment_logs/, serial_measurements/, song_000/, ...



    ################## IMPORT ##################
    ### Import:
    # load experiment files:
    log_frame = preprocessing.fetch_experiment_log(subject_experiment_data)
    data_analysis.get_qtc_measurement_start_end(log_frame, verbose=True)
    # log columns: ['Time', 'Music', 'Event', 'Questionnaire']
    serial_frame = preprocessing.fetch_serial_measurements(subject_experiment_data).set_index('Time')
    # serial columns: ['Time', 'fsr', 'ecg', 'gsr']

    # enrich log frame columns based on 'Event' and 'Questionnaire' data:
    enriched_log_frame = data_analysis.prepare_log_frame(log_frame, set_time_index=True)



    ################## SUBJECT-SPECIFIC AMENDMENTS ##################
    if subject_ind == 0:
        pass


    elif subject_ind == 1:
        log_frame = data_analysis.remove_song_entries(
            enriched_log_frame, log_frame,
            song_title_artist_id_tuples=[("Ain't No Sunshine", "Bill Withers", 17),
                                         ("Merry-Go-Round of Life - from 'Howl's Moving Castle'", "Joe Hisaishi", 21),
                                         ("As", "George Michael", 24),
                                         ("Dancing In the Dark", "Bruce Springsteen", 28)
                                         ])
        enriched_log_frame = data_analysis.prepare_log_frame(log_frame, set_time_index=True)
        enriched_log_frame = data_analysis.annotate_trial(enriched_log_frame,
                                                          "Talking and frustration because of briefly stuck measurement",
                                                          True, trial_id=11)

        enriched_log_frame = data_analysis.annotate_trial(enriched_log_frame,
                                                          "Talking and frustration because of briefly stuck measurement",
                                                          True, trial_id=15)


    elif subject_ind == 2:
        log_frame = data_analysis.remove_song_entries(enriched_log_frame, log_frame,
                                                      [("I Say a Little Prayer", "Aretha Franklin", 0),
                                                       ("Celebration", "Kool & The Gang", 1),
                                                       ("Uptown Funk (feat. Bruno Mars)", "Mark Ronson", 2)])
        #log_frame = data_analysis.remove_silence_trial(enriched_log_frame, log_frame, silence_ids=[0, 1, 2])
        enriched_log_frame = data_analysis.prepare_log_frame(log_frame, set_time_index=True)

        # mark some trials:
        enriched_log_frame = data_analysis.annotate_trial(enriched_log_frame,
                                                          "Flawed Dynamometer Measurement and Corresponding Talking",
                                                          True, trial_id=0)
        enriched_log_frame = data_analysis.annotate_trial(enriched_log_frame,
                                                          "Flawed Dynamometer Measurement and Corresponding Talking",
                                                          True,
                                                          trial_id=1)
        enriched_log_frame = data_analysis.annotate_trial(enriched_log_frame,
                                                          "Flawed Dynamometer Measurement and Corresponding Talking",
                                                          True,
                                                          trial_id=2)
        enriched_log_frame = data_analysis.annotate_trial(enriched_log_frame,
                                                          "Flawed Dynamometer Measurement and Corresponding Talking",
                                                          True,
                                                          trial_id=4)
        enriched_log_frame = data_analysis.annotate_trial(enriched_log_frame,
                                                          "Flawed Dynamometer Measurement and Corresponding Talking",
                                                          True,
                                                          trial_id=5)

        # mark idle state recording:
        enriched_log_frame.loc[pd.Timestamp("2026-01-17 21:05:20"):, 'Phase'] = 'Idle State'


    elif subject_ind == 3:
        log_frame = data_analysis.remove_song_entries(enriched_log_frame, log_frame,
                                                      [("Merry-Go-Round of Life - from 'Howl's Moving Castle'",
                                                        "Joe Hisaishi", 2),
                                                       ("Never Too Much", "Luther Vandross", 14)])
        log_frame = data_analysis.remove_single_row_by_timestamp(log_frame, timestamp = '2026-01-22 18:59:30.676946')
        enriched_log_frame = data_analysis.prepare_log_frame(log_frame, set_time_index=True)

        # mark idle state recording:
        enriched_log_frame.loc[pd.Timestamp("2026-01-22 19:08:00"):, 'Phase'] = 'Idle State'


    elif subject_ind == 4:
        log_frame = data_analysis.remove_song_entries(
            enriched_log_frame, log_frame,
            song_title_artist_id_tuples=[("Can't Get Enough! - Vocal Club Mix", "Soulsearcher", 8),])
        enriched_log_frame = data_analysis.prepare_log_frame(log_frame, set_time_index=True)

        enriched_log_frame = data_analysis.annotate_trial(enriched_log_frame, "Talking", False, song_id=8)
        # dont exclude the above, since we do not have THAT many measurements and it was only a brief comment
        enriched_log_frame = data_analysis.annotate_trial(enriched_log_frame, "Talking and then was repeated anyways", True, silence_id=1)

        # mark idle state recording:
        enriched_log_frame.loc[pd.Timestamp("2026-01-23 17:56:00"):, 'Phase'] = 'Idle State'




    ################## FINAL SONG + QUESTIONNAIRE VALIDATION ##################
    print("\n\n")
    print("-" * 80)
    print("Song Data Validation")
    print("-" * 80)
    song_data_consistency_report = data_analysis.validate_song_indices(enriched_log_frame, subject_experiment_data,
                                                                       verbose=True)

    print("\n\n")
    print("-" * 80)
    print("Questionnaire Data Validation")
    print("-" * 80)
    questionnaire_data_consistency_report = data_analysis.validate_trial_questionnaires(enriched_log_frame,
                                                                                        subject_experiment_data,
                                                                                        verbose=True)
    data_analysis.repair_trial_questionnaire_mismatches(enriched_log_frame, questionnaire_data_consistency_report)




    ################## INSPECT FLAWED ACCURACY MEASUREMENTS ##################
    print("\n\n")
    print("-" * 80)
    print("Dynamometer Data Validation")
    print("-" * 80)
    data_analysis.validate_force_measurements(enriched_log_frame, serial_frame)

    print("\n\n")
    print("-" * 80)
    print("Avg. Accuracy (RMSE) per Trial")
    print("-" * 80)
    for trial_id in enriched_log_frame['Trial ID'].unique():
        if np.isnan(trial_id): continue

        rmse = enriched_log_frame.loc[enriched_log_frame['Trial ID'] == trial_id]['Task Avg. RMSE'].astype(float).mean()
        print(f"Trial {int(trial_id)} RMSE: {rmse:.2f}")




    ################## EXPORT ##################
    if save_result: enriched_log_frame.to_csv(subject_experiment_data / "experiment_logs" / filemgmt.file_title("Enriched Experiment Log", ".csv"))
