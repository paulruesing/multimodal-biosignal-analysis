"""
This script contains the experiment workflow leveraging processes defined in measurement_processes.py.
It manages shared memory allocation, multiprocessing and runs through the whole experiment workflow.
®Paul Rüsing, INI ETH / UZH
"""

import serial
from typing import Callable, Literal
import multiprocessing
import ctypes
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from torchgen.gen import file_manager_from_dispatch_key

import src.utils.file_management as filemgmt

from src.pipeline.measurement_processes import RobustEventManager, dummy_sampling_process, sampling_process, \
    plot_input_view, qtc_control_master_view, dynamometer_force_mapping, plot_onboarding_form, \
    plot_pretrial_familiarity_check, SharedString, plot_posttrial_rating, save_terminate_process, plot_breakout_screen, \
    accuracy_sampler


### MULTIPROCESSING IMPLEMENTATION:
def start_experiment_processes(
        personal_data_dir: str | Path,
        measurement_saving_path: str | Path,
        mvc_saving_dir: str | Path,
        control_log_dir: str | Path,
        music_category_txt: str | Path | None = None,
        experiment_config_txt: str | Path | None = None,
) -> None:
    """
    Starts multiprocessing setup for simultaneous measurement sampling and live plotting for multiple biosignal channels.

    Parameters
    ----------
    measurement_definitions : tuple of tuples
        Configuration for each measurement, where each tuple contains:
        - measurement_label (str): the key identifying the measurement.
        - processing_callable (callable or None): optional function to process raw serial_measurements.
        - serial_input_marker (str): prefix identifying the measurement line in serial input.
        - exponential_moving_average smoothing alpha
    measurement_saving_path : str or Path, optional
        Path where recorded measurement data will be saved. If None, data will not be saved persistently.

    Returns
    -------
    None
        Function initializes and starts multiprocessing processes for sampling and plotting.
        Manages graceful shutdown and data saving on KeyboardInterrupt.

    Notes
    -----
    - Launches separate processes for each display: FSR (force-sensitive resistor), ECG, and GSR (galvanic skin response).
    - Uses a shared multiprocessing dictionary to communicate latest measurement values among processes.
    - Implements `RobustEventManager` for safe inter-process signaling of save events to avoid deadlocks.
    - Currently uses a dummy sampling process; replace with actual serial sampling target as needed.
    - Handles KeyboardInterrupt for clean termination and saves any buffered data before exit.
    """
    ### PREPARATION
    # sanity check:
    if experiment_config_txt is None: print("[WARNING] No experiment configuration provided. Will use default settings.")

    # load experiment config:
    if experiment_config_txt is not None:
        experiment_config_file = filemgmt.TxtConfig(experiment_config_txt)
    baud_rate = experiment_config_file.get_as_type("Serial BAUD Rate", "str") if experiment_config_txt is not None else 115200
    serial_port = experiment_config_file.get_as_type("Serial Port", "str") if experiment_config_txt is not None else '/dev/tty.usbmodem143309601'
    measurement_sampling_rate_hz = experiment_config_file.get_as_type("Serial Sampling Rate", "float") if experiment_config_txt is not None else 1000

    use_initial_mvc = experiment_config_file.get_as_type("Use Initial MVC", "bool") if experiment_config_txt is not None else False
    initial_mvc = experiment_config_file.get_as_type("Initial MVC", "float") if experiment_config_txt is not None and use_initial_mvc else None
    mvc_max_time = experiment_config_file.get_as_type("MVC Maximum Time", "float") if experiment_config_txt is not None else 30.0

    fsr_smoothing_alpha = experiment_config_file.get_as_type("Force Smoothing Alpha", "float") if experiment_config_txt is not None else .1
    ecg_smoothing_alpha = experiment_config_file.get_as_type("ECG Smoothing Alpha", "float") if experiment_config_txt is not None else .4
    gsr_smoothing_alpha = experiment_config_file.get_as_type("GSR Smoothing Alpha", "float") if experiment_config_txt is not None else .4

    display_ecg = experiment_config_file.get_as_type("Display ECG", "bool") if experiment_config_txt is not None else True
    display_gsr = experiment_config_file.get_as_type("Display GSR", "bool") if experiment_config_txt is not None else True

    target_sine_min = experiment_config_file.get_as_type("Target Sine Minimum", "float") if experiment_config_txt is not None else 5
    target_sine_max = experiment_config_file.get_as_type("Target Sine Maximum", "float") if experiment_config_txt is not None else 20
    use_relative_target_freq = experiment_config_file.get_as_type("Use Relative Target Frequency", "bool") if experiment_config_txt is not None else False
    ratio_target_sine_freq_bpm = experiment_config_file.get_as_type("Ratio Target Sine Frequency Music BPM", "float") if experiment_config_txt is not None else .1
    target_sine_freq_abs = experiment_config_file.get_as_type("Absolute Target Sine Frequency", "float") if experiment_config_txt is not None else .1
    target_display_corridor = experiment_config_file.get_as_type("Target Display Corridor", "float") if experiment_config_txt is not None else 10

    pre_trial_time = experiment_config_file.get_as_type("Pre Trial Time", "float") if experiment_config_txt is not None else 30
    pre_accuracy_phase_sec = experiment_config_file.get_as_type("Pre Accuracy Measurement Time", "float") if experiment_config_txt is not None else 5
    motor_trial_duration = experiment_config_file.get_as_type("Motor Trial Duration", "float") if experiment_config_txt is not None else 60

    pretrial_familiarity_question_str = experiment_config_file.get_as_type("Familiarity Question", "str") if experiment_config_txt is not None else "How well do you know this song? (0 = never heard it, 7 = can sing/hum along)"
    liking_question_str = experiment_config_file.get_as_type("Liking Question", "str") if experiment_config_txt is not None else "How did you like the song? (0: terrible, 7: extremely well)"
    emotion_question_str = experiment_config_file.get_as_type("Emotional Question", "str") if experiment_config_txt is not None else "Please rate your overall emotional state right now. (0: extremely unhappy/distressed, 7 = extremely happy/peaceful)"

    # serial connection intact?
    serial_connection_intact = False
    try:
        with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
            serial_connection_intact = True
            pass
    except serial.SerialException as e:
        print("[WARNING] Serial connection not working properly. Will show random samples for test purposes.")
        serial_connection_intact = False

    ### SHARED MEMORY DEFINITIONS
    # measurement dict:
    mp_manager = multiprocessing.Manager()
    shared_measurement_dict = mp_manager.dict()
    measurement_labels = []  # for dynamic object definition below
    for measurement_label in ['fsr', 'ecg', 'gsr']:
        shared_measurement_dict[measurement_label] = .0
        measurement_labels.append(measurement_label)

    # song info dict:
    shared_song_info_dict = mp_manager.dict()
    shared_dict_lock = mp_manager.Lock()  # holds for both dicts
    
    # accuracy measurement dict:
    shared_force_value_target_dict = mp_manager.dict()
    # current accuracy display str:
    shared_current_accuracy_str = SharedString(256, initial_value="Accuracy measurement will be displayed here...")

    # rating result str:
    shared_questionnaire_str = SharedString(256, initial_value="Questionnaire results will be displayed here...")

    # multiprocessing events:
    force_serial_save_event = RobustEventManager()  # force serial save
    serial_saving_done_event = RobustEventManager()
    force_log_saving_event = RobustEventManager()  # force master log save
    log_saving_done_event = RobustEventManager()
    start_trigger_event = RobustEventManager()  # send trigger via serial
    stop_trigger_event = RobustEventManager()
    start_onboarding_event = RobustEventManager()  # sent from master view, call new processes
    start_mvc_calibration_event = RobustEventManager()
    start_sampling_event = RobustEventManager()
    start_music_motor_task_event = RobustEventManager()  # called upon starting a song
    start_silent_motor_task_event = RobustEventManager()  # called upon silent trial
    start_test_motor_task_event = RobustEventManager()  # called upon test trial
    save_accuracy_and_close_event = RobustEventManager()  # called to force accuracy saving
    save_accuracy_done_event = RobustEventManager()   # called if force save is done

    ### PROCESS DEFINITIONS
    filemgmt.assert_dir(control_log_dir)
    master_displayer = multiprocessing.Process(
        target=qtc_control_master_view,
        args=(shared_measurement_dict, shared_dict_lock, start_trigger_event, stop_trigger_event,
              start_onboarding_event, start_mvc_calibration_event, start_sampling_event,
              start_music_motor_task_event, start_silent_motor_task_event,
              shared_questionnaire_str, shared_song_info_dict,
              force_log_saving_event, log_saving_done_event, start_test_motor_task_event),
        kwargs={'music_category_txt': music_category_txt,
                'control_log_dir': control_log_dir,
                'title': 'Experiment Master - Close this window to end the experiment.'
                },
        name="MasterDisplayProcess")

    filemgmt.assert_dir(personal_data_dir)
    onboarding_process = multiprocessing.Process(
        target=plot_onboarding_form,
        args=(personal_data_dir, shared_questionnaire_str,),
        kwargs={},
        name="OnboardingFormProcess")

    global mvc; mvc = initial_mvc
    def calibrate_mvc():
        # 0) ensure saving dir exists:
        filemgmt.assert_dir(mvc_saving_dir)  # else creates dir

        # 1) start sampling only fsr (without defining MVC):
        mvc_sampler = multiprocessing.Process(
            target=sampling_process if serial_connection_intact else dummy_sampling_process,
            args=(shared_measurement_dict, shared_dict_lock, force_serial_save_event, serial_saving_done_event, start_trigger_event, stop_trigger_event,),
            kwargs={'measurement_definitions': (("fsr", dynamometer_force_mapping, "FSR:",
                                                 fsr_smoothing_alpha),),  # smoothing alpha
                    'sampling_rate_hz': measurement_sampling_rate_hz,
                    'save_recordings_path': mvc_saving_dir,
                    'record_bool': serial_connection_intact,
                    'baud_rate': baud_rate, },
            name="MVCSamplingProcess")
        mvc_sampler.start()

        # 2) open live_input_view:
        mvc_displayer = multiprocessing.Process(
            target=plot_input_view,
            args=(shared_measurement_dict, shared_dict_lock, ),
            kwargs={'measurement_dict_label': 'fsr',
                    'include_gauge': True,
                    'title': f'Please apply as much force as possible! You have {mvc_max_time} seconds. If done, wait or close window.',
                    'window_title': 'Dynamometer Serial Input View' if serial_connection_intact else "SHOWING RANDOM DEVELOPMENT SAMPLES",
                    'input_unit_label': 'Force [kg]',
                    'y_limits': (0, 90),
                    },
            name="MVCDisplayProcess")
        mvc_displayer.start()

        # 3) close both processes after 1 min or if displayer was closed
        start = time.time()
        while mvc_displayer.is_alive() and (time.time() - start) < mvc_max_time:
           time.sleep(.1)  # dont check every second
        else:
            save_terminate_process(mvc_displayer)  # terminate display

            # force saving:
            force_serial_save_event.set()
            print('Waiting for MVC serial saving...')
            serial_saving_done_event.wait(timeout=5)  # wait until done
            print('MVC serial saving done!')
            time.sleep(2)  # wait briefly for file to be written

            # terminate sampler:
            save_terminate_process(mvc_sampler)

        # 4) load saved measurements and save max as MVC:
        try:
            force_series = pd.read_csv(filemgmt.most_recent_file(mvc_saving_dir, ".csv", ["fsr"])).loc[:, "fsr"]
            global mvc; mvc = np.nanmax(force_series.values)  # store mvc globally
            shared_questionnaire_str.write(f"Recorded MVC Value: {mvc:.2f}kg")  # for display in master process
            print("Derived MVC: ", mvc)  # print in console as well

        except ValueError:
            status_msg = "Force measurement for MVC calibration failed. Please try again."
            print(status_msg); shared_questionnaire_str.write(status_msg)  # print and display in master process

    ecg_displayer = multiprocessing.Process(
        target=plot_input_view,
        args=(shared_measurement_dict, shared_dict_lock,),
        kwargs={'measurement_dict_label': 'ecg',
                'target_value': None,
                'include_gauge': False,
                'title': 'ECG Input', 'window_title': 'ECG Serial Input View' if serial_connection_intact else "SHOWING RANDOM DEVELOPMENT SAMPLES"
                },
        name="ECGDisplayProcess")

    gsr_displayer = multiprocessing.Process(
            target=plot_input_view,
            args=(shared_measurement_dict, shared_dict_lock,),
            kwargs={'measurement_dict_label': 'gsr',
                    'include_gauge': False,
                    'title': 'GSR Input', 'window_title': 'GSR Serial Input View' if serial_connection_intact else "SHOWING RANDOM DEVELOPMENT SAMPLES"
                    },
            name="ECGDisplayProcess")


    # todo: define performance_displayer process


    ### PROCESS MANAGEMENT
    # todo: ponder whether more definitions should be included in separate functions for clarity
    try:
        ## Start Master
        print("Starting master process!")
        master_displayer.start()
        global song_counter
        song_counter = 0

        # check for commands:
        while master_displayer.is_alive():
            ## Registration Phase
            if start_onboarding_event.is_set():
                if not onboarding_process.is_alive():
                    print("Starting onboarding process!")
                    onboarding_process.start()
                    # wait for process to die before taking new commands:
                    while onboarding_process.is_alive():
                        pass
                    else:
                        start_onboarding_event.clear()


            ## Calibration Phase
            if start_mvc_calibration_event.is_set():
                print("Starting MVC calibration process!")
                calibrate_mvc()
                start_mvc_calibration_event.clear()


            if start_sampling_event.is_set():
                # define measurement definitions with MVC:
                measurement_definitions = (("fsr",  # measurement label
                                            (dynamometer_force_mapping, mvc),  # MVC [kg]
                                            "FSR:",  # serial connection measurement identifier
                                            fsr_smoothing_alpha),  # smoothing alpha
                                           ("ecg", None, "ECG:", ecg_smoothing_alpha),
                                           ("gsr", None, "GSR:", gsr_smoothing_alpha),
                                           )
                # define sampler with such:
                filemgmt.assert_dir(measurement_saving_path)  # create dir if necessary

                try:  # prevent double definition
                    _ = sampler  # if already defined:
                    print("Starting new sampling process.")
                    save_terminate_process(sampler)  # first terminate old
                except NameError:  # if not already defined, everything is fine
                    pass

                sampler = multiprocessing.Process(
                    target=sampling_process if serial_connection_intact else dummy_sampling_process,
                    args=(shared_measurement_dict, shared_dict_lock, force_serial_save_event, serial_saving_done_event, start_trigger_event, stop_trigger_event,),
                    kwargs={'measurement_definitions': measurement_definitions,  # reflects MVC
                            'sampling_rate_hz': measurement_sampling_rate_hz,
                            'save_recordings_path': measurement_saving_path,
                            'record_bool': serial_connection_intact,
                            'baud_rate': baud_rate, },
                    name="SamplingProcess")
                # start sampling:
                if not sampler.is_alive():
                    print("Starting sampling process!")
                    sampler.start()
                # start display of other modalities:
                if 'gsr' in measurement_labels and display_gsr:
                    if not gsr_displayer.is_alive():  # dont start twice
                        print("Starting GSR display process!")
                        gsr_displayer.start()
                if 'ecg' in measurement_labels and display_ecg:
                    if not ecg_displayer.is_alive():  # dont start twice
                        print("Starting ECG display process!")
                        ecg_displayer.start()

                start_sampling_event.clear()


            if start_test_motor_task_event.is_set():
                try:  # catch NameError (if sampler wasn't defined yet) and RuntimeError (if raised deliberately)
                    if not sampler.is_alive(): raise RuntimeError('Sampling process needs to be started before!')

                    ###### continue if sampler's running #####
                    target_freq = .1
                    test_motor_task = multiprocessing.Process(
                        target=plot_input_view,
                        args=(shared_measurement_dict, shared_dict_lock,),
                        kwargs={'measurement_dict_label': 'fsr',
                                'target_value': (target_sine_min, target_sine_max, target_sine_freq_abs),  # target value as sine wave with .1 Hz
                                'target_corridor': target_display_corridor,
                                'include_gauge': True,
                                'title': 'Your grip force controls the red line. Try to keep it close to the moving green target line within the green target corridor!',
                                'input_unit_label': 'Force [% MVC]',
                                'y_limits': (0, 100),
                                'window_title': 'Test Dynamic Motor Task' if serial_connection_intact else "SHOWING RANDOM DEVELOPMENT SAMPLES"
                                },
                        name=f"TestMotorTask")

                    # start motor task:
                    if not test_motor_task.is_alive():  # dont start twice
                        status_msg = f"Starting test motor task process with target frequency {target_freq}Hz!"
                        print(status_msg); shared_questionnaire_str.write(status_msg)
                        test_motor_task.start()

                    # wait for ending of test motor task:
                    start = time.time()  # run for 60 seconds or until window is closed
                    while time.time() - start < 60 and test_motor_task.is_alive():
                        time.sleep(0.1)
                    else:  # if done, terminate process
                        save_terminate_process(test_motor_task)

                    # clean-up:
                    status_msg = f"Test motor task done!"
                    print(status_msg); shared_questionnaire_str.write(status_msg)
                    start_test_motor_task_event.clear()

                except (RuntimeError, NameError):  # if sampler wasn't yet defined
                    shared_questionnaire_str.write("First start sampling process please!")
                    start_test_motor_task_event.clear()

            ## Motor Task
            if start_music_motor_task_event.is_set() or start_silent_motor_task_event.is_set():
                try:  # not if, because we would need to catch a NameError if not defined yet
                    assert sampler.is_alive()

                    ###### continue if sampler's running #####
                    # check whether it's silent or music trial:
                    is_music_trial = start_music_motor_task_event.is_set()

                    # prepare directory:
                    trial_label = f"song_{song_counter:03}" if is_music_trial else "silence"
                    current_song_data_dir = personal_data_dir / trial_label
                    filemgmt.assert_dir(current_song_data_dir)  # create dir

                    # wait shortly for spotify to start song:
                    time.sleep(1)

                    if is_music_trial:  # store song information (from shared_song_info_dict):
                        with shared_dict_lock:
                            temp_dict = dict(shared_song_info_dict)  # snapshot dict
                            # example structure: {
                            #     "Title": "Blurred Lines",
                            #     "Artist": "Robin Thicke",
                            #     "Album": "Blurred Lines (Deluxe)",
                            #     "Duration [ms]": 263826.0,
                            #     "Category": "Familiar Groovy",
                            #     "Category Index": 0
                            # }

                        # todo: include other information, calculate bpm

                        print(f"Starting song_{song_counter:03}. Song info: ", temp_dict)
                        save_path = current_song_data_dir / filemgmt.file_title(f"song_{song_counter:03} information", ".json")
                        with open(save_path, "w") as json_file:  # save as json file
                            json.dump(temp_dict, json_file, indent=4)  # Pretty print with indent=4
                        print('Saved song information to ', save_path)

                        # allow for relative target:
                        if use_relative_target_freq:
                            # todo: calculate relative target frequency (use_relative_target_freq, ratio_target_sine_freq_bpm)
                            raise NotImplementedError("Relative target frequency not implemented yet.")
                        else: target_freq = target_sine_freq_abs

                        # pretrial_familiarity_check:
                        pretrial_process = multiprocessing.Process(
                            target=plot_pretrial_familiarity_check,
                            args=(current_song_data_dir, shared_questionnaire_str,),
                            kwargs={"question_text": pretrial_familiarity_question_str},
                            name=f"Pretrial Process {trial_label}")
                        pretrial_process.start()

                    else:
                        target_freq = target_sine_freq_abs  # silence trial cannot use relative target

                        # breakout_screen:
                        pretrial_process = multiprocessing.Process(
                            target=plot_breakout_screen,
                            args=(pre_trial_time,  # waiting time
                                  ),
                            kwargs={'title': 'Have a break. Your trial will start soon.'},
                            name=f"Pretrial Process {trial_label}")
                        pretrial_process.start()

                    # prepare for possible restart of motor task during pre-trial phase:
                    start_music_motor_task_event.clear()  # clear event to allow for restarting
                    start_silent_motor_task_event.clear()
                    if is_music_trial: song_counter += 1  # increase song counter already (if music trial)

                    # wait at least 30seconds until pretrial process is closed, or experiment is restarted:
                    start = time.time()
                    while (pretrial_process.is_alive()) and not (
                            start_music_motor_task_event.is_set() or start_silent_motor_task_event.is_set()):  # allow for restarting
                        time.sleep(0.1)  # check every

                    # stop pre_trial process:
                    save_terminate_process(pretrial_process)

                    # breakout screen if time remains and no restart triggered
                    if time.time() - start < pre_trial_time and not (start_music_motor_task_event.is_set() or start_silent_motor_task_event.is_set()):
                        # if process is closed, show waiting screen:
                        pretrial_break_process = multiprocessing.Process(
                            target=plot_breakout_screen,
                            args=(pre_trial_time - (time.time() - start),  # remaining waiting time
                                  ),
                            kwargs={'title': 'Have a break. Your trial will start soon.'},
                            name=f"Pretrial Break Process {trial_label}")
                        # only show if not already breakup screen (during silence trial) was shown
                        if is_music_trial: pretrial_break_process.start()
                        while time.time() - start < pre_trial_time and not (  # waiting time but still allow for restart
                            start_music_motor_task_event.is_set() or start_silent_motor_task_event.is_set()):
                            time.sleep(.1)  # check every 100ms
                        else:
                            save_terminate_process(pretrial_break_process)  # end pretrial process

                    # pretrial time over:
                    if start_music_motor_task_event.is_set() or start_silent_motor_task_event.is_set():  # if restart
                        pass  # go to next iteration
                    else:  # if we can continue with trial
                        # define motor task process:
                        display_refresh_rate_hz = 15  # equals accuracy sampling rate
                        dynamic_motor_task = multiprocessing.Process(
                            target=plot_input_view,
                            args=(shared_measurement_dict, shared_dict_lock,),
                            kwargs={'measurement_dict_label': 'fsr',

                                    'target_value': (target_sine_min, target_sine_max, target_freq),  # target value as sine wave with .1 Hz
                                    # todo: change target frequency based on music characteristics (if not silent)
                                    'target_corridor': target_display_corridor,

                                    'include_gauge': True,
                                    'display_refresh_rate_hz': display_refresh_rate_hz,

                                    'shared_value_target_dict': shared_force_value_target_dict,
                                    'shared_current_accuracy_str': shared_current_accuracy_str,
                                    'save_accuracy_and_close_event': save_accuracy_and_close_event,

                                    'title': 'Your grip force controls the red line. Try to keep it close to the moving green target line within the green target corridor!',
                                    'input_unit_label': 'Force [% MVC]',
                                    'y_limits': (0, 100),
                                    'window_title': 'Dynamic Motor Task' if serial_connection_intact else "SHOWING RANDOM DEVELOPMENT SAMPLES"
                                    },
                            name=f"DynamicMotorTask {trial_label}")
                        
                        accuracy_sampling_process = multiprocessing.Process(
                            target=accuracy_sampler,
                            args=(display_refresh_rate_hz, ),
                            kwargs={'shared_value_target_dict': shared_force_value_target_dict,
                                    'shared_dict_lock': shared_dict_lock,
                                    'shared_current_se_str': shared_current_accuracy_str,
                                    'shared_questionnaire_result_str': shared_questionnaire_str,
                                    'accuracy_save_dir': current_song_data_dir,
                                    'save_accuracy_and_close_event': save_accuracy_and_close_event,
                                    'save_accuracy_done_event': save_accuracy_done_event,
                                    'pre_accuracy_phase_dur_sec': pre_accuracy_phase_sec,
                                    },
                        )

                        # start motor task:
                        if not dynamic_motor_task.is_alive():  # dont start twice
                            status_msg = f"Starting motor task process with target frequency {target_freq}Hz!"
                            print(status_msg); shared_questionnaire_str.write(status_msg)
                            dynamic_motor_task.start()
                        # start accuracy sampling:
                        if not accuracy_sampling_process.is_alive():
                            accuracy_sampling_process.start()

                        # wait for ending of motor task:
                        start = time.time()  # run for 60 seconds or until window is closed
                        while time.time() - start < motor_trial_duration and dynamic_motor_task.is_alive(): time.sleep(0.1)
                        else:  # if done:
                            # close motor task:
                            save_terminate_process(dynamic_motor_task)

                            # store accuracy measurements:
                            print("Waiting for accuracy saving...")
                            save_accuracy_and_close_event.set()

                            # wait until done:
                            save_accuracy_done_event.wait(timeout=5)  # wait until done
                            save_accuracy_done_event.clear()
                            print("Accuracy saved!")

                            # terminate process:
                            save_terminate_process(accuracy_sampling_process)

                        # save trial summary:
                        accuracy_array = pd.read_csv(filemgmt.most_recent_file(current_song_data_dir,
                                                                               ".csv",
                                                                               ["Accuracy Results"], ))
                        rmse = np.sqrt(np.mean(accuracy_array.iloc[:, 1]))
                        summary_dict = {'RMSE': rmse, 'Target Frequency': target_freq, 'Target Min': target_sine_min, 'Target Max': target_sine_max,}
                        save_path = current_song_data_dir / filemgmt.file_title(f"Trial Summary", ".json")
                        with open(save_path, "w") as json_file:
                            json.dump(summary_dict, json_file, indent=4)  # Pretty print with indent=4
                        print("Saved trial summary to ", save_path)

                        # todo: update performance view event

                        # define and start post trial rating: (includes music questions only if category string is provided)
                        posttrial_process = multiprocessing.Process(
                            target=plot_posttrial_rating,
                            args=(current_song_data_dir, shared_questionnaire_str,
                                  ),
                            kwargs={
                                "category_string": temp_dict['Category'].replace("Unfamiliar ",  # drop familiarity label
                                                                                 "").replace("Familiar ", "") if is_music_trial else None,
                                "liking_question_str": liking_question_str,
                                "emotion_question_str": emotion_question_str,
                            },
                            name=f"Posttrial Process {trial_label}")
                        posttrial_process.start()

                        # wait until post trial rating is submitted:
                        while posttrial_process.is_alive():
                            time.sleep(0.1)  # check every 100 ms
                        else:
                            # stop post_trial process:
                            save_terminate_process(posttrial_process)

                except (AssertionError, NameError):  # if sampler wasn't defined yet
                    shared_questionnaire_str.write("First start sampling process please!")
                    start_music_motor_task_event.clear()
                    start_silent_motor_task_event.clear()


        else:
            raise KeyboardInterrupt  # to fall into the except loop

    except KeyboardInterrupt:
        print("Terminating processes...")
        force_serial_save_event.set()  # trigger sampler saving
        print('Waiting for serial saving...')
        serial_saving_done_event.wait(timeout=5)  # wait until done
        print('Serial saving done!')

        force_log_saving_event.set()  # trigger master log saving
        print('Waiting for log saving...')
        log_saving_done_event.wait(timeout=5)  # wait until done
        print('Log saving done!')

        # kill remaining processes:
        try:
            save_terminate_process(sampler)
        except NameError:
            pass

        save_terminate_process(onboarding_process)
        save_terminate_process(gsr_displayer)
        save_terminate_process(ecg_displayer)
        save_terminate_process(master_displayer)

    finally:
        print("Cleanup completed")


if __name__ == '__main__':
    # define saving folder:
    ROOT = Path().resolve().parent
    CONFIG_DIR = ROOT / "config"
    MUSIC_CONFIG = CONFIG_DIR / "music_selection.txt"
    EXPERIMENT_CONFIG = CONFIG_DIR / "experiment_config.txt"
    # todo: add experiment config file (questionnaire strings, dynamic task Hz ratio)

    # important:
    SUBJECT_DIR = ROOT / "data" / "experiment_results" / "subject_00"
    SERIAL_MEASUREMENTS = SUBJECT_DIR / "serial_measurements"
    MVC_MEASUREMENTS = SUBJECT_DIR / "mvc_measurements"
    EXPERIMENT_LOG = SUBJECT_DIR / "experiment_logs"

    # start process:
    start_experiment_processes(
        mvc_saving_dir=MVC_MEASUREMENTS,
        personal_data_dir=SUBJECT_DIR,
        measurement_saving_path=SERIAL_MEASUREMENTS,
        music_category_txt=MUSIC_CONFIG,
        control_log_dir=EXPERIMENT_LOG,
        experiment_config_txt=EXPERIMENT_CONFIG,
    )
