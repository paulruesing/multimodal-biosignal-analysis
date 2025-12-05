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
    plot_pretrial_familiarity_check, SharedString, plot_posttrial_rating, save_terminate_process, plot_breakout_screen


### MULTIPROCESSING IMPLEMENTATION:
def start_experiment_processes(
        personal_data_dir: str | Path,
        measurement_saving_path: str | Path,
        mvc_saving_dir: str | Path,
        control_log_dir: str | Path,
        measurement_sampling_rate_hz: int = 1000,
        record_measurements: bool = True,
        music_category_txt: str | Path | None = None,

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
    measurement_sampling_rate_hz : int, optional
        Sampling frequency in Hertz for the measurement acquisition process (default is 1000).
    record_measurements : bool, optional
        Whether to record serial_measurements to disk (default is True).

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
    if not record_measurements: print("[WARNING] Measurement recording is deactivated! No measurements and control logs will be saved.")

    # serial connection intact?
    serial_connection_intact = False
    try:
        baud_rate: int = 115200
        serial_port: str = '/dev/tty.usbmodem143309601'
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

    global mvc; mvc = None
    def calibrate_mvc():
        # 0) ensure saving dir exists:
        filemgmt.assert_dir(mvc_saving_dir)  # else creates dir

        # 1) start sampling only fsr (without defining MVC):
        mvc_sampler = multiprocessing.Process(
            target=sampling_process if serial_connection_intact else dummy_sampling_process,
            args=(shared_measurement_dict, shared_dict_lock, force_serial_save_event, serial_saving_done_event, start_trigger_event, stop_trigger_event,),
            kwargs={'measurement_definitions': (("fsr", dynamometer_force_mapping, "FSR:",
                                                 .1),),  # smoothing alpha
                    'sampling_rate_hz': measurement_sampling_rate_hz,
                    'save_recordings_path': mvc_saving_dir,
                    'record_bool': True,
                    'baud_rate': 115200, },
            name="MVCSamplingProcess")
        mvc_sampler.start()

        # 2) open live_input_view:
        mvc_displayer = multiprocessing.Process(
            target=plot_input_view,
            args=(shared_measurement_dict, shared_dict_lock, ),
            kwargs={'measurement_dict_label': 'fsr',
                    'shared_questionnaire_result_str': shared_questionnaire_str,
                    'include_gauge': True,
                    'title': 'Please apply as much force as possible! You have 30 seconds. If done, wait or close window.',
                    'window_title': 'Dynamometer Serial Input View' if serial_connection_intact else "SHOWING RANDOM DEVELOPMENT SAMPLES",
                    'input_unit_label': 'Force [kg]',
                    'y_limits': (0, 90),
                    },
            name="MVCDisplayProcess")
        mvc_displayer.start()

        # 3) close both processes after 1 min or if displayer was closed
        start = time.time()
        while mvc_displayer.is_alive() and (time.time() - start) < 30:
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
                'shared_questionnaire_result_str': shared_questionnaire_str,
                'target_value': None,
                'include_gauge': False,
                'title': 'ECG Input', 'window_title': 'ECG Serial Input View' if serial_connection_intact else "SHOWING RANDOM DEVELOPMENT SAMPLES"
                },
        name="ECGDisplayProcess")

    gsr_displayer = multiprocessing.Process(
            target=plot_input_view,
            args=(shared_measurement_dict, shared_dict_lock,),
            kwargs={'measurement_dict_label': 'gsr',
                    'shared_questionnaire_result_str': shared_questionnaire_str,
                    'target_value': 1.2,
                    'include_gauge': False,
                    'title': 'GSR Input', 'window_title': 'GSR Serial Input View' if serial_connection_intact else "SHOWING RANDOM DEVELOPMENT SAMPLES"
                    },
            name="ECGDisplayProcess")


    ### PROCESS MANAGEMENT
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
                                            .1),  # smoothing alpha
                                           ("ecg", None, "ECG:", .4),
                                           ("gsr", None, "GSR:", .4),
                                           )
                # define sampler with such:
                filemgmt.assert_dir(measurement_saving_path)  # create dir if necessary
                sampler = multiprocessing.Process(
                    target=sampling_process if serial_connection_intact else dummy_sampling_process,
                    args=(shared_measurement_dict, shared_dict_lock, force_serial_save_event, serial_saving_done_event, start_trigger_event, stop_trigger_event,),
                    kwargs={'measurement_definitions': measurement_definitions,  # reflects MVC
                            'sampling_rate_hz': measurement_sampling_rate_hz,
                            'save_recordings_path': measurement_saving_path,
                            'record_bool': record_measurements,
                            'baud_rate': 115200, },
                    name="SamplingProcess")
                # start sampling:
                if not sampler.is_alive():
                    print("Starting sampling process!")
                    sampler.start()
                # start display of other modalities:
                if 'gsr' in measurement_labels:
                    if not gsr_displayer.is_alive():  # dont start twice
                        print("Starting GSR display process!")
                        gsr_displayer.start()
                if 'ecg' in measurement_labels:
                    if not ecg_displayer.is_alive():  # dont start twice
                        print("Starting ECG display process!")
                        ecg_displayer.start()

                start_sampling_event.clear()

            # todo: allow for test trial
            if start_test_motor_task_event.is_set():
                try:  # not if, because we would need to catch a NameError if not defined yet
                    assert sampler.is_alive()

                    ###### continue if sampler's running #####
                    target_freq = .1
                    test_motor_task = multiprocessing.Process(
                        target=plot_input_view,
                        args=(shared_measurement_dict, shared_dict_lock,),
                        kwargs={'measurement_dict_label': 'fsr',
                                'shared_questionnaire_result_str': shared_questionnaire_str,
                                'target_value': (5, 20, target_freq),  # target value as sine wave with .1 Hz
                                'accuracy_save_dir': None,  # no accuracy saving!
                                'target_corridor': 10,
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


                except (AssertionError, NameError):  # if sampler wasn't yet defined
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
                        print(f"Starting song_{song_counter:03}. Song info: ", temp_dict)
                        save_path = current_song_data_dir / filemgmt.file_title(f"song_{song_counter:03} information", ".json")
                        with open(save_path, "w") as json_file:  # save as json file
                            json.dump(temp_dict, json_file, indent=4)  # Pretty print with indent=4
                        print('Saved song information to ', save_path)

                        # pretrial_familiarity_check:
                        pretrial_process = multiprocessing.Process(
                            target=plot_pretrial_familiarity_check,
                            args=(current_song_data_dir, shared_questionnaire_str,),
                            kwargs={},
                            name=f"Pretrial Process {trial_label}")
                        pretrial_process.start()

                    else:
                        # breakout_screen:
                        pretrial_process = multiprocessing.Process(
                            target=plot_breakout_screen,
                            args=(30.0,  # waiting time
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
                    if time.time() - start < 30 and not (start_music_motor_task_event.is_set() or start_silent_motor_task_event.is_set()):
                        # if process is closed, show waiting screen:
                        pretrial_break_process = multiprocessing.Process(
                            target=plot_breakout_screen,
                            args=(30 - (time.time() - start),  # remaining waiting time
                                  ),
                            kwargs={'title': 'Have a break. Your trial will start soon.'},
                            name=f"Pretrial Break Process {trial_label}")
                        # only show if not already breakup screen (during silence trial) was shown
                        if is_music_trial: pretrial_break_process.start()
                        while time.time() - start < 30 and not (  # waiting time but still allow for restart
                            start_music_motor_task_event.is_set() or start_silent_motor_task_event.is_set()):
                            time.sleep(.1)  # check every 100ms
                        else:
                            save_terminate_process(pretrial_break_process)  # end pretrial process

                    # pretrial time over:
                    if start_music_motor_task_event.is_set() or start_silent_motor_task_event.is_set():  # if restart
                        pass  # go to next iteration
                    else:  # if we can continue with trial
                        # define motor task process:
                        target_freq = .1
                        dynamic_motor_task = multiprocessing.Process(
                            target=plot_input_view,
                            args=(shared_measurement_dict, shared_dict_lock,),
                            kwargs={'measurement_dict_label': 'fsr',
                                    'shared_questionnaire_result_str': shared_questionnaire_str,
                                    'target_value': (5, 20, target_freq),  # target value as sine wave with .1 Hz
                                    'accuracy_save_dir': current_song_data_dir,
                                    # todo: change target frequency based on music characteristics (if not silent)
                                    'target_corridor': 10,
                                    'include_gauge': True,
                                    'title': 'Your grip force controls the red line. Try to keep it close to the moving green target line within the green target corridor!',
                                    'input_unit_label': 'Force [% MVC]',
                                    'y_limits': (0, 100),
                                    'window_title': 'Dynamic Motor Task' if serial_connection_intact else "SHOWING RANDOM DEVELOPMENT SAMPLES"
                                    },
                            name=f"DynamicMotorTask {trial_label}")

                        # start motor task:
                        if not dynamic_motor_task.is_alive():  # dont start twice
                            status_msg = f"Starting motor task process with target frequency {target_freq}Hz!"
                            print(status_msg); shared_questionnaire_str.write(status_msg)
                            dynamic_motor_task.start()
                        # wait for ending of motor task:
                        start = time.time()  # run for 60 seconds or until window is closed
                        while time.time() - start < 60 and dynamic_motor_task.is_alive(): time.sleep(0.1)
                        else:  # if done, terminate process
                            save_terminate_process(dynamic_motor_task)

                        # define and start post trial rating: (includes music questions only if category string is provided)
                        posttrial_process = multiprocessing.Process(
                            target=plot_posttrial_rating,
                            args=(current_song_data_dir, shared_questionnaire_str,
                                  ),
                            kwargs={
                                "category_string": temp_dict['Category'].replace("Unfamiliar ",  # drop familiarity label
                                                                                 "").replace("Familiar ", "") if is_music_trial else None,
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
    # todo: eventually add experiment config file (questionnaire strings, dynamic task Hz ratio)

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
        record_measurements=True,  # False: start dummy_sampling, True: start real sampling
        music_category_txt=MUSIC_CONFIG,
        control_log_dir=EXPERIMENT_LOG,
    )
