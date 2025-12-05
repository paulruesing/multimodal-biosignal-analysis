"""
This script contains process definitions and relevant auxiliary functions to be called in an experiment workflow.
That external workflow needs to manage shared memory allocation and multiprocessing.
®Paul Rüsing, INI ETH / UZH
"""


import serial
import time
import json
import random
from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, RadioButtons, Slider
from matplotlib.animation import FuncAnimation
from pynput import keyboard
from scipy.optimize import curve_fit
from typing import Callable, Literal, Optional
import multiprocessing
from ctypes import c_char

import src.utils.file_management as filemgmt
from src.pipeline.music_control import SpotifyController

from pathlib import Path
import pandas as pd

matplotlib.use('TkAgg')


############### READOUT METHODS ###############
def read_serial_measurements(measurement_definitions: tuple[tuple[str, Callable[[float], float] | tuple[Callable, float], str, float]],
                             baud_rate: int = 115200,
                             serial_port: str = '/dev/tty.usbmodem143309601',
                             record_bool: bool = True,
                             command: Literal['A', 'B'] | None = None,
                             # measurement_label, processing_callable, serial_input_marker

                             allowed_input_range: tuple[float] = (.0, 3.3),
                             ) -> dict[str, float] | None:
    """
    Reads multiple sensor serial_measurements simultaneously from a serial port and processes each.

    Parameters
    ----------
    measurement_definitions : tuple of tuples
        Each tuple contains:
        - measurement_label (str): unique label for the measurement.
        - processing_func (callable or tuple with callable and MVC arg):
            optional post-processing function for the measurement's raw value. If provided a tuple (callable, float) the
            float defines the first argument to pass to such function.
        - serial_input_marker (str): prefix string that identifies the measurement in the serial input line.
        - smoothing_ema_alpha
    baud_rate : int, optional
        Baud rate for the serial connection (default is 115200).
    serial_port : str, optional
        The serial port identifier to connect to (default is '/dev/tty.usbmodem143309601').
    record_bool : bool, optional
        Whether to record the processed values with timestamps (default is True).
    command : str or None, optional
        Optional command to send to the device ('A' or 'B') before reading (default is None).
    allowed_input_range : tuple of float, optional
        Acceptable input value range; values outside are discarded (default is (0.0, 3.3)).
    smoothing_ema_alpha : float, optional
        Exponential moving average smoothing factor; 1 = no smoothing, closer to 0 = more smoothing (default is 0.4).

    Returns
    -------
    dict of {str: float} or None
        Returns a dictionary mapping measurement labels to their processed and smoothed values.
        If reading fails, returns last valid values for all serial_measurements.

    Notes
    -----
    Uses dynamic global variables for each measurement label to keep track of last valid readings, timestamps, and recorded serial_measurements.
    Handles serial communication and parsing errors gracefully by reverting to last valid readings.
    """
    # sanity check (whether last valid reading is initialised)
    for measurement_label, _, _, _ in measurement_definitions:
        try:
            _ = globals()["_last_valid_reading_" + measurement_label]
        except KeyError:
            raise AttributeError(f"Define global argument named {'_last_valid_reading_' + measurement_label} for {measurement_label} measurement before calling read_serial_measurements!")

    # we deploy globals() function here for dynamic global object naming and accessing (depending on included serial_measurements)
    try:
        output_dict = {}  # will be returned
        with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
            # check for output command:
            if command in ("A", "B"):
                ser.write(command.encode("ascii"))
                ser.flush()  # waits for all outgoing data to be transmitted

            # read new line and convert to float:
            line = ser.readline().decode('ascii', errors="ignore").strip()

            # process each measurement type:
            for measurement_label, processing_func, teensy_marker, smoothing_factor in measurement_definitions:
                if not line.startswith(teensy_marker):  # check whether line contains measurement result
                    # otherwise use last result:
                    output_dict[measurement_label] = globals()['_last_valid_reading_' + measurement_label]
                    continue  # jump to next measurement

                # format measurement:
                raw_str = line.replace(teensy_marker, "")  # formatting
                value = float(raw_str)

                # check whether input remains in feasible range:
                if not allowed_input_range[0] < value < allowed_input_range[1]:
                    # otherwise use last result:
                    output_dict[measurement_label] = globals()['_last_valid_reading_' + measurement_label]
                    continue  # and jump to next measurement

                # measurement-specific processing function:
                if processing_func is not None:  # apply processing func (with or without argument)
                    if isinstance(processing_func, tuple):  value = processing_func[0](value, processing_func[1])
                    else: value = processing_func(value)

                # Apply EMA smoothing:
                value = smoothing_factor * value + (1 - smoothing_factor) * globals()['_last_valid_reading_' + measurement_label]

                # overwrite last valid reading:
                globals()['_last_valid_reading_' + measurement_label] = value

                # save (if record_bool) and return:
                if record_bool:
                    globals()['timestamps_' + measurement_label].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    globals()['measurements_' + measurement_label].append(value)
                output_dict[measurement_label] = value

        return output_dict

    except (ValueError, serial.SerialException) as e:
        print(f"Serial error: {e}")
        return {measurement_label: globals()['_last_valid_reading_' + measurement_label] for measurement_label, _, _, _ in measurement_definitions}


def force_estimator_fsr(voltage: float,
                        fsr_a: float = 5.0869,
                        fsr_b: float = 1.8544) -> float:
    """
    Estimates force from voltage reading based on a calibration model for an FSR sensor.

    Parameters
    ----------
    voltage : float
        The input voltage measured from the FSR sensor.
    fsr_a : float, optional
        Calibration coefficient 'a' for the model (default is 5.0869).
    fsr_b : float, optional
        Calibration exponent 'b' for the model (default is 1.8544).

    Returns
    -------
    float
        Estimated force corresponding to the input voltage, based on the power-law relationship.
    """
    force_estimation = fsr_a * voltage ** fsr_b
    return force_estimation


def dynamometer_force_mapping(v, mvc_kg: float | None = None):  # here with default params
    """
    Fitted but added manual offset (-2) to force closer to 0
    Returns [kg] if global var. _current_mvc_kg is None else [% MVC].
    """
    factor = 1 if mvc_kg is None else 100 / mvc_kg  # consider MVC
    return (2.8708 * (v ** 4.1071) - 3) * factor


def sampling_process(shared_dict,
                     shared_dict_lock,
                     force_serial_save_event,  # save_event callable through other functions
                     serial_saving_done_event,  # saving_done event pausing other processes
                     start_trigger_event,  # send start trigger event ('A' via serial connection)
                     stop_trigger_event,  # send stop trigger event ('B' via serial connection)
                     measurement_definitions: tuple[tuple[str, Callable[[float], float] | None, str, float]],
                     # (measurement_label, processing_callable, serial_input_marker, smoothing_alpha)
                     sampling_rate_hz: int = 1000,
                     record_bool: bool = True,
                     save_recordings_path: str | Path = None,
                     store_every_n_measurements: int = 60000,  # equals 1 min
                     working_memory_size: int = 600000,  # equals 10 min, serial_measurements to store in RAM before clean-up
                     **read_serial_kwargs,
                     ):
    """
    Continuously samples sensor data from serial input and updates a shared dictionary for inter-process communication.

    Parameters
    ----------
    shared_dict : multiprocessing.Manager.dict
        Shared dictionary object to store the latest sample values for each measurement label.
    force_serial_save_event : threading.Event or multiprocessing.Event
        Event to trigger saving of the current buffered data to disk.
    serial_saving_done_event : threading.Event or multiprocessing.Event
        Event to send start trigger event ('A' via serial connection)
    start_trigger_event : threading.Event or multiprocessing.Event
        Event to signal sending of start trigger.
    stop_trigger_event : threading.Event or multiprocessing.Event
        Event to signal sending of stop trigger.
    measurement_definitions : tuple of tuples
        Each tuple contains measurement_label (str), optional processing callable, serial input marker (str) and smoothing alpha factor.
    record_bool : bool, optional
        Whether to record the processed values with timestamps (default is True).
    sampling_rate_hz : int, optional
        Desired sampling frequency in Hz (default is 1000).
    save_recordings_path : str or Path, optional
        Directory path where data recordings will be saved (default is None).
    store_every_n_measurements : int, optional
        Frequency (number of samples) to perform redundant saves (default is 10000).
    working_memory_size : int, optional
        Number of samples to hold in memory before final save and cleanup (default is 600000).
    **read_serial_kwargs : dict
        Additional keyword arguments passed to serial reading functions.

    Notes
    -----
    Initializes global variables for each measurement's historic data and applies smoothing as defined.
    Saves intermediate and final data as CSV files when triggered or memory limit reached.
    Mimics real-time sampling with sleep intervals to achieve approximate sampling rate.
    """
    # info:
    print("[INFO] If possible, operate the measuring computer in floating mode (from battery) to prevent power line noise influencing serial measurement!")

    # initialise global variables for read_sensor function:
    for measurement_label, _, _, _ in measurement_definitions:
        globals()['measurements_' + measurement_label] = []
        globals()['timestamps_' + measurement_label] = []
        globals()['_last_valid_reading_' + measurement_label] = .0

    # saving method to be called regularly and upon clean-up:
    def save_data(title_suffix: str = ''):
        # if len(serial_measurements) == 0: return  # only save if there's something to save
        if save_recordings_path is not None and record_bool:
            print(f"Saving recorded data to {save_recordings_path}")

            # prepare separate series for each measurement:
            measurement_labels = [measurement_label for measurement_label, _, _, _ in measurement_definitions]
            df_list = [pd.DataFrame(index=globals()['timestamps_' + measurement_label],
                                     data={measurement_label: globals()['measurements_' + measurement_label]},
                                     ) for measurement_label in measurement_labels]

            # merge series to df and save:
            save_df = df_list[0]
            for df in df_list[1:]:
                save_df = save_df.join(df, how='outer')
            savepath = save_recordings_path / filemgmt.file_title(f"Measurements ({' '.join(measurement_labels)}) {sampling_rate_hz}Hz{title_suffix}", '.csv')
            print(savepath)
            save_df.to_csv(savepath)

    try:
        sample_counter = 1
        while True:
            # check for command events:
            if start_trigger_event.is_set():
                command = 'A'
                print("Sending start trigger via serial!")
                start_trigger_event.clear()
            elif stop_trigger_event.is_set():
                command = 'B'
                print("Sending stop trigger via serial!")
                stop_trigger_event.clear()
            else: command = None
            # method retrieves and saves sample as well as sends command:
            samples = read_serial_measurements(measurement_definitions=measurement_definitions,
                                               record_bool=record_bool,
                                               command=command,
                                               **read_serial_kwargs)

            # store in shared memory:
            with shared_dict_lock:
                for measurement_label, _, _, _ in measurement_definitions:
                    shared_dict[measurement_label] = samples[measurement_label]

            # eventually store:
            if sample_counter % store_every_n_measurements == 0:
                save_data(title_suffix=f' Redundant Save')

            # eventually clean-up local memory:
            if sample_counter > working_memory_size:
                save_data(title_suffix=f' Interim Save WorkMem Full')
                sample_counter = 1

            if force_serial_save_event.is_set():
                save_data(title_suffix=f' Final Save')
                force_serial_save_event.clear()
                serial_saving_done_event.set()

            # simulate sampling frequency:
            time.sleep(1/sampling_rate_hz)
            sample_counter += 1
    finally:  # store data if saving path provided
        save_data()


def dummy_sampling_process(shared_dict,
                           shared_dict_lock,
                           force_serial_save_event,  # save_event callable through other functions
                           serial_saving_done_event,  # saving_done event pausing other processes
                           start_trigger_event,  # send start trigger event ('A' via serial connection)
                           stop_trigger_event,  # send stop trigger event ('B' via serial connection)
                           measurement_definitions: tuple[tuple[str, Callable[[float], float] | None, str, float]],
                           # measurement_label, processing_callable, serial_input_marker
                           custom_rand_maxs: tuple[float] = None,
                           sampling_rate_hz: int = 1000,

                           # the following parameters are unused but included to prevent errors when replacing the sampling rate function
                           save_recordings_path: str | Path = None,
                           store_every_n_measurements: int = 10000,
                           working_memory_size: int = 600000,
                           # equals 10 min, serial_measurements to store in RAM before clean-up
                           **read_serial_kwargs,
                           ):
    """
    Dummy process to simulate sensor data sampling for development and testing purposes.

    Parameters
    ----------
    shared_dict : multiprocessing.Manager.dict
        Shared dictionary to store generated sample values.
    force_serial_save_event : threading.Event or multiprocessing.Event
        Event to trigger dummy save operation.
    serial_saving_done_event : threading.Event or multiprocessing.Event
        Event to signal completion of dummy save.
    stop_trigger_event : threading.Event or multiprocessing.Event
        Event to send stop trigger event ('B' via serial connection).
    serial_saving_done_event : threading.Event or multiprocessing.Event
        Event to signal completion of saving data.
    measurement_definitions : tuple of tuples
        Each tuple contains measurement_label (str), optional processing callable, serial input marker (str) and smoothing float (0 = full smoothing, 1 = no smoothing).
    custom_rand_maxs : tuple of floats, optional
        Custom maximum values for random data generation per measurement label (default creates ascending range).
    sampling_rate_hz : int, optional
        Frequency to simulate in Hz (default is 1000).
    save_recordings_path : str or Path, optional
        Ignored in dummy process; present for API compatibility (default is None).
    store_every_n_measurements : int, optional
        Ignored in dummy process; present for API compatibility (default is 10000).
    working_memory_size : int, optional
        Ignored in dummy process; present for API compatibility (default is 600000).
    **read_serial_kwargs : dict
        Ignored in dummy process; present for API compatibility.

    Notes
    -----
    Generates random float values scaled by custom or default max values.
    Supports triggering of save events that print dummy save statements.
    Mimics sleep intervals to simulate real-time sampling frequency.
    """
    try:
        sample_counter = 1
        while True:
            # imitate check for command events:
            if start_trigger_event.is_set():
                command = 'A'
                print("Sending start trigger via serial!")
                start_trigger_event.clear()
            elif stop_trigger_event.is_set():
                command = 'B'
                print("Sending stop trigger via serial!")
                stop_trigger_event.clear()
            else:
                command = None

            # random dummy samples:
            rand_maxs = list(range(1, len(measurement_definitions) + 1)) if custom_rand_maxs is None else custom_rand_maxs
            samples = {measurement_label: np.random.rand() * rand_max for (measurement_label, _, _, _), rand_max in zip(measurement_definitions, rand_maxs)}
            with shared_dict_lock:
                for measurement_label, _, _, _ in measurement_definitions:
                    shared_dict[measurement_label] = samples[measurement_label]

            # imitate data saving:
            if force_serial_save_event.is_set():
                print('[SAMPLER] imitate (dummy) saving...')
                force_serial_save_event.clear()
                serial_saving_done_event.set()
                print('[SAMPLER] imitate (dummy) saved!')

            # simulate sampling frequency:
            time.sleep(1/sampling_rate_hz)
            sample_counter += 1
    finally:
        serial_saving_done_event.set()
        print('[SAMPLER] saved!')


############### PLOTTING METHODS ###############
def plot_onboarding_form(result_json_dir: str | Path,
                         shared_questionnaire_str):
    ### DISABLE MPL KEYBOARD SHORTCUTS
    # default keyboard shortcuts:
    keymaps = ['back', 'forward', 'fullscreen', 'grid', 'help', 'home', 'pan', 'save', 'xscale', 'yscale']

    for keymap in keymaps:  # try disabling:
        try:
            plt.rcParams[f'keymap.{keymap}'] = []
        except KeyError:
            print("Couldn't disable shortcut for ", keymap)
            pass

    ### PLOT
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(bottom=0.25)  # space for widgets
    manager = plt.get_current_fig_manager()  # change TkAgg window title
    manager.set_window_title("Participant Registration Form")
    ax.axis('off')  # hide axes (borders and ticklabels)
    fig.suptitle('Welcome to the study :)')
    ax.set_title("Please enter your personal details below. Thank you!")

    ### INPUT TEXTBOXES
    input_dict = {}  # input dict

    # define text boxes (callback_function, ax, object, on_submit(func)):
    # full name:
    def submit_name_textbox(text):
        input_dict["Name"] = text
    name_textbox_ax = fig.add_axes((0.55, 0.8, 0.39, 0.05))  # x, y, w, h
    name_textbox = TextBox(name_textbox_ax, 'Full Name (FIRST LAST):')
    name_textbox.on_submit(submit_name_textbox)

    # birthdate: text
    def submit_birthdate_textbox(text):
        input_dict["Birthdate"] = text
    birthdate_textbox_ax = fig.add_axes((0.55, 0.7, 0.39, 0.05))  # x, y, w, h
    birthdate_textbox = TextBox(birthdate_textbox_ax, 'Birthdate (DD/MM/YYYY):')
    birthdate_textbox.on_submit(submit_birthdate_textbox)

    # gender: (radiobutton) female / male / other
    gender_dropdown_label = ax.text(.42, .6, "Gender:", transform=ax.transAxes, va='center', ha='left')
    gender_dropdown_ax = fig.add_axes((0.51, 0.57, 0.4, 0.11))  # x, y, w, h
    gender_dropdown_ax.axis('off')
    gender_dropdown_ax.set_facecolor('gold')
    gender_options = ['Female', 'Male', 'Non-binary', 'Not selected']  # options for selector
    gender_dropdown = RadioButtons(gender_dropdown_ax, gender_options, active=3)
    def submit_gender_dropdown(label):
        if label != "Not selected": input_dict["Gender"] = label
    gender_dropdown.on_clicked(submit_gender_dropdown)

    # dominant hand: left / right
    dominand_hand_dropdown_label = ax.text(.3, .44, "Dominant hand:", transform=ax.transAxes, va='center', ha='left')
    dominand_hand_dropdown_ax = fig.add_axes((0.51, 0.47, 0.4, 0.1))  # x, y, w, h
    dominand_hand_dropdown_ax.axis('off')
    dominand_hand_dropdown_ax.set_facecolor('gold')
    dominant_hand_options = ['Left', 'Right', 'Not selected']  # options for selector
    dominand_hand_dropdown = RadioButtons(dominand_hand_dropdown_ax, dominant_hand_options, active=2)
    def submit_dominand_hand_dropdown(label):
        if label != "Not selected": input_dict["Dominant hand"] = label
    dominand_hand_dropdown.on_clicked(submit_dominand_hand_dropdown)

    # "Do you play an instrument? If yes, which:"
    def submit_instrument_textbox(text):
        input_dict["Instrument"] = text
    instrument_textbox_ax = fig.add_axes((0.55, 0.4, 0.39, 0.05))  # x, y, w, h
    instrument_textbox = TextBox(instrument_textbox_ax, 'Do you play an instrument? If yes, which:')
    instrument_textbox.on_submit(submit_instrument_textbox)

    # "If yes, how well:" 1-7
    skill_slider_ax = fig.add_axes((.55, .32, .39, .05))
    skill_slider = Slider(skill_slider_ax, 'If yes, how well:', 0, 7, valinit=0, valstep=1, valfmt='%i')
    def update_skill_slider(val):
        input_dict["Musical skill"] = int(val)
        fig.canvas.draw_idle()  # update view
    skill_slider.on_changed(update_skill_slider)

    # "How often do you listen to music?" Most of the day / a small part of the day / every 2 or 3 days / seldom
    listening_habit_dropdown_label = ax.text(.03, -.02, "How often do you listen to music?", transform=ax.transAxes, va='center', ha='left')
    listening_habit_dropdown_ax = fig.add_axes((0.51, 0.16, 0.4, 0.14))  # x, y, w, h
    listening_habit_dropdown_ax.axis('off')
    listening_habit_dropdown_ax.set_facecolor('gold')
    listening_habit_options = ['Most of the day', 'A small part of the day', 'Every 2 or 3 days', 'Seldom', 'Not selected']  # options for selector
    listening_habit_dropdown = RadioButtons(listening_habit_dropdown_ax, listening_habit_options, active=4)
    def submit_listening_habit_dropdown(label):
        if label != "Not selected": input_dict["Listening habit"] = label
    listening_habit_dropdown.on_clicked(submit_listening_habit_dropdown)


    ### SUBMISSION and SAVING
    # define submission_button (callback_function, ax, object, on_submit(func))
    #   on submit: check whether data is missing, otherwise save to result_json_dir and quit func
    def click_submission_button(event):
        # check for missing inputs:
        input_missing = False

        # check whether there is correct input for mandatory input fields:
        key_object_dict = {'Name': name_textbox.label, 'Birthdate': birthdate_textbox.label,
                           'Gender': gender_dropdown_label, 'Dominant hand': dominand_hand_dropdown_label,
                           'Listening habit': listening_habit_dropdown_label,}
        for key, object in key_object_dict.items():
            if key not in input_dict:  # check only mandatory fields
                key_object_dict[key].set_color('red')
                fig.canvas.draw_idle()  # update view
                input_missing = True
            else:
                incorrect_format = False  # distinct format checks below
                if key == 'Name':
                    if len(input_dict[key].split(" ")) <= 1: incorrect_format = True
                if key == 'Birthdate':  # 10 digits with two "/"
                    if len(input_dict[key].split("/")) != 3 or len(input_dict[key]) != 10: incorrect_format = True
                # if one failed -> mark cell:
                if incorrect_format:
                    key_object_dict[key].set_color('red')
                    fig.canvas.draw_idle()  # update view
                    input_missing = True
                else:  # if now correct, reset color to black
                    key_object_dict[key].set_color('black')
                    fig.canvas.draw_idle()  # update view

        if not input_missing:
            print("Input dict: ", input_dict)
            save_path = result_json_dir / filemgmt.file_title(f"Subject {input_dict['Name']} Data", ".json")
            with open(save_path, "w") as json_file:
                json.dump(input_dict, json_file, indent=4)  # Pretty print with indent=4
            print('Saved config to ', save_path)

            # write to shared memory:
            result = f"{input_dict['Name']} registered successfully!"
            shared_questionnaire_str.write(result)

            # close fig:
            plt.close()

    submission_button_ax = plt.axes([0.4, .05, 0.2, 0.075])
    submission_button = Button(submission_button_ax, 'Submit')
    submission_button.on_clicked(click_submission_button)

    plt.show()


# todo: updates too fast
def plot_breakout_screen(time_sec: float, title="Have a break. Please wait."):
    """ Plot countdown during break. Figure clouses after time_sec. """
    ### PLOT
    try:
        # initialise:
        matplotlib.use('TkAgg')  # select backend (suitable for animation)
        fig, ax = plt.subplots(figsize=(6, 3))
        manager = plt.get_current_fig_manager()  # change TkAgg window title
        manager.set_window_title("Breakout Screen")
        ax.axis('off')  # hide axes (borders and ticklabels)
        ax.set_title(title)

        # countdown:
        global remaining_time
        remaining_time = time_sec
        countdown_text = fig.text(0.3, 0.4, f"Remaining waiting time: {remaining_time:.2f}s", ha='left', va='center', fontsize=10)

        # animation:
        display_refresh_rate_hz = 10
        global display_start_time
        display_start_time = time.time()  # store to compute remaining time
        def update(frame):
            """ update view and fetch new observation. (frame is required although unused) """
            # reduce countdown:
            global remaining_time
            global display_start_time
            remaining_time = time_sec - (time.time() - display_start_time)  # total time - passed time

            # close figure upon countdown end:
            if remaining_time <= 0.0: plt.close()

            # else update text:
            countdown_text.set_text(f"Remaining waiting time: {remaining_time:.2f}s")

            # redraw and return:
            fig.canvas.draw_idle()
            return countdown_text,

        # run and show animation:
        ani = FuncAnimation(fig, update, frames=1, blit=False,
                            interval=int(1000/display_refresh_rate_hz), repeat=True)
        plt.show()

    finally:
        plt.close('all')




def plot_pretrial_familiarity_check(result_json_dir: str | Path,  # dir to save results to
                                    shared_questionnaire_str,  # shared memory for master process
                                    ):
    ### PLOT
    fig, ax = plt.subplots(figsize=(12, 2))
    manager = plt.get_current_fig_manager()  # change TkAgg window title
    manager.set_window_title("Pre-Trial Familiarity Check")
    ax.axis('off')  # hide axes (borders and ticklabels)
    ax.set_title("Please listen to this song and answer within 30 seconds. Your trial will start soon.")

    ### INPUT TEXTBOXES
    input_dict = {}  # input dict

    # define familiarity slider (callback_function, ax, object, on_submit(func)):
    slider_ax = fig.add_axes((.55, .5, .39, .1))
    slider = Slider(slider_ax, 'How well do you know this song? (0 = never heard it, 7 = can sing/hum along)', 0, 7, valinit=0, valstep=1, valfmt='%i')
    def update_slider(val):
        input_dict["Familiarity"] = int(val)
        fig.canvas.draw_idle()  # update view
    slider.on_changed(update_slider)

    ### SUBMISSION and SAVING
    # define submission_button (callback_function, ax, object, on_submit(func))
    #   on submit: check whether data is missing, otherwise save to result_json_dir and quit func
    def click_submission_button(event):
        # check for missing inputs:
        input_missing = False

        # check whether there is correct input for mandatory input fields:
        key_object_dict = {'Familiarity': slider.label,}
        for key, object in key_object_dict.items():
            if key not in input_dict:  # check only mandatory fields
                key_object_dict[key].set_color('red')
                fig.canvas.draw_idle()  # update view
                input_missing = True
            else:
                key_object_dict[key].set_color('black')
                fig.canvas.draw_idle()  # update view

        if not input_missing:
            print("Input dict: ", input_dict)
            save_path = result_json_dir / filemgmt.file_title(f"Pre-Trial Familiarity Check Data", ".json")
            with open(save_path, "w") as json_file:
                json.dump(input_dict, json_file, indent=4)  # Pretty print with indent=4
            print('Saved pre-trial check data to ', save_path)

            # write to shared string:
            result = f"Familiarity check result: {input_dict['Familiarity']}"
            shared_questionnaire_str.write(result)

            # close fig:
            plt.close()

    submission_button_ax = plt.axes([0.4, .05, 0.2, 0.15])
    submission_button = Button(submission_button_ax, 'Submit')
    submission_button.on_clicked(click_submission_button)
    plt.show()


# todo: ponder, whether to enter one lyric adds value
def plot_posttrial_rating(result_json_dir: str | Path,  # dir to save results to
                          shared_questionnaire_str,  # shared memory for master process
                          category_string: str | None = None,  # for question
                                    ):
    """ Includes music questions only if category string is provided. """
    ### PLOT
    fig, ax = plt.subplots(figsize=(15, 2))
    manager = plt.get_current_fig_manager()  # change TkAgg window title
    manager.set_window_title("Post-Trial Rating")
    ax.axis('off')  # hide axes (borders and ticklabels)
    # title (user instruction):
    ax.set_title(f"Please take a break and answer the below {'three' if category_string is not None else 'one'} question{'s' if category_string is not None else ''}.")

    ### INPUT TEXTBOXES
    input_dict = {}  # input dict

    if category_string is not None:
        # define liking slider (callback_function, ax, object, on_submit(func)):
        liking_slider_ax = fig.add_axes((.55, .7, .39, .1))
        liking_slider = Slider(liking_slider_ax, 'How did you like the song? (0: terrible, 7: extremely well)', 0, 7, valinit=0, valstep=1, valfmt='%i')
        def update_liking_slider(val):
            input_dict["Liking"] = int(val)
            fig.canvas.draw_idle()  # update view
        liking_slider.on_changed(update_liking_slider)

        # define category validation slider (callback_function, ax, object, on_submit(func)):
        category_slider_ax = fig.add_axes((.55, .5, .39, .1))
        category_slider = Slider(category_slider_ax, f"Do you think the song matches the category '{category_string.capitalize()}'? (0: not at all, 7: perfect match)", 0, 7,
                        valinit=0, valstep=1, valfmt='%i')
        def update_category_slider(val):
            input_dict["Fitting Category"] = int(val)
            fig.canvas.draw_idle()  # update view
        category_slider.on_changed(update_category_slider)

    # define mood slider (callback_function, ax, object, on_submit(func)):
    emotion_slider_ax = fig.add_axes((.55, .3 if category_string is not None else .5, .39, .1))
    emotion_slider = Slider(emotion_slider_ax,
                             f"Please rate your overall emotional state right now. (0: extremely unhappy/distressed, 7 = extremely happy/peaceful",
                             0, 7,
                             valinit=0, valstep=1, valfmt='%i')
    def update_emotion_slider(val):
        input_dict["Emotional State"] = int(val)
        fig.canvas.draw_idle()  # update view

    emotion_slider.on_changed(update_emotion_slider)

    ### SUBMISSION and SAVING
    # define submission_button (callback_function, ax, object, on_submit(func))
    #   on submit: check whether data is missing, otherwise save to result_json_dir and quit func
    def click_submission_button(event):
        # check for missing inputs:
        input_missing = False

        # check whether there is correct input for mandatory input fields:
        key_object_dict = {"Emotional State": emotion_slider.label}
        if category_string is not None:
            key_object_dict['Liking'] = liking_slider.label
            key_object_dict['Fitting Category'] = category_slider.label

        for key, object in key_object_dict.items():
            if key not in input_dict:  # check only mandatory fields
                key_object_dict[key].set_color('red')
                fig.canvas.draw_idle()  # update view
                input_missing = True
            else:
                key_object_dict[key].set_color('black')
                fig.canvas.draw_idle()  # update view

        if not input_missing:
            print("Input dict: ", input_dict)
            save_path = result_json_dir / filemgmt.file_title(f"Post-Trial Rating Data", ".json")
            with open(save_path, "w") as json_file:
                json.dump(input_dict, json_file, indent=4)  # Pretty print with indent=4
            print('Saved post-trial rating data to ', save_path)

            # write to shared string:
            result = f"Post trial rating result: {input_dict}"
            shared_questionnaire_str.write(result)

            # close fig:
            plt.close()
    submission_button_ax = plt.axes([0.4, .05, 0.2, 0.15])
    submission_button = Button(submission_button_ax, 'Submit')
    submission_button.on_clicked(click_submission_button)

    plt.show()


# todo: include pause screen (target=0 + no accuracy + text)
# todo: add accuracy metric (how to store?)
def plot_input_view(shared_dict: dict[str, float],  # shared memory from sampling process
                    shared_dict_lock,
                    measurement_dict_label: str,
                    shared_questionnaire_result_str,
                    include_gauge: bool = True,
                    display_window_len_s: int = 3,
                    display_refresh_rate_hz: int = 15,
                    y_limits: tuple[float, float] = (0, 3.3),
                    target_value: float | tuple[float, float, float] | None = None,  # either fixed line or sine-wave (tuple[min, max, freq])
                    target_corridor: float | None = None,  # draw corridor around target
                    accuracy_save_dir: Path | str | None = None,
                    pre_accuracy_phase_dur_sec: float = 5.0,
                    dynamically_update_y_limits: bool = True,
                    plot_size: tuple[float, float] = (15, 10),
                    input_unit_label: str = 'Input [V]',
                    x_label: str = 'Time [s]',
                    title: str = 'Live Input View',
                    window_title: str = 'Serial Input'):
    """
    Displays a live updating plot for a biosignal input from shared memory with optional gauge visualization.

    Parameters
    ----------
    shared_dict : dict of str to float
        Shared dictionary holding the latest measurement values keyed by measurement label.
    measurement_dict_label : str
        The key in shared_dict corresponding to the measurement to be visualized.
    include_gauge : bool, optional
        Whether to include a polar gauge visualizing the current input value (default is True).
    display_window_len_s : int, optional
        The length of the time window in seconds shown on the line plot (default is 3).
    display_refresh_rate_hz : int, optional
        Refresh rate of the display in Hertz (default is 15).
    y_limits : tuple of float, optional
        Initial y-axis limits for the line plot and gauge (default is (0, 3.3)).
    target_value : float or None, optional
        Optional target value to display as a reference line on the plot and gauge (default is None).
    dynamically_update_y_limits : bool, optional
        If True, adjusts y-axis limits dynamically based on incoming data outside current bounds (default is True).
    plot_size : tuple of float, optional
        Size of the matplotlib figure in inches (default is (15, 10)).
    input_unit_label : str, optional
        Label for the input units used on the y-axis and gauge labels (default is 'Input [V]').
    x_label : str, optional
        Label for the x-axis representing time (default is 'Time [s]').
    title : str, optional
        Title of the entire plot figure (default is 'Live Input View').

    Notes
    -----
    - Uses matplotlib's FuncAnimation for live updating views.
    - Provides a pause/continue button to control updating.
    - Implements exponential moving average smoothing internally via the sampling process.
    - Gauge is a semicircular polar plot showing current input relative to y-limits.
    - Handles dynamic rescaling of plots if incoming values exceed current y-limits.
    """
    try:
        ### PREPARE PLOT
        matplotlib.use('TkAgg')  # select backend (suitable for animation)

        global dynamic_y_limit  # variables that are dynamically adjusted during update() need to be defined globally
        dynamic_y_limit = y_limits
        global update_counter; update_counter = 0  # define display refreshment counter and sanity check
        if display_refresh_rate_hz > 20: print(f"Fps are {display_refresh_rate_hz}, which is > 20 and potentially leads to rendering issues.")

        # initial data:
        x = np.linspace(-display_window_len_s, 0, display_window_len_s*display_refresh_rate_hz)
        global y; y = np.zeros_like(x)

        if isinstance(target_value, tuple):  # for changing (sine) target
            sine_min, sine_max, sine_freq_hz = target_value  # tuple structure

            # shown y values:
            global target_y; target_y = np.zeros_like(x)  # will be updated later, same as measurement y

            # sine values (with distinct min., max. and freq. as defined in target_value) to fetch new targets from:
            sine_frames = int(display_refresh_rate_hz // sine_freq_hz)  # how many frames per sine wave (this captures sine_freq)
            target_sine_x = np.linspace(0, 2 * np.pi, sine_frames)  # whole rad range split on that amount of frames
            global target_sine_y  # these will be read out during update
            target_sine_y = sine_min + (np.sin(target_sine_x)*.5 + .5) * (sine_max - sine_min)  # scaled to min and max

        ## LINE PLOT
        # initialise figure:
        fig, dummy_ax = plt.subplots(figsize=plot_size)
        dummy_ax.grid(False)  # Disable grid lines
        dummy_ax.set_axis_off()  # Turn off the entire polar axis frame
        fig.suptitle(title)
        manager = plt.get_current_fig_manager()  # change TkAgg window title
        manager.set_window_title(window_title)

        # format and initialise line plot:
        line_ax = fig.add_subplot(122) if include_gauge else fig.add_subplot(111)
        line_ax.set_xlim(x.min(), x.max())
        line_ax.set_ylim(*y_limits)
        line_ax.set_xlabel(x_label)
        line_ax.set_ylabel(input_unit_label)
        line_ax.set_title('Rolling Input View')

        # include target:
        if target_value is not None:
            if isinstance(target_value, float):  # if fixed
                target_line = line_ax.axhline(y=target_value, color='green', lw=1, label='Target Value')
                assert y_limits[0] < target_value < y_limits[1]; "target_value must lie within defined y_limits!"

                if target_corridor is not None:  # mark target corridor
                    target_corridor_line_low = line_ax.axhline(y=target_value - target_corridor/2, color='darkgreen',
                                                               alpha=.5, lw=1, label='Target Corridor')
                    target_corridor_line_low = line_ax.axhline(y=target_value + target_corridor / 2, color='darkgreen',
                                                               alpha=.5, lw=1)

            elif isinstance(target_value, tuple):  # if sine-wave
                target_line, = line_ax.plot([], [], lw=1, color='green', label='Target Value')
                target_end_point, = line_ax.plot([], [], 'go')

                if target_corridor is not None:
                    target_corridor_line_low, = line_ax.plot([], [], lw=1, alpha=.5, color='darkgreen', label='Target Corridor')
                    target_corridor_line_high, = line_ax.plot([], [], lw=1, alpha=.5, color='darkgreen')

        line, = line_ax.plot([], [], lw=2, color='red', label='Measurement Value')
        end_point, = line_ax.plot([], [], 'ro', ms=9)

        ## GAUGE PLOT
        if include_gauge:  # format and initialise gauge plot:
            # parameters:
            gauge_radius = 10  # arbitrary, is scaled anyway
            n_xticks = 11  # number of ax ticklabels
            gauge_circumference = 7/4 * np.pi  # rad

            # initialise plot:
            gauge_ax = fig.add_subplot(121, projection='polar')
            gauge_ax.set_theta_offset(np.pi * ((gauge_circumference/np.pi-1)/2+1))  # Rotate start for gauge to be open downwards and "laying" on the ground
            gauge_ax.set_theta_direction(-1)  # Clockwise direction
            gauge_ax.set_ylim(0, gauge_radius)  # same y-limit as lineplot
            gauge_ax.grid(False)  # Disable grid lines
            gauge_ax.set_yticklabels([])  # turn off the radial ax labels

            # annotate ax:
            gauge_ax.set_xticks(np.linspace(0, gauge_circumference, n_xticks))
            gauge_ax.set_xticklabels([f"{tick:.2f}" for tick in np.linspace(dynamic_y_limit[0], dynamic_y_limit[1], n_xticks)])
            gauge_ax.set_xlabel(input_unit_label)
            gauge_ax.set_title('Force Level')

            # beautify gauge:
            gauge_ax.spines['polar'].set_visible(False)  # hide polar spine (replaced below) because we don't use full circle
            angles = np.linspace(0,  gauge_circumference, 100)  # gauge background semicircle
            radii = np.full_like(angles, gauge_radius)
            gauge_ax.plot(angles, radii, color='lightgray', linewidth=20, solid_capstyle='round')
            gauge_ax.bar([0, gauge_circumference], [gauge_radius]*2, width=0.03, color='black')  # mark ends

            # initialise current value line:
            needle_line, = gauge_ax.plot([], [], lw=3, color='red', label='Current Value')

            # include target:
            if target_value is not None:  # is set during update anyway so need to differentiate constant and sine here
                target_needle_line, = gauge_ax.plot([], [], lw=2, color='green', label='Target Value')
                if target_corridor is not None:
                    target_corridor_low_needle_line, = gauge_ax.plot([], [], lw=1, alpha=.5, color='darkgreen', label='Target Corridor')
                    target_corridor_high_needle_line, = gauge_ax.plot([], [], lw=1, alpha=.5, color='darkgreen')

            # function later required to update values:
            def convert_y_to_angle(y_value: float) -> float:
                return (y_value / dynamic_y_limit[1]) * gauge_circumference

        # variable, function and button for pausing:
        global is_running
        is_running = True

        def pause_button_click(event):
            global is_running
            is_running = not is_running
            if is_running: button.label.set_text("Pause")
            else: button.label.set_text("Continue")

        ax_button = plt.axes([0.8, .9, 0.1, 0.04])
        button = Button(ax_button, 'Pause')
        button.on_clicked(pause_button_click)

        ## gamification
        global record_accuracy_bool
        record_accuracy_bool = False  # will be set to True after trial phase, remains False if accuracy_save_dir not defined
        if accuracy_save_dir is not None:
            # accuracy measurement:
            global accuracy_list; accuracy_list = []  # will hold measurements
            global store_accuracy  # called to store measurements
            def store_accuracy(current: float, target: float) -> None:
                """ Squared distance. """
                accuracy_list.append((target - current) ** 2)

        # trial status:
        trial_status_text = line_ax.text(.0, 1.05, "", transform=line_ax.transAxes)
        global time_until_accuracy_measurement  # will define remaining pre-accuracy time
        time_until_accuracy_measurement = pre_accuracy_phase_dur_sec
        global display_start_time  # store to compute remaining pre-accuracy time
        display_start_time = time.time()

        ### ANIMATION METHODS
        def init():
            # initialise lineplot:
            line_ax.legend()
            line.set_data(x, y)  # set data of line
            end_point.set_data([x.max()], [0])

            if isinstance(target_value, tuple):
                target_line.set_data(x, target_y)
                target_end_point.set_data([x.max()], [0])

                if target_corridor is not None:
                    target_corridor_line_low.set_data(x, target_y)
                    target_corridor_line_high.set_data(x, target_y)

            # if gamification desired (accuracy storing):
            if accuracy_save_dir is not None:
                trial_status_text.set_text(f"Accuracy measurement starts in: {time_until_accuracy_measurement:.2f}sec")

            if include_gauge:  # initialise gauge
                gauge_ax.legend()
                needle_line.set_data([0, 0], [0, gauge_radius])
                if target_value is not None and isinstance(target_value, float):  # mark target if constant (= float type)
                    target_needle_line.set_data([0, convert_y_to_angle(target_value)], [0, gauge_radius])

                    if target_corridor is not None:  # mark target corridor
                        target_corridor_low_needle_line.set_data(
                            [0, convert_y_to_angle(target_value - target_corridor / 2)], [0, gauge_radius])
                        target_corridor_high_needle_line.set_data(
                            [0, convert_y_to_angle(target_value + target_corridor / 2)], [0, gauge_radius])

            # return only what's necessary:
            output_tuple = (line, end_point)  # measurement line and its endpoint
            if target_value is not None: output_tuple = output_tuple + (target_line,)   # target line
            if include_gauge:
                output_tuple = output_tuple + (needle_line,)   # gauge measurement needle
                if target_value is not None: output_tuple = output_tuple + (target_needle_line,)   # gauge target needle
            return output_tuple

        if isinstance(target_value, tuple):  # for sine varying target
            # define global counter (within sine wave data)
            global current_sine_ind
            current_sine_ind = 0

        def update(frame):
            """ update view and fetch new observation. (frame is required although unused) """
            global target_y  # global definition at begin of function
            global time_until_accuracy_measurement
            global record_accuracy_bool
            global accuracy_list
            global display_start_time

            # update only if is_running:
            if is_running:
                ## MEASUREMENTS
                with shared_dict_lock:
                    new_obs = shared_dict[measurement_dict_label]  # retrieve new information
                if dynamically_update_y_limits:  # update y limit if it doesn't fit
                    # check if update necessary and change parameters:
                    global dynamic_y_limit  # this also affects convert_y_to_angle
                    if new_obs > dynamic_y_limit[1]:
                        dynamic_y_limit = (dynamic_y_limit[0], new_obs)
                        update_y_lim = True

                    elif new_obs < dynamic_y_limit[0]:
                        dynamic_y_limit = (new_obs, dynamic_y_limit[1])
                        update_y_lim = True
                    else: update_y_lim = False

                    # refresh display:
                    if update_y_lim:
                        if include_gauge:  # adjust gauge ticks and target:
                            gauge_ax.set_xticklabels([f"{tick:.2f}" for tick in np.linspace(dynamic_y_limit[0], dynamic_y_limit[1], n_xticks)])

                            if target_value is not None and isinstance(target_value, float):  # constant target
                                target_needle_line.set_data([0, convert_y_to_angle(target_value)], [0, gauge_radius])

                                if target_corridor is not None:  # mark target corridor
                                    target_corridor_low_needle_line.set_data(
                                        [0, convert_y_to_angle(target_value - target_corridor / 2)], [0, gauge_radius])
                                    target_corridor_high_needle_line.set_data(
                                        [0, convert_y_to_angle(target_value + target_corridor / 2)], [0, gauge_radius])

                        line_ax.set_ylim(*dynamic_y_limit)  # adjust lineplot
                        fig.canvas.draw_idle()  # redraw

                # shift data and append new observation:
                global y; y = np.roll(y, -1); y[-1] = new_obs

                # update line plot:
                line.set_ydata(y); end_point.set_ydata([y[-1]])

                # update gauge plot:
                if include_gauge: needle_line.set_data([0, convert_y_to_angle(new_obs)], [0, gauge_radius])

                ## accuracy computation:
                # store measurement
                if record_accuracy_bool:  # remains False if accuracy_save_dir is None
                    store_accuracy(current=new_obs, target=target_y[-1] if isinstance(target_value, tuple) else target_value)
                # based on current target (before adapting because that is what user saw)

                # manage pre-measurement phase and show measurement status:
                if accuracy_save_dir is not None:
                    if time_until_accuracy_measurement > 0:
                        trial_status_text.set_text(f"Accuracy measurement starts in: {time_until_accuracy_measurement:.2f}sec")
                        time_until_accuracy_measurement = pre_accuracy_phase_dur_sec - (time.time() - display_start_time)
                        record_accuracy_bool = False
                    else:  # show recent accuracy and set record bool to True:
                        current_accuracy = accuracy_list[-1] if len(accuracy_list) > 0 else None
                        if current_accuracy is not None:
                            new_color = 'darkgreen' if current_accuracy < 50 else ('red' if current_accuracy > 250 else 'black')
                            trial_status_text.set_text(f"Current accuracy (sq. dist.): {current_accuracy:.2f}")
                            trial_status_text.set_color(new_color)
                        record_accuracy_bool = True

                ## TARGET VALUES
                if isinstance(target_value, tuple):  # for varying target
                    # shift and append new target:
                    global current_sine_ind  # current sine counter (counts within target_sine_y)
                    new_target = target_sine_y[current_sine_ind]  # read current sine position
                    target_y = np.roll(target_y, -1); target_y[-1] = new_target

                    # update line plot:
                    target_line.set_ydata(target_y); target_end_point.set_ydata([target_y[-1]])

                    if target_corridor is not None:  # update_target_corridor
                        target_corridor_line_low.set_ydata(target_y - target_corridor / 2)
                        target_corridor_line_high.set_ydata(target_y + target_corridor / 2)

                    # update gauge plot:
                    if include_gauge:
                        target_needle_line.set_data([0, convert_y_to_angle(target_y[-1])], [0, gauge_radius])
                        if target_corridor is not None:  # update target corridor
                            target_corridor_low_needle_line.set_data(
                                [0, convert_y_to_angle(target_y[-1] - target_corridor / 2)], [0, gauge_radius])
                            target_corridor_high_needle_line.set_data(
                                [0, convert_y_to_angle(target_y[-1] + target_corridor / 2)], [0, gauge_radius])

                    # update sine counter (pause is considered by having it in is_running condition):
                    current_sine_ind = (current_sine_ind + 1) % (len(target_sine_y)-1)

            # return only what's necessary:
            output_tuple = (line, end_point)  # measurement line and its endpoint
            if target_value is not None: output_tuple = output_tuple + (target_line,)  # target line
            if include_gauge:
                output_tuple = output_tuple + (needle_line,)  # gauge measurement needle
                if target_value is not None: output_tuple = output_tuple + (
                    target_needle_line,)  # gauge target needle
            return output_tuple

        # run and show animation:
        ani = FuncAnimation(fig, update, frames=len(x)+1,
                            init_func=init, blit=False,
                            interval=int(1000/display_refresh_rate_hz), repeat=True)
        plt.show()

    finally:
        if accuracy_save_dir is not None:
            # compute RMSE and display:
            rmse = np.sqrt(np.nanmean(accuracy_list))  # RMSE
            result_str = f"Achieved RMSE: {rmse:.3f}"
            print(result_str); shared_questionnaire_result_str.write(result_str)

            # store as csv:
            save_path = accuracy_save_dir / filemgmt.file_title("Trial Accuracy Results", ".csv")
            pd.Series(data=accuracy_list).to_csv(save_path)
            print("Saved accuracy results to: ", save_path)

        plt.close('all')


def qtc_control_master_view(shared_dict: dict[str, float],  # shared memory from sampling process
                            shared_dict_lock,
                            start_trigger_event,
                            stop_trigger_event,
                            start_onboarding_event,
                            start_mvc_calibration_event,
                            start_sampling_event,
                            start_music_motor_task_event,
                            start_silent_motor_task_event,
                            shared_questionnaire_result_str,
                            shared_song_info_dict,
                            force_log_saving_event,
                            log_saving_done_event,
                            start_test_motor_task_event,
                            plot_size: tuple[float, float] = (12, 3),
                            title: str = "Experiment Control Master",
                            display_refresh_rate_hz: float = 3,
                            music_category_txt: str | Path | None = None,
                            control_log_dir: str | Path | None = None,
                            save_log_working_memory_size: int = 60000,
                            window_title: str = "Master",
                            ):
    """
    Displays a control master view for managing start/stop triggers and monitoring shared biosignal measurement keys.

    Parameters
    ----------
    shared_dict : dict of str to float
        Shared dictionary holding the latest measurement values keyed by measurement label; used to display keys.
    start_trigger_event : threading.Event or multiprocessing.Event
        Event to signal the start of sampling; triggered by the Start button.
    stop_trigger_event : threading.Event or multiprocessing.Event
        Event to signal the stop of sampling; triggered by the Stop button.
    plot_size : tuple of float, optional
        Size of the matplotlib figure in inches (default is (5, 2)).
    title : str, optional
        The title shown on the figure window (default is "Quattrocento Control Master").
    display_refresh_rate_hz : float, optional
        Update frequency of the display in Hertz (default is 5).

    Returns
    -------
    None
        Creates an interactive window with Start and Stop buttons and a status text that updates regularly.

    Notes
    -----
    - Uses matplotlib’s FuncAnimation to periodically update status text reflecting keys in shared_dict.
    - Integrates two buttons to set/clear events used by a sampling process.
    - Designed for synchronization and control of data acquisition pipelines in multiprocessing contexts.
    - Automatically closes figure upon window close or termination.
    """
    try:
        matplotlib.use('TkAgg')  # select backend (suitable for animation)

        if music_category_txt is not None:  # initialise spotify controller
            print("Initialising SpotifyController instance. Remember opening spotify!")
            music_master = SpotifyController(category_url_dict=music_category_txt, randomly_shuffle_category_lists=True)
            include_music = True
        else: include_music = False

        # initialise figure:
        fig, dummy_ax = plt.subplots(figsize=plot_size)
        dummy_ax.grid(False)  # Disable grid lines
        dummy_ax.set_axis_off()  # Turn off the entire polar axis frame
        fig.suptitle(title)
        manager = plt.get_current_fig_manager()  # change TkAgg window title
        manager.set_window_title(window_title)

        ### BUTTONS
        # quattrocento triggers::
        global recent_event_str
        recent_event_str = None
        def start_button_click(event):
            start_trigger_event.set()
            global recent_event_str
            recent_event_str = 'Start Trigger'
            start_button.label.set_text("Start Trigger\n(Done)")
            stop_button.label.set_text("Stop Trigger")
        def stop_button_click(event):
            stop_trigger_event.set()
            global recent_event_str
            recent_event_str = 'Stop Trigger'
            start_button.label.set_text("Start Trigger")
            stop_button.label.set_text("Stop Trigger\n(Done)")
        otb_control_label = fig.text(0.1, 0.86, "OTB400 Control:", ha='left', va='center', fontsize=10)
        start_button_ax = plt.axes([0.1, .65, 0.175, 0.175])
        start_button = Button(start_button_ax, 'Start Trigger')
        start_button.on_clicked(start_button_click)
        stop_button_ax = plt.axes([0.3, .65, 0.175, 0.175])
        stop_button = Button(stop_button_ax, 'Stop Trigger')
        stop_button.on_clicked(stop_button_click)

        # experiment phase triggers:
        def click_onboarding_button(event):
            start_onboarding_event.set()
            global recent_event_str
            recent_event_str = 'Onboarding Phase'
            onboarding_button.label.set_text("Onboarding\n(Done)")
        experiment_control_label = fig.text(0.525, 0.86, "Experiment Control:", ha='left', va='center', fontsize=10)
        onboarding_button_ax = plt.axes([0.55, .65, 0.08, 0.175])
        onboarding_button = Button(onboarding_button_ax, 'Onboarding')
        onboarding_button.on_clicked(click_onboarding_button)

        def click_mvc_button(event):
            start_mvc_calibration_event.set()
            global recent_event_str
            recent_event_str = 'MVC Calibration Phase'
            mvc_button.label.set_text("MVC\n(Done)")
        mvc_button_ax = plt.axes([0.64, .65, 0.08, 0.175])
        mvc_button = Button(mvc_button_ax, 'MVC')
        mvc_button.on_clicked(click_mvc_button)

        def click_sampling_button(event):
            start_sampling_event.set()
            global recent_event_str
            recent_event_str = 'Sampling Phase'
            sampling_button.label.set_text("Sampling\n(Done)")
        sampling_button_ax = plt.axes([0.73, .65, 0.08, 0.175])
        sampling_button = Button(sampling_button_ax, 'Sampling')
        sampling_button.on_clicked(click_sampling_button)

        def click_test_task_button(event):
            start_test_motor_task_event.set()
            global recent_event_str
            recent_event_str = 'Test Task Phase'
            test_task_button.label.set_text("Test Task\n(Done)")
        test_task_button_ax = plt.axes([0.82, .65, 0.08, 0.175])
        test_task_button = Button(test_task_button_ax, 'Test Task')
        test_task_button.on_clicked(click_test_task_button)

        # define music control instruments:
        if include_music:
            # display random category order:
            n_categories = len(music_master.category_url_dict.keys()) / 2 + 1  # /2 to remove familiar/unfamiliar, +1 for silence
            n_buttons = len(music_master.category_url_dict.keys())
            button_indices = list(range(1, int(n_categories)+1))
            random.shuffle(button_indices)  # random sequence
            # extend so that all elements occur twice after silence:
            final_button_indices = button_indices[:1]
            for element in button_indices[1:]: final_button_indices += [element]*2

            # resume and pause buttons:
            music_button_label = fig.text(0.1, 0.56, "Music / Task Control: (button indices propose a random category sequence)", ha='left', va='center', fontsize=10)

            # silence trial:
            silence_trial_button_ax = plt.axes([0.1, .35, 0.1, 0.175])
            silence_trial_button = Button(silence_trial_button_ax, f'Silence Trial ({final_button_indices[0]})')
            def silence_trial_button_clicked(event):
                music_master.pause()
                start_silent_motor_task_event.set()  # start silent motor task
            silence_trial_button.on_clicked(silence_trial_button_clicked)

            # pause / resume:
            pause_resume_button_ax = plt.axes([0.8, .35, 0.1, 0.175])
            pause_resume_button = Button(pause_resume_button_ax, 'Pause/Resume')
            def pause_resume_button_clicked(event):
                if song_info_text.get_text() == "No track playing currently.":
                    music_master.resume()
                    pause_resume_button.label.set_text("Pause")
                else:
                    music_master.pause()
                    pause_resume_button.label.set_text("Resume")
            pause_resume_button.on_clicked(pause_resume_button_clicked)

            ## category buttons (dynamically created based on amount of defined categories):
            # define positions:
            n_rows = 2
            width_button = (.5 / n_buttons * n_rows) * 1
            button_x_positions = list(np.linspace(.225, .775-width_button, int(n_buttons / n_rows))) * n_rows
            button_y_positions = [.455] * int(n_buttons / n_rows)
            for row_ind in range(1, n_rows):  # add n_buttons / n_rows downshifted y_coords
                button_y_positions += [button_y_positions[-1] - .105 * row_ind] * int(n_buttons / n_rows)

            # create category buttons:
            for category, button_x_pos, button_y_pos, button_index in zip(music_master.category_url_dict.keys(),
                                                                          button_x_positions, button_y_positions,
                                                                          final_button_indices[1:]  # exclude silence
                                                                          ):
                temp_button_ax = plt.axes((button_x_pos, button_y_pos, width_button, .07))
                globals()[f'{category}_button'] = Button(temp_button_ax, f'{category} ({button_index})')
                # define function (category needs to be saved as default value due to late binding)
                globals()[f'{category}_button_clicked'] = lambda event, cat=category: (
                    music_master.play_next_from(cat), start_music_motor_task_event.set()  # play music and start music motor task
                )
                globals()[f'{category}_button'].on_clicked(globals()[f'{category}_button_clicked'])

        # status texts:
        measurement_info_text = fig.text(0.1, 0.25, "", ha='left', va='center', fontsize=10)
        if include_music: song_info_text = fig.text(0.1, 0.1, "", ha='left', va='center', fontsize=10)
        rating_result_info_text = fig.text(.1, .175, "", ha='left', va='center', fontsize=10)

        ### animation methods:
        def init():
            measurement_info_text.set_text("Initializing...")
            if include_music: song_info_text.set_text("Initializing...")
            rating_result_info_text.set_text("Initializing...")
            return (measurement_info_text, song_info_text) if include_music else measurement_info_text

        # for log saving:
        global save_log_counter
        save_log_counter = 1  # for saving log file
        global last_rating_result; last_rating_result = ""  # to only log new rating results
        global current_rating_result; current_rating_result = ""  # to store current rating results and compare

        # log helper functions:
        def initialise_log_dict(verbose: bool = True):
            if control_log_dir is not None:  # initialise log file
                global log_dict
                log_dict = {'Time': [], 'Music': [], 'Event': [], 'Questionnaire': []}  # content of log file
                if verbose: print(f"Initialising log file. Will save and reset full file every {(save_log_working_memory_size/display_refresh_rate_hz):.2f} s and do interim saves every {(save_log_working_memory_size/display_refresh_rate_hz/200):.2f} s!")
        initialise_log_dict()
        global save_log_dict
        def save_log_dict(suffix: str | None = None, verbose: bool = True):
            if control_log_dir is not None:
                savepath = control_log_dir / filemgmt.file_title(f"Experiment Log{f' {suffix}' if suffix is not None else ''}", ".csv")
                pd.DataFrame(log_dict).to_csv(savepath, index=False)
                if verbose:  # status message
                    print(f"Saved master control log to: {savepath}")

        def update(frame):
            """ update view and fetch new observation. (frame is required although unused) """
            # update only if is_running:
            ### update view (info texts):
            with shared_dict_lock:
                measurement_info_text.set_text(f"Receiving Serial Measurements: {list(shared_dict.keys())}")

            if include_music:
                if music_master.current_category_and_counter is not None:  # e.g. Groovy (1/8)
                    current_cat_str = f"{music_master.current_category_and_counter[0]} ({music_master.current_category_and_counter[1]+1}/{len(music_master.category_url_dict[music_master.current_category_and_counter[0]])})"" | "
                else: current_cat_str = ""
                try:
                    current_track_info = music_master.get_current_track(output_type='dict')
                    # update displayed song info text: (e.g. "Hallelujah by Leonard Cohen | 34.29s / 194.38s")
                    current_track_str = f"{current_track_info['Title']} by {current_track_info['Artist']} | {current_track_info['Position [s]']:.2f}s / {current_track_info['Duration [ms]']/1000:.2f}s"
                    song_info_text.set_text(current_cat_str + current_track_str)

                    # store in shared song info dict:
                    _ = current_track_info.pop('Position [s]')  # remove position argument
                    with shared_dict_lock:
                        shared_song_info_dict.update(current_track_info)  # store rest
                        if music_master.current_category_and_counter is not None:  # add category information
                            shared_song_info_dict['Category'] = music_master.current_category_and_counter[0]
                            shared_song_info_dict['Category Index'] = music_master.current_category_and_counter[1]

                except ValueError:  # no music playing currently
                    song_info_text.set_text("No track playing currently.")

            global current_rating_result
            current_rating_result = shared_questionnaire_result_str.read()
            rating_result_info_text.set_text(current_rating_result)

            ### logging:
            global save_log_counter
            global recent_event_str
            global last_rating_result

            # log updating:
            if control_log_dir is not None:
                global log_dict
                log_dict['Time'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                log_dict['Music'].append(song_info_text.get_text())

                if recent_event_str is not None:  # save and reset in one block to prevent resetting event str without logging
                    log_dict['Event'].append(recent_event_str)
                    recent_event_str = None
                else: log_dict['Event'].append(recent_event_str)

                if current_rating_result != last_rating_result:  # log only new rating results
                    log_dict['Questionnaire'].append(current_rating_result)
                    last_rating_result = current_rating_result
                else: log_dict['Questionnaire'].append(None)

                save_log_counter += 1  # because this is only increased here, we can omit the condition below:

            # log saving:
            if save_log_counter % save_log_working_memory_size == 0:  # save and working memory reset
                save_log_dict(suffix="Working Memory Full Save")
                initialise_log_dict(verbose=False)  # resets working memory
            elif save_log_counter % (save_log_working_memory_size // 200) == 0:  # interim save
                save_log_dict(suffix="Interim Save")
            elif force_log_saving_event.is_set():  # forced full save
                save_log_dict(suffix="Forced Full Save")
                force_log_saving_event.clear()
                log_saving_done_event.set()

            ## return
            return (measurement_info_text, song_info_text) if include_music else measurement_info_text

        # run and show animation:
        ani = FuncAnimation(fig, update, frames=1,
                            init_func=init, blit=False,
                            interval=int(1000/display_refresh_rate_hz), repeat=True)
        plt.show()

    finally:
        save_log_dict(suffix="Final Full Save")
        force_log_saving_event.clear()
        log_saving_done_event.set()
        plt.close('all')


############### MULTIPROCESSING METHODS ###############
class RobustEventManager:
    """ Triggers events and safely waits for triggers while preventing deadlocks through timeouts. """
    def __init__(self):
        self.event = multiprocessing.Event()
        self.lock = multiprocessing.Lock()
        self.trigger_count = multiprocessing.Value('i', 0)

    def set(self):
        with self.lock:
            self.trigger_count.value += 1
            self.event.set()

    def is_set(self):
        return self.event.is_set()

    def wait(self, timeout=None):
        initial_count = self.trigger_count.value

        if timeout is None:
            # Wait indefinitely
            while True:
                if self.event.wait(timeout=1):  # Short timeout for checks
                    with self.lock:
                        if self.trigger_count.value > initial_count:
                            return True
        else:
            # Wait with timeout
            while timeout > 0:
                if self.event.wait(timeout=min(1, timeout)):  # Short timeout for checks
                    with self.lock:
                        if self.trigger_count.value > initial_count:
                            return True
                timeout -= 1
                if timeout <= 0:
                    return False
            return False

    def clear(self):
        with self.lock:
            self.event.clear()
            self.trigger_count.value = 0


class SharedString:
    """
    Thread-safe wrapper for shared string storage using multiprocessing.Array.

    Creates an instance object that can be passed between processes and provides
    instance methods for safe read/write operations with automatic lock management.

    Attributes:
        buffer (multiprocessing.Array): Shared character buffer
        lock (multiprocessing.Lock): Synchronization lock
        max_size (int): Maximum buffer capacity
    """

    def __init__(self, size: int, initial_value: str = ""):
        """
        Initialize shared string instance with specified size.

        Parameters:
            size (int): Maximum buffer size in bytes (includes null terminator)
            initial_value (str): Optional initial string value

        Raises:
            ValueError: If initial_value exceeds size limit
            TypeError: If size is not positive integer
        """
        # Validate inputs
        if not isinstance(size, int) or size <= 0:
            raise TypeError(f"size must be positive integer, got {size}")

        if not isinstance(initial_value, str):
            raise TypeError(f"initial_value must be str, got {type(initial_value)}")

        # Check overflow
        encoded_init = initial_value.encode('utf-8')
        if len(encoded_init) >= size:
            raise ValueError(
                f"initial_value too long: {len(encoded_init)} bytes "
                f"exceeds buffer size {size}"
            )

        # Create shared buffer and lock
        self.buffer = multiprocessing.Array('c', size)
        self.lock = multiprocessing.Lock()
        self.max_size = size

        # Write initial value
        if initial_value:
            self.write(initial_value)

    def write(self, value: str) -> None:
        """
        Safely write string to shared buffer with null termination.

        Parameters:
            value (str): String to write

        Raises:
            ValueError: If value exceeds buffer capacity
            TypeError: If value is not string
        """
        if not isinstance(value, str):
            raise TypeError(f"value must be str, got {type(value)}")

        # Encode and validate size
        encoded = value.encode('utf-8')
        if len(encoded) >= self.max_size:
            raise ValueError(
                f"value too long: {len(encoded)} bytes "
                f"exceeds buffer capacity {self.max_size}"
            )

        # Write to buffer with lock
        with self.lock:
            # Clear previous data
            self.buffer[:] = [0] * self.max_size

            # Write encoded string as list of byte integers
            self.buffer[:len(encoded)] = list(encoded)

            # Add null terminator at end of string
            self.buffer[len(encoded)] = 0

    def read(self) -> str:
        """
        Safely read string from shared buffer with null-termination handling.

        Returns:
            str: Decoded string with null bytes stripped

        Raises:
            UnicodeDecodeError: If buffer contains invalid UTF-8
        """
        # Read from buffer with lock
        with self.lock:
            # Convert buffer slice to bytes
            raw_bytes = bytes(self.buffer[:])

            # Strip null bytes and decode
            try:
                decoded = raw_bytes.rstrip(b'\x00').decode('utf-8')
            except UnicodeDecodeError as e:
                raise UnicodeDecodeError(
                    e.encoding,
                    e.object,
                    e.start,
                    e.end,
                    f"Invalid UTF-8 in shared buffer: {e.reason}"
                ) from e

        return decoded

    def get_lock(self) -> multiprocessing.Lock:
        """
        Retrieve the synchronization lock for manual context management.

        Returns:
            multiprocessing.Lock: Lock object
        """
        return self.lock

    def get_size(self) -> int:
        """
        Get maximum buffer capacity.

        Returns:
            int: Max size in bytes
        """
        return self.max_size


def save_terminate_process(process: multiprocessing.Process) -> None:
    """ First request termination, then force-kill process."""
    if process.is_alive():  # terminate process (request)
        process.terminate()
        process.join(timeout=1.0)

        if process.is_alive():  # if still alive kill (force)
            process.kill()
            process.join()

    if process.pid is not None:  # then process was started at some point
        process.join()


### OLD MULTIPROCESSING IMPLEMENTATION:

def start_measurement_processes(measurement_definitions: tuple[tuple[str, Callable[[float], float] | None, str, float]],measurement_saving_path: str | Path = None,
                                measurement_sampling_rate_hz: int = 1000,
                                record_measurements: bool = True,
                                # measurement_label, processing_callable, serial_input_marker
                                music_category_txt: str | Path | None = None,
                                control_log_dir: str | Path | None = None,
                                ) -> None:
    # sanity check:
    if not record_measurements: print("[WARNING] Measurement recording is deactivated! No measurements and control logs will be saved.")

    # initialise shared:
    shared_dict = multiprocessing.Manager().dict()
    measurement_labels = []  # for dynamic object definition below
    for measurement_label, _, _, _ in measurement_definitions:
        shared_dict[measurement_label] = .0
        measurement_labels.append(measurement_label)

    # saving event:
    force_serial_save_event = RobustEventManager()
    serial_saving_done_event = RobustEventManager()
    start_trigger_event = RobustEventManager()
    stop_trigger_event = RobustEventManager()
    start_onboarding_event = RobustEventManager()
    start_mvc_calibration_event = RobustEventManager()
    start_sampling_event = RobustEventManager()
    start_music_motor_task_event = RobustEventManager()  # called upon starting a song

    # define processes:
    sampler = multiprocessing.Process(
        target=dummy_sampling_process if not record_measurements else sampling_process,  # if not recording use dummy,
        args=(shared_dict, force_serial_save_event, serial_saving_done_event, start_trigger_event, stop_trigger_event,),
        kwargs={'measurement_definitions': measurement_definitions,
                'sampling_rate_hz': measurement_sampling_rate_hz,
                'save_recordings_path': measurement_saving_path,
                'record_bool': record_measurements,
                'baud_rate': 115200,},
        name="SamplingProcess")

    if 'fsr' in measurement_labels:
        fsr_displayer = multiprocessing.Process(
            target=plot_input_view,
            args=(shared_dict,),
            kwargs={'measurement_dict_label': 'fsr',
                    'target_value': (15, 30, 1),  # target value as sine wave with .1 Hz
                    'target_corridor': 10,
                    'include_gauge': True,
                    'title': 'FSR Input',
                    'input_unit_label': 'Force [% MVC]',
                    'y_limits': (0, 100),
                    },
            name="FSRDisplayProcess")

    if 'ecg' in measurement_labels:
        ecg_displayer = multiprocessing.Process(
            target=plot_input_view,
            args=(shared_dict,),
            kwargs={'measurement_dict_label': 'ecg',
                    'target_value': None,
                    'include_gauge': False,
                    'title': 'ECG Input'
                    },
            name="ECGDisplayProcess")

    if 'gsr' in measurement_labels:
        gsr_displayer = multiprocessing.Process(
            target=plot_input_view,
            args=(shared_dict,),
            kwargs={'measurement_dict_label': 'gsr',
                    'target_value': 1.2,
                    'include_gauge': False,
                    'title': 'GSR Input'
                    },
            name="ECGDisplayProcess")

    master_displayer = multiprocessing.Process(
        target=qtc_control_master_view,
        args=(shared_dict, start_trigger_event, stop_trigger_event,
              start_onboarding_event, start_mvc_calibration_event, start_sampling_event, start_music_motor_task_event),
        kwargs={'music_category_txt': music_category_txt,
                'control_log_dir': control_log_dir if record_measurements else None,
                },
        name="MasterDisplayProcess")

    # start processes:
    try:
        sampler.start()
        if 'fsr' in measurement_labels: fsr_displayer.start()
        if 'ecg' in measurement_labels: ecg_displayer.start()
        if 'gsr' in measurement_labels: gsr_displayer.start()
        master_displayer.start()

        # Wait for processes with timeout
        sampler.join()  #timeout=300  # 5 minute timeout (unused currently, main script ends anyway)
        if 'fsr' in measurement_labels: fsr_displayer.join()  #timeout=300
        if 'ecg' in measurement_labels: ecg_displayer.join()
        if 'gsr' in measurement_labels: gsr_displayer.join()
        master_displayer.join()

    except KeyboardInterrupt:
        print("Terminating processes...")
        force_serial_save_event.set()  # trigger sampler saving
        print('Waiting for saving...')
        serial_saving_done_event.wait(timeout=5)  # wait until done
        print('Saving done!')

        # stop all processes:
        sampler.terminate()
        if 'fsr' in measurement_labels: fsr_displayer.terminate()
        if 'ecg' in measurement_labels: ecg_displayer.terminate()
        if 'gsr' in measurement_labels: gsr_displayer.terminate()
        master_displayer.terminate()

        sampler.join()
        if 'fsr' in measurement_labels: fsr_displayer.join()
        if 'ecg' in measurement_labels: ecg_displayer.join()
        if 'gsr' in measurement_labels: gsr_displayer.join()
        master_displayer.join()

    finally:
        print("Cleanup completed")

if __name__ == '__main__':
    # define saving folder:
    ROOT = Path().resolve().parent.parent
    SERIAL_MEASUREMENTS = ROOT / "data" / "serial_measurements"
    EXPERIMENT_LOG = ROOT / "data" / "experiment_logs"
    CONFIG_DIR = ROOT / "config"
    MUSIC_CONFIG = CONFIG_DIR / "music_selection.txt"

    # important:
    SUBJECT_DIR = ROOT / "data" / "experiment_results" / "subject_00"
    SONG_ONE_DIR = SUBJECT_DIR / "song_00"



    start_measurement_processes(measurement_definitions=(("fsr",  # measurement label
                                                          (dynamometer_force_mapping, 100),  # 60 = MVC [kg]
                                                          "FSR:",  # serial connection measurement identifier
                                                          .1),  # smoothing alpha
                                                         #("ecg", None, "ECG:", .4),
                                                         #("gsr", None, "GSR:", .4),
                                                         ),
                                measurement_saving_path=SERIAL_MEASUREMENTS,
                                record_measurements=True,  # False: start dummy_sampling, True: start real sampling
                                music_category_txt=MUSIC_CONFIG,
                                control_log_dir=EXPERIMENT_LOG,
                                )



