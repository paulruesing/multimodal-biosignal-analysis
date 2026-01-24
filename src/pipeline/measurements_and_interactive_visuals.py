"""
This script contains process definitions and relevant auxiliary functions to be called in an experiment workflow.
These processes conduct sampling and interactive visualizations.

That external workflow needs to manage shared memory allocation and multiprocessing.
®Paul Rüsing, INI ETH / UZH
"""


import serial
import time
import math
import warnings
import json
import os
import random
from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, RadioButtons, Slider
import matplotlib.animation as animation
from pynput import keyboard
from scipy.optimize import curve_fit
from typing import Callable, Literal, Optional
import multiprocessing
from ctypes import c_char

import src.utils.file_management as filemgmt
import src.utils.str_conversion as strconv
import src.utils.multiprocessing_tools as mptools
from src.pipeline.music_control import SpotifyController

from pathlib import Path
import pandas as pd

############### MATPLOTLIB PREP ###############
# prevent AttributeErrors if cancelling animation:
_original_step = animation.Animation._step
def patched_step(self):
    """Override _step to handle None event_source gracefully."""
    if self.event_source is None:
        return False  # Stop animation cleanly
    try:
        return _original_step(self)
    except AttributeError:  # sometimes TkAgg backend causes AttributeErrors upon animation closing
        return False  # Catch interval error, stop safely
animation.Animation._step = patched_step

matplotlib.use('Qt5Agg')  # TkAgg
theme: Literal['dark', 'light'] = 'dark'

matplotlib.rcParams["toolbar"] = "none"
plt.style.use('dark_background' if theme == 'dark' else 'seaborn-v0_8-whitegrid')
button_background_color = textbox_background_color = 'darkslategray' if theme == 'dark' else 'white'
button_hover_color = textbox_hover_color = 'darkcyan' if theme == 'dark' else 'lightgrey'
slider_background_color = 'darkslategray' if theme == 'dark' else 'grey'
slider_bar_color = 'darkcyan' if theme == 'dark' else 'darkturquoise'
font_color = "white" if theme == 'dark' else "black"
radio_button_selected_color = 'aqua' if theme == 'dark' else 'darkturquoise'

target_color = 'green'
dark_target_color = 'darkgreen'
measurement_color = 'red'

# Suppress the harmless Matplotlib animation timer race condition (doesn't work because it's no warning)
'''warnings.filterwarnings(
    'ignore',
    message=".*'NoneType' object has no attribute 'interval'.*"
)'''

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

        # read new line and convert to float:
        buf = b""

        with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
            # check for output command:
            if command in ("A", "B"):
                ser.write(command.encode("ascii"))
                ser.flush()  # waits for all outgoing data to be transmitted

            line = ser.readline().decode('ascii', errors="ignore").strip()

            # process each measurement type:
            for measurement_label, processing_func, teensy_marker, smoothing_factor in measurement_definitions:
                if not line.startswith(teensy_marker):  # check whether line contains measurement result
                    # otherwise use last result:
                    output_dict[measurement_label] = globals()['_last_valid_reading_' + measurement_label]
                    if record_bool:
                        globals()['timestamps_' + measurement_label].append(datetime.now())
                        globals()['measurements_' + measurement_label].append(
                            globals()['_last_valid_reading_' + measurement_label])
                    continue  # jump to next measurement

                # format measurement:
                raw_str = line.replace(teensy_marker, "")  # formatting
                value = float(raw_str)

                # check whether input remains in feasible range:
                if not allowed_input_range[0] < value < allowed_input_range[1]:
                    # otherwise use last result:
                    output_dict[measurement_label] = globals()['_last_valid_reading_' + measurement_label]
                    if record_bool:
                        globals()['timestamps_' + measurement_label].append(datetime.now())
                        globals()['measurements_' + measurement_label].append(globals()['_last_valid_reading_' + measurement_label])
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
                    globals()['timestamps_' + measurement_label].append(datetime.now())  # is expensive:.strftime("%Y-%m-%d %H:%M:%S.%f"))
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
    return (2.2 * (v ** 4.1071) - 10) * factor  # 2.8708 * (v ** 4.1071) - 3 before!


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

            # use shortest datetime for all measurements (they are written sequentially within a short interval inside read_serial_measurement)
            #   hence we short all series in order for them to match
            shortest_len = np.min([len(globals()['timestamps_' + measurement_label]) for measurement_label in measurement_labels])
            dt_list = globals()['timestamps_' + measurement_labels[0]][:shortest_len]
            datetime_strings = pd.to_datetime(dt_list).strftime('%Y-%m-%d %H:%M:%S.%f')
            df_list = [pd.DataFrame(index=datetime_strings,
                                     data={measurement_label: globals()['measurements_' + measurement_label][:shortest_len]},
                                     ) for measurement_label in measurement_labels]

            # merge series to df and save:
            save_df = df_list[0]
            for df in df_list[1:]:
                save_df = save_df.join(df, how='outer')
            savepath = save_recordings_path / filemgmt.file_title(f"Measurements ({' '.join(measurement_labels)}) {sampling_rate_hz}Hz{title_suffix}", '.csv')
            print(savepath)
            save_df.to_csv(savepath)

    try:
        from tqdm import tqdm
        from itertools import count
        sample_counter = 1

        # store attributes locally to increase speed:
        read_serial = read_serial_measurements
        mdefs = measurement_definitions
        lock = shared_dict_lock
        while True:  #for _ in tqdm(count(start=0, step=1)):  # to display iterations per second. replace by while in production
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
            samples = read_serial(measurement_definitions=mdefs,
                                               record_bool=record_bool,
                                               command=command,
                                               **read_serial_kwargs)

            # store in shared memory:
            with lock:
                for measurement_label, _, _, _ in mdefs:
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
            # time.sleep(1/sampling_rate_hz) (with: 300 iterations per second), (without: 360it / sec)
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
                           custom_rand_means: tuple[float] = (10, 1, 1),
                           custom_rand_stds: tuple[float] = (.2, 1, 1),
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
    custom_rand_stds : tuple of floats, optional
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
            rand_means = [1 for _ in range(len(measurement_definitions))] if custom_rand_means is None else custom_rand_means
            rand_stds = [1 for _ in range(len(measurement_definitions))] if custom_rand_stds is None else custom_rand_stds
            samples = {measurement_label: rand_min + (-.5 + np.random.rand()) * rand_max for (measurement_label, _, _, _), rand_min, rand_max in zip(measurement_definitions, rand_means, rand_stds)}
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
class AnimationManager:
    def __init__(self, shutdown_event=None):
        self.anim = None
        self.fig = None
        self.shutdown_event = shutdown_event

    def start(self, fig, update_func, interval, init_func=None):
        self.fig = fig
        self.anim = animation.FuncAnimation(fig, update_func, interval=interval,
                                  blit=False, repeat=True, init_func=init_func,
                                  cache_frame_data=False)  # CRITICAL: disable frame caching)

    def stop(self):
        if self.anim is not None:  # first stop timer
            self.anim.pause()  # Pause timer first

            try:
                es = getattr(self.anim, "event_source", None)
                if es is not None:
                    es.stop()
            except AttributeError:  # accessing event_source sometimes causes AttributeErrors if animation is closed parallely
                pass

        if self.fig is not None:  # then close figure
            plt.close(self.fig)

    def check_shutdown(self) -> bool:
        if self.shutdown_event is not None:
            if self.shutdown_event.is_set():
                self.stop()
                self.shutdown_event.clear()  # reset event
                return True
        return False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()


# auxiliary functions:
def create_textbox(fig: plt.Figure,
                   input_dict: dict,
                   key: str,
                   label: str,
                   position: tuple[float, float, float, float],
                   button_background_color: str,
                   button_hover_color: str,
                   label_valign: str = 'center') -> TextBox:
    """
    Create and configure a TextBox widget with automatic input dictionary storage.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure object
    ax : plt.Axes
        Matplotlib axes object (for coordinate reference)
    input_dict : dict
        Input dictionary to store textbox value under key
    key : str
        Dictionary key for storing the textbox value
    label : str
        Label text displayed next to textbox
    position : tuple
        TextBox position as (x, y, width, height) in figure coordinates
    button_background_color : str
        Background color for textbox
    button_hover_color : str
        Hover color for textbox

    Returns
    -------
    TextBox
        Configured TextBox widget
    """

    # Define submission callback that stores value in input_dict
    def submit_callback(text: str) -> None:
        input_dict[key] = text

    # Create axes for textbox at specified position
    textbox_ax = fig.add_axes(position)

    # Create textbox with label and colors
    textbox = TextBox(
        textbox_ax,
        label + "  ",  # looks better
        color=button_background_color,
        hovercolor=button_hover_color
    )

    # adjust vertical alignment
    if hasattr(textbox, 'label'):
        textbox.label.set_verticalalignment(label_valign)

    # Register submission callback
    textbox.on_submit(submit_callback)

    return textbox


def create_radio_buttons(fig: plt.Figure,
                         ax: plt.Axes,
                         input_dict: dict,
                         key: str,
                         label: str,
                         options: list[str] | tuple[str, ...],
                         position: tuple[float, float, float, float],
                         label_position: tuple[float, float],
                         active_index: int,
                         radio_button_selected_color: str = 'gold',
                         background_color: str  = 'black',
                         skip_value: str | None = None,
                         horizontal: bool = False) -> tuple[RadioButtons, plt.Text]:
    """
    Create and configure a RadioButtons widget with automatic input dictionary storage.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure object
    ax : plt.Axes
        Matplotlib axes object for text label placement
    input_dict : dict
        Input dictionary to store selected value under key
    key : str
        Dictionary key for storing the selected option
    label : str
        Label text displayed to the left of radio buttons
    options : list or tuple
        Available options for radio button selection
    position : tuple
        RadioButtons position as (x, y, width, height) in figure coordinates
    label_position : tuple
        Label position as (x, y) in axes coordinates
    active_index : int
        Index of initially selected option (0-based)
    radio_button_selected_color : str
        Color for selected radio button
    background_color : str, optional
        Background color for radio button axes (default: 'gold')
    skip_value : str, optional
        Value to skip storing in input_dict (e.g. "Not selected")
        If provided, selection only stores if label != skip_value
    horizontal : bool, optional
        If True, arrange buttons horizontally left-to-right (default: False for vertical)

    Returns
    -------
    tuple
        (RadioButtons widget, label Text object)
    """

    # Create label text at specified position
    label_text = ax.text(
        label_position[0],
        label_position[1],
        label,
        transform=ax.transAxes,
        va='center',
        ha='right'
    )

    # Create axes for radio buttons
    radio_ax = fig.add_axes(position)
    radio_ax.axis('off')
    radio_ax.set_facecolor(background_color)

    # Create radio buttons with manual layout adjustment for horizontal
    options = list(options)
    if skip_value not in options: options.append(skip_value)

    radio_buttons = RadioButtons(
        radio_ax,
        options,
        active=active_index,
        activecolor=radio_button_selected_color,
        radio_props={'edgecolor': radio_button_selected_color}
    )

    # If horizontal layout, manually reposition radio button elements
    if horizontal:
        num_options = len(options)

        # Access internal Line2D objects for radio circles and labels
        for i, (radio_line, label_obj) in enumerate(zip(radio_buttons.lines, radio_buttons.labels)):
            # Distribute buttons evenly across horizontal space (0.1 to 0.9)
            x_pos = 0.1 + (i / max(1, num_options - 1)) * 0.8 if num_options > 1 else 0.5

            # Update line (radio circle) position
            radio_line.set_xdata([x_pos, x_pos])

            # Update label position
            label_obj.set_x(x_pos + 0.03)
            label_obj.set_y(0.5)
            label_obj.set_va('center')

    # Define selection callback
    def on_click_callback(label_clicked: str) -> None:
        """
        Store selected option in input_dict if not skipped value.
        """
        if skip_value is None or label_clicked != skip_value:
            input_dict[key] = label_clicked

    # Register callback
    radio_buttons.on_clicked(on_click_callback)

    return radio_buttons, label_text



def create_slider(fig: plt.Figure,
                  input_dict: dict,
                  key: str,
                  label: str,
                  position: tuple[float, float, float, float],
                  vmin: float,
                  vmax: float,
                  valinit: float,
                  valstep: float | int,
                  slider_bar_color: str,
                  slider_background_color: str,
                  valfmt: str = '%i') -> Slider:
    """
    Create and configure a Slider widget with automatic input dictionary storage.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure object
    input_dict : dict
        Input dictionary to store slider value under key
    key : str
        Dictionary key for storing the slider value
    label : str
        Label text displayed next to slider
    position : tuple
        Slider position as (x, y, width, height) in figure coordinates
    vmin : float
        Minimum slider value
    vmax : float
        Maximum slider value
    valinit : float
        Initial slider value
    valstep : float or int
        Step size for slider increments
    slider_bar_color : str
        Color for slider bar
    slider_background_color : str
        Color for slider background track
    valfmt : str, optional
        Format string for value display (default: '%i' for integers)

    Returns
    -------
    Slider
        Configured Slider widget
    """

    # Create axes for slider at specified position
    slider_ax = fig.add_axes(position)

    # Create slider with styling
    slider = Slider(
        slider_ax,
        label + "  ",  # looks better
        vmin,
        vmax,
        valinit=valinit,
        valstep=valstep,
        valfmt=valfmt,
        color=slider_bar_color,
        track_color=slider_background_color,
        initcolor='None',
        handle_style={'facecolor': 'white', 'edgecolor': 'white', 'size': 10}
    )

    # Define value change callback that stores in input_dict and redraws
    def on_changed_callback(val: float) -> None:
        """
        Store slider value in input_dict and trigger figure redraw.
        """
        input_dict[key] = int(val)
        fig.canvas.draw_idle()

    # Register callback
    slider.on_changed(on_changed_callback)

    return slider

def plot_onboarding_form(result_json_dir: str | Path,
                         shared_questionnaire_str,
                         instrument_question_str: str = "Do you play an instrument? If yes, which:",
                         listening_habit_question: str = "How often do you listen to music?",
                         listening_habit_options: list[str] | tuple[str, ...] = ('Most of the day', 'A small part of the day', 'Every 2 or 3 days', 'Seldom'),
                         dancing_question: str = "How much do you like moving to music? (0 = not at all, 7 = love dancing)",
                         athletic_ability_question_str: str = "Please rate your current athleticism. (0 = unfit, 7 = professional)",
                         health_questions_intro_str: str = "To ensure this study is safe for you and to help us account for individual differences in nervous system function, we need to understand your motor health history. Please answer truthfully and know that your data is being treated confidentially.",
                         known_diseases_question_str: str = "Have you ever been diagnosed by a healthcare professional with any neural condition? If yes, which:",
                         motor_symptoms_question_str: str = "In the past 6 months, have you experienced any difficulties with fine motor tasks?  If yes, which:",
                         medication_question_str: str = "Are you currently taking any medications or substances that could affect your nervous system or muscle function? If yes, which:",
                         ):
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
    # initialise:
    fig, ax = plt.subplots(figsize=(24, 12))
    fig.subplots_adjust(top=.93, bottom=0.25)  # space for widgets
    manager = plt.get_current_fig_manager()  # change TkAgg window title
    manager.set_window_title("Participant Registration Form")
    ax.axis('off')  # hide axes (borders and ticklabels)
    fig.suptitle('Welcome to the study :)')
    ax.set_title("Please enter your personal details below. Thank you!")

    # attributes:
    text_box_height = slider_height = .04


    ### INPUT TEXTBOXES
    input_dict = {}  # input dict

    # define text boxes (callback_function, ax, object, on_submit(func)):
    # full name:
    name_textbox = create_textbox(fig, input_dict, key="Name", label="Full Name (FIRST LAST):",
                                  position=(0.55, .88, 0.39, text_box_height),
                                  button_background_color=button_background_color,
                                  button_hover_color=button_hover_color,)
    # birthdate: text
    birthdate_textbox = create_textbox(fig, input_dict, key="Birthdate", label="Birthdate (DD/MM/YYYY):",
                                       position=(0.55, .82, 0.39, text_box_height),
                                       button_background_color=button_background_color,
                                       button_hover_color=button_hover_color,)

    # gender: (radiobutton) female / male / other
    gender_dropdown, gender_dropdown_label = create_radio_buttons(
        fig, ax, input_dict, key="Gender", label="Gender:",
        options=["Male", "Female", "Non-binary"], skip_value="Not selected", active_index=3,
        label_position=(.53, .76), position=(0.51, .725, .4, .08),
        radio_button_selected_color=button_hover_color, horizontal=False
    )
    # dominant hand: left / right
    dominant_hand_dropdown, dominand_hand_dropdown_label = create_radio_buttons(
        fig, ax, input_dict, key="Dominant hand", label="Dominant Hand:",
        options=["Left", "Right"], skip_value="Not selected", active_index=2,
        label_position=(.53, .66), position=(0.51, .665, .39, .065),
        radio_button_selected_color=button_hover_color, horizontal=False
    )

    # "Do you play an instrument? If yes, which:"
    instrument_textbox = create_textbox(fig, input_dict, key="Instrument", label=instrument_question_str,
                                        position=(0.55, 0.61, 0.39, text_box_height),
                                        button_background_color=button_background_color,
                                        button_hover_color=button_hover_color,)

    # "If yes, how well:" 1-7
    musical_skill_slider = create_slider(fig, input_dict, key="Musical skill", label='If yes, how well:',
                                 position=(0.55, 0.55, 0.39, slider_height),
                                 vmin=0, vmax=7, valinit=0, valstep=1, valfmt="%.0f",
                                 slider_bar_color=slider_bar_color, slider_background_color=slider_background_color,)

    # "How often do you listen to music?" Most of the day / a small part of the day / every 2 or 3 days / seldom
    listening_habit_dropdown, listening_habit_dropdown_label = create_radio_buttons(
        fig, ax, input_dict, key="Listening habit", label=listening_habit_question,
        options=list(listening_habit_options), skip_value="Not selected", active_index=len(listening_habit_options),
        label_position=(.53, .37), position=(0.51, .45, .39, .1),
        radio_button_selected_color=button_hover_color)

    # dancing slider:
    dancing_slider = create_slider(fig, input_dict, key="Dancing habit",
                                       label=strconv.enter_line_breaks(dancing_question, 100, 10),
                                 position=(.5, .4, .39, slider_height),
                                 vmin=0, vmax=7, valinit=0, valstep=1, valfmt="%.0f",
                                 slider_bar_color=slider_bar_color, slider_background_color=slider_background_color, )

    # how athletic are you:
    athleticism_slider = create_slider(fig, input_dict, key="Athleticism",
                                       label=strconv.enter_line_breaks(athletic_ability_question_str, 100, 10),
                                 position=(0.5, 0.35, 0.39, slider_height),
                                 vmin=0, vmax=7, valinit=0, valstep=1, valfmt="%.0f",
                                 slider_bar_color=slider_bar_color, slider_background_color=slider_background_color, )

    # health related questions:
    health_intro = ax.text(
        -.125, .07,
        strconv.enter_line_breaks(health_questions_intro_str, 190, 10),
        transform=ax.transAxes, va='center', ha='left'
    )

    max_letters_health_labels = 90
    condition_textbox = create_textbox(fig, input_dict, key="Condition",
                                       label=strconv.enter_line_breaks(known_diseases_question_str, max_letters_health_labels, 10),
                                       position=(0.55, 0.22, 0.39, text_box_height),
                                       button_background_color=button_background_color,
                                       button_hover_color=button_hover_color,)

    symptom_textbox = create_textbox(fig, input_dict, key="Motor Symptoms",
                                       label=strconv.enter_line_breaks(motor_symptoms_question_str, max_letters_health_labels, 10),
                                       position=(0.55, 0.15
                                                     , 0.39, text_box_height),
                                       button_background_color=button_background_color,
                                       button_hover_color=button_hover_color, )

    medication_textbox = create_textbox(fig, input_dict, key="Condition",
                                     label=strconv.enter_line_breaks(medication_question_str, max_letters_health_labels, 10),
                                     position=(0.55, 0.08
                                                   , 0.39, text_box_height),
                                     button_background_color=button_background_color,
                                     button_hover_color=button_hover_color, )




    ### SUBMISSION and SAVING
    # define submission_button (callback_function, ax, object, on_submit(func))
    #   on submit: check whether data is missing, otherwise save to result_json_dir and quit func
    def click_submission_button(event):
        # check for missing inputs:
        input_missing = False

        # check whether there is correct input for mandatory input fields:
        key_object_dict = {'Name': name_textbox.label, 'Birthdate': birthdate_textbox.label,
                           'Gender': gender_dropdown_label, 'Dominant hand': dominand_hand_dropdown_label,
                           'Listening habit': listening_habit_dropdown_label, 'Dancing habit': dancing_slider.label,
                           'Athleticism': athleticism_slider.label,}
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
                    key_object_dict[key].set_color(font_color)
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

    submission_button_ax = plt.axes([0.4, 0.02, 0.2, text_box_height])
    submission_button = Button(submission_button_ax, 'Submit',
                               color=button_background_color,
                               hovercolor=button_hover_color)
    submission_button.on_clicked(click_submission_button)

    plt.show()


def plot_offboarding_form(result_json_dir: str | Path,
                         fatigue_question_str: str = "How fatiguing was the overall experiment to you? (0 = completely easy, 7 = very tiring)",
                         pleasure_question_str: str = "How much did you enjoy the experiment? (0 = very dull/unpleasant, 7 = very fun)",
                         ):
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
    # initialise:
    fig, ax = plt.subplots(figsize=(24, 4))
    fig.subplots_adjust(top=.8, bottom=0.25)  # space for widgets
    manager = plt.get_current_fig_manager()  # change TkAgg window title
    manager.set_window_title("Participant Offboarding Form")
    ax.axis('off')  # hide axes (borders and ticklabels)
    fig.suptitle('Thank you for participating! :)')
    ax.set_title("Please enter some final feedback below. Much appreciated!")

    # attributes:
    text_box_height = slider_height = .04


    ### INPUT TEXTBOXES
    input_dict = {}  # input dict

    # "If yes, how well:" 1-7
    fatigue_slider = create_slider(fig, input_dict, key="Total fatigue", label=fatigue_question_str,
                                 position=(0.55, 0.55, 0.39, slider_height),
                                 vmin=0, vmax=7, valinit=0, valstep=1, valfmt="%.0f",
                                 slider_bar_color=slider_bar_color, slider_background_color=slider_background_color,)

    # how athletic are you:
    pleasure_slider = create_slider(fig, input_dict, key="Total pleasure",
                                       label=pleasure_question_str,
                                 position=(0.55, 0.4, 0.39, slider_height),
                                 vmin=0, vmax=7, valinit=0, valstep=1, valfmt="%.0f",
                                 slider_bar_color=slider_bar_color, slider_background_color=slider_background_color, )



    ### SUBMISSION and SAVING
    # define submission_button (callback_function, ax, object, on_submit(func))
    #   on submit: check whether data is missing, otherwise save to result_json_dir and quit func
    def click_submission_button(event):
        # check for missing inputs:
        input_missing = False

        # check whether there is correct input for mandatory input fields:
        key_object_dict = {'Total fatigue': fatigue_slider.label, 'Total pleasure': pleasure_slider.label,}
        for key, object in key_object_dict.items():
            if key not in input_dict:  # check only mandatory fields
                key_object_dict[key].set_color('red')
                fig.canvas.draw_idle()  # update view
                input_missing = True  # some input missing -> don't save yet
            else:
                key_object_dict[key].set_color(font_color)
                fig.canvas.draw_idle()  # update view

        if not input_missing:
            print("Input dict: ", input_dict)
            save_path = result_json_dir / filemgmt.file_title(f"Post-Study Feedback Data", ".json")
            with open(save_path, "w") as json_file:
                json.dump(input_dict, json_file, indent=4)  # Pretty print with indent=4
            print('Saved feedback data to ', save_path)

            # close fig:
            plt.close()

    submission_button_ax = plt.axes([0.4, 0.1, 0.2, text_box_height])
    submission_button = Button(submission_button_ax, 'Submit',
                               color=button_background_color,
                               hovercolor=button_hover_color)
    submission_button.on_clicked(click_submission_button)

    plt.show()


def legacy_plot_onboarding_form(result_json_dir: str | Path,
                         shared_questionnaire_str,
                         athletic_ability_question_str: str = "How would you rate your current athletic performance (training state)? (0 = very unfit, 7 = professional athlete)",
                         health_questions_intro_str: str = "To ensure this study is safe for you and to help us account for individual differences in nervous system function, we need to understand your motor health history. Please answer truthfully and know that your data is being treated confidentially.",
                         known_diseases_question_str: str = "Have you ever been diagnosed by a healthcare professional with any of the following conditions? (select all that apply)",
                         known_disease_options: list[str] | tuple[str, ...] = (
                                 "Stroke or transient ischemic attack (TIA/mini-stroke)",
                                 "Parkinson's disease or other movement disorder",
                                 "Multiple sclerosis or other demyelinating disease", "Cerebral palsy",
                                 "Brain or spinal cord injury",
                                 "Amyotrophic lateral sclerosis (ALS) or motor neuron disease", "Essential tremor",
                                 "Epilepsy or seizure disorder"),
                         motor_symptoms_question_str: str = "In the past 6 months, have you experienced any of the following? (select all that apply)",
                         motor_symptoms_options: list[str] | tuple[str, ...] = (
                                 "Weakness / numbness / tingling in your arms or hands",
                                 "Tremor / shaking / involuntary movements in your arms or hands",
                                 "Difficulty with fine motor tasks",
                                 "Stiffness or reduced range of motion in your arms or shoulders",
                                 "Loss of coordination or balance problems",
                                 "Difficulty controlling force or grip strength",
                                 "Pain that limits arm/hand movement"),
                         medication_question_str: str = "Are you currently taking any medications or substances that could affect your nervous system or muscle function? (select all that apply)",
                         medication_options: list[str] | tuple[str] = (
                                 "Medications for neurological conditions (Parkinson's / epilepsy / spasticity medications)",
                                 "Muscle relaxants",
                                 "Antipsychotic medications",
                                 "Benzodiazepines or sedatives",
                                 "Stimulant medications (prescription or otherwise)",
                                 "Alcohol on 5+ days per week or heavy use on any day",
                                 "Cannabis on 5+ days per week or heavy use on any day")
                         ):
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
    # define y positions:
    n_categories = 12  # 4 personal, 3 musical, 1 athletical, 4 health-related
    y_positions = np.linspace(.8, .15, n_categories)
    text_box_height = .03
    slider_height = .03
    dropdown_height_per_element = .015

    #  fig.subplots_adjust(bottom=0.25)  # space for widgets
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
    name_textbox = TextBox(name_textbox_ax, 'Full Name (FIRST LAST):',
                           color=button_background_color,
                           hovercolor=button_hover_color)
    name_textbox.on_submit(submit_name_textbox)

    # birthdate: text
    def submit_birthdate_textbox(text):
        input_dict["Birthdate"] = text

    birthdate_textbox_ax = fig.add_axes((0.55, 0.7, 0.39, 0.05))  # x, y, w, h
    birthdate_textbox = TextBox(birthdate_textbox_ax, 'Birthdate (DD/MM/YYYY):',
                                color=button_background_color,
                                hovercolor=button_hover_color)
    birthdate_textbox.on_submit(submit_birthdate_textbox)

    # gender: (radiobutton) female / male / other
    gender_dropdown_label = ax.text(.42, .6, "Gender:", transform=ax.transAxes, va='center', ha='left')
    gender_dropdown_ax = fig.add_axes((0.51, 0.57, 0.4, 0.11))  # x, y, w, h
    gender_dropdown_ax.axis('off')
    gender_dropdown_ax.set_facecolor('gold')
    gender_options = ['Female', 'Male', 'Non-binary', 'Not selected']  # options for selector
    gender_dropdown = RadioButtons(gender_dropdown_ax, gender_options, active=3,
                                   activecolor=radio_button_selected_color,
                                   radio_props={'edgecolor': radio_button_selected_color, })

    def submit_gender_dropdown(label):
        if label != "Not selected": input_dict["Gender"] = label

    gender_dropdown.on_clicked(submit_gender_dropdown)

    # dominant hand: left / right
    dominand_hand_dropdown_label = ax.text(.3, .44, "Dominant hand:", transform=ax.transAxes, va='center', ha='left')
    dominand_hand_dropdown_ax = fig.add_axes((0.51, 0.47, 0.4, 0.1))  # x, y, w, h
    dominand_hand_dropdown_ax.axis('off')
    dominand_hand_dropdown_ax.set_facecolor('gold')
    dominant_hand_options = ['Left', 'Right', 'Not selected']  # options for selector
    dominand_hand_dropdown = RadioButtons(dominand_hand_dropdown_ax, dominant_hand_options, active=2,
                                          activecolor=radio_button_selected_color,
                                          radio_props={'edgecolor': radio_button_selected_color, })

    def submit_dominand_hand_dropdown(label):
        if label != "Not selected": input_dict["Dominant hand"] = label

    dominand_hand_dropdown.on_clicked(submit_dominand_hand_dropdown)

    # "Do you play an instrument? If yes, which:"
    def submit_instrument_textbox(text):
        input_dict["Instrument"] = text

    instrument_textbox_ax = fig.add_axes((0.55, 0.4, 0.39, 0.05))  # x, y, w, h
    instrument_textbox = TextBox(instrument_textbox_ax, 'Do you play an instrument? If yes, which:',
                                 color=button_background_color,
                                 hovercolor=button_hover_color)
    instrument_textbox.on_submit(submit_instrument_textbox)

    # "If yes, how well:" 1-7
    skill_slider_ax = fig.add_axes((.55, .32, .39, .05))
    skill_slider = Slider(skill_slider_ax, 'If yes, how well: ', 0, 7, valinit=0, valstep=1, valfmt='%i',
                          color=slider_bar_color, track_color=slider_background_color,
                          initcolor='None', handle_style={'facecolor': 'white', 'edgecolor': 'white', 'size': 10})

    def update_skill_slider(val):
        input_dict["Musical skill"] = int(val)
        fig.canvas.draw_idle()  # update view

    skill_slider.on_changed(update_skill_slider)

    # "How often do you listen to music?" Most of the day / a small part of the day / every 2 or 3 days / seldom
    listening_habit_dropdown_label = ax.text(.03, -.02, "How often do you listen to music?", transform=ax.transAxes,
                                             va='center', ha='left')
    listening_habit_dropdown_ax = fig.add_axes((0.51, 0.16, 0.4, 0.14))  # x, y, w, h
    listening_habit_dropdown_ax.axis('off')
    listening_habit_dropdown_ax.set_facecolor('white')
    listening_habit_options = ['Most of the day', 'A small part of the day', 'Every 2 or 3 days', 'Seldom',
                               'Not selected']  # options for selector
    listening_habit_dropdown = RadioButtons(listening_habit_dropdown_ax, listening_habit_options, active=4,
                                            activecolor=radio_button_selected_color,
                                            radio_props={'edgecolor': radio_button_selected_color, })

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
                           'Listening habit': listening_habit_dropdown_label, }
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
                    key_object_dict[key].set_color(font_color)
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
    submission_button = Button(submission_button_ax, 'Submit',
                               color=button_background_color,
                               hovercolor=button_hover_color)
    submission_button.on_clicked(click_submission_button)

    plt.show()


def plot_breakout_screen(time_sec: float, title="Have a break. Please wait.",
                         anim_shutdown_event=None):
    """ Plot countdown during break. Figure clouses after time_sec. """
    ### PLOT
    try:
        with AnimationManager(anim_shutdown_event) as anim_mgr:
            # initialise:
            fig, ax = plt.subplots(figsize=(6, 3))
            manager = plt.get_current_fig_manager()  # change TkAgg window title
            manager.set_window_title("Breakout Screen")
            ax.axis('off')  # hide axes (borders and ticklabels)
            ax.set_title(title)

            # countdown:
            #global remaining_time
            remaining_time = time_sec
            countdown_text = fig.text(0.3, 0.4, f"Remaining waiting time: {remaining_time:.2f}s", ha='left', va='center', fontsize=10)

            # animation:
            display_refresh_rate_hz = 10
            #global display_start_time
            display_start_time = time.time()  # store to compute remaining time
            def update(frame):
                """ update view and fetch new observation. (frame is required although unused) """
                if anim_mgr.check_shutdown(): return 1,  # allow for forced shutdown

                # reduce countdown:
                #global remaining_time
                #global display_start_time
                remaining_time = time_sec - (time.time() - display_start_time)  # total time - passed time

                # close figure upon countdown end:
                if remaining_time <= 0.0: anim_mgr.stop()

                # else update text:
                countdown_text.set_text(f"Remaining waiting time: {remaining_time:.2f}s")

                # redraw and return:
                fig.canvas.draw_idle()
                return countdown_text,

            # run and show animation:
            anim_mgr.start(fig, update, int(1000 / display_refresh_rate_hz))
            plt.show()

    except AttributeError:  # sometimes TkAgg backend causes AttributeErrors upon animation closing
        print("Animation ended safely.")


def plot_pretrial_familiarity_check(result_json_dir: str | Path,  # dir to save results to
                                    shared_questionnaire_str,  # shared memory for master process
                                    question_text: str = 'How well do you know this song? (0 = never heard it, 7 = can sing/hum along)',
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
    slider = Slider(slider_ax, question_text + " ", 0, 7, valinit=0, valstep=1, valfmt='%i',
                          color=slider_bar_color, track_color=slider_background_color,
                          initcolor='None', handle_style={'facecolor': 'white', 'edgecolor': 'white', 'size': 10})
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
                key_object_dict[key].set_color(font_color)
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
    submission_button = Button(submission_button_ax, 'Submit',
                               color=button_background_color,
                               hovercolor=button_hover_color)
    submission_button.on_clicked(click_submission_button)
    plt.show()


# todo: ponder, whether to enter one lyric adds value
def plot_posttrial_rating(result_json_dir: str | Path,  # dir to save results to
                          shared_questionnaire_str,  # shared memory for master process
                          category_string: str | None = None,  # for question
                          liking_question_str: str = 'How did you like the song? (0: terrible, 7: extremely well)',
                          emotion_question_str: str = "Please rate your overall emotional state right now. (0: extremely unhappy/distressed, 7 = extremely happy/peaceful",
                                    ):
    """ Includes music questions only if category string is provided. """
    ### PLOT
    fig, ax = plt.subplots(figsize=(16, 3))
    manager = plt.get_current_fig_manager()  # change TkAgg window title
    manager.set_window_title("Post-Trial Rating")
    ax.axis('off')  # hide axes (borders and ticklabels)
    # title (user instruction):
    ax.set_title(f"Please take a break and answer the below {'three' if category_string is not None else 'one'} question{'s' if category_string is not None else ''}.")

    ### INPUT TEXTBOXES
    input_dict = {}  # input dict

    if category_string is not None:
        # define liking slider (callback_function, ax, object, on_submit(func)):
        liking_slider = create_slider(fig, input_dict, "Liking", liking_question_str,
                                      position=(.55, .7, .39, .1),
                                      vmin=0, vmax=7, valinit=0, valstep=1, valfmt='%i',
                                      slider_bar_color=slider_bar_color, slider_background_color=slider_background_color,)


        # define category validation slider (callback_function, ax, object, on_submit(func)):
        category_slider = create_slider(fig, input_dict, "Fitting Category", f"Do you think the song matches the category '{category_string.capitalize()}'? (0: not at all, 7: perfect match)",
                                      position=(.55, .6, .39, .1),
                                      vmin=0, vmax=7, valinit=0, valstep=1, valfmt='%i',
                                      slider_bar_color=slider_bar_color,
                                      slider_background_color=slider_background_color, )

        # suggest other category:
        other_category_options = [cat for cat in ['Groovy', 'Classic', 'Happy', 'Sad'] if cat != category_string.capitalize()] + ['None of them']
        other_category_dropdown, other_category_dropdown_label = create_radio_buttons(fig, ax, input_dict, "Other category", "If not (<=3), which other category would you assign to it?",
                                              options=other_category_options, skip_value="Not specified",
                                              position=(.5, .3, .39, .3), label_position=(.53, .445),
                                              active_index=len(other_category_options), radio_button_selected_color=slider_bar_color,)


    # define mood slider (callback_function, ax, object, on_submit(func)):
    emotion_slider_ax = fig.add_axes((.55, .2 if category_string is not None else .5, .39, .1))
    emotion_slider = Slider(emotion_slider_ax,
                             emotion_question_str + " ",
                             0, 7,
                             valinit=0, valstep=1, valfmt='%i',
                          color=slider_bar_color, track_color=slider_background_color,
                          initcolor='None', handle_style={'facecolor': 'white', 'edgecolor': 'white', 'size': 10})
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
                if key == "Fitting Category":  # if fitting category key
                    if input_dict[key] <= 3:  # if unfitting
                        if "Other category" not in input_dict:  # other needs to be specified
                            other_category_dropdown_label.set_color("red")
                            input_missing = True
                            fig.canvas.draw_idle()  # update view
                        else:
                            other_category_dropdown_label.set_color(font_color)
                            fig.canvas.draw_idle()  # update view
                    else:  # if fitting category
                        other_category_dropdown_label.set_color(font_color)

                key_object_dict[key].set_color(font_color)



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
    submission_button = Button(submission_button_ax, 'Submit',
                               color=button_background_color,
                               hovercolor=button_hover_color)
    submission_button.on_clicked(click_submission_button)

    plt.show()


def plot_input_view(shared_dict: dict[str, float],  # shared memory from sampling process
                    shared_dict_lock,
                    measurement_dict_label: str,
                    include_gauge: bool = True,
                    display_window_len_s: int = 3,
                    display_refresh_rate_hz: int = 30,
                    y_limits: tuple[float, float] = (0, 3.3),
                    target_value: float | tuple[float, float, float] | None = None,  # either fixed line or sine-wave (tuple[min, max, freq])
                    target_corridor: float | None = None,  # draw corridor around target
                    shared_value_target_dict: dict[str, float] | None = None,
                    shared_current_accuracy_str=None,
                    anim_shutdown_event=None,
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
    - Uses matplotlib's animation.FuncAnimation for live updating views.
    - Provides a pause/continue button to control updating.
    - Implements exponential moving average smoothing internally via the sampling process.
    - Gauge is a semicircular polar plot showing current input relative to y-limits.
    - Handles dynamic rescaling of plots if incoming values exceed current y-limits.
    """
    try:
        with AnimationManager(anim_shutdown_event) as anim_mgr:
            ### PREPARE PLOT
            global dynamic_y_limit  # variables that are dynamically adjusted during update() need to be defined globally
            dynamic_y_limit = y_limits
            global update_counter; update_counter = 0  # define display refreshment counter and sanity check
            if display_refresh_rate_hz > 60: print(f"Fps are {display_refresh_rate_hz}, which is > 60 and potentially leads to rendering issues.")

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
                    target_line = line_ax.axhline(y=target_value, color=target_color, lw=1, label='Target Value')
                    assert y_limits[0] < target_value < y_limits[1]; "target_value must lie within defined y_limits!"

                    if target_corridor is not None:  # mark target corridor
                        target_corridor_line_low = line_ax.axhline(y=target_value - target_corridor/2, color=dark_target_color,
                                                                   alpha=.5, lw=1, label='Target Corridor')
                        target_corridor_line_low = line_ax.axhline(y=target_value + target_corridor / 2, color=dark_target_color,
                                                                   alpha=.5, lw=1)

                elif isinstance(target_value, tuple):  # if sine-wave
                    target_line, = line_ax.plot([], [], lw=1, color=target_color, label='Target Value')
                    target_end_point, = line_ax.plot([], [], 'go')

                    if target_corridor is not None:
                        target_corridor_line_low, = line_ax.plot([], [], lw=1, alpha=.5, color=dark_target_color, label='Target Corridor')
                        target_corridor_line_high, = line_ax.plot([], [], lw=1, alpha=.5, color=dark_target_color)

            line, = line_ax.plot([], [], lw=2, color=measurement_color, label='Measurement Value')
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
                gauge_ax.bar([0, gauge_circumference], [gauge_radius]*2, width=0.03, color=font_color)  # mark ends

                # initialise current value line:
                needle_line, = gauge_ax.plot([], [], lw=3, color=measurement_color, label='Current Value')

                # include target:
                if target_value is not None:  # is set during update anyway so need to differentiate constant and sine here
                    target_needle_line, = gauge_ax.plot([], [], lw=2, color=target_color, label='Target Value')
                    if target_corridor is not None:
                        target_corridor_low_needle_line, = gauge_ax.plot([], [], lw=1, alpha=.5, color=dark_target_color, label='Target Corridor')
                        target_corridor_high_needle_line, = gauge_ax.plot([], [], lw=1, alpha=.5, color=dark_target_color)

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
            button = Button(ax_button, 'Pause',
                                   color=button_background_color,
                                   hovercolor=button_hover_color)
            button.on_clicked(pause_button_click)

            ## gamification
            global record_accuracy_bool
            record_accuracy_bool = False  # will be set to True after trial phase, remains False if accuracy_save_dir not defined

            # trial status:
            trial_status_text = line_ax.text(.0, 1.05, "", transform=line_ax.transAxes)

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
                # check for requested shutdown:
                if anim_mgr.check_shutdown(): return 1,

                global target_y  # global definition at begin of function

                # allow for ending of view
                #if save_accuracy_and_close_event.is_set(): plt.close('all')

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

                    ## accuracy display:  (is sampled and computed in separate accuracy_sampler process
                    if shared_value_target_dict is not None:  # communicate to sampler
                        with shared_dict_lock:
                            shared_value_target_dict['value'] = new_obs
                            shared_value_target_dict['target'] = target_y[-1]
                    if shared_current_accuracy_str is not None:  # ponder whether we can include color here
                        trial_status_text.set_text(shared_current_accuracy_str.read())


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
            anim_mgr.start(fig, update, interval=int(1000/display_refresh_rate_hz),
                           init_func=init)
            plt.show()

    except AttributeError:  # sometimes TkAgg backend causes AttributeErrors upon animation closing
        print("Animation ended safely.")



def accuracy_sampler(sampling_rate: float,
                     shared_value_target_dict,
                     shared_dict_lock,
                     shared_current_se_str,
                     shared_questionnaire_result_str,
                     accuracy_save_dir: str | Path,
                     save_accuracy_and_close_event,
                     save_accuracy_done_event,
                     pre_accuracy_phase_dur_sec: float = 5.0,
                     ):
    """
    Computes, stores and finally saves RMSE between a measurement and its target.

    Receives from a shared_value_target_dict.

    Communicates runtime result via a shared_current_se_str shared string.

    Returns final result via shared_questionnaire_result_str.
    """
    # waiting time before accuracy measurement:
    display_start_time = time.time()
    remaining_wait_time = pre_accuracy_phase_dur_sec
    while remaining_wait_time > 0:
        remaining_wait_time = pre_accuracy_phase_dur_sec - (time.time() - display_start_time)
        shared_current_se_str.write(f"Accuracy measurement will start in {remaining_wait_time:.2f} sec...")
        time.sleep(1/sampling_rate)  # simulate sampling rate

    # sample and store accuracies:
    accuracy_list = []
    while not save_accuracy_and_close_event.is_set():
        try:
            with shared_dict_lock:
                current_val = shared_value_target_dict['value']
                current_target = shared_value_target_dict['target']
        except KeyError:
            raise KeyError("shared_value_target_dict must contain 'value' and 'target' to sample accuracy!")

        # compute, store and communciate current acc.:
        current_se = (current_target - current_val) ** 2
        accuracy_list.append(current_se)
        shared_current_se_str.write(f"Current accuracy (sq. dist.): {current_se:.2f}")

        # simulate sampling frequency:
        time.sleep(1/sampling_rate)

    # first clear event:
    save_accuracy_and_close_event.clear()

    # store final RMSE:
    rmse = np.sqrt(np.nanmean(accuracy_list))
    result_str = f"Achieved RMSE: {rmse:.3f}"
    print(result_str); shared_questionnaire_result_str.write(result_str)

    # store as csv:
    save_path = accuracy_save_dir / filemgmt.file_title("Trial Accuracy Results", ".csv")
    pd.Series(data=accuracy_list).to_csv(save_path)
    print("Saved accuracy results to: ", save_path)
    save_accuracy_done_event.set()


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
    - Uses matplotlib’s animation.FuncAnimation to periodically update status text reflecting keys in shared_dict.
    - Integrates two buttons to set/clear events used by a sampling process.
    - Designed for synchronization and control of data acquisition pipelines in multiprocessing contexts.
    - Automatically closes figure upon window close or termination.
    """
    try:
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
        start_button = Button(start_button_ax, 'Start Trigger',
                               color=button_background_color,
                               hovercolor=button_hover_color)
        start_button.on_clicked(start_button_click)
        stop_button_ax = plt.axes([0.29, .65, 0.175, 0.175])
        stop_button = Button(stop_button_ax, 'Stop Trigger',
                               color=button_background_color,
                               hovercolor=button_hover_color)
        stop_button.on_clicked(stop_button_click)

        # experiment phase triggers:
        def click_onboarding_button(event):
            start_onboarding_event.set()
            global recent_event_str
            recent_event_str = 'Onboarding Phase'
            onboarding_button.label.set_text("Onboarding\n(Done)")
        experiment_control_label = fig.text(0.55, 0.86, "Experiment Control:", ha='left', va='center', fontsize=10)
        onboarding_button_ax = plt.axes([0.55, .65, 0.08, 0.175])
        onboarding_button = Button(onboarding_button_ax, 'Onboarding',
                               color=button_background_color,
                               hovercolor=button_hover_color)
        onboarding_button.on_clicked(click_onboarding_button)

        def click_mvc_button(event):
            start_mvc_calibration_event.set()
            global recent_event_str
            recent_event_str = 'MVC Calibration Phase'
            mvc_button.label.set_text("MVC\n(Done)")
        mvc_button_ax = plt.axes([0.64, .65, 0.08, 0.175])
        mvc_button = Button(mvc_button_ax, 'MVC',
                               color=button_background_color,
                               hovercolor=button_hover_color)
        mvc_button.on_clicked(click_mvc_button)

        def click_sampling_button(event):
            start_sampling_event.set()
            global recent_event_str
            recent_event_str = 'Sampling Phase'
            sampling_button.label.set_text("Restart\nSampling")
        sampling_button_ax = plt.axes([0.73, .65, 0.08, 0.175])
        sampling_button = Button(sampling_button_ax, 'Sampling',
                               color=button_background_color,
                               hovercolor=button_hover_color)
        sampling_button.on_clicked(click_sampling_button)

        def click_test_task_button(event):
            start_test_motor_task_event.set()
            global recent_event_str
            recent_event_str = 'Test Task Phase'
            test_task_button.label.set_text("Test Task\n(Done)")
        test_task_button_ax = plt.axes([0.82, .65, 0.08, 0.175])
        test_task_button = Button(test_task_button_ax, 'Test Task',
                               color=button_background_color,
                               hovercolor=button_hover_color)
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
            silence_trial_button = Button(silence_trial_button_ax, f'Silence Trial ({final_button_indices[0]})',
                               color=button_background_color,
                               hovercolor=button_hover_color)
            def silence_trial_button_clicked(event):
                music_master.pause()
                start_silent_motor_task_event.set()  # start silent motor task
            silence_trial_button.on_clicked(silence_trial_button_clicked)

            # pause / resume:
            pause_resume_button_ax = plt.axes([0.8, .35, 0.1, 0.175])
            pause_resume_button = Button(pause_resume_button_ax, 'Pause/Resume',
                               color=button_background_color,
                               hovercolor=button_hover_color)
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
                globals()[f'{category}_button'] = Button(temp_button_ax, f'{category} ({button_index})',
                               color=button_background_color,
                               hovercolor=button_hover_color)
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
                if music_master.current_category is not None:  # e.g. Groovy (1/8)
                    current_cat_str = f"{music_master.current_category} ({music_master.category_counter_dict[music_master.current_category]+1}/{len(music_master.category_url_dict[music_master.current_category])})"" | "
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
                        if music_master.current_category is not None:  # add category information
                            shared_song_info_dict['Category'] = music_master.current_category
                            shared_song_info_dict['Category Index'] = music_master.category_counter_dict[music_master.current_category]

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
                log_dict['Time'].append(datetime.now())  # is expensive: .strftime("%Y-%m-%d %H:%M:%S")
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
                save_log_dict(suffix="Final Full Save")
                force_log_saving_event.clear()
                log_saving_done_event.set()

            ## return
            return (measurement_info_text, song_info_text) if include_music else measurement_info_text

        # run and show animation:
        global master_ani
        master_ani = animation.FuncAnimation(fig, update, frames=1,
                            init_func=init, blit=False,
                            interval=int(1000/display_refresh_rate_hz), repeat=True)
        plt.show()

    finally:
        save_log_dict(suffix="Final Full Save")
        force_log_saving_event.clear()
        log_saving_done_event.set()
        plt.close('all')


def plot_performance_view(this_subject_dir: str | Path,
                          other_subject_dirs: list[str | Path] | None = None,
                          register_new_performance_event = None,
                          refresh_rate_hz: float = 10,
                          plot_size=(5, 4),
                          plot_title: str = "Motor Task Performance Overview",
                          window_title: str = "Performance Monitor",
                          anim_shutdown_event=None):
    if other_subject_dirs is not None:  # read all other historic performances:
        if not isinstance(other_subject_dirs, list): other_subject_dirs = [other_subject_dirs]
        global other_performances
        other_performances = []
        for dir in other_subject_dirs:
            other_performances.extend(
                filemgmt.fetch_json_recursively(dir, file_identifier="Trial Summary", value_key="RMSE"))

        # filter NaNs:
        other_performances = [x for x in other_performances if not math.isnan(x)]

        # define positoin for user boxplot:
        pos_user_bp = .5
    else: pos_user_bp = 0

    # initialise plot:
    fig, ax = plt.subplots(1, 1, figsize=plot_size)
    # format:
    fig.suptitle(plot_title)
    ax.set_title('(lower = better)')
    ax.set_axis_off()  # temporarily
    manager = plt.get_current_fig_manager()  # change TkAgg window title
    manager.set_window_title(window_title)

    # define function to create new user boxplot:
    def plot_new_bps():
        # clear previous and reshow:
        ax.clear()  # Remove old plots
        ax.set_title('(lower = better)')  # Restore title after clear
        ax.set_axis_on()  # reshow ax

        # fetch own performances with timestamps:
        own_performances = filemgmt.fetch_json_recursively(this_subject_dir,
                                                           file_identifier="Trial Summary", value_key="RMSE",
                                                           with_time_from_file_title=True)
        # sort by ascending time:
        sorted_performance_dict = dict(sorted(own_performances.items(), key=lambda x: x[0]))
        sorted_performances = list(sorted_performance_dict.values())

        # filter nans:
        sorted_performances = [x for x in sorted_performances if not math.isnan(x)]

        # plotting:
        bp_dict = ax.boxplot([other_performances, sorted_performances] if pos_user_bp != 0 else [sorted_performances],
                              positions=[0, pos_user_bp] if pos_user_bp != 0 else [pos_user_bp], showfliers=True, patch_artist=True)

        # now color patches:
        colors = [measurement_color, slider_bar_color] if pos_user_bp != 0 else [button_hover_color]
        for box, color in zip(bp_dict['boxes'], colors):
            box.set_facecolor(color)
            box.set_edgecolor(color)

        # mark most recent performance:
        last_value = sorted_performances[-1]
        ax.annotate('Your Last',
                         xy=(pos_user_bp + .05, last_value), xytext=(pos_user_bp + .3, last_value),
                         arrowprops=dict(arrowstyle='->', color=slider_bar_color, lw=2),
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=slider_bar_color, alpha=0.7),
                         fontsize=10, ha='center')

        # formatting:
        if pos_user_bp != 0: ax.set_xticks([0, pos_user_bp]); ax.set_xticklabels(["Previous", "You"])
        else: ax.set_xticks([0]); ax.set_xticklabels(["You"])
        ax.set_ylabel('Motor Task RMSE')
        ax.set_xlabel('Participant')

        return bp_dict


    # if refresh event is provided, update plot upon such:
    if register_new_performance_event is not None:
        try:
            with AnimationManager(anim_shutdown_event) as anim_mgr:  # try + finally to clean-up after terminating process
                def update(frame):  # animation method
                    if anim_mgr.check_shutdown(): return 1,

                    if register_new_performance_event.is_set():
                        register_new_performance_event.clear()  # clear again
                        #ax.set_title("UPDATE")
                        boxplots = plot_new_bps()
                        fig.canvas.draw_idle()

                    return 1,


                anim_mgr.start(fig, update, interval=int(1000/refresh_rate_hz))
                plt.show()

        except AttributeError:  # sometimes TkAgg backend causes AttributeErrors upon animation closing
            print("Animation ended safely.")

    else:  # otherwise plot only once:
        plot_new_bps()
        plt.show()


########## TESTING ##########
if __name__ == '__main__':
    # define saving folder:
    ROOT = Path().resolve().parent.parent
    SERIAL_MEASUREMENTS = ROOT / "data" / "serial_measurements"
    EXPERIMENT_LOG = ROOT / "data" / "experiment_logs"
    CONFIG_DIR = ROOT / "config"
    MUSIC_CONFIG = CONFIG_DIR / "music_selection.txt"
    RESULT_DIR = ROOT / "data" / "experiment_results"

    # important:
    SUBJECT_DIR = RESULT_DIR / "subject_00"
    SONG_ONE_DIR = SUBJECT_DIR / "song_00"

    from src.utils.multiprocessing_tools import SharedString
    plot_onboarding_form(SONG_ONE_DIR, SharedString(256, ""))

    #print(filemgmt.fetch_json_recursively(SUBJECT_DIR, "Trial Summary", "RMSE", True))