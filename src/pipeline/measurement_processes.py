import serial
import time
from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from pynput import keyboard
from scipy.optimize import curve_fit
from typing import Callable, Literal
import multiprocessing
from ctypes import c_char

import src.utils.file_management as filemgmt
from src.pipeline.music_control import SpotifyController

from pathlib import Path
import pandas as pd


############### READOUT METHODS ###############
# todo: ponder, whether prefixes are the best way to distinguish serial_measurements (I think they might work well)
def read_serial_measurements(measurement_definitions: tuple[tuple[str, Callable[[float], float], str, float]],
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
        - processing_func (callable or None): optional post-processing function for the measurement's raw value.
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
                if processing_func is not None: value = processing_func(value)

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


def force_estimator(voltage: float,
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


def sampling_process(shared_dict,
                     force_save_event,  # save_event callable through other functions
                     saving_done_event,  # saving_done event pausing other processes
                     start_trigger_event,  # send start trigger event ('A' via serial connection)
                     stop_trigger_event,  # send stop trigger event ('B' via serial connection)
                     measurement_definitions: tuple[tuple[str, Callable[[float], float] | None, str, float]],
                     # (measurement_label, processing_callable, serial_input_marker)
                     sampling_rate_hz: int = 1000,
                     record_bool: bool = True,
                     save_recordings_path: str | Path = None,
                     store_every_n_measurements: int = 10000,
                     working_memory_size: int = 600000,  # equals 10 min, serial_measurements to store in RAM before clean-up
                     **read_serial_kwargs,
                     ):
    """
    Continuously samples sensor data from serial input and updates a shared dictionary for inter-process communication.

    Parameters
    ----------
    shared_dict : multiprocessing.Manager.dict
        Shared dictionary object to store the latest sample values for each measurement label.
    force_save_event : threading.Event or multiprocessing.Event
        Event to trigger saving of the current buffered data to disk.
    saving_done_event : threading.Event or multiprocessing.Event
        Event to send start trigger event ('A' via serial connection)
    stop_trigger_event : threading.Event or multiprocessing.Event
        Event to send stop trigger event ('B' via serial connection).
    saving_done_event : threading.Event or multiprocessing.Event
        Event to signal completion of saving data.
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
            for measurement_label, _, _, _ in measurement_definitions:
                shared_dict[measurement_label] = samples[measurement_label]
                print(shared_dict)

            # eventually store:
            if sample_counter % store_every_n_measurements == 0:
                save_data(title_suffix=f' Redundant Save')

            # eventually clean-up local memory:
            if sample_counter > working_memory_size:
                save_data(title_suffix=f' Interim Save WorkMem Full')
                sample_counter = 1

            if force_save_event.is_set():
                save_data(title_suffix=f' Final Save')
                force_save_event.clear()
                saving_done_event.set()

            # simulate sampling frequency:
            time.sleep(1/sampling_rate_hz)
            sample_counter += 1
    finally:  # store data if saving path provided
        save_data()


def dummy_sampling_process(shared_dict,
                           force_save_event,  # save_event callable through other functions
                           saving_done_event,  # saving_done event pausing other processes
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
    force_save_event : threading.Event or multiprocessing.Event
        Event to trigger dummy save operation.
    saving_done_event : threading.Event or multiprocessing.Event
        Event to signal completion of dummy save.
    stop_trigger_event : threading.Event or multiprocessing.Event
        Event to send stop trigger event ('B' via serial connection).
    saving_done_event : threading.Event or multiprocessing.Event
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
            for measurement_label, _, _, _ in measurement_definitions:
                shared_dict[measurement_label] = samples[measurement_label]

            # imitate data saving:
            if force_save_event.is_set():
                print('[SAMPLER] saving...')
                force_save_event.clear()
                saving_done_event.set()
                print('[SAMPLER] saved!')

            # simulate sampling frequency:
            time.sleep(1/sampling_rate_hz)
            sample_counter += 1
    finally:
        saving_done_event.set()
        print('[SAMPLER] saved!')

############### PLOTTING METHODS ###############
def plot_input_view(shared_dict: dict[str, float],  # shared memory from sampling process
                    measurement_dict_label: str,
                    include_gauge: bool = True,
                    display_window_len_s: int = 3,
                    display_refresh_rate_hz: int = 15,
                    y_limits: tuple[float, float] = (0, 3.3),
                    target_value: float | None = None,
                    dynamically_update_y_limits: bool = True,
                    plot_size: tuple[float, float] = (15, 10),
                    input_unit_label: str = 'Input [V]',
                    x_label: str = 'Time [s]',
                    title: str = 'Live Input View'):
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
        matplotlib.use('TkAgg')  # select backend (suitable for animation)

        global dynamic_y_limit  # variables that are dynamically adjusted during update() need to be defined globally
        dynamic_y_limit = y_limits

        # define display refreshment counter and sanity check:
        global update_counter; update_counter = 0
        if display_refresh_rate_hz > 20: print(f"Fps are {display_refresh_rate_hz}, which is > 20 and potentially leads to rendering issues.")

        # Initial data
        x = np.linspace(-display_window_len_s, 0, display_window_len_s*display_refresh_rate_hz)
        global y; y = np.zeros_like(x)

        # initialise figure:
        fig, dummy_ax = plt.subplots(figsize=plot_size)
        dummy_ax.grid(False)  # Disable grid lines
        dummy_ax.set_axis_off()  # Turn off the entire polar axis frame
        fig.suptitle(title)

        # format and initialise line plot:
        line_ax = fig.add_subplot(122) if include_gauge else fig.add_subplot(111)
        line_ax.set_xlim(x.min(), x.max())
        line_ax.set_ylim(*y_limits)
        line_ax.set_xlabel(x_label)
        line_ax.set_ylabel(input_unit_label)
        line_ax.set_title('Rolling Input View')
        if target_value is not None:
            line_ax.axhline(y=target_value, color='green', lw=1, label='Target Value')
            assert y_limits[0] < target_value < y_limits[1]; "target_value must lie within defined y_limits!"
        line, = line_ax.plot([], [], lw=2, label='History')
        end_point, = line_ax.plot([], [], 'ro', ms=9, label='Current Value')

        if include_gauge:  # format and initialise gauge plot:
            gauge_radius = 10  # arbitrary, is scaled anyway
            n_xticks = 8
            gauge_circumference = 7/4 * np.pi  # rad
            gauge_ax = fig.add_subplot(121, projection='polar')
            gauge_ax.set_theta_offset(np.pi * ((gauge_circumference/np.pi-1)/2+1))  # Rotate start for gauge to be open downwards and "laying" on the ground
            gauge_ax.set_theta_direction(-1)  # Clockwise direction
            gauge_ax.set_ylim(0, gauge_radius)  # same y-limit as lineplot
            gauge_ax.grid(False)  # Disable grid lines
            gauge_ax.set_yticklabels([])  # turn off the radial ax labels
            gauge_ax.set_xticks(np.linspace(0, gauge_circumference, n_xticks))
            gauge_ax.set_xticklabels([f"{tick:.2f}" for tick in np.linspace(dynamic_y_limit[0], dynamic_y_limit[1], n_xticks)])
            gauge_ax.set_xlabel(input_unit_label)
            gauge_ax.set_title('Force Level')
            gauge_ax.spines['polar'].set_visible(False)  # hide polar spine (replaced below) because we don't use full circle
            angles = np.linspace(0,  gauge_circumference, 100)  # gauge background semicircle
            radii = np.full_like(angles, gauge_radius)
            gauge_ax.plot(angles, radii, color='lightgray', linewidth=20, solid_capstyle='round')
            gauge_ax.bar([0, gauge_circumference], [gauge_radius]*2, width=0.03, color='black')  # mark ends
            needle_line, = gauge_ax.plot([], [], lw=3, color='red', label='Current Value')
            if target_value is not None:  # initialise target line
                target_needle_line, = gauge_ax.plot([], [], lw=2, color='green', label='Target Value')

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

        ax_button = plt.axes([0.8, .9, 0.1, 0.075])
        button = Button(ax_button, 'Pause')
        button.on_clicked(pause_button_click)

        def init():
            # initialise lineplot:
            line_ax.legend()
            line.set_data(x, y)  # set data of line
            end_point.set_data([x.max()], [0])

            if include_gauge:  # initialise gauge
                gauge_ax.legend()
                needle_line.set_data([0, 0], [0, gauge_radius])
                if target_value is not None:  # mark target
                    target_needle_line.set_data([0, convert_y_to_angle(target_value)], [0, gauge_radius])

            return (line, end_point) if not include_gauge else ((needle_line, line, end_point) if target_value is None else (needle_line, line, end_point, target_needle_line))

        def update(frame):
            """ update view and fetch new observation. (frame is required although unused) """
            # update only if is_running:
            if is_running:
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
                        if include_gauge:
                            gauge_ax.set_xticklabels([f"{tick:.2f}" for tick in np.linspace(dynamic_y_limit[0], dynamic_y_limit[1], n_xticks)])
                            if target_value is not None: target_needle_line.set_data([0, convert_y_to_angle(target_value)], [0, gauge_radius])
                        line_ax.set_ylim(*dynamic_y_limit)
                        fig.canvas.draw_idle()

                # Shift data and append
                global y
                y = np.roll(y, -1)
                y[-1] = new_obs

                # update line plot:
                line.set_ydata(y)
                end_point.set_ydata([y[-1]])

                # update gauge plot:
                if include_gauge:
                    needle_line.set_data([0, convert_y_to_angle(new_obs)], [0, gauge_radius])


            return (line, end_point) if not include_gauge else ((needle_line, line, end_point) if target_value is None else (needle_line, line, end_point, target_needle_line))

        # run and show animation:
        ani = FuncAnimation(fig, update, frames=len(x)+1,
                            init_func=init, blit=False,
                            interval=int(1000/display_refresh_rate_hz), repeat=True)
        plt.show()

    finally:
        plt.close('all')


# todo: eventually include analog input from QTC?
def qtc_control_master_view(shared_dict: dict[str, float],  # shared memory from sampling process
                        start_trigger_event,
                        stop_trigger_event,
                        plot_size: tuple[float, float] = (10, 2),
                        title: str = "Quattrocento Control Master",
                        display_refresh_rate_hz: float = 3,
                        music_category_txt: str | Path | None = None,
                        control_log_path: str | Path | None = None,
                        save_log_working_memory_size: int = 60000,
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
            music_master = SpotifyController(category_url_dict=music_category_txt)
            include_music = True
        else: include_music = False

        # initialise figure:
        fig, dummy_ax = plt.subplots(figsize=plot_size)
        dummy_ax.grid(False)  # Disable grid lines
        dummy_ax.set_axis_off()  # Turn off the entire polar axis frame
        fig.suptitle(title)

        # trigger buttons (events relate to sampling-process):
        global recent_event_str
        recent_event_str = None
        def start_button_click(event):
            start_trigger_event.set()
            global recent_event_str
            recent_event_str = 'Start Trigger'
            start_button.label.set_text("Start Trigger (Done)")
            stop_button.label.set_text("Stop Trigger")
        def stop_button_click(event):
            stop_trigger_event.set()
            global recent_event_str
            recent_event_str = 'Stop Trigger'
            start_button.label.set_text("Start Trigger")
            stop_button.label.set_text("Stop Trigger (Done)")

        start_button_ax = plt.axes([0.1, .7, 0.35, 0.15])
        start_button = Button(start_button_ax, 'Start Trigger')
        start_button.on_clicked(start_button_click)
        stop_button_ax = plt.axes([0.55, .7, 0.35, 0.15])
        stop_button = Button(stop_button_ax, 'Stop Trigger')
        stop_button.on_clicked(stop_button_click)

        # define music control instruments:
        if include_music:
            # resume and pause buttons:
            music_button_label = fig.text(0.1, 0.6, "Music Control:", ha='left', va='center', fontsize=10)
            resume_button_ax = plt.axes([0.1, .4, 0.1, 0.15])
            resume_button = Button(resume_button_ax, 'Resume')
            def resume_button_clicked(event): music_master.resume()
            resume_button.on_clicked(resume_button_clicked)
            pause_button_ax = plt.axes([0.8, .4, 0.1, 0.15])
            pause_button = Button(pause_button_ax, 'Pause')
            def pause_button_clicked(event): music_master.pause()
            pause_button.on_clicked(pause_button_clicked)

            # category buttons:
            n_categories = len(music_master.category_url_dict.keys())
            width_button = (.5 / n_categories) * .95
            button_positions = np.linspace(.225, .775-width_button, n_categories)
            for category, button_pos in zip(music_master.category_url_dict.keys(), button_positions):
                temp_button_ax = plt.axes([button_pos, .4, width_button, .15])
                globals()[f'{category}_button'] = Button(temp_button_ax, f'Next {category}')
                # define function (category needs to be saved as default value due to late binding)
                globals()[f'{category}_button_clicked'] = lambda event, cat=category: music_master.play_next_from(cat)
                globals()[f'{category}_button'].on_clicked(globals()[f'{category}_button_clicked'])

        # status text:
        info_text = fig.text(0.1, 0.25, "", ha='left', va='center', fontsize=10)
        if include_music: music_text = fig.text(0.1, 0.1, "", ha='left', va='center', fontsize=10)

        ### animation methods:
        def init():
            info_text.set_text("Initializing...")
            if include_music: music_text.set_text("Initializing...")
            return (info_text, music_text) if include_music else info_text

        global save_log_counter
        save_log_counter = 1  # for saving log file
        if control_log_path is not None:
            global log_dict
            log_dict = {'Time': [], 'Music': [], 'Event': []}  # content of log file
            print(f"Initialising log file. Will save and reset full file every {(save_log_working_memory_size/display_refresh_rate_hz):.2f} s and do interim saves every {(save_log_working_memory_size/display_refresh_rate_hz/200):.2f} s!")
        def update(frame):
            """ update view and fetch new observation. (frame is required although unused) """
            # update only if is_running:
            ### update view:
            info_text.set_text(f"Receiving Serial Measurements: {list(shared_dict.keys())}")
            if include_music:
                if music_master.current_category_and_counter is not None:  # e.g. Groovy (1/8)
                    current_cat_str = f"{music_master.current_category_and_counter[0]} ({music_master.current_category_and_counter[1]+1}/{len(music_master.category_url_dict[music_master.current_category_and_counter[0]])})"" | "
                else: current_cat_str = ""
                try:
                    current_track_info = music_master.get_current_track(output_type='dict')
                    # e.g. Hallelujah by Leonard Cohen | 34.29s / 194.38s
                    current_track_str = f"{current_track_info['Title']} by {current_track_info['Artist']} | {current_track_info['Position [s]']:.2f}s / {current_track_info['Duration [ms]']/1000:.2f}s"
                    music_text.set_text(current_cat_str + current_track_str)
                except ValueError:  # no music playing currently
                    music_text.set_text("No track playing currently.")

            ### logging:
            global save_log_counter
            global recent_event_str
            # log updating:
            if control_log_path is not None:
                global log_dict
                log_dict['Time'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                log_dict['Music'].append(music_text.get_text())
                if recent_event_str is not None:  # save and reset in one block to prevent resetting event str without logging
                    log_dict['Event'].append(recent_event_str)
                    recent_event_str = None
                else: log_dict['Event'].append(recent_event_str)

                save_log_counter += 1  # because this is only increased here, we can omit the condition below:
            # log saving:
            if save_log_counter % save_log_working_memory_size == 0:  # save and working memory reset
                pd.DataFrame(log_dict).to_csv(control_log_path / filemgmt.file_title("Experiment Log Working Memory Full Save", ".csv"), index=False)
                log_dict = {'Time': [], 'Music': [], 'Event': []}
            elif save_log_counter % (save_log_working_memory_size // 200) == 0:  # interim save
                pd.DataFrame(log_dict).to_csv(control_log_path / filemgmt.file_title("Experiment Log Interim Save", ".csv"), index=False)

            return (info_text, music_text) if include_music else info_text

        # run and show animation:
        ani = FuncAnimation(fig, update, frames=1,
                            init_func=init, blit=False,
                            interval=int(1000/display_refresh_rate_hz), repeat=True)
        plt.show()

    finally:
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


### MULTIPROCESSING IMPLEMENTATION:
def start_measurement_processes(measurement_definitions: tuple[tuple[str, Callable[[float], float] | None, str, float]],measurement_saving_path: str | Path = None,
                                measurement_sampling_rate_hz: int = 1000,
                                record_measurements: bool = True,
                                # measurement_label, processing_callable, serial_input_marker
                                music_category_txt: str | Path | None = None,
                                control_log_path: str | Path | None = None,
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
    # initialise shared :
    shared_dict = multiprocessing.Manager().dict()
    measurement_labels = []  # for dynamic object definition below
    for measurement_label, _, _, _ in measurement_definitions:
        shared_dict[measurement_label] = .0
        measurement_labels.append(measurement_label)

    # saving event:
    force_save_event = RobustEventManager()
    saving_done_event = RobustEventManager()
    start_trigger_event = RobustEventManager()
    stop_trigger_event = RobustEventManager()

    # define processes:
    sampler = multiprocessing.Process(
        target=sampling_process,  # dummy_sampling_process,
        args=(shared_dict, force_save_event, saving_done_event, start_trigger_event, stop_trigger_event,),
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
                    'target_value': 1.2,
                    'include_gauge': True,
                    'title': 'FSR Input'
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
        args=(shared_dict, start_trigger_event, stop_trigger_event,),
        kwargs={'music_category_txt': music_category_txt,
                'control_log_path': control_log_path,
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
        force_save_event.set()  # trigger sampler saving
        print('Waiting for saving...')
        saving_done_event.wait(timeout=5)  # wait until done
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
    MUSIC_CONFIG = ROOT / "config" / "music_selection.txt"

    # start process:
    start_measurement_processes(measurement_definitions=(("fsr", None, "FSR:", .2),
                                                         ("ecg", None, "ECG:", .4),
                                                         ("gsr", None, "GSR:", .4),
                                                         ),
                                measurement_saving_path=SERIAL_MEASUREMENTS,
                                record_measurements=True,
                                music_category_txt=MUSIC_CONFIG,
                                control_log_path=EXPERIMENT_LOG,
                                )

