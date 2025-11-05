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
from typing import Callable
import multiprocessing
from ctypes import c_char

import src.utils.file_management as filemgmt

from pathlib import Path
import pandas as pd


############### READOUT METHODS ###############
def read_fsr_sensor(baud_rate: int = 115200,
                    serial_port: str = '/dev/tty.usbmodem143309601',
                    record_bool: bool = True,
                    command: str | None = None,
                    allowed_input_range: tuple[float] = (.0, 3.3),
                    processing_func: Callable[[float], float] | None = None,
                    smoothing_ema_alpha: float = 0.4,  # 1 = no smoothing, -> 0 more smoothing
                    ) -> float | None:
    """ To be commented. """
    global _last_valid_reading, measurements, timestamps
    try:
        with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
            # check for output command:
            if command in ("A", "B"):
                ser.write(command.encode("ascii"))
                ser.flush()  # waits for all outgoing data to be transmitted

            # read new line and convert to float:
            line = ser.readline().decode('ascii', errors="ignore").strip()
            if not line.startswith("VAL:"):  # check whether line contains measurement result
                return _last_valid_reading
            raw_str = line.replace("VAL:", "")  # formatting
            value = float(raw_str)

            # check whether input remains in feasible range:
            if not allowed_input_range[0] < value < allowed_input_range[1]: return _last_valid_reading

            if processing_func is not None: value = processing_func(value)

            # Apply EMA smoothing:
            value = smoothing_ema_alpha * value + (1 - smoothing_ema_alpha) * _last_valid_reading
            _last_valid_reading = value

            # save (if record_bool) and return:
            if record_bool:
                timestamps.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                measurements.append(value)
            return value

    except (ValueError, serial.SerialException) as e:
        print(f"Serial error: {e}")
        return _last_valid_reading


# todo: ponder, whether prefixes are the best way to distinguish measurements (I think they might work well)
def read_serial_measurements(measurement_definitions: tuple[tuple[str, Callable[[float], float], str, float]],
                             baud_rate: int = 115200,
                             serial_port: str = '/dev/tty.usbmodem143309601',
                             record_bool: bool = True,
                             command: str | None = None,
                             # measurement_label, processing_callable, serial_input_marker

                             allowed_input_range: tuple[float] = (.0, 3.3),
                             smoothing_ema_alpha: float = 0.4,  # 1 = no smoothing, -> 0 more smoothing
                             ) -> dict[str, float] | None:
    """ To be commented. """
    # we deploy globals() function here for dynamic global object naming and accessing (depending on included measurements)
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
            for measurement_label, processing_func, teensy_marker in measurement_definitions:
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
                value = smoothing_ema_alpha * value + (1 - smoothing_ema_alpha) * globals()['_last_valid_reading_' + measurement_label]

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
        return {measurement_label: globals()['_last_valid_reading_' + measurement_label] for measurement_label, _, _ in measurement_definitions}


def force_estimator(voltage: float,
                    fsr_a: float = 5.0869,
                    fsr_b: float = 1.8544) -> float:
    """ Converts the voltage input to estimated dynanometer force. """
    force_estimation = fsr_a * voltage ** fsr_b
    return force_estimation


def sampling_process(shared_dict,
                     force_save_event,  # save_event callable through other functions
                     saving_done_event,  # saving_done event pausing other processes
                     measurement_definitions: tuple[tuple[str, Callable[[float], float] | None, str]],
                     # (measurement_label, processing_callable, serial_input_marker)
                     sampling_rate_hz: int = 1000,
                     save_recordings_path: str | Path = None,
                     store_every_n_measurements: int = 10000,
                     working_memory_size: int = 600000,  # equals 10 min, measurements to store in RAM before clean-up
                     **read_serial_kwargs,
                     ):
    # initialise global variables for read_sensor function:
    for measurement_label, processing_func, teensy_marker in measurement_definitions:
        globals()['measurements_' + measurement_label] = []
        globals()['timestamps_' + measurement_label] = []
        globals()['_last_valid_reading_' + measurement_label] = .0

    # saving method to be called regularly and upon clean-up:
    def save_data(title_suffix: str = ''):
        # if len(measurements) == 0: return  # only save if there's something to save
        if save_recordings_path is not None:
            print(f"Saving recorded data to {save_recordings_path}")

            # prepare separate series for each measurement:
            measurement_labels = [measurement_label for measurement_label, _, _ in measurement_definitions]
            series_list = [pd.Series(index=globals()['timestamps_' + measurement_label],
                                     data=globals()['measurements_' + measurement_label],
                                     name=measurement_label,) for measurement_label in measurement_labels]

            # merge series to df and save:
            save_df = pd.concat(series_list, axis=1)
            savepath = save_recordings_path / filemgmt.file_title(f"Measurements ({' '.join(measurement_labels)}) {sampling_rate_hz}Hz{title_suffix}", '.csv')
            print(savepath)
            save_df.to_csv(savepath)

    try:
        sample_counter = 1
        while True:
            # method retrieves and saves sample:
            samples = read_serial_measurements(measurement_definitions=measurement_definitions,
                                               **read_serial_kwargs)

            # store in shared memory:
            for measurement_label, _, _ in measurement_definitions:
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

                           # important:
                           measurement_definitions: tuple[tuple[str, Callable[[float], float] | None, str]],
                           # measurement_label, processing_callable, serial_input_marker
                           custom_rand_maxs: tuple[float] = None,

                           sampling_rate_hz: int = 1000,
                           save_recordings_path: str | Path = None,
                           store_every_n_measurements: int = 10000,
                           working_memory_size: int = 600000,
                           # equals 10 min, measurements to store in RAM before clean-up
                           **read_serial_kwargs,
                           ):
    """ Imitates sampling from serial connection for development purposes. """
    try:
        sample_counter = 1
        while True:
            # random dummy samples:
            rand_maxs = list(range(1, len(measurement_definitions) + 1)) if custom_rand_maxs is None else custom_rand_maxs
            samples = {measurement_label: np.random.rand() * rand_max for (measurement_label, _, _), rand_max in zip(measurement_definitions, rand_maxs)}
            for measurement_label, _, _ in measurement_definitions:
                shared_dict[measurement_label] = samples[measurement_label]

            # imitate data saving:
            if force_save_event.is_set():
                print('[SAMPLER] saving...')
                force_save_event.clear()
                saving_done_event.set()
                print('[SAMPLER] saved!')
            print(shared_dict)
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
def start_measurement_processes(measurement_definitions: tuple[tuple[str, Callable[[float], float] | None, str]],measurement_saving_path: str | Path = None,
                                measurement_sampling_rate_hz: int = 1000,
                                record_measurements: bool = True,
                                # measurement_label, processing_callable, serial_input_marker
                                ) -> None:
    # initialise shared :
    shared_dict = multiprocessing.Manager().dict()
    for measurement_label, _, _ in measurement_definitions:
        shared_dict[measurement_label] = .0


    # saving event:
    force_save_event = RobustEventManager()
    saving_done_event = RobustEventManager()

    # define processes:
    sampler = multiprocessing.Process(
        target=dummy_sampling_process,
        args=(
            shared_dict,
            force_save_event, saving_done_event,  # events
        ),
        kwargs={'measurement_definitions': measurement_definitions,
                'sampling_rate_hz': measurement_sampling_rate_hz,
                'save_recordings_path': measurement_saving_path,
                'record_bool': record_measurements,
                'baud_rate': 115200,},
        name="SamplingProcess")

    fsr_displayer = multiprocessing.Process(
        target=plot_input_view,
        args=(shared_dict,),
        kwargs={'measurement_dict_label': 'fsr',
                'target_value': 1.2,
                'include_gauge': True,
                'title': 'FSR Input'
                },
        name="FSRDisplayProcess")

    ecg_displayer = multiprocessing.Process(
        target=plot_input_view,
        args=(shared_dict,),
        kwargs={'measurement_dict_label': 'ecg',
                'target_value': None,
                'include_gauge': False,
                'title': 'ECG Input'
                },
        name="ECGDisplayProcess")

    gsr_displayer = multiprocessing.Process(
        target=plot_input_view,
        args=(shared_dict,),
        kwargs={'measurement_dict_label': 'gsr',
                'target_value': 1.2,
                'include_gauge': False,
                'title': 'GSR Input'
                },
        name="ECGDisplayProcess")

    # start processes:
    try:
        sampler.start()
        fsr_displayer.start()
        ecg_displayer.start()
        gsr_displayer.start()

        # Wait for processes with timeout
        sampler.join()  #timeout=300  # 5 minute timeout (unused currently, main script ends anyway)
        fsr_displayer.join()  #timeout=300
        ecg_displayer.join()
        gsr_displayer.join()

    except KeyboardInterrupt:
        print("Terminating processes...")
        force_save_event.set()  # trigger sampler saving
        print('Waiting for saving...')
        saving_done_event.wait(timeout=5)  # wait until done
        print('Saving done!')

        # stop all processes:
        sampler.terminate()
        fsr_displayer.terminate()
        ecg_displayer.terminate()
        gsr_displayer.terminate()
        sampler.join()
        fsr_displayer.join()
        ecg_displayer.join()
        gsr_displayer.join()

    finally:
        print("Cleanup completed")

if __name__ == '__main__':
    # define saving folder:
    ROOT = Path().resolve().parent.parent
    DATA = ROOT / "data" / "measurements"

    # start process:
    start_measurement_processes(measurement_definitions=(("fsr", None, "VAL:"),
                                                         ("ecg", None, "ECG:"),
                                                         ("gsr", None, "GSR:"),),
                                measurement_saving_path=DATA, record_measurements=True,
                                )

