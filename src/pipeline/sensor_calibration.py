from time import sleep
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import serial
import time
from datetime import datetime
from typing import Callable, Literal
import src.utils.file_management as filemgmt

from scipy.optimize import curve_fit

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



if __name__ == '__main__':
    # file dirs:
    ROOT = Path().resolve().parent.parent
    CONFIG_DIR = ROOT / "config"

    new_calibration: bool = False

    if new_calibration:
        force_voltage_dict = {}  # will hold average voltages per force level

        for force_lvl in [2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 35, 40]:  # iterate over force levels
            repeat = True  # bool to judge whether session was successful

            # measurement (user interaction required)
            while repeat:
                # user instructions:
                print(f"Next task will be hold {force_lvl}kg.")
                sleep(1)
                print(f"Please, increase force to {force_lvl}kg now, don't overshoot and try to keep the dynamometer in live update display.")
                for _ in tqdm(range(0, 10), "Preparation time"): sleep(1)
                print(f"Now hold {force_lvl}kg for the next .5 seconds...")

                # measure:
                temp_voltage_list = []
                global _last_valid_reading_fsr; _last_valid_reading_fsr = 0.0  # initialise before reading from serial

                for _ in tqdm(range(0, 500, 10), "Measurement Progress"):  # sample every 10 ms
                    new_measure = read_serial_measurements([("fsr", None, "FSR:", 1),], record_bool=False)['fsr']
                    if new_measure != 0.0:  # only store if not equal to last_valid_reading (defined above as 0.0)
                        temp_voltage_list.append(new_measure)
                    sleep(.008)  # assume measurement takes 2ms

                repeat = False if input("Press enter, if measurement was successful, otherwise enter anything to repeat: ") == "" else True

            # take average and store:
            print(f"Measured voltages for {force_lvl}kg are: {temp_voltage_list}")
            temp_average_voltage = np.nanmedian(temp_voltage_list)
            print(f"Median voltage for {force_lvl}kg is: {temp_average_voltage}")
            force_voltage_dict[force_lvl] = temp_average_voltage

        # save to pandas and to csv:
        result_frame = pd.DataFrame(index=list(force_voltage_dict.keys()), data=list(force_voltage_dict.values()))
        result_frame.to_csv(CONFIG_DIR / filemgmt.file_title("Dynamometer Calibration Results", ".csv"))

    else:
        result_frame = pd.read_csv(filemgmt.most_recent_file(CONFIG_DIR, ".csv",
                                                             ["Dynamometer Calibration Results"]))
        forces = result_frame.iloc[:, 0]; voltages = result_frame.iloc[:, 1]

        # learn mapping:
        def monomial_model(v, a, b):  # defines model complexity
            return a * (v ** b)  # monomial (power law) model

        def dual_ponomial_model(v, a, b, c, d):
            return a * (v ** b) + c * (v ** d)

        params, test = curve_fit(monomial_model, voltages, forces)
        a, b = params
        print(f"Model: F = {a:.4f} * V^{b:.4f}")# + {c:.4f} * V^{d:.4f}")

        # result for 2015-12-02 15_13_38's calibration is F = 2.8708 * V^4.1071

        # calculate RMSE:
        def model(v, a: float = 2.8708, b: float = 4.1071):  # here with default params
            return a * (v ** b)

        preds = [model(val) for val in voltages]
        print("Predictions: ", preds)
        rmse = ((preds - forces) ** 2).mean() ** 0.5
        print(f"RMSE: {rmse}")
