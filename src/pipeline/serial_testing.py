from pathlib import Path
import src.utils.file_management as filemgmt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Literal

import src.pipeline.signal_features as features
import src.pipeline.data_integration as data_integration
import src.pipeline.data_analysis as data_analysis

if __name__ == '__main__':
    # define saving folder:
    ROOT = Path().resolve().parent.parent
    subject_ind: int = 4
    SERIAL_MEASUREMENTS = ROOT / "data" / "experiment_results" / f"subject_{subject_ind:02}" / "serial_measurements"
    subject_experiment_data = ROOT / "data" / "experiment_results" / f"subject_{subject_ind:02}"

    mpl.use('Qt5Agg')

    # Adjustable plot parameters
    start_idx, end_idx = 0, 1000000  # Change these indices as needed
    modality: Literal['ecg', 'fsr', 'gsr'] = 'fsr'

    do_plot_heart_rate: bool = False

    y_label = {'ecg': 'ECG Potential [V]', 'gsr': 'GSR Potential [V]', 'fsr': 'FSR Input [% MVC]'}[modality]

    # Load and prepare data
    path = filemgmt.most_recent_file(SERIAL_MEASUREMENTS, ".csv", [modality])
    print("Opening ", path)
    frame = pd.read_csv(path)
    frame.rename(columns={"Unnamed: 0": "datetime"}, inplace=True)
    frame['datetime'] = pd.to_datetime(frame['datetime'])
    serial_df = data_integration.fetch_serial_measurements(subject_experiment_data)

    # print(frame.datetime.dt.microsecond)

    # Calculate time in seconds from start for proper x-axis:
    #sampling_rate = len(frame) / (frame.datetime.iloc[-1].second - frame.datetime.iloc[0].second)
    #frame['time [s]'] = frame.index / sampling_rate

    log_df = data_integration.fetch_enriched_log_frame(subject_experiment_data)

    trial_start_ends = data_integration.get_all_task_start_ends(log_df, output_type='list')






    # Plot HRV data
    if do_plot_heart_rate:
        bpm_series, hrv_series = features.compute_heart_rate_and_variability(serial_df.set_index('datetime')['ecg'])

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(serial_df.reset_index()['Time'].iloc[start_idx:end_idx],
                hrv_series.iloc[start_idx:end_idx],
                linewidth=0.8, color='darkblue', label="HRV")
        ax2 = ax.twinx()
        # Plot BPM data
        ax2.plot(serial_df.reset_index()['Time'].iloc[start_idx:end_idx],
                 bpm_series.iloc[start_idx:end_idx],
                 linewidth=1.5, color='red', label='BPM')

        # Formatting
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('HRV (s)', fontsize=12, color='darkblue')
        ax2.set_ylabel('BPM', fontsize=12, color='red')
        ax.set_title(f'Estimated HRV + BPM', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=10)

        # Tight layout for better spacing
        plt.tight_layout()
        plt.show()



    # Plot HRV data
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(serial_df.reset_index()['Time'].iloc[start_idx:end_idx],
            serial_df[modality].iloc[start_idx:end_idx],
            linewidth=0.8, color='darkblue', label=modality)

    # Formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12, color='darkblue')
    ax.set_title(f'{modality}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=10)

    # Tight layout for better spacing
    plt.tight_layout()
    plt.show()