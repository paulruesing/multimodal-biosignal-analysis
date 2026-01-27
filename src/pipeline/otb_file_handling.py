from PyQt5 import QtWidgets
import pyqtgraph as pg
import os
import shutil
import numpy as np
import xmltodict
import tarfile
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm

########################################################################################################################
# The below code is an amended version from
# https://github.com/OTBioelettronica/OTB-Python/tree/main/Python%20Open%20and%20Processing%20OTBFiles/OpenOTB4
########################################################################################################################

def show_graph(time, data, title="Signal", shift=0.5):
    win = QtWidgets.QMainWindow()
    win.setWindowTitle(title)

    central_widget = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout()
    central_widget.setLayout(layout)

    tabs = QtWidgets.QTabWidget()
    layout.addWidget(tabs)

    n_channels = data.shape[0]
    maximus = np.max(np.abs(data))

    # --- Tab 1: Normalized Data ---
    normalized_widget = pg.PlotWidget(title="Normalized Data")
    for ch in range(n_channels):
        color = pg.intColor(ch, hues=n_channels)
        y = data[ch, :] / 2 / maximus - ch
        normalized_widget.plot(time, y, pen=pg.mkPen(color=color, width=1))
    tabs.addTab(normalized_widget, "Normalized Data")

    # --- Tab 2: Raw Data ---
    shifted_widget = pg.PlotWidget(title="Raw Data")
    for ch in range(n_channels):
        color = pg.intColor(ch, hues=n_channels)
        y = data[ch, :] - ch * shift
        shifted_widget.plot(time, y, pen=pg.mkPen(color=color, width=1))
    tabs.addTab(shifted_widget, "Raw Data")

    win.setCentralWidget(central_widget)
    win.resize(1000, 600)
    win.show()
    return win


def _save_signal_to_csv(
        data: np.ndarray,
        time_axis: np.ndarray,
        signal_name: str,
        base_filename: str,
        output_dir: str,
        output_title: str = None,
        combine_channels: bool = True,
        output_files: list = None,
        channel_range: Tuple[int, int] = None,
        append_filename_suffixes: bool = False
) -> str:
    """
    Save signal data to CSV file(s) with optional channel selection.

    Parameters
    ----------
    data : np.ndarray
        Signal data array of shape (channels, samples)
    time_axis : np.ndarray
        Time values corresponding to samples
    signal_name : str
        Name of the signal (used in filename)
    base_filename : str
        Base filename from original OTB4 file
    output_dir : str
        Output directory for CSV files
    output_title : str, optional
        Custom title for output filename. If None, uses base_filename
    combine_channels : bool, default=True
        If True, save all channels in single CSV
        If False, save separate CSV per channel
    output_files : list, optional
        List to append output filenames to
    channel_range : Tuple[int, int], optional
        Tuple (start, end) to select subset of channels (0-indexed, end exclusive).
        Example: (0, 64) selects channels 0-63. If None, uses all channels.

    Returns
    -------
    str
        Path to the saved CSV file (or first file if multiple)
    """
    if output_files is None:
        output_files = []

    n_ch, n_samples = data.shape
    filename_base = output_title if output_title else base_filename

    # Apply channel selection
    if channel_range is not None:
        start, end = channel_range
        if start < 0 or end > n_ch or start >= end:
            raise ValueError(
                f"Invalid channel_range ({start}, {end}). "
                f"Must be 0 <= start < end <= {n_ch}"
            )
        data = data[start:end, :]
        n_ch = end - start
        channel_offset = start
    else:
        channel_offset = 0

    if combine_channels:
        # Single CSV: time column + selected channels
        csv_data = {'Time_s': time_axis}
        for ch in range(n_ch):
            actual_ch = ch + channel_offset
            csv_data[f'Channel_{actual_ch + 1}'] = data[ch, :]

        df = pd.DataFrame(csv_data)

        # Generate filename: use signal_name if not generic "Signal"
        if signal_name != "Signal":
            if channel_range is not None:
                output_file = os.path.join(
                    output_dir,
                    f'{filename_base}_{signal_name}_ch{channel_range[0]}-{channel_range[1] - 1}.csv' if append_filename_suffixes else f'{filename_base}.csv',
                )
            else:
                output_file = os.path.join(output_dir, f'{filename_base}_{signal_name}.csv' if append_filename_suffixes else f'{filename_base}.csv',)
        else:
            if channel_range is not None:
                output_file = os.path.join(
                    output_dir,
                    f'{filename_base}_ch{channel_range[0]}-{channel_range[1] - 1}.csv' if append_filename_suffixes else f'{filename_base}.csv',
                )
            else:
                output_file = os.path.join(output_dir, f'{filename_base}.csv')

        df.to_csv(output_file, index=False)
        output_files.append(output_file)
        return output_file

    else:
        # Separate CSV per channel
        first_file = None
        for ch in range(n_ch):
            actual_ch = ch + channel_offset
            df = pd.DataFrame({
                'Time_s': time_axis,
                f'Channel_{actual_ch + 1}': data[ch, :]
            })

            if signal_name != "Signal":
                output_file = os.path.join(
                    output_dir,
                    f'{filename_base}_{signal_name}_ch{actual_ch + 1}.csv'
                )
            else:
                output_file = os.path.join(
                    output_dir,
                    f'{filename_base}_ch{actual_ch + 1}.csv'
                )

            df.to_csv(output_file, index=False)
            output_files.append(output_file)

            if first_file is None:
                first_file = output_file

        return first_file


def import_otb4_to_csv(
        otb4_path: str,
        output_dir: str,
        output_title: str = None,
        combine_channels: bool = True,
        channel_range: Tuple[int, int] = None,
        verbose: bool = True
) -> Dict:
    """
    Import an OTB4 file and export signals to CSV.

    Follows the exact logic of the original OTB4 import script,
    but saves to CSV files instead of displaying GUI.

    Parameters
    ----------
    otb4_path : str
        Path to the .otb4 file to import
    output_dir : str
        Directory where CSV file(s) will be saved
    output_title : str, optional
        Custom title for output filename. If None, uses base filename from otb4_path.
        Example: 'my_recording' creates files like 'my_recording.csv'
    combine_channels : bool, default=True
        If True, save all channels in single CSV with time column.
        If False, create separate CSV per channel.
    channel_range : Tuple[int, int], optional
        Tuple (start, end) to select subset of channels (0-indexed, end exclusive).
        Example: (0, 64) selects channels 0-63 (64 channels total).
        If None, exports all channels.
    verbose : bool, default=True
        Print progress messages to console

    Returns
    -------
    dict
        Metadata dictionary containing:
        - 'device': Device type (e.g., 'Novecento+')
        - 'sampling_freq': Sampling frequency (Hz)
        - 'n_channels': Total number of channels in original recording
        - 'n_channels_exported': Number of channels actually exported
        - 'channel_range': Channel range if specified, None otherwise
        - 'output_files': List of created CSV file paths
        - 'track_info': Full track information from XML

    Raises
    ------
    FileNotFoundError
        If .otb4 file doesn't exist or contains no .sig files
    ValueError
        If data reshape fails (corrupted file) or invalid channel_range

    Examples
    --------
    >>> # Export all channels with custom title
    >>> result = import_otb4_to_csv(
    ...     'recording.otb4',
    ...     './output',
    ...     output_title='patient_123',
    ...     combine_channels=True
    ... )

    >>> # Export only channels 0-63
    >>> result = import_otb4_to_csv(
    ...     'recording.otb4',
    ...     './output',
    ...     output_title='patient_123',
    ...     channel_range=(0, 64),
    ...     combine_channels=True
    ... )
    >>> print(f"Exported {result['n_channels_exported']} channels")
    """

    # --- Validate inputs ---
    if not os.path.exists(otb4_path):
        raise FileNotFoundError(f"OTB4 file not found: {otb4_path}")

    os.makedirs(output_dir, exist_ok=True)
    base_filename = output_title if output_title else Path(otb4_path).stem

    if verbose:
        print(f"[1/5] Extracting {base_filename}...")

    # --- Setup temporary directory ---
    tmp_dir = f'_tmp_otb4_{Path(otb4_path).stem}'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)

    try:
        # --- Extract TAR archive ---
        try:
            with tarfile.open(otb4_path, 'r') as tar:
                tar.extractall(tmp_dir)
        except tarfile.ReadError:
            raise FileNotFoundError(f"Failed to extract {otb4_path}. File may be corrupted.")

        if verbose:
            print(f"[2/5] Parsing XML metadata...")

        # --- Parse XML metadata ---
        xml_files = [f for f in os.listdir(tmp_dir) if f.endswith('Tracks_000.xml')]
        if not xml_files:
            raise FileNotFoundError("No Tracks_000.xml found in archive.")

        xml_file = xml_files[0]
        with open(os.path.join(tmp_dir, xml_file)) as fd:
            abs_xml = xmltodict.parse(fd.read())

        track_info = abs_xml['ArrayOfTrackInfo']['TrackInfo']
        if not isinstance(track_info, list):
            track_info = [track_info]

        device = track_info[0]['Device'].split(';')[0]

        # --- Extract recording parameters ---
        Gains = [float(track['Gain']) for track in track_info]
        nADBit = [int(track['ADC_Nbits']) for track in track_info]
        PowerSupply = [float(track['ADC_Range']) for track in track_info]
        Fsample = [int(track['SamplingFrequency']) for track in track_info]
        path = [track['SignalStreamPath'] for track in track_info]
        nChannel = [0] + [int(track['NumberOfChannels']) for track in track_info]
        startIndex = [int(track['AcquisitionChannel']) for track in track_info]

        TotCh = sum(nChannel)

        # Validate channel_range
        if channel_range is not None:
            start, end = channel_range
            if start < 0 or end > TotCh or start >= end:
                raise ValueError(
                    f"Invalid channel_range ({start}, {end}). "
                    f"Recording has {TotCh} channels. "
                    f"Must be 0 <= start < end <= {TotCh}"
                )
            n_channels_exported = end - start
        else:
            n_channels_exported = TotCh

        if verbose:
            print(f"   Device: {device}")
            print(f"   Total channels: {TotCh}")
            if channel_range is not None:
                print(
                    f"   Exporting channels: {channel_range[0]}-{channel_range[1] - 1} ({n_channels_exported} channels)")
            else:
                print(f"   Exporting: {TotCh} channels")
            print(f"   Sampling rate: {Fsample[0]} Hz")

        # --- Find .sig files ---
        signals = sorted([f for f in os.listdir(tmp_dir) if f.endswith('.sig')])
        if not signals:
            raise FileNotFoundError("No .sig files found in archive.")

        if verbose:
            print(f"[3/5] Reading binary signal data...")

        output_files = []

        if device == 'Novecento+':
            # Novecento+ has multiple signal blocks
            for sig_name in signals[1:]:  # Skip first (typically empty)
                matching_blocks = [j for j, p in enumerate(path) if p == sig_name]
                if not matching_blocks:
                    if verbose:
                        print(f"   Warning: No block found for {sig_name}")
                    continue

                nCh = sum([nChannel[j + 1] for j in matching_blocks])
                file_path = os.path.join(tmp_dir, sig_name)

                # Read binary data (32-bit for Novecento+)
                with open(file_path, 'rb') as f:
                    raw_data = np.fromfile(f, dtype=np.int32)
                    try:
                        data = raw_data.reshape((nCh, -1), order='F').astype(np.float32)
                    except ValueError as e:
                        if verbose:
                            print(f"   Error: Failed to reshape {sig_name}")
                        raise ValueError(f"Data reshape failed for {sig_name}") from e

                # Convert ADC counts to voltage (mV)
                current_ch = 0
                for j in matching_blocks:
                    n_ch_block = nChannel[j + 1]
                    psup = PowerSupply[j]
                    adbit = nADBit[j]
                    gain = Gains[j]
                    for ch in range(current_ch, current_ch + n_ch_block):
                        data[ch, :] = data[ch, :] * psup / (2 ** adbit) * 1000 / gain
                    current_ch += n_ch_block

                # --- Save to CSV ---
                Fs = Fsample[matching_blocks[0]]
                t = np.arange(data.shape[1]) / Fs
                output_file = _save_signal_to_csv(
                    data, t, sig_name, base_filename, output_dir,
                    output_title=base_filename,
                    combine_channels=combine_channels,
                    output_files=output_files,
                    channel_range=channel_range
                )

                if verbose:
                    duration = data.shape[1] / Fs
                    print(f"   Saved: {os.path.basename(output_file)} "
                          f"({n_channels_exported} channels, {duration:.2f}s)")

        else:
            # Other devices: single signal file
            sig_name = signals[0]
            file_path = os.path.join(tmp_dir, sig_name)

            # Read binary data (16-bit for other devices)
            with open(file_path, 'rb') as f:
                raw_data = np.fromfile(f, dtype=np.int16)
                if raw_data.size % TotCh != 0:
                    raise ValueError(
                        f"Data size {raw_data.size} not divisible by "
                        f"channel count {TotCh}"
                    )
                data = raw_data.reshape((TotCh, -1), order='F').astype(np.float32)

            # Convert ADC counts to voltage (mV)
            idx = [nChannel[0]]
            for val in nChannel[1:]:
                idx.append(idx[-1] + val)

            for ntype in range(1, len(track_info) + 1):
                for ch in range(idx[ntype - 1], idx[ntype]):
                    data[ch, :] = (data[ch, :] * PowerSupply[ntype - 1] /
                                   (2 ** nADBit[ntype - 1]) * 1000 / Gains[ntype - 1])

            # --- Save to CSV ---
            Fs = Fsample[0]
            t = np.arange(data.shape[1]) / Fs
            output_file = _save_signal_to_csv(
                data, t, "Signal", base_filename, output_dir,
                output_title=base_filename,
                combine_channels=combine_channels,
                output_files=output_files,
                channel_range=channel_range
            )

            if verbose:
                duration = data.shape[1] / Fs
                print(f"   Saved: {os.path.basename(output_file)} "
                      f"({n_channels_exported} channels, {duration:.2f}s)")

        if verbose:
            print(f"[4/5] Complete!")
            print(f"   Output files: {len(output_files)}")

        return {
            'device': device,
            'sampling_freq': Fs,
            'n_channels': TotCh,
            'n_channels_exported': n_channels_exported,
            'channel_range': channel_range,
            'output_files': output_files,
            'track_info': track_info
        }

    finally:
        # Clean up temporary directory
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)




















    """
    # --- GUI File Selection ---
    tk.Tk().withdraw()
    filetypes = [("OTB4 files", "*.otb4"), ("Zip files", "*.zip"), ("TAR files", "*.tar")]
    file_path = filedialog.askopenfilename(filetypes=filetypes)
    
    if not file_path:
        print("No file selected.")
        sys.exit()
    
    # --- File Handling ---
    tmp_dir = 'tmpopen'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    
    # --- Extract tar ---
    with tarfile.open(file_path, 'r') as tar:
        tar.extractall(tmp_dir)
    
    # --- Parse XML ---
    xml_file = [f for f in os.listdir(tmp_dir) if f.endswith('Tracks_000.xml')][0]
    with open(os.path.join(tmp_dir, xml_file)) as fd:
        abs_xml = xmltodict.parse(fd.read())
    
    track_info = abs_xml['ArrayOfTrackInfo']['TrackInfo']
    if not isinstance(track_info, list):
        track_info = [track_info]
    
    device = track_info[0]['Device'].split(';')[0]
    
    # --- Read Parameters ---
    Gains, nADBit, PowerSupply, Fsample, Path = [], [], [], [], []
    nChannel, startIndex = [0], []
    
    for track in track_info:
        Gains.append(float(track['Gain']))
        nADBit.append(int(track['ADC_Nbits']))
        PowerSupply.append(float(track['ADC_Range']))
        Fsample.append(int(track['SamplingFrequency']))
        Path.append(track['SignalStreamPath'])
        nChannel.append(int(track['NumberOfChannels']))
        startIndex.append(int(track['AcquisitionChannel']))
    
    TotCh = sum(nChannel)
    
    # --- Read .sig files ---
    signals = sorted([f for f in os.listdir(tmp_dir) if f.endswith('.sig')])
    if not signals:
        raise FileNotFoundError("No file .sig found.")
    
    Data = []
    windows = []
    app = QtWidgets.QApplication(sys.argv)
    
    if device == 'Novecento+':
        for sig_name in signals[1:]: 
            matching_blocks = [j for j, p in enumerate(Path) if p == sig_name]
            if not matching_blocks:
                print(f"No block found for {sig_name}")
                continue
    
            nCh = sum([nChannel[j + 1] for j in matching_blocks])
            file_path = os.path.join(tmp_dir, sig_name)
    
            # --- Read Binary Data ---
            with open(file_path, 'rb') as f:
                raw_data = np.fromfile(f, dtype=np.int32)
                try:
                    data = raw_data.reshape((nCh, -1), order='F').astype(np.float32)
                except ValueError:
                    print(f"Error in reshape of {sig_name}")
                    continue
    
            current_ch = 0
            for j in matching_blocks:
                n_ch_block = nChannel[j + 1]
                psup = PowerSupply[j]
                adbit = nADBit[j]
                gain = Gains[j]
                for ch in range(current_ch, current_ch + n_ch_block):
                    data[ch, :] = data[ch, :] * psup / (2 ** adbit) * 1000 / gain
                current_ch += n_ch_block
    
            Data.append(data)
            Fs = Fsample[matching_blocks[0]]
            # --- Plotting ---
            t = np.arange(data.shape[1]) / Fs
            windows.append(show_graph(t, data, title=f"Signal: {sig_name}", shift=0.5))
    
    else:
        sig_name = signals[0]
        file_path = os.path.join(tmp_dir, sig_name)
    
        # --- Read Binary Data ---
        with open(file_path, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.int16)
            if raw_data.size % TotCh != 0:
                print("Error in reshape of signal.")
                sys.exit()
            data = raw_data.reshape((TotCh, -1), order='F').astype(np.float32)
    
        idx = [nChannel[0]]
        for val in nChannel[1:]:
            idx.append(idx[-1] + val)
    
        for ntype in range(1, len(track_info) + 1):
            for ch in range(idx[ntype - 1], idx[ntype]):
                data[ch, :] = data[ch, :] * PowerSupply[ntype - 1] / (2 ** nADBit[ntype - 1]) * 1000 / Gains[ntype - 1]
    
        Data.append(data)
        Fs = Fsample[0]
        # --- Plotting ---
        t = np.arange(data.shape[1]) / Fs
        windows.append(show_graph(t, data, title="Signal", shift=0.5))
    
    sys.exit(app.exec_())"""