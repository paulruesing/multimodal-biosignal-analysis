import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Union, Tuple, List, Callable
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgba
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.widgets import Button, Slider
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing
import os
import textwrap
from scipy.stats import gaussian_kde
import math

import src.utils.file_management as filemgmt
from src.utils.str_conversion import enter_line_breaks
import src.pipeline.signal_features as features


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

mpl.use('Qt5Agg')

##############  CONSTANT PARAMETERS ##############
### EEG:
EEG_CHANNELS = ['Fp1', 'Fpz', 'Fp2',
                'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
                'F9', 'F7', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F8', 'F10',
                'FT9', 'FT7',
                'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
                'FT8', 'FT10',
                'T9', 'T7',
                'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                'T8', 'T10',
                'TP9', 'TP7',
                'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
                'TP8', 'TP10',
                'P9', 'P7', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P8', 'P10',
                'PO7', 'POz', 'PO8',
                'O1', 'O2',
                ]  # according to printout of quattrocento
EEG_CHANNELS_BY_AREA = {
    area_label: [ch for ch in EEG_CHANNELS if (ch[:len(area_abbr)] == area_abbr) and (
                (ch[len(area_abbr):].isnumeric()) or ch[len(area_abbr):] == 'z')] for area_label, area_abbr in [
        ('Frontal Pole', 'Fp'), ('Anterior Frontal', 'AF'), ('Fronto-Central', 'FC'), ('Frontal', 'F'),
        ('Fronto-Temporal', 'FT'), ('Temporal', 'T'), ('Central', 'C'), ('Temporo-Parietal', 'TP'),
        ('Centro-Parietal', 'CP'), ('Parietal', 'P'), ('Parieto-Occipital', 'PO'), ('Occipital', 'O')]}
EEG_CHANNEL_IND_DICT = {ch: ind for ind, ch in enumerate(EEG_CHANNELS)}

EEG_POSITIONS = {'Fpz': (0.0, 0.602),
                 'Fp1': (-0.165, 0.5599999999999999),
                 'Fp2': (0.165, 0.5599999999999999),
                 'AF7': (-0.30800000000000005, 0.48999999999999994),
                 'AF3': (-0.15400000000000003, 0.44799999999999995),
                 'AFz': (0.0, 0.42),  # maybe slightly off
                 'AF4': (0.15400000000000003, 0.44799999999999995),
                 'AF8': (0.30800000000000005, 0.48999999999999994),
                 'F9': (-0.5060000000000001, 0.45499999999999996),
                 'F7': (-0.4100000000000005, 0.385),
                 #'F5': (-0.33, 0.32899999999999996),
                 'F3': (-0.22000000000000003, 0.294),
                 'F1': (-0.11000000000000001, 0.26599999999999996),
                 'Fz': (0.0, 0.252),
                 'F2': (0.11000000000000001, 0.26599999999999996),
                 'F4': (0.22000000000000003, 0.294),
                 #'F6': (0.33, 0.32899999999999996),
                 'F8': (0.4100000000000005, 0.385),
                 'F10': (0.5060000000000001, 0.45499999999999996),
                 'FT9': (-0.5940000000000001, 0.238),
                 'FT7': (-0.48400000000000004, 0.196),
                 'FC5': (-0.36850000000000005, 0.16799999999999998),
                 'FC3': (-0.25300000000000006, 0.147),
                 'FC1': (-0.12925, 0.13299999999999998),
                 'FCz': (0.0, 0.126),
                 'FC2': (0.12925, 0.13299999999999998),
                 'FC4': (0.25300000000000006, 0.147),
                 'FC6': (0.36850000000000005, 0.16799999999999998),
                 'FT8': (0.48400000000000004, 0.196),
                 'FT10': (0.5940000000000001, 0.238),
                 'T9': (-0.640000000000001, 0.0),  # maybe slightly off
                 'T7': (-0.53, 0.0),
                 'C5': (-0.41250000000000003, 0.0),
                 'C3': (-0.275, 0.0),
                 'C1': (-0.1375, 0.0),
                 'Cz': (0.0, 0.0),
                 'C2': (0.1375, 0.0),
                 'C4': (0.275, 0.0),
                 'C6': (0.41250000000000003, 0.0),
                 'T8': (0.53, 0.0),
                 'T10': (0.640000000000001, 0.0),  # maybe slightly off
                 'TP9': (-0.6, -.24),  # maybe slightly off
                 'TP7': (-0.48400000000000004, -0.196),
                 'CP5': (-0.36850000000000005, -0.16799999999999998),
                 'CP3': (-0.25300000000000006, -0.147),
                 'CP1': (-0.12925, -0.13299999999999998),
                 'CPz': (0.0, -0.126),
                 'CP2': (0.12925, -0.13299999999999998),
                 'CP4': (0.25300000000000006, -0.147),
                 'CP6': (0.36850000000000005, -0.16799999999999998),
                 'TP8': (0.48400000000000004, -0.196),
                 'TP10': (0.6, -.24),  # maybe slightly off
                 'P9': (-0.47, -0.42),  # maybe slightly off
                 'P7': (-0.370000000000005, -0.355),
                 #'P5': (-0.33, -0.32899999999999996),
                 'P3': (-0.22000000000000003, -0.294),
                 'P1': (-0.11000000000000001, -0.26599999999999996),
                 'Pz': (0.0, -0.252),
                 'P2': (0.11000000000000001, -0.26599999999999996),
                 'P4': (0.22000000000000003, -0.294),
                 # 'P6': (0.33, -0.32899999999999996),
                 'P8': (0.3700000000000005, -0.355),
                 'P10': (0.47, -0.42),  # maybe slightly off
                 'PO7': (-0.30800000000000005, -0.48999999999999994),
                 #'PO3': (-0.15400000000000003, -0.44799999999999995),
                 'POz': (0.0, -0.42),
                 #'PO4': (0.15400000000000003, -0.44799999999999995),
                 'PO8': (0.30800000000000005, -0.48999999999999994),
                 'O1': (-0.165, -0.5599999999999999),
                 #'Oz': (0.0, -0.602),
                 'O2': (0.165, -0.5599999999999999)}
# from -.6 to +.6 in both dimensions:
x_list = list(np.linspace(-.6, .6, 8))*8
y_list = []
for element in list(np.linspace(-.6, .6, 8))[::-1]:
    y_list += [element] * 8
EMG_POSITIONS = {str(ind): (x, y) for ind, (x, y) in enumerate(zip(x_list, y_list))}


##############  AUXILIARY FUNCTIONS ##############
def smart_save_fig(dir: str | Path,
                   title: str | None = None,
                   format: str = '.svg'):
    """ Saves figure with a timestamp file-title. """
    save_dir = Path(dir) / filemgmt.file_title(title if title is not None else 'Plot', format)
    plt.savefig(save_dir, bbox_inches='tight')

##############  PLOTTING FUNCTIONS ##############
def initialise_electrode_heatmap(
        values: np.ndarray | list[float],
        positions: np.ndarray | dict[str, tuple[float, float, float]] | list[tuple[float, float, float]],
        add_head_shape: bool = False,  # only senseful for EEG
        include_labels: bool = True,
        colormap='viridis',
        color_scale=None,
        value_label: str = 'Heatmap values',
        plot_size: tuple[int, int] = (8, 8),
        plot_title: str = 'EEG Heatmap',
        hidden: bool = False,
        save_dir: str | Path = None,
) -> tuple[plt.Figure, plt.Axes, PatchCollection]:
    """ Needs to return figure, ax and circle_collection (PatchCollection) for animation. """
    # type conversion:
    if isinstance(values, list): values = np.array(values)
    if not isinstance(positions, dict):  # dummy channel labels if no dict provided
        positions = {ind: entry for ind, entry in enumerate(positions)}

    # sanity checks:
    n_channels = values.shape[0]
    assert n_channels == 64, f"Provided number of channels {n_channels} isn't 64 as required by this method."

    ##### initialise figure:
    print('Initializing EEG animation...')
    fig, ax = plt.subplots(figsize=plot_size)
    ax.set_title(plot_title)

    # extract value range for normalization (if not provided take minimum and maximum along both dimensions)
    vmin, vmax = color_scale if color_scale is not None else (np.min(values), np.max(values))
    # normalize and setup colormap:
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.colormaps[colormap]

    ### eeg heatmap:
    # initialise circles for heatmap with PatchCollection for seamless animation (freezes animation):
    circles = [patches.Circle(pos, radius=0.05) for pos in positions.values()]
    circle_collection = PatchCollection(circles,
                                        edgecolors='black',
                                        linewidths=0.8,
                                        cmap=cmap,
                                        # this leads to circle_collection.set_array directly changing the colors
                                        norm=norm)
    ax.add_collection(circle_collection)

    # channel labels:
    if include_labels:
        for label, (x, y) in positions.items():
            txt = ax.text(x, y, label, ha='center', va='center', fontsize=8, color='black')

    if add_head_shape:
        # add ellipse around head:
        width, height = 1.256, 1.505
        ellipse = patches.Ellipse([0, 0], width=width, height=height,
                                  edgecolor='black', facecolor='none', lw=1.5, ls='-')
        ax.add_patch(ellipse)

        # add nose:
        half_circle = patches.Wedge(center=(0, height / 2), r=0.1, theta1=-5, theta2=185,
                                    facecolor='none', edgecolor='black', ls='-', lw=1.5)
        ax.add_patch(half_circle)

    # colorbar:
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(np.mean(values, axis=0))  # derive color bar from time-averaged values
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(value_label)

    # scaling and removing axes:
    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.set_xticks([])
    ax.set_yticks([])

    if not hidden: plt.show()

    # eventually save:
    if save_dir is not None: smart_save_fig(save_dir, plot_title)

    return fig, ax, circle_collection


def animate_electrode_heatmap(values: np.ndarray | list[list[float]],
                              positions: np.ndarray | dict[str, tuple[float, float, float]] | list[tuple[float, float, float]],
                              sampling_rate: float,
                              animation_fps: int = 15,
                              **heatmap_kwargs, ):
    """
    Animate a heatmap of 64 channel electrode values plotted as circles at predefined electrode positions.
    based on the 64-channel 10-20 system.

    Parameters
    ----------
    values : np.ndarray or list[float]
        Input data values. If a list of lists or array is provided, it should have shape (n_timesteps, 64).
    sampling_rate : float
        Original sampling rate (Hz) of the input EEG data, used to resample data for animation.
    include_labels : bool, default True
        Flag to indicate whether to display channel labels on the heatmap.
    colormap : str or matplotlib.colors.Colormap, default 'viridis'
        Colormap name or instance used to map data values to colors on the heatmap.
    color_scale : tuple of (float, float) or None, optional
        Minimum and maximum values for color normalization (vmin, vmax). If None, color
        limits are derived from the min and max in the data.
    value_label : str, default 'Heatmap values'
        Label for the heatmap's colorbar.
    plot_size : tuple of int, default (7, 7)
        Width and height (in inches) of the matplotlib figure displaying the heatmap.
    plot_title : str, default 'EEG Heatmap'
        Title text displayed above the plot.
    animation_fps : int, default 15
        Frame rate (frames per second) used for animating the heatmap.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object containing the animated EEG heatmap plot.

    Notes
    -----
    - The function initializes a matplotlib figure showing a topographic map with circles representing EEG electrodes.
    - Circles are colored according to values that update over time to produce the animation effect.
    - Playback controls such as pause and speed slider are provided to interact with the animation.
    - The function automatically rescales the input data to match the specified animation FPS if needed.
    """
    ##### input preparation:
    # type conversion:
    if isinstance(values, list): values = np.array(values)

    # sanity checks:
    n_channels = values.shape[1]
    assert n_channels == 64, f"Provided number of channels {n_channels} isn't 64 as required by this method."
    if animation_fps > 20: print(f"Animation fps are {animation_fps} which is beyond the recommended maximum of 20.")

    # resample values for time-plausible animation:
    if sampling_rate != animation_fps:
        print('Resampling data for efficient animation (to prevent this, provide data with sampling_rate equivalent to animation_fps)...')
        values = features.resample_data(values, axis=0,
                                        original_sampling_freq=sampling_rate,
                                        new_sampling_freq=animation_fps)

    ### initialise figure:
    # heatmap:
    color_scale = (np.min(np.min(values)), np.max(np.max(values)))  # to account for all timesteps
    fig, ax, circle_collection = initialise_electrode_heatmap(values=values[0, :],
                                                              positions=positions,
                                                              color_scale=color_scale,
                                                              hidden=True,  # should be shown only upon animation start
                                                              **heatmap_kwargs
                                                              )
    # info text:
    info_text = ax.text(-.5, -.8, s="Initialising...", ha='center', va='center', fontsize=8)

    # add pause button:
    global is_running; is_running = True

    def pause_button_click(event):  # button clicked method
        global is_running; is_running = not is_running  # change running status
        # change button text:
        if is_running: button.label.set_text("Pause")
        else: button.label.set_text("Continue")

    ax_button = plt.axes((.1, .025, 0.15, 0.05))  # [left, bottom, width, height]
    button = Button(ax_button, 'Pause')
    button.on_clicked(pause_button_click)

    # add playback speed slider:
    global playback_speed; playback_speed = 1.0  # multiplied to frame counter increment

    def update(val):  # update function
        global playback_speed; playback_speed = val

    slider_ax = plt.axes((0.425, 0.035, 0.4, 0.03))
    freq_slider = Slider(slider_ax, 'Playback Speed', valmin=0.1, valmax=2.0, valinit=1.0)
    freq_slider.on_changed(update)

    ### animation:
    total_frames = len(values); total_time = total_frames / animation_fps

    # actual frame counter (pausable and controlled by playback speed)
    global pausable_frame_ind; pausable_frame_ind = 0
    def update(frame):
        if is_running:  # if not paused
            # info title:
            global pausable_frame_ind
            current_time = pausable_frame_ind / animation_fps
            info_text.set_text(f"Time: {current_time:.1f}s / {total_time:.1f}s")

            # update circle colors dynamically:
            rounded_pausable_frame_ind = int(math.floor(pausable_frame_ind))
            circle_collection.set_array(values[rounded_pausable_frame_ind, :])

            # take into account playback speed:
            global playback_speed
            pausable_frame_ind = (pausable_frame_ind + 1*playback_speed) % total_frames

        return circle_collection, [info_text]  # if doesn't work -> switch back to return circles + [info_text]

    global ani  # store animation in global namespace
    ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=False,
                        interval=int(1000 / animation_fps), repeat=True)
    plt.show()


def plot_freq_domain(amplitude_spectrum: np.ndarray[float, float] | np.ndarray[float],
                     frequencies: np.ndarray[float],
                     frequency_range: tuple[float, float] = None,
                     channel_labels: np.ndarray[str] = None,
                     plot_size: tuple[float, float] = (12, 6),
                     plot_title: str = "Magnitude Spectrum",
                     save_dir: str | Path = None,
                     continue_code: bool = False,
                     include_legend: bool = False,):
    """
    Plot the magnitude spectrum of one or more channels in the frequency domain.

    This function visualizes the amplitude spectrum as a line plot over the given
    frequency axis. It supports single-channel or multi-channel amplitude spectra,
    allowing for separate line plots per channel, optionally labeled with provided
    channel names.

    Parameters
    ----------
    amplitude_spectrum : np.ndarray[float, float] | np.ndarray[float]
        Array containing amplitude values. Can be 1D (single channel)
        or 2D (frequency × channels).
    frequencies : np.ndarray[float]
        Frequency axis values in Hertz corresponding to the amplitude spectrum.
    frequency_range : tuple[float, float], optional
        Tuple specifying the minimum and maximum frequencies to display,
        e.g., (0, 100). If None, the full range of frequencies is shown.
    channel_labels : np.ndarray[str], optional
        Array of labels for each channel. If None, channels will be labeled
        numerically as "Channel 1", "Channel 2", etc.
    plot_size : tuple[float, float], default (12, 6)
        Size of the plot (width, height) in inches.
    plot_title : str, default "Magnitude Spectrum"
        Title displayed at the top of the plot.
    save_dir : str or Path, optional
        Directory to save the figure.
    continue_code: bool, default False
        Whether to continue code execution while fig is shown.
    include_legend: bool, default False
        Whether to include legend in plot (slow for many channels).

    Returns
    -------
    None
        Displays a matplotlib plot of the amplitude spectrum.
    """
    # initialise figure:
    plt.figure(figsize=plot_size)

    # plot lines per channel:
    if len(amplitude_spectrum.shape) == 1:  # extend by pseudo-axis if only 1D array provided
        amplitude_spectrum = amplitude_spectrum[:, np.newaxis]
    for ch in range(amplitude_spectrum.shape[1]):
        plt.plot(frequencies, amplitude_spectrum[:, ch], label=f'Channel {ch}' if channel_labels is None else channel_labels[ch])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title(plot_title)
    if include_legend: plt.legend(ncols=amplitude_spectrum.shape[1] // 16 + 1)
    if frequency_range is not None: plt.xlim(frequency_range)
    else: print("Consider defining frequency range for improved display.")
    
    # eventually save:
    if save_dir is not None: smart_save_fig(save_dir, plot_title)

    plt.show(block=not continue_code)


def plot_spectrogram(
        spectrogram: np.ndarray,
        timestamps: np.ndarray,
        frequencies: np.ndarray | list[float] = None,
        channels: np.ndarray | list[str] = None,
        plot_type: str = 'time-frequency',
        cmap: str = 'viridis',
        frequency_range: tuple[float, float] | None = None,
        channel_range: tuple[int, int] | None = None,
        apply_log_scale: bool = False,
        is_log_scale: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        title: str = 'Spectrogram',
        xlabel: str = 'Time [s]',
        ylabel: str = 'Frequency [Hz]',
        cbar_label: str = 'Power [V²/Hz]',
        figsize: tuple[float, float] = (14, 6),
        aspect: str = 'auto',
        phase_series: pd.Series | None = None,
        phase_cmap: str = 'tab10',
        save_dir: str | Path = None,
        continue_code: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize time-frequency or time-channel spectrograms with optional phase annotations.

    Parameters
    ----------
    spectrogram : ndarray
        Power spectral density matrix or channel data matrix.
        Shape: (n_windows, n_frequencies) or (n_windows, n_channels)
    timestamps : ndarray
        Time centers of each window (seconds), shape (n_windows,)
    frequencies : ndarray or list[float], optional
        Frequency array (Hz) for 'time-frequency' plots, shape (n_frequencies,).
        Required if plot_type='time-frequency'. Default: None
    channels : ndarray or list[str], optional
        Channel identifiers (indices or labels) for 'time-channel' plots.
        Shape: (n_channels,). Required if plot_type='time-channel'. Default: None
    plot_type : str, optional
        Type of plot: 'time-frequency' or 'time-channel'. Default: 'time-frequency'
    cmap : str, optional
        Matplotlib colormap name for spectrogram. Default: 'viridis'
    frequency_range : tuple, optional
        (fmin, fmax) to restrict displayed frequency range in Hz (time-frequency only).
        Default: None
    channel_range : tuple, optional
        (ch_min, ch_max) to restrict displayed channel range (time-channel only).
        Default: None
    apply_log_scale : bool, optional
        Apply log10 scaling to data. Recommended for PSD with high variance.
        Default: True
    vmin, vmax : float, optional
        Min/max values for colormap normalization. Auto-determined if None.
        Default: None
    title : str, optional
        Title of the plot. Default: 'Spectrogram'
    xlabel : str, optional
        X-axis label. Default: 'Time [s]'
    ylabel : str, optional
        Y-axis label. Auto-set based on plot_type if not modified.
        Default: 'Frequency [Hz]'
    cbar_label : str, optional
        Colorbar label. Default: 'Power [V²/Hz]'
    figsize : tuple, optional
        Figure size (width, height) in inches. Default: (14, 6)
    aspect : str, optional
        Aspect ratio control. 'auto' preserves data proportions.
        Default: 'auto'
    phase_series : pd.Series, optional
        Time-indexed Series with string phase labels. NaN values are ignored.
        Time index will be resampled to match timestamps using nearest neighbor.
        Default: None
    phase_cmap : str, optional
        Colormap for phase regions. Default: 'tab10'
    save_dir : str or Path, optional
        Directory to save the figure. Default: None
    continue_code : bool, optional
        Whether to continue code execution while fig is shown. Default: False

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object (use for saving with fig.savefig())
    ax : matplotlib.axes.Axes
        Axes object with the spectrogram plot (or first subplot if phase_series provided)

    Raises
    ------
    ValueError
        If array shapes don't match or required parameters missing
    """

    # Validate plot_type
    if plot_type not in ('time-frequency', 'time-channel'):
        raise ValueError(
            f"plot_type must be 'time-frequency' or 'time-channel', got '{plot_type}'"
        )

    # Validate required parameters based on plot type
    if plot_type == 'time-frequency' and frequencies is None:
        raise ValueError("frequencies parameter required for plot_type='time-frequency'")
    if plot_type == 'time-channel' and channels is None:
        raise ValueError("channels parameter required for plot_type='time-channel'")

    # Input validation: array dimensions
    n_windows, n_features = spectrogram.shape
    if n_windows != len(timestamps):
        raise ValueError(
            f"Number of windows ({n_windows}) does not match "
            f"timestamps array length ({len(timestamps)})"
        )

    if plot_type == 'time-frequency':
        if n_features != len(frequencies):
            raise ValueError(
                f"Number of frequencies ({n_features}) does not match "
                f"frequencies array length ({len(frequencies)})"
            )
    elif plot_type == 'time-channel':
        if n_features != len(channels):
            raise ValueError(
                f"Number of channels ({n_features}) does not match "
                f"channels array length ({len(channels)})"
            )

    # Copy to avoid modifying input
    spec = spectrogram.T.copy()  # transpose: feature dimension on y-axis
    times = timestamps.copy()

    # Extract and filter features (frequency or channel)
    if plot_type == 'time-frequency':
        features = frequencies.copy()
        feature_range = frequency_range
        default_ylabel = 'Frequency [Hz]'
        default_cbar = 'Power [V²/Hz]'
    else:  # time-channel
        features = np.arange(len(channels)) if isinstance(channels, np.ndarray) else np.arange(len(channels))
        feature_labels = channels
        feature_range = channel_range
        default_ylabel = 'Channel'
        default_cbar = 'Amplitude'

    # Apply range filter if specified
    if feature_range is not None:
        fmin, fmax = feature_range
        if plot_type == 'time-frequency':
            feat_mask = (features >= fmin) & (features <= fmax)
        else:  # time-channel: range is index-based
            feat_mask = (np.arange(len(features)) >= fmin) & (np.arange(len(features)) <= fmax)
        spec = spec[feat_mask, :]
        features = features[feat_mask]
        if plot_type == 'time-channel':
            feature_labels = [feature_labels[i] for i in range(len(feature_labels)) if feat_mask[i]]

    # Apply log scaling if requested
    if apply_log_scale:
        spec = np.log10(np.abs(spec) + 1e-10)
        log_suffix = ' (log10)'
    elif is_log_scale:
        log_suffix = ' (log10)'
    else:
        log_suffix = ''

    # Set default labels based on plot type if not customized
    if ylabel == 'Frequency [Hz]' and plot_type == 'time-channel':
        ylabel = default_ylabel
    if cbar_label == 'Power [V²/Hz]' and plot_type == 'time-channel':
        cbar_label = default_cbar

    # =========================================================================
    # CREATE FIGURE WITH OPTIONAL PHASE SUBPLOT
    # =========================================================================

    has_phases = phase_series is not None
    fig, ax = plt.subplots(figsize=figsize)# constrained_layout=True)

    # Calculate extent for proper axis labeling
    extent = (
        times[0] - (times[1] - times[0]) / 2,
        times[-1] + (times[-1] - times[-2]) / 2,
        features[0] - (features[1] - features[0]) / 2,
        features[-1] + (features[-1] - features[-2]) / 2
    )

    # Plot spectrogram
    im = ax.imshow(
        spec, aspect=aspect, origin='lower',
        extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation='nearest'
    )

    # Configure axes labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Set yticks for channel plot
    if plot_type == 'time-channel':
        ax.set_yticks(np.arange(len(feature_labels)))
        ax.set_yticklabels(feature_labels)

    # Use make_axes_locatable for clean layout
    divider = make_axes_locatable(ax)

    # Append phase subplot as bottom axis if needed
    if has_phases:
        ax_phase = divider.append_axes("bottom", size="20%", pad="10%", sharex=ax)

    # Colorbar
    cax = divider.append_axes("right", size="5%", pad="3%")
    cbar = plt.colorbar(im, cax=cax, label=cbar_label + log_suffix)
    cbar.ax.tick_params(labelsize=10)

    # =========================================================================
    # PROCESS AND DISPLAY PHASES
    # =========================================================================

    if has_phases:
        _plot_phase_subplot(
            ax_phase=ax_phase,
            phase_series=phase_series,
            timestamps=times,
            time_extent=(extent[0], extent[1]),
            phase_cmap=phase_cmap
        )
        ax_phase.set_xlabel('Time [s]')
        ax.set_xlabel('')
        #ax.set_xticklabels(['' for _ in range(len(ax.get_xticklabels()))])

    plt.tight_layout()

    # Save figure if directory provided
    if save_dir is not None:
        smart_save_fig(save_dir, title)

    plt.show(block=not continue_code)
    return fig, ax


def _plot_phase_subplot(
        ax_phase: plt.Axes,
        phase_series: pd.Series,
        timestamps: np.ndarray,
        time_extent: tuple[float, float],
        phase_cmap: str = 'tab10'
) -> None:
    """Plot phase annotations as shaded regions with text labels."""
    if phase_series is None or len(phase_series) == 0:
        return

    # derive unique phases:
    phase_clean = phase_series.dropna()
    if len(phase_clean) == 0:
        ax_phase.text(0.5, 0.5, 'No phases within time snippet', ha='center', va='center',
                      transform=ax_phase.transAxes, fontsize=10, color='gray')
        ax_phase.set_xlim(0, time_extent[1] - time_extent[0])
        ax_phase.set_ylim(0, 1)
    else:
        unique_phases = phase_clean.unique()
        cmap = plt.colormaps[phase_cmap]
        phase_color_dict = {phase: color for phase, color in zip(unique_phases,
                                                                 cmap(np.linspace(0, 1, len(unique_phases))))}

        # convert phase datetime index to seconds from start:
        if isinstance(phase_series.index, pd.DatetimeIndex):
            phase_times_sec = (phase_series.index - phase_series.index[0]).total_seconds().to_series().reset_index(drop=True)
        else:
            phase_times_sec = phase_series.index.to_series().reset_index(drop=True)

        # derive phase on- and offsets:
        filled_phase = phase_series.fillna('No phase').reset_index(drop=True)  # required for pd comparisons
        phase_ids = (filled_phase != filled_phase.shift(1)).cumsum().reset_index(drop=True)
        starts = phase_times_sec.groupby(phase_ids).min()
        ends = phase_times_sec.groupby(phase_ids).max()
        phases = filled_phase.groupby(phase_ids).first()

        for phase, start, end in zip(phases, starts, ends):
            if phase == 'No phase': continue
            color = phase_color_dict[phase]
            ax_phase.axvspan(start, end, alpha=.6, color=color)
            text_x = (start + end) / 2  # mean time

            import textwrap
            formatted_text = textwrap.fill(phase, 12)
            ax_phase.text(text_x, .5, formatted_text, ha='center', va='center',
                          color='white' if _is_dark(color) else 'black')

    # formatting:
    ax_phase.set_ylim(0, 1)
    ax_phase.set_xlim( time_extent[0], time_extent[1])
    ax_phase.set_yticks([])
    ax_phase.set_ylabel('Phase', fontsize=10)
    ax_phase.spines['top'].set_visible(False)
    ax_phase.spines['right'].set_visible(False)
    ax_phase.spines['left'].set_visible(False)


def _is_dark(color) -> bool:
    """
    Determine if a color is dark (for contrast text color selection).

    Parameters
    ----------
    color : tuple or str
        RGBA color tuple or color string

    Returns
    -------
    bool
        True if color is dark, False if light
    """
    rgba = to_rgba(color)
    # Relative luminance formula (WCAG)
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return luminance < 0.5



def plot_scatter(x, y, x_label: str | None = None, y_label: str | None = None,
                 category_list: list | None = None, category_label: str | None = None,
                 cmap: str | list[str] = 'Set1', plot_size=(8, 8),
                 marker_size: float = 25,
                 include_kdes: bool = True, kde_alpha: float = 0.5,
                 categorical_kdes: bool = True,
                 add_kde_legend: bool = False,
                 save_dir: str | Path | None = None):
    """Scatter plot with categorical coloring and optional marginal KDE distributions.

    Parameters
    ----------
    x : array-like
        X-axis values.
    y : array-like
        Y-axis values.
    x_label : str, optional
        Label for x-axis.
    y_label : str, optional
        Label for y-axis.
    category_list : list, optional
        Categorical values (str or numeric) for coloring. If None, uses single color.
    category_label : str, optional
        Legend title for categories.
    cmap : str or list[str], default 'Set1'
        Colormap name (e.g., 'viridis') or list of color strings.
    plot_size : tuple, default (8, 8)
        Figure size as (width, height).
    include_kdes : bool, default True
        Whether to include marginal KDE distributions.
    kde_alpha : float, default 0.5
        Alpha transparency for KDE fill (0-1).
    categorical_kdes : bool, default True
        If True and category_list is provided, plot separate KDE for each category.
    add_kde_legend : bool, default False
        If True, add legend to KDE marginal axes.
    save_dir : str | Path, optional
        Directory to save figure. If None, doesn't save.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Main scatter axes object.
    """
    from scipy.stats import gaussian_kde

    # =========================================================================
    # CREATE FIGURE AND AXES LAYOUT
    # =========================================================================

    fig = plt.figure(figsize=plot_size)

    if include_kdes:
        # Create 2x2 gridspec with marginal KDE axes
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[4, 1],
                              hspace=0.05, wspace=0.05)
        ax_main = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    else:
        # Simple single axes layout
        ax_main = fig.add_subplot(1, 1, 1)

    # =========================================================================
    # PLOT SCATTER POINTS WITH OPTIONAL CATEGORIES
    # =========================================================================

    if category_list is None:
        # No categories: single color scatter
        ax_main.scatter(x, y, s=marker_size, color='steelblue', alpha=0.7)
        colors = None
        unique_cats = None
    else:
        # Get unique categories and map to color indices
        unique_cats = sorted(set(category_list))
        cat_to_idx = {cat: i for i, cat in enumerate(unique_cats)}
        indices = [cat_to_idx[cat] for cat in category_list]

        # Get colors from colormap
        if isinstance(cmap, str):
            colors = plt.cm.get_cmap(cmap, len(unique_cats))(
                np.linspace(0, 1, len(unique_cats))
            )
        else:
            colors = cmap[:len(unique_cats)]

        # Plot scatter with category colors
        point_colors = [colors[idx] for idx in indices]
        ax_main.scatter(x, y, c=point_colors, s=marker_size)

        # Create legend for scatter plot
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                          markersize=8, label=str(cat))
                   for i, cat in enumerate(unique_cats)]
        ax_main.legend(handles=handles, title=category_label, loc='best')

    # =========================================================================
    # PLOT MARGINAL KDE DISTRIBUTIONS
    # =========================================================================

    if include_kdes:
        if categorical_kdes and category_list is not None:
            # Plot separate KDE for each category
            x_range = np.linspace(np.asarray(x).min(), np.asarray(x).max(), 200)
            y_range = np.linspace(np.asarray(y).min(), np.asarray(y).max(), 200)

            for i, cat in enumerate(unique_cats):
                mask = np.array(category_list) == cat
                x_cat = np.asarray(x)[mask]
                y_cat = np.asarray(y)[mask]

                # Plot X-axis marginal KDE (top)
                if len(x_cat) > 1:  # Need at least 2 points for KDE
                    kde_x = gaussian_kde(x_cat)
                    ax_top.fill_between(x_range, kde_x(x_range), alpha=kde_alpha,
                                        color=colors[i],
                                        label=str(cat) if add_kde_legend else None)

                # Plot Y-axis marginal KDE (right)
                if len(y_cat) > 1:
                    kde_y = gaussian_kde(y_cat)
                    ax_right.fill_betweenx(y_range, kde_y(y_range), alpha=kde_alpha,
                                           color=colors[i],
                                           label=str(cat) if add_kde_legend else None)

            # Add legends only if labels were assigned
            if add_kde_legend:
                ax_top.legend(loc='upper right', fontsize=8)
                ax_right.legend(loc='upper right', fontsize=8)
        else:
            # Plot overall KDE (no categories or categorical_kdes=False)
            x_range = np.linspace(np.asarray(x).min(), np.asarray(x).max(), 200)
            kde_x = gaussian_kde(x)
            ax_top.fill_between(x_range, kde_x(x_range), alpha=kde_alpha,
                                color='steelblue')

            y_range = np.linspace(np.asarray(y).min(), np.asarray(y).max(), 200)
            kde_y = gaussian_kde(y)
            ax_right.fill_betweenx(y_range, kde_y(y_range), alpha=kde_alpha,
                                   color='steelblue')

        # Configure KDE axes
        ax_top.set_ylabel('Density')
        ax_top.tick_params(labelbottom=False)
        ax_right.set_xlabel('Density')
        ax_right.tick_params(labelleft=False)

    # =========================================================================
    # CONFIGURE MAIN AXES AND LAYOUT
    # =========================================================================

    ax_main.set_xlabel(x_label)
    ax_main.set_ylabel(y_label)

    # Use subplots_adjust for gridspec layouts to avoid tight_layout warnings
    if include_kdes:
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1,
                            hspace=0.05, wspace=0.05)
    else:
        plt.tight_layout()

    # =========================================================================
    # SAVE FIGURE IF DIRECTORY PROVIDED
    # =========================================================================

    if save_dir is not None:
        smart_save_fig(save_dir,
                       title=f"Scatter_KDEs_x_{x_label}_y_{y_label}")

    plt.show()

    return fig, ax_main


def plot_psd_avg_with_std(
        freq_psd_dict,
        sampling_freq,
        figsize=(14, 7),
        linewidth=2.5,
        std_factor: float = .1,
        std_alpha=0.25,
        colors=None,
        title='Frequency Bands - Power Spectral Density Over Time'
):
    """
    Plot PSD average per frequency band data with shaded standard deviation regions.

    Parameters
    ----------
    freq_psd_dict : dict
        Dictionary with frequency band names as keys and (n_samples, n_channels)
        numpy arrays as values.
    sampling_freq : float
        Sampling frequency in Hz.
    figsize : tuple, optional
        Figure size as (width, height). Default is (14, 7).
    linewidth : float, optional
        Line width for mean traces. Default is 2.5.
    std_alpha : float, optional
        Transparency of shaded std regions (0-1). Default is 0.25.
    colors : dict, optional
        Custom color mapping {band_name: color}. If None, uses default palette.
    title : str, optional
        Plot title. Default is descriptive EEG title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """

    # Default colors if not provided
    if colors is None:
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        colors = {band: default_colors[i % len(default_colors)]
                  for i, band in enumerate(freq_psd_dict.keys())}

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate time axis based on sampling frequency
    n_samples = list(freq_psd_dict.values())[0].shape[0]
    time = np.arange(n_samples) / sampling_freq  # Convert to seconds

    # Plot each frequency band with std shading
    for band_name, band_data in freq_psd_dict.items():
        # Calculate mean and std across channels (axis=1)
        mean_signal = band_data.mean(axis=1)
        std_signal = band_data.std(axis=1) * std_factor

        # Get color for this band
        color = colors.get(band_name, 'C0')

        # Plot mean line
        ax.plot(time, mean_signal, label=band_name, linewidth=linewidth,
                color=color)

        # Shade ± std region around the mean
        ax.fill_between(time,
                        mean_signal - std_signal,
                        mean_signal + std_signal,
                        alpha=std_alpha,
                        color=color)

    # Customize the plot
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power Spectral Density', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    return fig, ax


def plot_array_with_ci(
        data: np.ndarray,
        time_axis: int = 0,
        hue_axis: Optional[int] = None,
        hue_name: Optional[str] = None,
        hue_labels: Optional[List[str]] = None,
        input_lower_ci: Optional[np.ndarray] = None,
        input_upper_ci: Optional[np.ndarray] = None,
        sampling_freq: Optional[float] = None,
        figsize: tuple = (14, 7),
        linewidth: float = 2.5,
        ci_alpha: float = 0.25,
        colors: Optional[Union[dict, list, str]] = None,
        title: str = 'Data Plot',
        xlabel: str = 'Time',
        ylabel: str = 'Value',
        legend: bool = True,
        legend_kws: Optional[dict] = None,
        phase_series: Optional[pd.Series] = None,
        phase_cmap: str = 'tab10'
) -> Tuple[Figure, Axes]:
    """
    Plot n-dimensional array with optional confidence intervals, hue-based line grouping, and phase annotations.

    Generalizes the original `plot_psd_avg_with_std` to work with arbitrary numpy arrays,
    allowing flexible specification of time and hue dimensions, plus explicit confidence
    interval arrays instead of std-based shading. Optionally includes a phase subplot
    showing temporal phase annotations.

    Parameters
    ----------
    data : np.ndarray
        Input data array. Shape is arbitrary; time_axis and hue_axis specify which
        dimensions correspond to time and hue (line grouping).
    time_axis : int, optional
        Axis index representing time samples. Default is 0.
    hue_axis : Optional[int], optional
        Axis index for grouping into multiple lines (e.g., channels, conditions).
        If None, plots single line from aggregated data. Default is None.
    hue_name : Optional[str], optional
        Name for hue dimension (used in legend if hue_axis provided).
        E.g., 'Channel', 'Condition'. Default is None.
    hue_labels : Optional[List[str]], optional
        Custom labels for each hue level (e.g., channel names).
        Must match data.shape[hue_axis] length if provided. Default is None.
    input_lower_ci : Optional[np.ndarray], optional
        Lower confidence interval bounds. Must match data.shape. If None, no CI shading.
        Default is None.
    input_upper_ci : Optional[np.ndarray], optional
        Upper confidence interval bounds. Must match data.shape. If None, no CI shading.
        Default is None.
    sampling_freq : Optional[float], optional
        Sampling frequency in Hz. If provided, time axis is converted to seconds.
        If None, time axis is integer indices. Default is None.
    figsize : tuple, optional
        Figure size as (width, height). Default is (14, 7).
    linewidth : float, optional
        Line width for traces. Default is 2.5.
    ci_alpha : float, optional
        Transparency of shaded CI regions (0-1). Default is 0.25.
    colors : Optional[Union[dict, list, str]], optional
        Color specification. Can be:
        - dict: mapping hue indices/names to colors {0: '#1f77b4', 1: '#ff7f0e', ...}
        - list: sequence of colors to cycle through
        - str: single color for all lines
        - None: uses default matplotlib palette
        Default is None.
    title : str, optional
        Plot title. Default is 'Data Plot'.
    xlabel : str, optional
        X-axis label. Default is 'Time'.
    ylabel : str, optional
        Y-axis label. Default is 'Value'.
    legend : bool, optional
        Whether to show legend. Default is True.
    legend_kws : Optional[dict], optional
        Additional keyword arguments passed to ax.legend(). Default is None.
    phase_series : Optional[pd.Series], optional
        Time-indexed Series with string phase labels. NaN values are ignored.
        Adds a phase subplot below main plot. Default is None.
    phase_cmap : str, optional
        Colormap for phase regions. Default is 'tab10'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The main axes object (data plot).

    Raises
    ------
    ValueError
        If time_axis or hue_axis out of bounds, if CI array shapes don't match data,
        or if hue_labels length doesn't match hue_axis dimension.

    Examples
    --------
    Plot 2D array (time x channels) with multiple lines per channel:

    >>> data = np.random.randn(100, 5)  # 100 time points, 5 channels
    >>> fig, ax = plot_array_with_ci(data, time_axis=0, hue_axis=1,
    ...                               hue_name='Channel', sampling_freq=100)

    Plot with confidence intervals and custom channel labels:

    >>> lower_ci = data - 0.1
    >>> upper_ci = data + 0.1
    >>> channel_names = ['C3', 'C4', 'Cz', 'FC3', 'FC4']
    >>> fig, ax = plot_array_with_ci(data, time_axis=0, hue_axis=1,
    ...                               input_lower_ci=lower_ci,
    ...                               input_upper_ci=upper_ci,
    ...                               hue_labels=channel_names)

    Plot with phase annotations:

    >>> phase_series = pd.Series(['Rest', 'Task', 'Rest', 'Task'],
    ...                          index=pd.date_range('2024-01-01', periods=100, freq='10ms'))
    >>> fig, ax = plot_array_with_ci(data, phase_series=phase_series)
    """

    # Input validation
    _validate_inputs(data, time_axis, hue_axis, input_lower_ci, input_upper_ci, hue_labels)

    # Build time axis
    n_time_samples = data.shape[time_axis]
    time = _build_time_axis(n_time_samples, sampling_freq)

    # Prepare color mapping
    color_map = _prepare_colors(colors, hue_axis, data)

    # Create figure with optional phase subplot
    has_phases = phase_series is not None
    if has_phases:
        # GridSpec for flexible layout: 80% main plot, 20% phase subplot
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 1, figure=fig, height_ratios=[4, 1], hspace=0.3)
        ax = fig.add_subplot(gs[0])
        ax_phase = fig.add_subplot(gs[1], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_phase = None

    # Plot data by hue groups
    if hue_axis is None:
        # Single line: aggregate along all non-time axes
        mean_signal = _aggregate_non_time_axes(data, time_axis)
        lower = upper = None

        if input_lower_ci is not None:
            lower = _aggregate_non_time_axes(input_lower_ci, time_axis)
        if input_upper_ci is not None:
            upper = _aggregate_non_time_axes(input_upper_ci, time_axis)

        color = color_map.get(0, 'C0') if isinstance(color_map, dict) else color_map
        _plot_line_with_ci(ax, time, mean_signal, lower, upper, color,
                           label=None, linewidth=linewidth, ci_alpha=ci_alpha)

    else:
        # Multiple lines: iterate over hue dimension
        hue_size = data.shape[hue_axis]

        for hue_idx in range(hue_size):
            # Extract data slice for this hue level
            mean_signal = _extract_hue_slice(data, hue_axis, hue_idx, time_axis)
            lower = upper = None

            if input_lower_ci is not None:
                lower = _extract_hue_slice(input_lower_ci, hue_axis, hue_idx, time_axis)
            if input_upper_ci is not None:
                upper = _extract_hue_slice(input_upper_ci, hue_axis, hue_idx, time_axis)

            # Determine label and color
            label = _get_line_label(hue_idx, hue_name, hue_labels, color_map)
            color = _get_line_color(hue_idx, color_map)

            _plot_line_with_ci(ax, time, mean_signal, lower, upper, color,
                               label=label, linewidth=linewidth, ci_alpha=ci_alpha)

    # Customize main plot appearance
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    if legend and (hue_axis is not None or any(h is not None for h in
                                               _get_all_labels(hue_axis, color_map))):
        legend_params = {'loc': 'best', 'fontsize': 11}
        if legend_kws:
            legend_params.update(legend_kws)
        ax.legend(**legend_params)

    # Add phase subplot if provided
    if has_phases and ax_phase is not None:
        time_extent = (time[0], time[-1])
        _plot_phase_subplot(
            ax_phase=ax_phase,
            phase_series=phase_series,
            timestamps=time,
            time_extent=time_extent,
            phase_cmap=phase_cmap
        )
        # Remove x-label from main plot to avoid duplication
        ax.set_xticklabels([])
        ax_phase.set_xlabel(xlabel, fontsize=12, fontweight='bold')

    plt.tight_layout()

    return fig, ax


def old_plot_array_with_ci(
        data: np.ndarray,
        time_axis: int = 0,
        hue_axis: Optional[int] = None,
        hue_name: Optional[str] = None,
        hue_labels: Optional[List[str]] = None,
        input_lower_ci: Optional[np.ndarray] = None,
        input_upper_ci: Optional[np.ndarray] = None,
        sampling_freq: Optional[float] = None,
        figsize: tuple = (14, 7),
        linewidth: float = 2.5,
        ci_alpha: float = 0.25,
        colors: Optional[Union[dict, list, str]] = None,
        title: str = 'Data Plot',
        xlabel: str = 'Time',
        ylabel: str = 'Value',
        legend: bool = True,
        legend_kws: Optional[dict] = None
) -> Tuple[Figure, Axes]:
    """
    Plot n-dimensional array with optional confidence intervals and hue-based line grouping.

    Generalizes the original `plot_psd_avg_with_std` to work with arbitrary numpy arrays,
    allowing flexible specification of time and hue dimensions, plus explicit confidence
    interval arrays instead of std-based shading.

    Parameters
    ----------
    data : np.ndarray
        Input data array. Shape is arbitrary; time_axis and hue_axis specify which
        dimensions correspond to time and hue (line grouping).
    time_axis : int, optional
        Axis index representing time samples. Default is 0.
    hue_axis : Optional[int], optional
        Axis index for grouping into multiple lines (e.g., channels, conditions).
        If None, plots single line from aggregated data. Default is None.
    hue_name : Optional[str], optional
        Name for hue dimension (used in legend if hue_axis provided).
        E.g., 'Channel', 'Condition'. Default is None.
    hue_labels : Optional[List[str]], optional
        Custom labels for each hue level (e.g., channel names).
        Must match data.shape[hue_axis] length if provided. Default is None.
    input_lower_ci : Optional[np.ndarray], optional
        Lower confidence interval bounds. Must match data.shape. If None, no CI shading.
        Default is None.
    input_upper_ci : Optional[np.ndarray], optional
        Upper confidence interval bounds. Must match data.shape. If None, no CI shading.
        Default is None.
    sampling_freq : Optional[float], optional
        Sampling frequency in Hz. If provided, time axis is converted to seconds.
        If None, time axis is integer indices. Default is None.
    figsize : tuple, optional
        Figure size as (width, height). Default is (14, 7).
    linewidth : float, optional
        Line width for traces. Default is 2.5.
    ci_alpha : float, optional
        Transparency of shaded CI regions (0-1). Default is 0.25.
    colors : Optional[Union[dict, list, str]], optional
        Color specification. Can be:
        - dict: mapping hue indices/names to colors {0: '#1f77b4', 1: '#ff7f0e', ...}
        - list: sequence of colors to cycle through
        - str: single color for all lines
        - None: uses default matplotlib palette
        Default is None.
    title : str, optional
        Plot title. Default is 'Data Plot'.
    xlabel : str, optional
        X-axis label. Default is 'Time'.
    ylabel : str, optional
        Y-axis label. Default is 'Value'.
    legend : bool, optional
        Whether to show legend. Default is True.
    legend_kws : Optional[dict], optional
        Additional keyword arguments passed to ax.legend(). Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.

    Raises
    ------
    ValueError
        If time_axis or hue_axis out of bounds, if CI array shapes don't match data,
        or if hue_labels length doesn't match hue_axis dimension.

    Examples
    --------
    Plot 2D array (time x channels) with multiple lines per channel:

    >>> data = np.random.randn(100, 5)  # 100 time points, 5 channels
    >>> fig, ax = plot_array_with_ci(data, time_axis=0, hue_axis=1,
    ...                               hue_name='Channel', sampling_freq=100)

    Plot with confidence intervals and custom channel labels:

    >>> lower_ci = data - 0.1
    >>> upper_ci = data + 0.1
    >>> channel_names = ['C3', 'C4', 'Cz', 'FC3', 'FC4']
    >>> fig, ax = plot_array_with_ci(data, time_axis=0, hue_axis=1,
    ...                               input_lower_ci=lower_ci,
    ...                               input_upper_ci=upper_ci,
    ...                               hue_labels=channel_names)

    Plot single 1D array with explicit colors:

    >>> data_1d = np.random.randn(100)
    >>> fig, ax = plot_array_with_ci(data_1d, colors='steelblue')
    """

    # Input validation
    _validate_inputs(data, time_axis, hue_axis, input_lower_ci, input_upper_ci, hue_labels)

    # Build time axis
    n_time_samples = data.shape[time_axis]
    time = _build_time_axis(n_time_samples, sampling_freq)

    # Prepare color mapping
    color_map = _prepare_colors(colors, hue_axis, data)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot data by hue groups
    if hue_axis is None:
        # Single line: aggregate along all non-time axes
        mean_signal = _aggregate_non_time_axes(data, time_axis)
        lower = upper = None

        if input_lower_ci is not None:
            lower = _aggregate_non_time_axes(input_lower_ci, time_axis)
        if input_upper_ci is not None:
            upper = _aggregate_non_time_axes(input_upper_ci, time_axis)

        color = color_map.get(0, 'C0') if isinstance(color_map, dict) else color_map
        _plot_line_with_ci(ax, time, mean_signal, lower, upper, color,
                           label=None, linewidth=linewidth, ci_alpha=ci_alpha)

    else:
        # Multiple lines: iterate over hue dimension
        hue_size = data.shape[hue_axis]

        for hue_idx in range(hue_size):
            # Extract data slice for this hue level
            mean_signal = _extract_hue_slice(data, hue_axis, hue_idx, time_axis)
            lower = upper = None

            if input_lower_ci is not None:
                lower = _extract_hue_slice(input_lower_ci, hue_axis, hue_idx, time_axis)
            if input_upper_ci is not None:
                upper = _extract_hue_slice(input_upper_ci, hue_axis, hue_idx, time_axis)

            # Determine label and color
            label = _get_line_label(hue_idx, hue_name, hue_labels, color_map)
            color = _get_line_color(hue_idx, color_map)

            _plot_line_with_ci(ax, time, mean_signal, lower, upper, color,
                               label=label, linewidth=linewidth, ci_alpha=ci_alpha)

    # Customize plot appearance
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    if legend and (hue_axis is not None or any(h is not None for h in
                                               _get_all_labels(hue_axis, color_map))):
        legend_params = {'loc': 'best', 'fontsize': 11}
        if legend_kws:
            legend_params.update(legend_kws)
        ax.legend(**legend_params)

    plt.tight_layout()

    return fig, ax


# ============================================================================
# Private Helper Functions
# ============================================================================

def _validate_inputs(
        data: np.ndarray,
        time_axis: int,
        hue_axis: Optional[int],
        input_lower_ci: Optional[np.ndarray],
        input_upper_ci: Optional[np.ndarray],
        hue_labels: Optional[List[str]]
) -> None:
    """
    Validate all input parameters for consistency and validity.

    Raises ValueError if:
    - Axes are out of bounds
    - Axes are identical
    - CI array shapes don't match data shape
    - hue_labels length doesn't match hue_axis dimension
    """
    ndim = data.ndim

    if not (0 <= time_axis < ndim):
        raise ValueError(f"time_axis={time_axis} out of bounds for data.ndim={ndim}")

    if hue_axis is not None:
        if not (0 <= hue_axis < ndim):
            raise ValueError(f"hue_axis={hue_axis} out of bounds for data.ndim={ndim}")
        if time_axis == hue_axis:
            raise ValueError(f"time_axis and hue_axis cannot be identical (both={time_axis})")

    if input_lower_ci is not None:
        if input_lower_ci.shape != data.shape:
            raise ValueError(
                f"input_lower_ci shape {input_lower_ci.shape} != data shape {data.shape}"
            )

    if input_upper_ci is not None:
        if input_upper_ci.shape != data.shape:
            raise ValueError(
                f"input_upper_ci shape {input_upper_ci.shape} != data shape {data.shape}"
            )

    if hue_labels is not None:
        if hue_axis is None:
            raise ValueError("hue_labels provided but hue_axis is None")
        if len(hue_labels) != data.shape[hue_axis]:
            raise ValueError(
                f"hue_labels length {len(hue_labels)} != data.shape[{hue_axis}]={data.shape[hue_axis]}"
            )


def _build_time_axis(n_samples: int, sampling_freq: Optional[float]) -> np.ndarray:
    """
    Build time axis array.

    Parameters
    ----------
    n_samples : int
        Number of time samples.
    sampling_freq : Optional[float]
        Sampling frequency in Hz. If provided, time in seconds; else integer indices.

    Returns
    -------
    np.ndarray
        Time axis values.
    """
    if sampling_freq is not None:
        return np.arange(n_samples) / sampling_freq
    return np.arange(n_samples)


def _aggregate_non_time_axes(
        data: np.ndarray,
        time_axis: int
) -> np.ndarray:
    """
    Aggregate data along all axes except time_axis via mean.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    time_axis : int
        Index of time axis (to preserve).

    Returns
    -------
    np.ndarray
        1D array of shape (n_time_samples,).
    """
    # Move time axis to position 0, then mean over remaining axes
    data_moved = np.moveaxis(data, time_axis, 0)
    return data_moved.reshape(data_moved.shape[0], -1).mean(axis=1)


def _extract_hue_slice(
        data: np.ndarray,
        hue_axis: int,
        hue_idx: int,
        time_axis: int
) -> np.ndarray:
    """
    Extract 1D time series for a specific hue level, aggregating other dimensions.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    hue_axis : int
        Index of hue axis.
    hue_idx : int
        Index along hue axis to extract.
    time_axis : int
        Index of time axis.

    Returns
    -------
    np.ndarray
        1D array of shape (n_time_samples,).
    """
    # Index along hue axis
    slices = [slice(None)] * data.ndim
    slices[hue_axis] = hue_idx
    sliced = data[tuple(slices)]

    # Move time axis to position 0
    sliced_moved = np.moveaxis(sliced, time_axis, 0)

    # Aggregate remaining dimensions
    return sliced_moved.reshape(sliced_moved.shape[0], -1).mean(axis=1)


def _prepare_colors(
        colors: Optional[Union[dict, list, str]],
        hue_axis: Optional[int],
        data: np.ndarray
) -> Union[dict, list, str]:
    """
    Normalize color specification into usable format.

    Parameters
    ----------
    colors : Optional[Union[dict, list, str]]
        User-provided colors.
    hue_axis : Optional[int]
        If provided, number of hue levels determines default palette size.
    data : np.ndarray
        Data array (for shape info if needed).

    Returns
    -------
    Union[dict, list, str]
        Normalized colors.
    """
    if colors is not None:
        return colors

    # Default palette
    default_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    if hue_axis is not None:
        hue_size = data.shape[hue_axis]
        return {i: default_palette[i % len(default_palette)] for i in range(hue_size)}

    return 'C0'


def _get_line_label(
        hue_idx: int,
        hue_name: Optional[str],
        hue_labels: Optional[List[str]],
        color_map: Union[dict, list, str]
) -> Optional[str]:
    """
    Generate label for a line based on hue index, hue_name, and hue_labels.

    Parameters
    ----------
    hue_idx : int
        Index along hue axis.
    hue_name : Optional[str]
        Name of hue dimension.
    hue_labels : Optional[List[str]]
        Custom labels for each hue level.
    color_map : Union[dict, list, str]
        Color mapping (used to infer if labels expected).

    Returns
    -------
    Optional[str]
        Label string or None.
    """
    # Priority: custom hue_labels > hue_name + index
    if hue_labels is not None:
        return hue_labels[hue_idx]

    if hue_name is None and not isinstance(color_map, dict):
        return None

    if hue_name:
        return f"{hue_name} {hue_idx}"

    return str(hue_idx)


def _get_line_color(
        hue_idx: int,
        color_map: Union[dict, list, str]
) -> str:
    """
    Get color for a specific hue index.

    Parameters
    ----------
    hue_idx : int
        Index along hue axis.
    color_map : Union[dict, list, str]
        Color mapping.

    Returns
    -------
    str
        Color specification.
    """
    if isinstance(color_map, dict):
        return color_map.get(hue_idx, 'C0')
    elif isinstance(color_map, list):
        return color_map[hue_idx % len(color_map)]
    else:
        return color_map


def _get_all_labels(hue_axis: Optional[int], color_map: Union[dict, list, str]) -> list:
    """
    Check if any labels will be generated (for legend decision).

    Parameters
    ----------
    hue_axis : Optional[int]
        Hue axis specification.
    color_map : Union[dict, list, str]
        Color mapping.

    Returns
    -------
    list
        Placeholder for label check.
    """
    if hue_axis is None:
        return [None]
    return [None]  # Simplified; actual labels generated in plot loop


def _plot_line_with_ci(
        ax: Axes,
        time: np.ndarray,
        signal: np.ndarray,
        lower_ci: Optional[np.ndarray],
        upper_ci: Optional[np.ndarray],
        color: str,
        label: Optional[str],
        linewidth: float,
        ci_alpha: float
) -> None:
    """
    Plot a single line with optional confidence interval shading.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    time : np.ndarray
        Time axis values.
    signal : np.ndarray
        1D signal to plot.
    lower_ci : Optional[np.ndarray]
        Lower CI bound (1D, same length as signal).
    upper_ci : Optional[np.ndarray]
        Upper CI bound (1D, same length as signal).
    color : str
        Line color.
    label : Optional[str]
        Line label for legend.
    linewidth : float
        Line width.
    ci_alpha : float
        Transparency of CI shading.
    """
    # Plot mean line
    ax.plot(time, signal, label=label, linewidth=linewidth, color=color)

    # Plot CI shading if both bounds provided
    if lower_ci is not None and upper_ci is not None:
        ax.fill_between(time, lower_ci, upper_ci, alpha=ci_alpha, color=color)


def draw_forest_plot(ax, effects_frame: pd.DataFrame,
                     hypothesis_column: str = 'Hypothesis',
                     param_column: str = 'Parameter',
                     comparison_lvl_column: str = 'Comparison_Level',
                     model_type_column: str = 'Model_Type',
                     coeff_column: str = 'Coefficient',
                     se_column: str = 'SE',
                     p_column: str = 'p_value',
                     CI_z_score: float = 1.96,  # 90%: 1.645, 95%: 1.96, 99%: 2.576
                     significant_pos_color: str = 'green',
                     significant_neg_color: str = 'red',
                     insignificant_color: str = '#AAAAAA',
                     include_y_labels: bool = True,
                     title_max_width: int = 40,
                     show_significance_legend: bool = False,
                     ):
    """
    Creates a beautiful forest plot for statistical effects.

    Takes a subset of the overall results frame with only one hypothesis!

    Parameters:
    -----------
    include_y_labels : bool
        If True, shows y-axis labels (use for first column).
        If False, hides y-axis labels but keeps ticks (use for subsequent columns).
    title_max_width : int
        Maximum character width for title before wrapping (default: 40)
    show_significance_legend : bool
        If True, shows significance legend (default: False)
    significant_pos_color : str
        Color for significant positive effects (default: 'green')
    significant_neg_color : str
        Color for significant negative effects (default: 'red')
    insignificant_color : str
        Color for non-significant effects (default: '#AAAAAA')
    """

    assert len(effects_frame[hypothesis_column].unique()) == 1, \
        "Please provide a subset of the results frame with only one hypothesis!"

    # Make a copy to avoid modifying original
    df = effects_frame.copy().reset_index(drop=True)

    # Create enumerated comparison level mapping
    unique_levels = sorted(df[comparison_lvl_column].unique())
    level_mapping = {level: f'lvl{i}' for i, level in enumerate(unique_levels)}
    df['comparison_lvl_enum'] = df[comparison_lvl_column].map(level_mapping)

    # Extract numeric level for sorting
    df['level_num'] = df['comparison_lvl_enum'].str.extract(r'(\d+)').astype(int)

    # Sort by parameter name, then by level number, then by model type
    df = df.sort_values(by=[param_column, 'level_num', model_type_column]).reset_index(drop=True)

    # Create y-axis labels combining parameter, enumerated level, and model type
    df['y_label'] = (df[param_column].astype(str) + ' | ' +
                     df['comparison_lvl_enum'].astype(str) + ' | ' +
                     df[model_type_column].astype(str))

    # Calculate confidence intervals
    df['ci_lower'] = df[coeff_column] - (CI_z_score * df[se_column])
    df['ci_upper'] = df[coeff_column] + (CI_z_score * df[se_column])

    # Determine significance levels and create labels
    def get_significance(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''  # Empty string for non-significant

    df['sig_label'] = df[p_column].apply(get_significance)
    df['is_significant'] = df[p_column] < 0.05

    # Assign colors based on significance AND effect direction
    def assign_color(row):
        if row['is_significant']:
            if row[coeff_column] > 0:
                return significant_pos_color
            else:
                return significant_neg_color
        else:
            return insignificant_color

    df['color'] = df.apply(assign_color, axis=1)

    # Assign y-positions with spacing between parameter groups
    y_position = 0
    separator_positions = []
    current_param = None

    for idx, row in df.iterrows():
        if current_param is not None and row[param_column] != current_param:
            # Add separator position BEFORE incrementing y_position
            separator_positions.append(y_position)
            # Add spacing between parameter groups
            y_position += 1

        df.at[idx, 'y_pos'] = y_position
        current_param = row[param_column]
        y_position += 1

    # Reverse y-positions so first row is at top
    max_y = df['y_pos'].max()
    df['y_pos'] = max_y - df['y_pos']
    separator_positions = [max_y - pos for pos in separator_positions]

    # Plot confidence interval lines (whiskers)
    for idx, row in df.iterrows():
        ax.plot([row['ci_lower'], row['ci_upper']],
                [row['y_pos'], row['y_pos']],
                color=row['color'],
                linewidth=2,
                alpha=0.8,
                zorder=1)

    # Plot coefficient points
    ax.scatter(df[coeff_column], df['y_pos'],
               c=df['color'],
               s=100,
               zorder=2,
               edgecolors='white',
               linewidths=1.5,
               alpha=0.9)

    # Add vertical reference line at zero (null effect / H0)
    ax.axvline(x=0.0, color='black', linestyle='--', linewidth=1.5,
               alpha=0.6, zorder=0, label='H₀ (no effect)')

    # Add horizontal separator lines between parameter groups
    x_min = df['ci_lower'].min()
    x_max = df['ci_upper'].max()
    x_range = x_max - x_min

    # Draw separator lines exactly at the empty row positions
    for sep_pos in separator_positions:
        ax.axhline(y=sep_pos, color='lightgray', linestyle='--',
                   linewidth=1, alpha=0.5, zorder=0)

    # Add significance labels on the right
    text_x = x_max + 0.05 * x_range

    for idx, row in df.iterrows():
        if row['sig_label']:  # Only show if not empty
            ax.text(text_x, row['y_pos'], row['sig_label'],
                    va='center', ha='left', fontsize=10,
                    fontweight='bold',
                    color=row['color'])

    # Set y-axis
    y_positions = df['y_pos'].values
    ax.set_yticks(y_positions)

    if include_y_labels:
        # All y-tick labels in black
        ax.set_yticklabels(df['y_label'], fontsize=9)
    else:
        # Hide labels but keep ticks visible
        ax.set_yticklabels([])

    ax.set_ylim(-0.5, max_y + 0.5)

    # Set x-axis
    ax.set_xlabel('Effect Size (β coefficient)', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', labelsize=10)

    # Add grid for readability
    ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_axisbelow(True)

    # Set title from hypothesis with text wrapping
    hypothesis_name = df[hypothesis_column].iloc[0]
    wrapped_title = '\n'.join(textwrap.wrap(hypothesis_name, width=title_max_width))
    ax.set_title(wrapped_title, fontsize=9, fontweight='bold', pad=10)

    # Add a subtle spine styling - always show left spine and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['left'].set_visible(True)  # Always visible
    ax.spines['bottom'].set_linewidth(1.2)

    # Adjust x-limits to accommodate significance labels
    x_margin = x_range * 0.15
    ax.set_xlim(x_min - x_margin, text_x + x_margin)

    # Add optional legend for significance markers
    if show_significance_legend and include_y_labels:
        sig_text = "* p<0.05  ** p<0.01  *** p<0.001"
        ax.text(0.02, 0.98, sig_text,
                transform=ax.transAxes,
                fontsize=8,
                va='top',
                ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

    # Return both the axis and the level mapping for reference
    return ax, level_mapping




def plot_hypothesis_forest_mosaic(result_frame: pd.DataFrame,
                                  hypotheses: list[str],
                                  exclude_intercepts: bool = True,
                                  model_type: str | None  = 'LME',
                                  output_dir: Path = None,
                                  file_identifier_suffix: str | None = None,
                                  hidden: bool = False,
                                  ):
    # slice results_frame:
    results_frame_subset = result_frame.copy()
    if exclude_intercepts:
        results_frame_subset = results_frame_subset[results_frame_subset['Parameter'] != 'Intercept']
    if model_type is not None:
        results_frame_subset = results_frame_subset[results_frame_subset['Model_Type'] == model_type]
    # formatting:
    results_frame_subset['Parameter'] = results_frame_subset['Parameter'].str.replace('C(', '')
    results_frame_subset['Parameter'] = results_frame_subset['Parameter'].str.replace('Q(', '')
    results_frame_subset['Parameter'] = results_frame_subset['Parameter'].str.replace("'", "")
    results_frame_subset['Parameter'] = results_frame_subset['Parameter'].str.replace(")", "")

    # prepare plot mosaic:
    fig, axs = plt.subplots(1, len(hypotheses), figsize=(15, 10))
    # plot hypothesis forest plots:
    for col_ind, hypothesis in enumerate(hypotheses):
        print(f"Plotting forest plot ({col_ind}) for hypothesis: {hypothesis}")

        axs[col_ind], _ = draw_forest_plot(
            axs[col_ind],
            effects_frame=results_frame_subset.loc[results_frame_subset['Hypothesis'] == hypothesis, :],
            include_y_labels=(col_ind == 0)  # Only True for first column
        )

    fig_title = f"Effect Size Overview{f' ({model_type} models)' if model_type is not None else ''}{f' ({file_identifier_suffix})' if file_identifier_suffix is not None else ''}"
    fig.suptitle(fig_title)

    fig.tight_layout()

    if output_dir is not None:
        save_path = filemgmt.file_title(fig_title, '.pdf')
        fig.savefig(output_dir / save_path, bbox_inches='tight')

    if not hidden:
        plt.show()



if __name__ == '__main__':
    initialise_electrode_heatmap(values=[0]*64,
                                 positios=EEG_POSITIONS, add_head_shape=True)