import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
from typing import Callable, Literal, Tuple
import multiprocessing
import os
import math

import src.utils.file_management as filemgmt
import src.pipeline.signal_features as features

mpl.use('Qt5Agg')

##############  CONSTANT PARAMETERS ##############
### EEG:
EEG_CHANNELS = ['Fpz', 'Fp1', 'Fp2',
                'AF7', 'AF3', 'AF4', 'AF8',
                'F9', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'F10',
                'FT9', 'FT7',
                'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
                'FT8', 'FT10',
                'T7',
                'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                'T8',
                'TP7',
                'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
                'TP8',
                'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                'O1', 'Oz', 'O2',
                ]  # according to https://www.bitbrain.com/neurotechnology-products/water-based-eeg/versatile-eeg
EEG_CHANNELS_BY_AREA = {
    area_label: [ch for ch in EEG_CHANNELS if (ch[:len(area_abbr)] == area_abbr) and (len(ch) == len(area_abbr)+1)] for area_label, area_abbr in [
        ('Frontal Pole', 'Fp'), ('Anterior Frontal', 'AF'), ('Fronto-Central', 'FC'), ('Frontal', 'F'),
        ('Fronto-Temporal', 'FT'), ('Temporal', 'T'), ('Central', 'C'), ('Temporo-Parietal', 'TP'),
        ('Centro-Parietal', 'CP'), ('Parietal', 'P'), ('Parieto-Occipital', 'PO'), ('Occipital', 'O')]}
EEG_CHANNEL_IND_DICT = {ch: ind+1 for ind, ch in enumerate(EEG_CHANNELS)}

EEG_POSITIONS = {'Fpz': (0.0, 0.602),
                 'Fp1': (-0.165, 0.5599999999999999),
                 'Fp2': (0.165, 0.5599999999999999),
                 'AF7': (-0.30800000000000005, 0.48999999999999994),
                 'AF3': (-0.15400000000000003, 0.44799999999999995),
                 'AF4': (0.15400000000000003, 0.44799999999999995),
                 'AF8': (0.30800000000000005, 0.48999999999999994),
                 'F9': (-0.5060000000000001, 0.45499999999999996),
                 'F7': (-0.42900000000000005, 0.385),
                 'F5': (-0.33, 0.32899999999999996),
                 'F3': (-0.22000000000000003, 0.294),
                 'F1': (-0.11000000000000001, 0.26599999999999996),
                 'Fz': (0.0, 0.252),
                 'F2': (0.11000000000000001, 0.26599999999999996),
                 'F4': (0.22000000000000003, 0.294),
                 'F6': (0.33, 0.32899999999999996),
                 'F8': (0.42900000000000005, 0.385),
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
                 'T7': (-0.55, 0.0),
                 'C5': (-0.41250000000000003, 0.0),
                 'C3': (-0.275, 0.0),
                 'C1': (-0.1375, 0.0),
                 'Cz': (0.0, 0.0),
                 'C2': (0.1375, 0.0),
                 'C4': (0.275, 0.0),
                 'C6': (0.41250000000000003, 0.0),
                 'T8': (0.55, 0.0),
                 'TP7': (-0.48400000000000004, -0.196),
                 'CP5': (-0.36850000000000005, -0.16799999999999998),
                 'CP3': (-0.25300000000000006, -0.147),
                 'CP1': (-0.12925, -0.13299999999999998),
                 'CPz': (0.0, -0.126),
                 'CP2': (0.12925, -0.13299999999999998),
                 'CP4': (0.25300000000000006, -0.147),
                 'CP6': (0.36850000000000005, -0.16799999999999998),
                 'TP8': (0.48400000000000004, -0.196),
                 'P7': (-0.42900000000000005, -0.385),
                 'P5': (-0.33, -0.32899999999999996),
                 'P3': (-0.22000000000000003, -0.294),
                 'P1': (-0.11000000000000001, -0.26599999999999996),
                 'Pz': (0.0, -0.252),
                 'P2': (0.11000000000000001, -0.26599999999999996),
                 'P4': (0.22000000000000003, -0.294),
                 'P6': (0.33, -0.32899999999999996),
                 'P8': (0.42900000000000005, -0.385),
                 'PO7': (-0.30800000000000005, -0.48999999999999994),
                 'PO3': (-0.15400000000000003, -0.44799999999999995),
                 'POz': (0.0, -0.42),
                 'PO4': (0.15400000000000003, -0.44799999999999995),
                 'PO8': (0.30800000000000005, -0.48999999999999994),
                 'O1': (-0.165, -0.5599999999999999),
                 'Oz': (0.0, -0.602),
                 'O2': (0.165, -0.5599999999999999)}
# from -.6 to +.6 in both dimensions:
x_list = list(np.linspace(-.6, .6, 8))*8
y_list = []
for element in list(np.linspace(-.6, .6, 8))[::-1]:
    y_list += [element] * 8
EMG_POSITIONS = {str(ind+1): (x, y) for ind, (x, y) in enumerate(zip(x_list, y_list))}


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
        positions = {ind+1: entry for ind, entry in enumerate(positions)}

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
    ani = FuncAnimation(fig, update, frames=total_frames, blit=False,
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
        plt.plot(frequencies, amplitude_spectrum[:, ch], label=f'Channel {ch+1}' if channel_labels is None else channel_labels[ch])
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
        frequencies: np.ndarray,
        cmap: str = 'viridis',
        frequency_range: tuple[float, float] | None = None,
        log_scale: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        title: str = 'Spectrogram',
        xlabel: str = 'Time [s]',
        ylabel: str = 'Frequency [Hz]',
        cbar_label: str = 'Power [V²/Hz]',
        figsize: tuple[float, float] = (14, 6),
        aspect: str = 'auto',
        save_dir: str | Path = None,
        continue_code: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize a time-frequency spectrogram.

    Parameters
    ----------
    spectrogram : ndarray
        Power spectral density matrix, shape (n_windows, n_frequencies)
        Each column represents a time window, each row a frequency bin
    timestamps : ndarray
        Time centers of each window (seconds), shape (n_windows,)
    frequencies : ndarray
        Frequency array (Hz), shape (n_frequencies,)
    cmap : str, optional
        Matplotlib colormap name. Default: 'viridis'
        Options: 'viridis', 'magma', 'plasma', 'inferno', 'turbo', 'jet', 'cool'
    frequency_range : tuple, optional
        (fmin, fmax) to restrict displayed frequency range in Hz.
        If None, uses full range. Default: None
    log_scale : bool, optional
        Apply log10 scaling to the spectrogram for better dynamic range visualization.
        Recommended for PSD data with high variance. Default: True
    vmin, vmax : float, optional
        Min/max values for colormap normalization.
        If None, automatically determined from data.
        Useful for fixing the color scale across multiple plots.
        Default: None
    title : str, optional
        Title of the plot. Default: 'Spectrogram'
    xlabel : str, optional
        X-axis label. Default: 'Time (s)'
    ylabel : str, optional
        Y-axis label. Default: 'Frequency (Hz)'
    cbar_label : str, optional
        Colorbar label. Default: 'Power (V²/Hz)'
    figsize : tuple, optional
        Figure size (width, height) in inches. Default: (14, 6)
    aspect : str, optional
        Aspect ratio control. 'auto' preserves data proportions.
        Default: 'auto'
    save_dir : str or Path, optional
        Directory to save the figure.
    continue_code: bool, default False
        Whether to continue code execution while fig is shown.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object (use for saving with fig.savefig())
    ax : matplotlib.axes.Axes
        Axes object with the spectrogram plot

    Raises
    ------
    ValueError
        If array shapes don't match

    Examples
    --------
    >>> # Basic usage
    >>> fig, ax = plot_spectrogram(spec, times, freqs)
    >>> plt.show()

    >>> # With frequency range restriction
    >>> fig, ax = plot_spectrogram(
    ...     spec, times, freqs,
    ...     frequency_range=(0, 100),
    ...     cmap='magma',
    ...     log_scale=True
    ... )

    >>> # Save figure
    >>> fig, ax = plot_spectrogram(spec, times, freqs)
    >>> fig.savefig('spectrogram.png', dpi=300, bbox_inches='tight')
    """

    # Input validation
    if spectrogram.shape[1] != len(frequencies):
        raise ValueError(
            f"Number of frequencies ({spectrogram.shape[1]}) does not match "
            f"frequencies array length ({len(frequencies)})"
        )
    if spectrogram.shape[0] != len(timestamps):
        raise ValueError(
            f"Number of windows ({spectrogram.shape[0]}) does not match "
            f"timestamps array length ({len(timestamps)})"
        )

    # Copy to avoid modifying input
    spec = spectrogram.T.copy()  # transpose spectrogram for visualization (frequency on y-axis)
    freqs = frequencies.copy()
    times = timestamps.copy()

    # Apply frequency range filter if specified
    if frequency_range is not None:
        fmin, fmax = frequency_range
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        spec = spec[freq_mask, :]
        freqs = freqs[freq_mask]

    # Apply log scaling if requested
    if log_scale:
        spec = np.log10(spec + 1e-10)
        log_suffix = ' (log10)'  # to amend colorbar label:
    else: log_suffix = ''

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate extent for proper axis labeling
    extent = (
        times[0] - (times[1] - times[0]) / 2,
        times[-1] + (times[-1] - times[-2]) / 2,
        freqs[0] - (freqs[1] - freqs[0]) / 2,
        freqs[-1] + (freqs[-1] - freqs[-2]) / 2
    )

    # plot spectrogram:
    im = ax.imshow(spec, aspect=aspect, origin='lower',  # frequency 0 at bottom
                   extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')

    # labels:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # colorbar:
    cbar = plt.colorbar(im, ax=ax, label=cbar_label + log_suffix)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()

    # eventually save fig:
    if save_dir is not None:
        smart_save_fig(save_dir, title)

    plt.show(block=not continue_code)
    return fig, ax