from __future__ import annotations

import numpy as np
import warnings
from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Union, Tuple, List, Callable, Literal, TYPE_CHECKING
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
from tqdm import tqdm
import os
import textwrap
from scipy.stats import gaussian_kde
import math


import src.utils.file_management as filemgmt
from src.utils.str_conversion import enter_line_breaks
import src.pipeline.data_analysis as data_analysis
import src.pipeline.signal_features as features
import src.pipeline.data_integration as data_integration
from src.pipeline.channel_layout import EEG_CHANNELS, EEG_CHANNELS_BY_AREA, EEG_CHANNEL_IND_DICT, EMG_CHANNELS

if TYPE_CHECKING:
    from src.pipeline.cbpa import CBPAConfig




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


def plot_category_reassignment_sankey(category_reassignment_frame: pd.DataFrame,
                                      song_colors: dict[str, str],
                                      preferred_order: list[str] | None = None,
                                      show_title: bool = False,
                                      output_dir: str | Path | None = None,
                                      width: int = 700,
                                      height: int = 400):
    """Plot category reassignments as a two-column Sankey (original -> perceived)."""
    import plotly.graph_objects as go

    if preferred_order is None:
        preferred_order = ['Happy', 'Groovy', 'Sad', 'Classic']

    sankey_source = category_reassignment_frame[['from', 'to']].dropna()
    if len(sankey_source) == 0:
        print("No category reassignments available for Sankey plot.")
        return None

    transition_counts = (
        sankey_source
        .groupby(['from', 'to'], as_index=False)
        .size()
        .rename(columns={'size': 'value'})
    )

    def ordered_categories(values: list[str]) -> list[str]:
        present = [cat for cat in preferred_order if cat in values]
        remaining = sorted([cat for cat in values if cat not in preferred_order])
        return present + remaining

    def spaced_positions(n: int, top: float = 0.08, bottom: float = 0.92) -> list[float]:
        if n <= 1:
            return [0.50]
        return np.linspace(top, bottom, n).tolist()

    def color_with_alpha(category: str, alpha: float) -> str:
        base_color = song_colors.get(category, 'gray')
        r, g, b, _ = mcolors.to_rgba(base_color)
        return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha:.2f})"

    left_categories = ordered_categories(transition_counts['from'].unique().tolist())
    right_categories = ordered_categories(transition_counts['to'].unique().tolist())

    left_ids = [f"L::{cat}" for cat in left_categories]
    right_ids = [f"R::{cat}" for cat in right_categories]
    node_ids = left_ids + right_ids
    node_id_to_index = {node_id: ind for ind, node_id in enumerate(node_ids)}

    left_y = spaced_positions(len(left_categories))
    right_y = spaced_positions(len(right_categories))
    node_x = [0.13] * len(left_categories) + [0.87] * len(right_categories)
    node_y = left_y + right_y
    node_color = [color_with_alpha(cat, 0.85) for cat in (left_categories + right_categories)]

    link_source = [node_id_to_index[f"L::{cat}"] for cat in transition_counts['from']]
    link_target = [node_id_to_index[f"R::{cat}"] for cat in transition_counts['to']]
    link_value = transition_counts['value'].to_list()
    link_color = [color_with_alpha(cat, 0.55) for cat in transition_counts['from']]

    fig = go.Figure(data=[go.Sankey(
        arrangement='fixed',
        node=dict(
            # Render labels outside the node columns via annotations.
            label=[''] * len(node_ids),
            x=node_x,
            y=node_y,
            pad=36,
            thickness=18,
            color=node_color,
            line=dict(color='rgba(70,70,70,1)', width=1.0),
        ),
        link=dict(
            source=link_source,
            target=link_target,
            value=link_value,
            color=link_color,
        )
    )])

    annotations = [
        dict(text='Original Category', x=0.06, y=1.05, xref='paper', yref='paper', xanchor='right', showarrow=False),
        dict(text='Perceived Category', x=0.94, y=1.05, xref='paper', yref='paper', xanchor='left', showarrow=False),
    ]

    # Plotly Sankey node.y uses top-origin coordinates, while paper annotations
    # use bottom-origin coordinates. Convert to keep labels aligned to nodes.
    def node_y_to_paper_y(y_pos: float) -> float:
        return 1.0 - y_pos

    annotations += [
        dict(text=cat, x=0.06, y=node_y_to_paper_y(y_pos), xref='paper', yref='paper', xanchor='right', yanchor='middle', showarrow=False)
        for cat, y_pos in zip(left_categories, left_y)
    ]
    annotations += [
        dict(text=cat, x=0.94, y=node_y_to_paper_y(y_pos), xref='paper', yref='paper', xanchor='left', yanchor='middle', showarrow=False)
        for cat, y_pos in zip(right_categories, right_y)
    ]

    fig.update_layout(
        title='Category Reassignments' if show_title else None,
        font_size=12,
        width=width,
        height=height,
        margin=dict(l=140, r=140, t=70, b=30),
        annotations=annotations,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        filemgmt.assert_dir(output_dir)
        save_path = output_dir / filemgmt.file_title("Category Reassignment Sankey Plot", ".svg")
        try:
            fig.write_image(str(save_path))
        except ValueError as e:
            warnings.warn(
                "Could not export Sankey plot as SVG. "
                "Please install kaleido (pip install --upgrade kaleido). "
                f"Original error: {e}"
            )

    fig.show()
    return fig

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
    else: plt.close()

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

    # Handle NaN values in timestamps for task-wise data
    # Filter to only include valid (non-NaN) time windows
    valid_time_mask = ~np.isnan(times)
    if np.any(valid_time_mask):
        # Filter both spectrogram and timestamps to valid entries
        spec = spec[:, valid_time_mask]  # Keep features, filter windows (time axis)
        times = times[valid_time_mask]
    
    # If no valid times remain, raise informative error
    if len(times) == 0:
        raise ValueError(
            f"No valid timestamps found in input. All {len(timestamps)} timestamps are NaN. "
            f"Cannot plot spectrogram without valid time information."
        )

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
    # Safe to use times directly since NaN values have been filtered
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
        def can_compute_kde(values: np.ndarray, eps: float = 1e-12) -> tuple[bool, np.ndarray]:
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size < 2:
                return False, arr
            if np.unique(arr).size < 2:
                return False, arr
            if np.nanstd(arr) <= eps:
                return False, arr
            return True, arr

        if categorical_kdes and category_list is not None:
            # Plot separate KDE for each category
            x_range = np.linspace(np.asarray(x).min(), np.asarray(x).max(), 200)
            y_range = np.linspace(np.asarray(y).min(), np.asarray(y).max(), 200)

            for i, cat in enumerate(unique_cats):
                mask = np.array(category_list) == cat
                x_cat = np.asarray(x)[mask]
                y_cat = np.asarray(y)[mask]

                # Plot X-axis marginal KDE (top)
                can_kde_x, x_cat_clean = can_compute_kde(x_cat)
                if can_kde_x:
                    try:
                        kde_x = gaussian_kde(x_cat_clean)
                        ax_top.fill_between(x_range, kde_x(x_range), alpha=kde_alpha,
                                            color=colors[i],
                                            label=str(cat) if add_kde_legend else None)
                    except (np.linalg.LinAlgError, ValueError) as e:
                        print(f"[WARNING] Skipping X-KDE for category '{cat}': {e}")
                else:
                    print(f"[WARNING] Skipping X-KDE for category '{cat}': insufficient variation or too few valid values")

                # Plot Y-axis marginal KDE (right)
                can_kde_y, y_cat_clean = can_compute_kde(y_cat)
                if can_kde_y:
                    try:
                        kde_y = gaussian_kde(y_cat_clean)
                        ax_right.fill_betweenx(y_range, kde_y(y_range), alpha=kde_alpha,
                                               color=colors[i],
                                               label=str(cat) if add_kde_legend else None)
                    except (np.linalg.LinAlgError, ValueError) as e:
                        print(f"[WARNING] Skipping Y-KDE for category '{cat}': {e}")
                else:
                    print(f"[WARNING] Skipping Y-KDE for category '{cat}': insufficient variation or too few valid values")

            # Add legends only if labels were assigned
            if add_kde_legend:
                ax_top.legend(loc='upper right', fontsize=8)
                ax_right.legend(loc='upper right', fontsize=8)
        else:
            # Plot overall KDE (no categories or categorical_kdes=False)
            x_range = np.linspace(np.asarray(x).min(), np.asarray(x).max(), 200)
            can_kde_x, x_clean = can_compute_kde(np.asarray(x))
            if can_kde_x:
                try:
                    kde_x = gaussian_kde(x_clean)
                    ax_top.fill_between(x_range, kde_x(x_range), alpha=kde_alpha,
                                        color='steelblue')
                except (np.linalg.LinAlgError, ValueError) as e:
                    print(f"[WARNING] Skipping overall X-KDE: {e}")
            else:
                print("[WARNING] Skipping overall X-KDE: insufficient variation or too few valid values")

            y_range = np.linspace(np.asarray(y).min(), np.asarray(y).max(), 200)
            can_kde_y, y_clean = can_compute_kde(np.asarray(y))
            if can_kde_y:
                try:
                    kde_y = gaussian_kde(y_clean)
                    ax_right.fill_betweenx(y_range, kde_y(y_range), alpha=kde_alpha,
                                           color='steelblue')
                except (np.linalg.LinAlgError, ValueError) as e:
                    print(f"[WARNING] Skipping overall Y-KDE: {e}")
            else:
                print("[WARNING] Skipping overall Y-KDE: insufficient variation or too few valid values")

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

def _resolve_p_column(
    df: pd.DataFrame,
    significance_source: Literal["autocorr", "fdr", "auto"] = "auto",
    fdr_col: str = "p_value_fdr",
    autocorr_col: str = "p_value_adjusted",
    fallback_col: str = "p_value_for_plot",
) -> str:
    """Return the name of the p-value column to use for significance colouring.

    'auto'     → use p_value_for_plot if present (FDR where available,
                 autocorr-adjusted elsewhere), else p_value_adjusted
    'fdr'      → use p_value_fdr; warn if column is absent or all-NaN
    'autocorr' → always use p_value_adjusted
    """
    if significance_source == "autocorr":
        return autocorr_col
    if significance_source == "fdr":
        if fdr_col not in df.columns or df[fdr_col].isna().all():
            warnings.warn(
                f"[Forest plot] significance_source='fdr' but '{fdr_col}' is "
                f"absent or all-NaN. Falling back to '{autocorr_col}'."
            )
            return autocorr_col
        return fdr_col
    # 'auto'
    if fallback_col in df.columns and not df[fallback_col].isna().all():
        return fallback_col
    return autocorr_col


def _rename_parameter_label(label: str, rename_dict: dict[str, str] | None = None) -> str:
    """Return renamed parameter label if mapping exists; otherwise return original label."""
    label_str = str(label)
    if rename_dict is None:
        return label_str
    return rename_dict.get(label_str, label_str)


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
                     title_max_width: int = 30,
                     show_significance_legend: bool = False,
                     rename_dict: dict[str, str] | None = None,
                     parameter_label_colors: dict[str, str] | None = None,
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
    rename_dict : dict[str, str] | None
        Optional mapping for display names of parameter labels.
    parameter_label_colors : dict[str, str] | None
        Optional mapping from raw parameter name to y-tick font color.
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
    df['param_label_display'] = df[param_column].astype(str).apply(
        lambda x: _rename_parameter_label(x, rename_dict)
    )
    df['y_label'] = (df['param_label_display'].astype(str) + ' | ' +
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
        y_tick_labels = ax.set_yticklabels(df['y_label'], fontsize=7)
        if parameter_label_colors:
            for tick, param_name in zip(y_tick_labels, df[param_column].astype(str).tolist()):
                tick.set_color(parameter_label_colors.get(param_name, 'black'))
    else:
        # Hide labels but keep ticks visible
        ax.set_yticklabels([])

    ax.set_ylim(-0.5, max_y + 0.5)

    # Set x-axis
    ax.set_xlabel('Coefficient (β)', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', labelsize=10)

    # Add grid for readability
    ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_axisbelow(True)

    # Set title from hypothesis with text wrapping
    hypothesis_name = df[hypothesis_column].iloc[0]
    wrapped_title = '\n'.join(textwrap.wrap(hypothesis_name, width=title_max_width))
    ax.set_title(wrapped_title, fontsize=7, fontweight='bold', pad=10)

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



def draw_time_resolution_forest_plot(
        ax,
        effects_frame: pd.DataFrame,
        parameter: str,
        comparison_level: str | int,
        time_resolution_column: str,
        hypothesis: str | None = None,
        hypothesis_column: str = 'Hypothesis',
        param_column: str = 'Parameter',
        comparison_lvl_column: str = 'Comparison_Level',
        model_type_column: str = 'Model_Type',
        coeff_column: str = 'Coefficient',
        se_column: str = 'SE',
        p_column: str = 'p_value',
        y_axis_label: str = 'Model Time Resolution [sec]',
        CI_z_score: float = 1.96,
        significant_pos_color: str = 'green',
        significant_neg_color: str = 'red',
        insignificant_color: str = '#AAAAAA',
        include_y_labels: bool = True,
        show_significance_legend: bool = False,
        rename_dict: dict[str, str] | None = None,
):
    """
    Forest plot comparing one parameter at one comparison level across time resolutions.

    The parameter name is shown as the plot title. The y-axis holds time resolution
    labels (from `time_resolution_column`), one row per model type per resolution.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    effects_frame : pd.DataFrame
        Full results frame — will be filtered to `parameter` x `comparison_level`.
    parameter : str
        Exact value to match in `param_column`.
    comparison_level : str
        Exact value to match in `comparison_lvl_column`.
    time_resolution_column : str
        Column whose values label the y-axis rows (e.g. 'n_segments', 'Time Resolution').
    y_axis_label : str
        Y-axis label. Defaults to 'Model Time Resolution'.
    """
    # --- filter to the single parameter x level of interest ---
    df = effects_frame.copy()
    display_parameter = _rename_parameter_label(parameter, rename_dict)
    if hypothesis is not None:
        df = df[df[hypothesis_column] == hypothesis]
    df = df[df[param_column] == parameter]
    is_multiple_model_types = (df[model_type_column].nunique(dropna=True) > 1)

    # allow for integer comparison levels:
    comparison_level_strs: list[str] = []
    if isinstance(comparison_level, int):
        for comparison_levl_str in df[comparison_lvl_column].unique():
            # check for level string
            if f"Level {comparison_level} " in comparison_levl_str:
                comparison_level_strs.append(comparison_levl_str)

    # allow for multiple slightly different level names:
    df = df[df[comparison_lvl_column].isin(comparison_level_strs)]

    if df.empty:
        ax.text(0.5, 0.5, f'No data\n"{display_parameter}"\n@ "{comparison_level}"',
                ha='center', va='center', transform=ax.transAxes, fontsize=8, color='gray')
        ax.set_title('\n'.join(textwrap.wrap(display_parameter, width=40)),
                     fontsize=7, fontweight='bold', pad=10)
        return ax

    df = df.reset_index(drop=True)

    # --- sort by time resolution, then model type ---
    df = df.sort_values(by=[time_resolution_column, model_type_column], ascending=False).reset_index(drop=True)

    # --- y-axis labels: resolution | model_type ---
    if is_multiple_model_types:
        df['y_label'] = (df[time_resolution_column].round(2).astype(str) + ' | ' +
                         df[model_type_column].astype(str))
    else: df['y_label'] = df[time_resolution_column].round(2).astype(str)

    # --- confidence intervals ---
    df['ci_lower'] = df[coeff_column] - CI_z_score * df[se_column]
    df['ci_upper'] = df[coeff_column] + CI_z_score * df[se_column]

    # --- significance ---
    def get_significance(p):
        if p < 0.001:  return '***'
        elif p < 0.01: return '**'
        elif p < 0.05: return '*'
        return ''

    df['sig_label'] = df[p_column].apply(get_significance)
    df['is_significant'] = df[p_column] < 0.05

    def assign_color(row):
        if row['is_significant']:
            return significant_pos_color if row[coeff_column] > 0 else significant_neg_color
        return insignificant_color

    df['color'] = df.apply(assign_color, axis=1)

    # --- y-positions with spacing between resolution groups ---
    y_position = 0
    separator_positions = []
    current_resolution = None

    df['y_pos'] = 0.0  # initialise before loop assignment to set dtype
    for idx, row in df.iterrows():
        if current_resolution is not None and row[time_resolution_column] != current_resolution:
            separator_positions.append(y_position)
            y_position += 1
        df.at[idx, 'y_pos'] = y_position
        current_resolution = row[time_resolution_column]
        y_position += 1

    # flip so first row is at top
    max_y = df['y_pos'].max()
    df['y_pos'] = max_y - df['y_pos']
    separator_positions = [max_y - pos for pos in separator_positions]

    # --- whiskers ---
    for _, row in df.iterrows():
        ax.plot([row['ci_lower'], row['ci_upper']], [row['y_pos'], row['y_pos']],
                color=row['color'], linewidth=2, alpha=0.8, zorder=1)

    for model_type, group in df.groupby(model_type_column):
        group = group.sort_values('y_pos', ascending=False)  # top-to-bottom order
        rows = list(group.itertuples())
        for i in range(len(rows) - 1):
            curr, nxt = rows[i], rows[i + 1]
            # connect lower CI ends
            ax.plot([curr.ci_lower, nxt.ci_lower],
                    [curr.y_pos, nxt.y_pos],
                    color='lightgray', linestyle='-', linewidth=0.8,
                    alpha=0.6, zorder=0)
            # connect upper CI ends
            ax.plot([curr.ci_upper, nxt.ci_upper],
                    [curr.y_pos, nxt.y_pos],
                    color='lightgray', linestyle='-', linewidth=0.8,
                    alpha=0.6, zorder=0)

    # --- coefficient points ---
    ax.scatter(df[coeff_column], df['y_pos'],
               c=df['color'], s=100, zorder=2,
               edgecolors='white', linewidths=1.5, alpha=0.9)

    # --- zero line ---
    ax.axvline(x=0.0, color='black', linestyle='--', linewidth=1.5, alpha=0.6, zorder=0)

    # --- separator lines between resolution groups ---
    x_min, x_max = df['ci_lower'].min(), df['ci_upper'].max()
    x_range = x_max - x_min
    x_range = x_range if x_range > 0 else abs(x_max) * 0.1 or 0.01
    for sep_pos in separator_positions:
        ax.axhline(y=sep_pos, color='lightgray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)

    # --- significance labels ---
    text_x = x_max + 0.05 * x_range
    for _, row in df.iterrows():
        if row['sig_label']:
            ax.text(text_x, row['y_pos'], row['sig_label'],
                    va='center', ha='left', fontsize=10,
                    fontweight='bold', color=row['color'])

    # --- y-axis ---
    ax.set_yticks(df['y_pos'].values)
    if include_y_labels:
        ax.set_yticklabels(df['y_label'], fontsize=7)
    else:
        ax.set_yticklabels([])

    ax.set_ylabel(y_axis_label, fontsize=10, fontweight='bold')
    ax.set_ylim(-0.5, max_y + 0.5)

    # --- x-axis ---
    ax.set_xlabel(f'{display_parameter} β', fontsize=max(min(250/len(display_parameter), 11), 6), fontweight='bold')
    ax.tick_params(axis='x', labelsize=10)

    # --- grid & spines ---
    ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # --- title = hypothesis if provided, else parameter name ---
    title_text = hypothesis if hypothesis is not None else display_parameter
    ax.set_title('\n'.join(textwrap.wrap(title_text, width=40)),
                 fontsize=8, fontweight='bold', pad=10)

    # --- x limits ---
    x_margin = x_range * 0.15
    ax.set_xlim(x_min - x_margin, text_x + x_margin)

    # --- optional significance legend ---
    if show_significance_legend and include_y_labels:
        ax.text(0.02, 0.92, "* p<0.05  ** p<0.01  *** p<0.001",
                transform=ax.transAxes, fontsize=8, va='top',
                ha='left', style='italic', color='dimgray',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='lightgray', alpha=0.8))

    return ax


def plot_time_resolution_forest_mosaic(
        result_frame: pd.DataFrame,
        hypotheses: list[str],
        parameter: str,
        comparison_level: str | int,
        time_resolution_column: str = 'Time Resolution',
        exclude_intercepts: bool = True,
        model_type: str | None = None,
        y_axis_label: str = 'Time Res. [sec]',
        output_dir: Path = None,
        file_identifier_suffix: str | None = None,
        hidden: bool = False,
        plot_size: tuple[int, int] | Literal['auto'] = 'auto',
        show_legend: bool = False,
        significance_source: Literal["autocorr", "fdr", "auto"] = "auto",
        show_title: bool = False,
        rename_dict: dict[str, str] | None = None,
):
    """
    Mosaic of time-resolution forest plots — one column per hypothesis,
    all showing a single fixed parameter at a single comparison level,
    with time resolutions on the y-axis.

    Parameters
    ----------
    result_frame : pd.DataFrame
        Full results frame from store_model_results.
    hypotheses : list[str]
        Hypotheses to plot, one subplot column each.
    parameter : str
        The exact parameter name to plot (matched after formatting).
    comparison_level : str
        The exact comparison level name to filter to.
    time_resolution_column : str
        Column holding the time resolution label per row
        (e.g. 'n_segments' or 'Time Resolution').
    model_type : str or None
        If set, filters to that model type (e.g. 'LME'). If None, both
        OLS and LME are shown as separate y-rows per resolution.
    y_axis_label : str
        Y-axis label passed to draw_time_resolution_forest_plot.
    output_dir : Path, optional
        Directory to save the figure. If None, not saved.
    file_identifier_suffix : str, optional
        Appended to figure title and filename.
    hidden : bool
        If True, plt.show() is suppressed.
    plot_size : tuple or 'auto'
        Figure size in inches. 'auto' sizes by number of time resolution rows.
    """
    # --- shared preprocessing (mirrors plot_hypothesis_forest_mosaic) ---
    df = result_frame.copy()
    if exclude_intercepts:
        df = df[df['Parameter'] != 'Intercept']
    if model_type is not None:
        df = df[df['Model_Type'] == model_type]
    is_multiple_model_types = (df['Model_Type'].nunique() > 1)

    # clean up formula syntax from parameter names
    for pat in ['C(', 'Q(', "'", ')']:
        df['Parameter'] = df['Parameter'].str.replace(pat, '', regex=False)

    # --- auto plot size: height driven by number of time-resolution rows ---
    if plot_size == 'auto':
        n_rows = df[time_resolution_column].nunique() * (1 if model_type else 2)
        plot_size = (3 * len(hypotheses), max(2, n_rows * 0.6))

    fig, axs = plt.subplots(1, len(hypotheses), figsize=plot_size, constrained_layout=True)

    # ensure axs is always iterable even for a single hypothesis
    if len(hypotheses) == 1:
        axs = [axs]

    for col_ind, hypothesis in enumerate(hypotheses):

        hyp_subset = df[df["Hypothesis"] == hypothesis]
        p_col = _resolve_p_column(hyp_subset, significance_source)

        print(f"Plotting time-resolution forest plot ({col_ind}) for hypothesis: {hypothesis}")

        draw_time_resolution_forest_plot(
            ax=axs[col_ind],
            effects_frame=df,
            parameter=parameter,
            comparison_level=comparison_level,
            time_resolution_column=time_resolution_column,
            hypothesis=hypothesis,
            p_column=p_col,
            y_axis_label=y_axis_label if col_ind == 0 else '',
            include_y_labels=(col_ind == 0),  # only label y-axis on first column
            show_significance_legend=(col_ind == 0) and show_legend,
            rename_dict=rename_dict,
        )

    display_parameter = _rename_parameter_label(parameter, rename_dict)
    fig_title = (
        f"Time Resolution Comparison: {display_parameter}"
        f"{f' ({model_type})' if model_type else ''}"
        f"{f' — {file_identifier_suffix}' if file_identifier_suffix else ''}"
    )
    if show_title: fig.suptitle(fig_title, fontsize=9, fontweight='bold')

    if output_dir is not None:
        save_path = filemgmt.file_title(fig_title, '.svg')
        fig.savefig(output_dir / save_path, bbox_inches='tight')

    if not hidden:
        plt.show()


def plot_hypothesis_forest_mosaic(
    result_frame: pd.DataFrame,
    hypotheses: list[str],
    exclude_intercepts: bool = True,
    model_type: str | None = "LME",
    output_dir: Path = None,
    file_identifier_suffix: str | None = None,
    hidden: bool = False,
    plot_size: tuple[int, int] | Literal["auto"] = "auto",
    significance_source: Literal["autocorr", "fdr", "auto"] = "auto",
    show_title: bool = False,
    rename_dict: dict[str, str] | None = None,
):
    # slice results_frame:
    results_frame_subset = result_frame.copy()
    if exclude_intercepts:
        results_frame_subset = results_frame_subset[results_frame_subset['Parameter'] != 'Intercept']
    if model_type is not None:
        results_frame_subset = results_frame_subset[results_frame_subset['Model_Type'] == model_type]

    # exclude variance/residual components that are not hypothesis-relevant
    _EXCLUDE_PARAMS = {'__re_std__', '__residual_std__'}
    results_frame_subset = results_frame_subset[
        ~results_frame_subset['Parameter'].isin(_EXCLUDE_PARAMS)
    ]

    # formatting:
    results_frame_subset['Parameter'] = results_frame_subset['Parameter'].str.replace('C(', '')
    results_frame_subset['Parameter'] = results_frame_subset['Parameter'].str.replace('Q(', '')
    results_frame_subset['Parameter'] = results_frame_subset['Parameter'].str.replace("'", "")
    results_frame_subset['Parameter'] = results_frame_subset['Parameter'].str.replace(")", "")

    # prepare plot mosaic:
    if plot_size == 'auto':
        n_predictors = results_frame_subset['Parameter'].nunique(dropna=True)
        plot_size = (12, n_predictors / 3)

    # squeeze=False guarantees a numpy array even with a single hypothesis subplot;
    # flatten() collapses the (1, n) 2-D result to 1-D for uniform col_ind indexing
    fig, axs = plt.subplots(1, len(hypotheses), figsize=plot_size,
                            constrained_layout=True, squeeze=False)
    axs = axs.flatten()

    # plot hypothesis forest plots:
    selected_hypotheses = results_frame_subset[
        results_frame_subset['Hypothesis'].isin(hypotheses)
    ].copy()
    selected_hypotheses['__plot_p'] = np.nan
    for hypothesis in hypotheses:
        hyp_mask = selected_hypotheses['Hypothesis'] == hypothesis
        hyp_subset = selected_hypotheses.loc[hyp_mask]
        if hyp_subset.empty:
            continue
        p_col = _resolve_p_column(hyp_subset, significance_source)
        if p_col in selected_hypotheses.columns:
            selected_hypotheses.loc[hyp_mask, '__plot_p'] = selected_hypotheses.loc[hyp_mask, p_col]

    # Determine one color per parameter label across all subplots.
    parameter_label_colors: dict[str, str] = {}
    for parameter_name, parameter_rows in selected_hypotheses.groupby('Parameter'):
        sig_rows = parameter_rows[
            (parameter_rows['__plot_p'] < 0.05) & parameter_rows['Coefficient'].notna()
        ]
        signs = set(np.sign(sig_rows['Coefficient'].to_numpy(dtype=float)))
        signs.discard(0.0)
        if not signs or len(signs) > 1:
            parameter_label_colors[str(parameter_name)] = 'black'
        elif 1.0 in signs:
            parameter_label_colors[str(parameter_name)] = 'green'
        else:
            parameter_label_colors[str(parameter_name)] = 'red'

    for col_ind, hypothesis in enumerate(hypotheses):
        hyp_subset = results_frame_subset.loc[
            results_frame_subset["Hypothesis"] == hypothesis
            ]
        p_col = _resolve_p_column(hyp_subset, significance_source)

        axs[col_ind], _ = draw_forest_plot(
            axs[col_ind],
            effects_frame=hyp_subset,
            p_column=p_col,
            include_y_labels=(col_ind == 0),
            rename_dict=rename_dict,
            parameter_label_colors=parameter_label_colors if col_ind == 0 else None,
        )

    fig_title = f"Coefficient Overview{f' ({model_type} models)' if model_type is not None else ''}{f' ({file_identifier_suffix})' if file_identifier_suffix is not None else ''}"
    if show_title: fig.suptitle(fig_title)

    if output_dir is not None:
        save_path = filemgmt.file_title(fig_title, '.svg')
        fig.savefig(output_dir / save_path, bbox_inches='tight')

    if not hidden:
        plt.show()
    else:
        plt.close()




def plot_cmc_lineplots_per_category(
        all_subject_data_frame: pd.DataFrame,
        category_column: str,
        muscle: str,
        cmc_operator: str,
        n_within_trial_segments: int,
        cmc_plot_min: float = 0.7,
        cmc_plot_max: float = 1.0,
        n_yticks: int = 4,
        show_fig_title: bool = False,
        include_std_dev: bool = True,
        extended_y_label: bool = False,
        std_dev_factor: float = 0.2,
        colormap: Union[str, List[str], Tuple[str, ...]] = 'tab20',
        save_dir: Path = None,
        show_significance_threshold: bool = True,
        n_tapers: int = 5,
        alpha: float = 0.2,
        subject_ids_subset: list[int] | None = None,
        plot_size: tuple[float, float] = (12, 6),
        show_legend: bool = True,
        show_grid: bool = False,
) -> None:
    """
    Create line plots of CMC values across trial time for different categories.

    Args:
        all_subject_data_frame: DataFrame containing all subject data
        category_column: Column name to use for categorization
        muscle: Muscle name ('Flexor' or 'Extensor')
        cmc_operator: CMC operator ('mean' or 'max')
        n_within_trial_segments: Number of segments per trial
        cmc_plot_min: Minimum y-axis value
        cmc_plot_max: Maximum y-axis value
        n_yticks: Number of y-axis ticks
        include_std_dev: Whether to plot standard deviation bands
        std_dev_factor: Multiplier for standard deviation bands
        colormap: Matplotlib colormap name or explicit list/tuple of color strings.
            If a list/tuple is passed, entries are used in order and must be valid
            matplotlib colors (e.g., '#1f77b4', 'steelblue').
        save_dir: Directory to save plots
        show_significance_threshold: Whether to show CMC significance threshold line
        n_tapers: Number of tapers for CMC significance threshold
        alpha: Alpha level for CMC significance threshold
        subject_ids_subset: Optional list of subject IDs to include. If None,
            all available subjects are plotted.
        plot_size: Figure size as (width, height) in inches.
        show_legend: Whether to show a figure-level legend.
        show_grid: Whether to show subplot grids.
    """
    from matplotlib.lines import Line2D

    if category_column == 'Subject ID':
        print(f"Skipping category '{category_column}' (incompatible with subject-wise plots)")
        return

    print(f"Plotting CMC lineplot for category: {category_column}")

    # Get unique category labels
    unique_labels = all_subject_data_frame[category_column].dropna().unique().tolist()
    unique_labels.sort()

    # Resolve colors from colormap name or explicit color list/tuple.
    colors = _resolve_category_colors(colormap=colormap, n_colors=len(unique_labels))

    # Create legend handles for all unique labels
    legend_handles = [Line2D([0], [0], color=color, lw=2, label=label)
                      for color, label in zip(colors, unique_labels)]

    # Add CMC significance threshold to legend if requested
    if show_significance_threshold:
        cmc_threshold = features.compute_cmc_independence_threshold(n_tapers, alpha)
        threshold_handle = Line2D([0], [0], color='grey', lw=2, linestyle='--',
                                  label=f"CMC Sig. Threshold ({int(alpha * 100)}%)")
        legend_handles.append(threshold_handle)

    # Create subplots (rows: beta/gamma, columns: subjects)
    subject_ids = all_subject_data_frame['Subject ID'].unique().tolist()
    if subject_ids_subset is not None:
        selected_subject_set = set(subject_ids_subset)
        subject_ids = [subject_id for subject_id in subject_ids if subject_id in selected_subject_set]
    if len(subject_ids) == 0:
        warnings.warn("No subjects selected for CMC line plot.", RuntimeWarning)
        return

    fig, axs = plt.subplots(2, len(subject_ids), figsize=plot_size, squeeze=False)

    # Prepare x-axis values
    x_ticks = np.linspace(0, 1, max(n_within_trial_segments, 2))

    # Loop over frequency bands
    for row_ind, freq_band in enumerate(['beta', 'gamma']):

        # Loop over subjects
        for col_ind, subject_id in enumerate(subject_ids):
            ax = axs[row_ind, col_ind]

            # Get subject-specific data
            subject_frame = all_subject_data_frame[
                all_subject_data_frame['Subject ID'] == subject_id
                ]

            # Plot each category
            for color, category in zip(colors, unique_labels):
                category_frame = subject_frame[subject_frame[category_column] == category]

                if len(category_frame) == 0:
                    continue

                # Average across trials for each time point
                within_trial_counter = category_frame.groupby('Trial ID').cumcount()
                grouped_cmc = category_frame[f"CMC_{muscle}_{cmc_operator}_{freq_band}"].groupby(
                    within_trial_counter
                )
                cmc_series = grouped_cmc.mean().to_numpy()
                cmc_std = grouped_cmc.std().to_numpy()

                if len(cmc_series) == 0:
                    continue

                # Handle single-point case
                if len(cmc_series) == 1:
                    cmc_series = np.array([cmc_series[0], cmc_series[0]])
                    cmc_std = np.array([cmc_std[0], cmc_std[0]])

                # Plot line with optional std dev bands
                ax.plot(x_ticks, cmc_series, label=category, color=color)
                if include_std_dev:
                    ax.fill_between(
                        x_ticks,
                        cmc_series - std_dev_factor * cmc_std,
                        cmc_series + std_dev_factor * cmc_std,
                        alpha=0.2,
                        color=color
                    )

            # Plot CMC significance threshold if requested
            if show_significance_threshold:
                ax.axhline(y=cmc_threshold, color='grey', linestyle='--', linewidth=2)

            # Format subplot
            _format_cmc_subplot(
                ax=ax,
                row_ind=row_ind,
                col_ind=col_ind,
                subject_id=subject_id,
                muscle=muscle,
                freq_band=freq_band,
                y_ticks=np.linspace(cmc_plot_min, cmc_plot_max, n_yticks),
                cmc_plot_min=cmc_plot_min,
                cmc_plot_max=cmc_plot_max,
                include_std_dev=include_std_dev,
                std_dev_factor=std_dev_factor,
                x_ticks=x_ticks,
                extended_y_label=extended_y_label,
                show_grid=show_grid,
                grid_color='lightgrey',
                grid_alpha=0.8,
            )

    if show_legend and legend_handles:
        legend_ax = axs[-1, -1]
        legend_ax.legend(
            handles=legend_handles,
            ncol=len(legend_handles),
            loc='upper right',
            bbox_to_anchor=(1.0, -0.23),
            title=category_column,
            borderaxespad=0,
            frameon=True,
        )

    if show_fig_title:
        fig.suptitle(f"CMC per Subject and '{category_column}'")

    fig.subplots_adjust(top=.95, bottom=0.17, left=0.075, right=0.98, hspace=0.17, wspace=.1)

    if save_dir is not None:
        save_path = save_dir / filemgmt.file_title(
            f"CMC {muscle} per Subject per {category_column}", ".svg"
        )
        fig.savefig(save_path, bbox_inches='tight')  # bbox_inches='tight' now captures the legend correctly

    plt.show()


def plot_cmc_lineplot_normalised(
        all_subject_data_frame: pd.DataFrame,
        muscle: str,
        cmc_operator: str,
        n_within_trial_segments: int,
        cmc_plot_min: float = 80.0,
        cmc_plot_max: float = 120.0,
        n_yticks: int = 5,
        show_fig_title: bool = False,
        trial_color: str = 'tab:blue',
        trial_alpha: float = 0.4,
        line_width: float = 0.8,
        show_grid: bool = False,
        corridor_std_factor: float = 0.5,
        corridor_color: str = 'grey',
        corridor_alpha: float = 0.15,
        save_dir: Path = None,
        show_significance_threshold: bool = False,
        n_tapers: int = 5,
        alpha: float = 0.2,
        subject_ids_subset: list[int] | None = None,
        plot_size: tuple[float, float] = (12, 6),
        show_legend: bool = True,
) -> None:
    """Plot normalized CMC time series per trial for each subject and frequency band.

    Args:
        subject_ids_subset: Optional list of subject IDs to include. If None,
            all available subjects are plotted.
        plot_size: Figure size as (width, height) in inches.
        show_legend: Whether to show a figure-level legend.
    """

    print("Plotting normalised CMC lineplot per trial")
    from matplotlib.lines import Line2D

    # Backward-compatible guard: callers that still pass [0.7, 1.0] are
    # interpreted as fractional values and converted to percent range.
    if cmc_plot_max <= 2.0:
        warnings.warn(
            "plot_cmc_lineplot_normalised expected percentage limits; converting fractional y-limits to percent.",
            RuntimeWarning,
        )
        cmc_plot_min *= 100.0
        cmc_plot_max *= 100.0

    subject_ids = all_subject_data_frame['Subject ID'].unique().tolist()
    if subject_ids_subset is not None:
        selected_subject_set = set(subject_ids_subset)
        subject_ids = [subject_id for subject_id in subject_ids if subject_id in selected_subject_set]
    if len(subject_ids) == 0:
        warnings.warn("No subjects selected for normalised CMC line plot.", RuntimeWarning)
        return

    fig, axs = plt.subplots(2, len(subject_ids), figsize=plot_size, squeeze=False)

    x_ticks = np.linspace(0, 1, max(n_within_trial_segments, 2))

    if show_significance_threshold:
        cmc_threshold = features.compute_cmc_independence_threshold(n_tapers, alpha)

    n_plotted_lines = 0

    for row_ind, freq_band in enumerate(['beta', 'gamma']):
        cmc_col = f"CMC_{muscle}_{cmc_operator}_{freq_band}"

        for col_ind, subject_id in enumerate(subject_ids):
            ax = axs[row_ind, col_ind]
            aligned_trials_for_corridor: list[np.ndarray] = []

            subject_frame = all_subject_data_frame[
                all_subject_data_frame['Subject ID'] == subject_id
            ]
            if len(subject_frame) == 0:
                continue

            subject_frame = subject_frame.assign(
                _within_trial_idx=subject_frame.groupby('Trial ID').cumcount()
            )

            for _, trial_frame in subject_frame.groupby('Trial ID', sort=False):
                trial_series = trial_frame.sort_values('_within_trial_idx')[cmc_col].to_numpy(dtype=float)
                if len(trial_series) == 0:
                    continue

                finite_mask = np.isfinite(trial_series)
                if not np.any(finite_mask):
                    continue

                first_valid_idx = int(np.flatnonzero(finite_mask)[0])
                trial_series = trial_series[first_valid_idx:]
                trial_start = trial_series[0]
                if (not np.isfinite(trial_start)) or np.isclose(trial_start, 0.0):
                    continue

                normalised_series = trial_series / trial_start * 100.0
                normalised_series[0] = 100.0

                if len(normalised_series) == 1:
                    x_values = np.array([0.0, 1.0])
                    normalised_series = np.array([normalised_series[0], normalised_series[0]])
                elif len(normalised_series) == len(x_ticks):
                    x_values = x_ticks
                else:
                    x_values = np.linspace(0, 1, len(normalised_series))

                ax.plot(
                    x_values,
                    normalised_series,
                    color=trial_color,
                    linewidth=line_width,
                    alpha=trial_alpha,
                    marker='o',
                    markevery=[0],
                    markersize=max(2.5, line_width * 4.0),
                    markeredgewidth=0,
                )
                n_plotted_lines += 1

                finite_plot_mask = np.isfinite(x_values) & np.isfinite(normalised_series)
                if np.sum(finite_plot_mask) >= 2:
                    aligned_trials_for_corridor.append(
                        np.interp(
                            x_ticks,
                            x_values[finite_plot_mask],
                            normalised_series[finite_plot_mask],
                            left=np.nan,
                            right=np.nan,
                        )
                    )
                elif np.sum(finite_plot_mask) == 1:
                    aligned_trials_for_corridor.append(
                        np.full(len(x_ticks), normalised_series[finite_plot_mask][0], dtype=float)
                    )

            if len(aligned_trials_for_corridor) > 0:
                trial_matrix = np.vstack(aligned_trials_for_corridor)
                mean_series = np.nanmean(trial_matrix, axis=0)
                std_series = np.nanstd(trial_matrix, axis=0)
                corridor_half_width = corridor_std_factor * std_series
                valid_corridor_mask = np.isfinite(mean_series) & np.isfinite(corridor_half_width)
                if np.any(valid_corridor_mask):
                    ax.fill_between(
                        x_ticks[valid_corridor_mask],
                        mean_series[valid_corridor_mask] - corridor_half_width[valid_corridor_mask],
                        mean_series[valid_corridor_mask] + corridor_half_width[valid_corridor_mask],
                        color=corridor_color,
                        alpha=corridor_alpha,
                        linewidth=0,
                        zorder=0,
                    )

            if show_significance_threshold:
                ax.axhline(y=cmc_threshold, color='grey', linestyle='--', linewidth=2)

            _format_cmc_subplot(
                ax=ax,
                row_ind=row_ind,
                col_ind=col_ind,
                subject_id=subject_id,
                muscle=muscle,
                freq_band=freq_band,
                y_ticks=np.linspace(cmc_plot_min, cmc_plot_max, n_yticks),
                cmc_plot_min=cmc_plot_min,
                cmc_plot_max=cmc_plot_max,
                include_std_dev=False,
                std_dev_factor=0.0,
                x_ticks=x_ticks,
                y_label_override=f'{muscle} {freq_band.capitalize()} Normalized CMC [%]',
                y_label_all_columns=False,
                show_grid=show_grid,
                grid_color='lightgrey',
                grid_alpha=0.8,
            )

    if n_plotted_lines == 0:
        warnings.warn(
            "No trial lines were plotted. Check CMC column names and whether trial starts are finite and non-zero.",
            RuntimeWarning,
        )

    legend_handles = [
        Line2D(
            [0], [0],
            color=trial_color,
            linewidth=line_width,
            alpha=trial_alpha,
            marker='o',
            markersize=max(2.5, line_width * 4.0),
            markeredgewidth=0,
            label='Single Trial Trajectory',
        ),
        patches.Patch(
            facecolor=corridor_color,
            edgecolor='none',
            alpha=corridor_alpha,
            label=f'Mean Corridor (±{corridor_std_factor:.2g}x SD)',
        ),
    ]
    if show_significance_threshold:
        legend_handles.append(
            Line2D(
                [0], [0],
                color='grey',
                linestyle='--',
                linewidth=2,
                label=f"CMC Sig. Threshold ({int(alpha * 100)}%)",
            )
        )

    if show_legend:
        legend_ax = axs[-1, -1]
        legend_ax.legend(
            handles=legend_handles,
            loc='upper right',
            bbox_to_anchor=(1.0, -0.23),
            borderaxespad=0,
            frameon=True,
        )

    if show_fig_title:
        fig.suptitle(f"Normalised CMC per Subject ({muscle}, {cmc_operator})")

    fig.subplots_adjust(top=.95, bottom=0.17, left=0.075, right=0.98, hspace=0.17, wspace=.1)

    if save_dir is not None:
        save_path = save_dir / filemgmt.file_title(
            f"Normalised CMC {muscle} per Subject", ".svg"
        )
        fig.savefig(save_path, bbox_inches='tight')

    plt.show()


def _resolve_category_colors(
        colormap: Union[str, List[str], Tuple[str, ...]],
        n_colors: int,
) -> List[Tuple[float, float, float, float]]:
    """Resolve a discrete list of category colors from a colormap name or color list."""
    if n_colors <= 0:
        return []

    if isinstance(colormap, str):
        cmap = plt.colormaps[colormap]
        return [cmap(i / max(n_colors - 1, 1)) for i in range(n_colors)]

    if isinstance(colormap, (list, tuple)):
        if len(colormap) < n_colors:
            raise ValueError(
                f"Explicit color list is too short: got {len(colormap)} colors for {n_colors} categories."
            )

        invalid_colors = [
            color for color in colormap
            if (not isinstance(color, str)) or (not mcolors.is_color_like(color))
        ]
        if invalid_colors:
            raise ValueError(
                f"Invalid color entries in explicit color list: {invalid_colors}."
            )

        return [mcolors.to_rgba(color) for color in colormap[:n_colors]]

    raise TypeError(
        "colormap must be a matplotlib colormap name (str) or a list/tuple of color strings."
    )


def _format_cmc_subplot(
        ax, row_ind: int, col_ind: int, subject_id: int,
        muscle: str, freq_band: str, y_ticks: np.ndarray,
        cmc_plot_min: float, cmc_plot_max: float,
        include_std_dev: bool, std_dev_factor: float,
        x_ticks: np.ndarray, extended_y_label: bool = False,
        y_label_override: Optional[str] = None,
        y_label_all_columns: bool = False,
        show_grid: bool = True,
        grid_color: Optional[str] = None,
        grid_alpha: Optional[float] = None,
) -> None:
    """Format individual subplot for CMC line plot."""

    # Title for top row
    if row_ind == 0:
        ax.set_title(f"Subject {subject_id:02}")

    # Y-axis formatting
    if col_ind == 0 or y_label_all_columns:
        if y_label_override is not None:
            ax.set_ylabel(y_label_override)
        else:
            ylabel = f"{muscle} {freq_band.capitalize()} CMC (Mean"
            if include_std_dev and extended_y_label:
                ylabel += f" ± {std_dev_factor:.1f}x Std.Dev."
            ylabel += ")"
            ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel('')

    ax.set_yticks(y_ticks)
    if col_ind != 0:
        ax.set_yticklabels([''] * len(y_ticks))
    ax.set_ylim([cmc_plot_min, cmc_plot_max])

    # x_ticks:
    ax.set_xticks(x_ticks)
    if row_ind == 1:
        if len(x_ticks) > 2: x_tick_labels = ['Start'] + [''] * (len(x_ticks) - 2) + ['End']
        else: x_tick_labels = ['Start', 'End']
        ax.set_xlabel('Trial Duration')
    else:
        x_tick_labels = [''] * len(x_ticks)
    ax.set_xticklabels(x_tick_labels)#, rotation=90)
    ax.get_xticklabels()[0].set_ha('left')  # 'Start' anchors left
    ax.get_xticklabels()[1].set_ha('right')  # 'Start' anchors left

    # Grid
    if show_grid:
        grid_kwargs = {}
        if grid_color is not None:
            grid_kwargs['color'] = grid_color
        if grid_alpha is not None:
            grid_kwargs['alpha'] = grid_alpha
        ax.grid(**grid_kwargs)

    """# Legend (only lower right subplot)
    if row_ind == 1 and col_ind == (n_subjects - 1):
        ax.legend(
            handles=legend_handles,
            bbox_to_anchor=(1.0, 0.0),
            loc='lower left',
            title=category_column
        )"""


def _phase_normalize_accuracy_cycles(
    accuracy: np.ndarray,
    phase_grid: np.ndarray,
    task_freq: float,
    trial_dur_sec: float,
    min_samples_per_cycle: int,
    start_offset_sec: float,
    end_cutoff_sec: float = 0.0,
    expected_sampling_rate: float | None = None,
) -> list[np.ndarray]:
    """
    Phase-normalize accuracy samples into per-cycle profiles.

    Accuracy is recorded only after the warm-up offset, so the synthetic time
    axis starts at ``start_offset_sec``. The sampling interval is inferred from
    the effective duration and sample count, which reflects the true elapsed
    time regardless of Python loop overhead.

    Parameters
    ----------
    end_cutoff_sec : float
        Seconds to discard from the *end* of the reconstructed time axis before
        phase normalization.  Other modalities are protected from post-task
        transients by ``get_task_start_end()``'s ``cut_off_sec_to_prevent_transients``;
        because accuracy timestamps are reconstructed (not sliced), this parameter
        provides the equivalent tail exclusion.  Should match the cutoff used
        when computing the other modalities' trial spans (default 0.0; set via
        the caller's ``accuracy_trial_end_cutoff_sec``).
    """
    if accuracy is None or len(accuracy) == 0 or task_freq <= 0:
        return []

    effective_dur = trial_dur_sec - start_offset_sec
    if effective_dur <= 0:
        return []

    # amount of timesteps:
    n = len(accuracy)

    # Infer actual sampling rate from known duration and sample count.
    inferred_rate = n / effective_dur

    if expected_sampling_rate is not None and expected_sampling_rate > 0:
        deviation = abs(inferred_rate - expected_sampling_rate) / expected_sampling_rate
        if deviation > 0.15:
            print(
                f"[WARNING] Accuracy sampling rate mismatch: "
                f"inferred {inferred_rate:.2f} Hz vs expected {expected_sampling_rate:.2f} Hz "
                f"({deviation * 100:.1f}% deviation). Using inferred rate."
            )

    # reconstruct relative time axis:
    t_rel = start_offset_sec + np.arange(n) / inferred_rate

    # DEBUG STATEMENT:
    # print(f"Assumed duration of accuracy sampling: {effective_dur:.2f} sec.\nResulting sampling rate: {inferred_rate:.2f} Hz.")

    # Discard post-task transients from the tail, mirroring the end-cutoff that
    # get_task_start_end() applies to CMC/PSD/force via cut_off_sec_to_prevent_transients.
    # The reconstructed time axis may extend into this tail (by design, to correct
    # cycle counting); we trim it here before phase normalization.
    effective_end = trial_dur_sec - end_cutoff_sec
    if end_cutoff_sec > 0.0 and effective_end > start_offset_sec:
        keep = t_rel < effective_end
        accuracy = accuracy[keep]
        t_rel = t_rel[keep]
        if len(accuracy) == 0:
            return []
    else:
        effective_end = trial_dur_sec

    return data_analysis.phase_normalize_cycles(
        signal=accuracy,
        t_rel=t_rel,
        task_freq=task_freq,
        trial_dur_sec=effective_end,
        phase_grid=phase_grid,
        min_samples_per_cycle=min_samples_per_cycle,
        start_offset_sec=start_offset_sec,
        use_interpolation=True,
        min_cycle_coverage_ratio=.9,
    )


def _derive_cmc_accuracy_hypothesis_label(cfg: CBPAConfig) -> str:
    """Build a deterministic label from modality settings for this plot type."""
    return f"{cfg.modality}_{cfg.modality_file_id}_{cfg.freq_band}_phase_avg_vs_accuracy"


def _create_cbpa_dual_panel_figure(
    show_target_sine: bool,
    figure_size_with_target: tuple[float, float] = (16, 6),
    figure_size_without_target: tuple[float, float] = (16, 5),
    grid_width_ratios: tuple[float, float, float, float] = (1.0, 0.05, 0.14, 1.0),
    grid_height_ratios_with_target: tuple[float, float] = (5.0, 1.0),
    grid_wspace: float = 0.12,
    grid_hspace_with_target: float = 0.28,
) -> tuple[Figure, Axes, Axes, Axes, Axes | None, Axes | None]:
    """Create shared two-panel layout with optional target-sine row."""
    if show_target_sine:
        fig = plt.figure(figsize=figure_size_with_target)
        gs = fig.add_gridspec(
            2,
            4,
            width_ratios=grid_width_ratios,
            height_ratios=grid_height_ratios_with_target,
            wspace=grid_wspace,
            hspace=grid_hspace_with_target,
        )
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 3])
        ax_tgt_left = fig.add_subplot(gs[1, 0], sharex=ax)
        ax_tgt_right = fig.add_subplot(gs[1, 3], sharex=ax2)
        fig.add_subplot(gs[1, 1]).axis("off")
        fig.add_subplot(gs[0, 2]).axis("off")
        fig.add_subplot(gs[1, 2]).axis("off")
        return fig, ax, cax, ax2, ax_tgt_left, ax_tgt_right

    fig = plt.figure(figsize=figure_size_without_target)
    gs = fig.add_gridspec(1, 4, width_ratios=grid_width_ratios, wspace=grid_wspace)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 3])
    fig.add_subplot(gs[0, 2]).axis("off")
    return fig, ax, cax, ax2, None, None


def _apply_phase_axis_style(
    axes: list[Axes],
    phase_xticks: tuple[float, ...],
    phase_marker_lines: tuple[float, ...],
) -> None:
    """Apply shared phase x-ticks and vertical marker lines to all axes."""
    for axis in axes:
        axis.set_xticks(list(phase_xticks))
        for marker_x in phase_marker_lines:
            axis.axvline(marker_x, color="grey", lw=0.5, ls=":")


def _resolve_cluster_mask(cluster, n_times: int, n_ch: int) -> np.ndarray:
    """Resolve any MNE cluster format to a (n_times, n_ch) boolean mask."""
    n_flat = n_times * n_ch
    flat_mask = np.zeros(n_flat, dtype=bool)

    if isinstance(cluster, tuple) and len(cluster) == 1:
        cluster = cluster[0]

    if isinstance(cluster, np.ndarray) and cluster.dtype == bool:
        return cluster.reshape(n_times, n_ch)

    if isinstance(cluster, slice):
        flat_mask[cluster] = True
        return flat_mask.reshape(n_times, n_ch)

    if isinstance(cluster, tuple) and len(cluster) == 2 and isinstance(cluster[0], np.ndarray):
        mask = np.zeros((n_times, n_ch), dtype=bool)
        mask[cluster[0], cluster[1]] = True
        return mask

    if isinstance(cluster, np.ndarray):
        idx = cluster.ravel().astype(int)
        idx = idx[(idx >= 0) & (idx < n_flat)]
        flat_mask[idx] = True
        return flat_mask.reshape(n_times, n_ch)

    try:
        idx = np.asarray(cluster).ravel().astype(int)
        idx = idx[(idx >= 0) & (idx < n_flat)]
        flat_mask[idx] = True
    except Exception as e:
        warnings.warn(f"[CBPA] _resolve_cluster_mask: unrecognised cluster format {type(cluster)}. Error: {e}")

    return flat_mask.reshape(n_times, n_ch)

def plot_cmc_accuracy_phase_average(
        cfg: CBPAConfig,
        experiment_results_dir: Path,
        *,
        accuracy_sd_factor: float = 0.25,
        cmc_percentile_limits: tuple[float, float] = (3.0, 97.0),
        figure_size_with_target: tuple[float, float] = (16, 6),
        figure_size_without_target: tuple[float, float] = (16, 5),
        grid_width_ratios: tuple[float, float, float, float] = (1.0, 0.05, 0.14, 1.0),
        grid_height_ratios_with_target: tuple[float, float] = (5.0, 1.0),
        grid_wspace: float = 0.12,
        grid_hspace_with_target: float = 0.28,
        phase_xticks: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0, 360.0),
        phase_marker_lines: tuple[float, ...] = (90.0, 270.0),
        channel_label_fontsize: float = 6.0,
        legend_fontsize: float = 8.0,
        subplot_margins: tuple[float, float, float, float] = (0.06, 0.985, 0.90, 0.10),
        save_dpi: int = 150,
        freq_pooling: Literal["max", "mean"] = "max",
        channel_pooling: Literal["max", "mean"] = "max",
        use_unscaled_force: bool = True,
        accuracy_trial_dur_offset_sec: float = 6.0,
        accuracy_trial_end_cutoff_sec: float = 2.0,
        plot_accuracy_per_cycle_id: bool = False,
        accuracy_cycles_to_plot: int = 4,
        accuracy_cycle_colors: tuple[str, ...] = ("tab:orange", "tab:red", "purple", "black"),
        min_accuracy_cycle_count: int = 20,
) -> None:
    """Create a CBPA-like plot with mean CMC map and phase-normalized accuracy."""
    import src.pipeline.cbpa as cbpa

    if cfg.modality != "CMC":
        raise ValueError("plot_cmc_accuracy_phase_average requires cfg.modality='CMC'.")
    if not cfg.use_phase_normalization:
        raise ValueError("plot_cmc_accuracy_phase_average requires phase normalization enabled.")
    if accuracy_sd_factor < 0:
        raise ValueError("accuracy_sd_factor must be >= 0.")
    p_low, p_high = cmc_percentile_limits
    if not (0.0 <= p_low < p_high <= 100.0):
        raise ValueError("cmc_percentile_limits must satisfy 0 <= low < high <= 100.")
    if accuracy_cycles_to_plot <= 0:
        raise ValueError("accuracy_cycles_to_plot must be > 0.")
    if min_accuracy_cycle_count < 1:
        raise ValueError("min_accuracy_cycle_count must be >= 1.")
    if len(accuracy_cycle_colors) == 0:
        raise ValueError("accuracy_cycle_colors must contain at least one color.")

    filemgmt.assert_dir(cfg.output_dir)
    plot_label = _derive_cmc_accuracy_hypothesis_label(cfg)

    stats_df = cbpa.load_stats_frame(cfg.data_root)
    subject_ids = sorted(stats_df["Subject ID"].astype(int).unique())
    exclude = set(cfg.exclude_subjects or [])
    subject_ids = [sid for sid in subject_ids if sid not in exclude]

    phase_grid = np.linspace(0, 360, cfg.n_phase_bins, endpoint=False)
    ch_names = cfg.channels if cfg.channels is not None else cbpa.CMC_EEG_CHANNEL_SUBSET

    subject_cmc_profiles: list[np.ndarray] = []
    subject_acc_profiles: list[np.ndarray] = []  # used only for legacy (single-line) behaviour
    valid_subjects: list[int] = []

    # Option A for cycle-wise view: pooled across all subjects + all trials by cycle index
    pooled_acc_cycles_by_idx: dict[int, list[np.ndarray]] = {
        cyc_idx: [] for cyc_idx in range(accuracy_cycles_to_plot)
    }

    for subj in tqdm(subject_ids, desc="Importing Subject Data"):
        try:
            spectrogram, freqs, timestamps, log_df = cbpa._load_subject_data(cfg, subj)
        except Exception as exc:
            warnings.warn(f"Subject {subj:02}: failed to load data ({exc}). Skipping.")
            continue

        trial_spans = {int(k): v for k, v in cbpa._get_trial_spans(log_df).items()}
        if len(trial_spans) == 0:
            warnings.warn(f"Subject {subj:02}: no trial spans found. Skipping.")
            continue

        band_power = cbpa._extract_band_power(
            cfg,
            spectrogram,
            freqs,
            channel_indices=None,
            freq_pooling=freq_pooling,
            channel_pooling=channel_pooling,
        )
        trial_cond_map = {trial_id: "ALL" for trial_id in trial_spans}
        cmc_cycles = cbpa._band_power_per_phase(
            cfg=cfg,
            band_power=band_power,
            timestamps=timestamps,
            trial_spans=trial_spans,
            trial_cond_map=trial_cond_map,
            log_df=log_df,
        ).get("ALL", [])

        if len(cmc_cycles) == 0:
            warnings.warn(f"Subject {subj:02}: no valid CMC cycles. Skipping.")
            continue

        # Accuracy handling
        accuracy_cycles_subject: list[np.ndarray] = []
        subj_has_any_accuracy = False
        subj_exp_dir = experiment_results_dir / f"subject_{subj:02}"

        for trial_id, (t_start, t_end) in trial_spans.items():
            task_freq = cbpa._get_task_freq_for_trial(log_df, t_start, t_end)
            if task_freq is None or task_freq <= 0:
                continue

            accuracy = data_integration.fetch_trial_accuracy(
                experiment_data_dir=subj_exp_dir,
                trial_id=int(trial_id),
                log_df=log_df,
                error_handling="continue",
            )
            if accuracy is None:
                continue

            trial_dur_sec = (t_end - t_start).total_seconds()
            trial_cycles = _phase_normalize_accuracy_cycles(
                accuracy=accuracy,
                phase_grid=phase_grid,
                task_freq=float(task_freq),
                trial_dur_sec=float(trial_dur_sec) + accuracy_trial_dur_offset_sec,
                min_samples_per_cycle=cfg.min_samples_per_cycle,
                start_offset_sec=float(data_integration.TRIAL_ACCURACY_START_OFFSET_SEC),
                end_cutoff_sec=accuracy_trial_end_cutoff_sec,
            )
            if len(trial_cycles) == 0:
                continue

            subj_has_any_accuracy = True

            if plot_accuracy_per_cycle_id:
                for cyc_idx, cyc in enumerate(trial_cycles):
                    if cyc_idx >= accuracy_cycles_to_plot:
                        break
                    pooled_acc_cycles_by_idx[cyc_idx].append(cyc)
            else:
                accuracy_cycles_subject.extend(trial_cycles)

        if not subj_has_any_accuracy:
            warnings.warn(f"Subject {subj:02}: no valid accuracy cycles. Skipping.")
            continue

        subject_cmc_profiles.append(np.nanmean(np.stack(cmc_cycles, axis=0), axis=0))
        if not plot_accuracy_per_cycle_id:
            subject_acc_profiles.append(np.nanmean(np.stack(accuracy_cycles_subject, axis=0), axis=0))
        valid_subjects.append(subj)

    # Validate data availability
    if len(subject_cmc_profiles) == 0:
        warnings.warn("No valid subjects for CMC+accuracy phase plot. Nothing will be plotted.")
        return

    if not plot_accuracy_per_cycle_id and len(subject_acc_profiles) == 0:
        warnings.warn("No valid accuracy profiles for legacy single-line plot. Nothing will be plotted.")
        return

    if plot_accuracy_per_cycle_id:
        has_any_cycle = any(
            len(pooled_acc_cycles_by_idx[idx]) >= min_accuracy_cycle_count
            for idx in range(accuracy_cycles_to_plot)
        )
        if not has_any_cycle:
            warnings.warn(
                "No cycle index has enough samples for plotting "
                f"(min_accuracy_cycle_count={min_accuracy_cycle_count})."
            )
            return

    cmc_stack = np.stack(subject_cmc_profiles, axis=0)  # (n_subj, n_phase, n_ch)
    cmc_mean = np.nanmean(cmc_stack, axis=0)

    # Legacy single-line accuracy aggregates
    if not plot_accuracy_per_cycle_id:
        acc_stack = np.stack(subject_acc_profiles, axis=0)  # (n_subj, n_phase)
        acc_mean = np.nanmean(acc_stack, axis=0)
        acc_mean_smooth = data_analysis.circular_smooth(acc_mean, kernel_bins=5)
        acc_std = np.nanstd(acc_stack, axis=0)
        acc_std_smooth = data_analysis.circular_smooth(acc_std, kernel_bins=5)

    fig, ax, cax, ax2, ax_tgt_left, ax_tgt_right = _create_cbpa_dual_panel_figure(
        show_target_sine=cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization,
        figure_size_with_target=figure_size_with_target,
        figure_size_without_target=figure_size_without_target,
        grid_width_ratios=grid_width_ratios,
        grid_height_ratios_with_target=grid_height_ratios_with_target,
        grid_wspace=grid_wspace,
        grid_hspace_with_target=grid_hspace_with_target,
    )

    if cfg.include_suptitle:
        mode_txt = "cycle-wise pooled accuracy" if plot_accuracy_per_cycle_id else "mean accuracy"
        fig.suptitle(
            f"{plot_label}\n"
            f"Average {cfg.modality} ({cfg.modality_file_id}, {cfg.freq_band}) + Task Error (RMSE, {mode_txt})  |  "
            f"n = {len(valid_subjects)} subjects",
            fontsize=10,
        )

    cmc_vmin = float(np.nanpercentile(cmc_mean, p_low))
    cmc_vmax = float(np.nanpercentile(cmc_mean, p_high))
    if not np.isfinite(cmc_vmin) or not np.isfinite(cmc_vmax) or cmc_vmin == cmc_vmax:
        cmc_vmin, cmc_vmax = None, None

    im = ax.imshow(
        cmc_mean.T,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=cmc_vmin,
        vmax=cmc_vmax,
        extent=(phase_grid[0], 360.0, -0.5, len(ch_names) - 0.5),
    )
    plt.colorbar(im, cax=cax, label=f"{cfg.freq_band.lower()}-band CMC Value")
    if not (cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization):
        ax.set_xlabel("Force Cycle Phase (°)")
    ax.set_ylabel("Channel index")
    ax.set_yticks(range(len(ch_names)))
    ax.set_yticklabels(ch_names, fontsize=channel_label_fontsize)
    ax.set_title("Averaged phase-normalized CMC map")
    ax.set_xlim(0, 360)

    # Accuracy panel
    if plot_accuracy_per_cycle_id:
        plotted_cycle_count = 0

        for cyc_idx in range(accuracy_cycles_to_plot):
            cycle_samples = pooled_acc_cycles_by_idx.get(cyc_idx, [])
            n_cycles = len(cycle_samples)

            if n_cycles < min_accuracy_cycle_count:
                continue

            cycle_stack = np.stack(cycle_samples, axis=0)  # (n_cycles, n_phase)
            cyc_mean = np.nanmean(cycle_stack, axis=0)
            cyc_std = np.nanstd(cycle_stack, axis=0)

            cyc_mean_smooth = data_analysis.circular_smooth(cyc_mean, kernel_bins=5)
            cyc_std_smooth = data_analysis.circular_smooth(cyc_std, kernel_bins=5)
            cyc_band = accuracy_sd_factor * cyc_std_smooth

            # Circular wraparound to close 0°/360° gap
            phase_grid_wrapped = np.concatenate([phase_grid, [360.0]])
            cyc_mean_wrapped = np.concatenate([cyc_mean_smooth, [cyc_mean_smooth[0]]])
            cyc_band_wrapped = np.concatenate([cyc_band, [cyc_band[0]]])

            color = accuracy_cycle_colors[cyc_idx % len(accuracy_cycle_colors)]
            label_mean = f"Cycle {cyc_idx + 1} mean (n={n_cycles})"
            label_band = f"Cycle {cyc_idx + 1} ±{accuracy_sd_factor:g}x SD"

            ax2.plot(
                phase_grid_wrapped,
                cyc_mean_wrapped,
                color=color,
                linewidth=1.8,
                label=label_mean,
            )
            ax2.fill_between(
                phase_grid_wrapped,
                cyc_mean_wrapped - cyc_band_wrapped,
                cyc_mean_wrapped + cyc_band_wrapped,
                color=color,
                alpha=0.18,
                label=label_band,
            )
            plotted_cycle_count += 1

        if plotted_cycle_count == 0:
            ax2.text(
                0.5,
                0.5,
                f"No cycle index passed min count ({min_accuracy_cycle_count}).",
                transform=ax2.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="grey",
            )
        else:
            ax2.legend(fontsize=legend_fontsize, ncol=accuracy_cycles_to_plot // 2 if plot_accuracy_per_cycle_id else 1)

        ax2.set_title("Averaged phase-normalized accuracy (cycle-wise pooled)")
    else:
        # Legacy behaviour: single mean ± SD line over subjects
        acc_band = accuracy_sd_factor * acc_std_smooth

        phase_grid_wrapped = np.concatenate([phase_grid, [360.0]])
        acc_mean_wrapped = np.concatenate([acc_mean_smooth, [acc_mean_smooth[0]]])
        acc_band_wrapped = np.concatenate([acc_band, [acc_band[0]]])

        ax2.plot(phase_grid_wrapped, acc_mean_wrapped, color="tab:blue", linewidth=1.8, label="Mean RMSE")
        ax2.fill_between(
            phase_grid_wrapped,
            acc_mean_wrapped - acc_band_wrapped,
            acc_mean_wrapped + acc_band_wrapped,
            color="tab:blue",
            alpha=0.2,
            label=f"±{accuracy_sd_factor:g} x SD",
        )
        ax2.legend(fontsize=legend_fontsize)
        ax2.set_title("Averaged phase-normalized accuracy")

    if not (cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization):
        ax2.set_xlabel("Force Cycle Phase (°)")
    ax2.set_ylabel("Task Error (RMSE)")
    ax2.set_xlim(0, 360)

    if (cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization) and ax_tgt_left is not None and ax_tgt_right is not None:
        dyno_force = None
        if cfg.include_dynamometer_force:
            dyno_force = _load_avg_dynamometer_force_per_phase(
                valid_subjects,
                experiment_results_dir,
                phase_grid,
                cfg,
                use_unscaled_force=use_unscaled_force,
            )
        _plot_target_sine_panel(
            ax_tgt_left,
            phase_grid,
            cfg,
            x_label="Force Cycle Phase (°)",
            show_ylabel=True,
            dynamometer_force_y=dyno_force,
            is_unscaled_force=use_unscaled_force,
        )
        _plot_target_sine_panel(
            ax_tgt_right,
            phase_grid,
            cfg,
            x_label="Force Cycle Phase (°)",
            show_ylabel=True,
            dynamometer_force_y=dyno_force,
            is_unscaled_force=use_unscaled_force,
            show_legend=False,
        )

    phase_axes = [ax, ax2]
    if (cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization) and ax_tgt_left is not None and ax_tgt_right is not None:
        phase_axes.extend([ax_tgt_left, ax_tgt_right])
    _apply_phase_axis_style(
        phase_axes,
        phase_xticks=phase_xticks,
        phase_marker_lines=phase_marker_lines,
    )

    margin_left, margin_right, margin_top, margin_bottom = subplot_margins
    fig.subplots_adjust(
        left=margin_left,
        right=margin_right,
        top=margin_top,
        bottom=margin_bottom,
    )

    if cfg.save_plots:
        out = cfg.output_dir / filemgmt.file_title(plot_label + "_cmc_accuracy_phase", ".png")
        fig.savefig(out, dpi=save_dpi, bbox_inches="tight")
        print(f"  Plot saved: {out}")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)


def plot_emg_psd_phase_average_plot(
        cfg: CBPAConfig,
        *,
        flexor_file_identifier: str = "emg_1_flexor",
        extensor_file_identifier: str = "emg_2_extensor",
        emg_percentile_limits: tuple[float, float] = (3.0, 97.0),
        figure_size_with_target: tuple[float, float] = (16, 6),
        figure_size_without_target: tuple[float, float] = (16, 5),
        grid_width_ratios: tuple[float, float, float, float] = (1.0, 0.05, 0.14, 1.0),
        grid_height_ratios_with_target: tuple[float, float] = (5.0, 1.0),
        grid_wspace: float = 0.12,
        grid_hspace_with_target: float = 0.28,
        phase_xticks: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0, 360.0),
        phase_marker_lines: tuple[float, ...] = (90.0, 270.0),
        channel_label_fontsize: float = 6.0,
        show_channel_labels: bool = True,
        subplot_margins: tuple[float, float, float, float] = (0.06, 0.985, 0.90, 0.10),
        save_dpi: int = 150,
        use_unscaled_force: bool = True,
) -> None:
    """Create a phase-normalized average EMG-PSD plot (left=flexor, right=extensor)."""
    import src.pipeline.cbpa as cbpa
    from dataclasses import replace

    if not cfg.use_phase_normalization:
        raise ValueError("plot_emg_psd_phase_average_plot requires phase normalization enabled.")

    p_low, p_high = emg_percentile_limits
    if not (0.0 <= p_low < p_high <= 100.0):
        raise ValueError("emg_percentile_limits must satisfy 0 <= low < high <= 100.")

    filemgmt.assert_dir(cfg.output_dir)

    stats_df = cbpa.load_stats_frame(cfg.data_root)
    subject_ids = sorted(stats_df["Subject ID"].astype(int).unique())
    exclude = set(cfg.exclude_subjects or [])
    subject_ids = [sid for sid in subject_ids if sid not in exclude]

    def _load_subject_emg_profile(subject_id: int, file_identifier: str) -> np.ndarray | None:
        local_cfg = replace(cfg, modality="PSD", modality_file_id=file_identifier)
        try:
            spectrogram, freqs, timestamps, log_df = cbpa._load_subject_data(local_cfg, subject_id)
        except Exception as exc:
            warnings.warn(
                f"Subject {subject_id:02}: failed to load EMG PSD '{file_identifier}' ({exc}). Skipping."
            )
            return None

        trial_spans = {int(k): v for k, v in cbpa._get_trial_spans(log_df).items()}
        if len(trial_spans) == 0:
            return None

        band_power = cbpa._extract_band_power(local_cfg, spectrogram, freqs, channel_indices=None)
        trial_cond_map = {trial_id: "ALL" for trial_id in trial_spans}

        cycles = cbpa._band_power_per_phase(
            cfg=local_cfg,
            band_power=band_power,
            timestamps=timestamps,
            trial_spans=trial_spans,
            trial_cond_map=trial_cond_map,
            log_df=log_df,
        ).get("ALL", [])

        if len(cycles) == 0:
            return None

        return np.nanmean(np.stack(cycles, axis=0), axis=0)

    subject_flexor_profiles: dict[int, np.ndarray] = {}
    subject_extensor_profiles: dict[int, np.ndarray] = {}

    for subject_id in tqdm(subject_ids, desc='Importing Subject Data'):
        flexor_profile = _load_subject_emg_profile(subject_id, flexor_file_identifier)
        extensor_profile = _load_subject_emg_profile(subject_id, extensor_file_identifier)
        if flexor_profile is None or extensor_profile is None:
            continue
        subject_flexor_profiles[subject_id] = flexor_profile
        subject_extensor_profiles[subject_id] = extensor_profile

    common_subjects = sorted(set(subject_flexor_profiles) & set(subject_extensor_profiles))
    if len(common_subjects) == 0:
        warnings.warn("No valid subjects for EMG-PSD phase plot. Nothing will be plotted.")
        return

    flexor_stack = np.stack([subject_flexor_profiles[sid] for sid in common_subjects], axis=0)
    extensor_stack = np.stack([subject_extensor_profiles[sid] for sid in common_subjects], axis=0)

    flexor_mean = np.nanmean(flexor_stack, axis=0)
    extensor_mean = np.nanmean(extensor_stack, axis=0)

    n_phase = flexor_mean.shape[0]
    phase_grid = np.linspace(0.0, 360.0, n_phase, endpoint=False)

    combined = np.concatenate([flexor_mean.ravel(), extensor_mean.ravel()])
    emg_vmin = float(np.nanpercentile(combined, p_low))
    emg_vmax = float(np.nanpercentile(combined, p_high))
    if not np.isfinite(emg_vmin) or not np.isfinite(emg_vmax) or emg_vmin == emg_vmax:
        emg_vmin, emg_vmax = None, None

    fig, ax, cax, ax2, ax_tgt_left, ax_tgt_right = _create_cbpa_dual_panel_figure(
        show_target_sine=(cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization),
        figure_size_with_target=figure_size_with_target,
        figure_size_without_target=figure_size_without_target,
        grid_width_ratios=grid_width_ratios,
        grid_height_ratios_with_target=grid_height_ratios_with_target,
        grid_wspace=grid_wspace,
        grid_hspace_with_target=grid_hspace_with_target,
    )

    if cfg.include_suptitle:
        fig.suptitle(
            f"EMG PSD phase-normalized average ({cfg.freq_band})\n"
            f"Left: {flexor_file_identifier} | Right: {extensor_file_identifier} | "
            f"n = {len(common_subjects)} subjects",
            fontsize=10,
        )

    channel_labels = [f"Ch {idx + 1}" for idx in range(flexor_mean.shape[1])]
    if flexor_mean.shape[1] == 64:
        channel_labels = EMG_CHANNELS
        # Show only every 8th channel label to reduce clutter.
        channel_tick_idx = list(range(0, len(channel_labels), 8))
        if channel_tick_idx[-1] != len(channel_labels) - 1:  # also show the last:
            channel_tick_idx.append(len(channel_labels) - 1)

    im = ax.imshow(
        flexor_mean.T,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=emg_vmin,
        vmax=emg_vmax,
        extent=(phase_grid[0], 360.0, -0.5, flexor_mean.shape[1] - 0.5),
    )
    plt.colorbar(im, cax=cax, label=f"{cfg.freq_band.lower()}-band EMG PSD (log10)")

    if not (cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization):
        ax.set_xlabel("Force Cycle Phase (°)")
    ax.set_ylabel("Channel index")
    ax.set_yticks(channel_tick_idx)
    ax.set_yticklabels([channel_labels[i] for i in channel_tick_idx] if show_channel_labels else [''] * len(channel_tick_idx), fontsize=channel_label_fontsize)
    ax.set_title("Phase-normalized average EMG PSD: Flexor")

    ax2.imshow(
        extensor_mean.T,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=emg_vmin,
        vmax=emg_vmax,
        extent=(phase_grid[0], 360.0, -0.5, extensor_mean.shape[1] - 0.5),
    )
    if not (cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization):
        ax2.set_xlabel("Force Cycle Phase (°)")
    ax2.set_ylabel("")
    ax2.set_yticks(channel_tick_idx)
    ax2.set_yticklabels(
        [channel_labels[i] for i in channel_tick_idx] if show_channel_labels else [''] * len(channel_tick_idx),
        fontsize=channel_label_fontsize)
    ax2.set_title("Phase-normalized average EMG PSD: Extensor")

    if (cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization) and ax_tgt_left is not None and ax_tgt_right is not None:
        # Optionally load and average dynamometer force if requested
        dyno_force = None
        # todo: replace with data analysis function
        if cfg.include_dynamometer_force:
            experiment_results_dir = cfg.data_root / "data" / "experiment_results"
            dyno_force = _load_avg_dynamometer_force_per_phase(
                common_subjects, experiment_results_dir, phase_grid, cfg, use_unscaled_force=use_unscaled_force,
            )
        _plot_target_sine_panel(ax_tgt_left, phase_grid, cfg, x_label="Force Cycle Phase (°)",
                                show_ylabel=True, dynamometer_force_y=dyno_force, is_unscaled_force=use_unscaled_force)
        _plot_target_sine_panel(ax_tgt_right, phase_grid, cfg, x_label="Force Cycle Phase (°)",
                                show_ylabel=True, dynamometer_force_y=dyno_force, is_unscaled_force=use_unscaled_force,
                                show_legend=False,  # legend only for left plot
                                )

    phase_axes = [ax, ax2]
    if (cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization) and ax_tgt_left is not None and ax_tgt_right is not None:
        phase_axes.extend([ax_tgt_left, ax_tgt_right])
    _apply_phase_axis_style(
        phase_axes,
        phase_xticks=phase_xticks,
        phase_marker_lines=phase_marker_lines,
    )

    margin_left, margin_right, margin_top, margin_bottom = subplot_margins
    fig.subplots_adjust(
        left=margin_left,
        right=margin_right,
        top=margin_top,
        bottom=margin_bottom,
    )

    if cfg.save_plots:
        out = cfg.output_dir / filemgmt.file_title("EMG_PSD_phase_average_plot", ".png")
        fig.savefig(out, dpi=save_dpi, bbox_inches="tight")
        print(f"  Plot saved: {out}")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _load_avg_dynamometer_force_per_phase(
        subject_ids: list[int],
        experiment_results_dir: Path,
        phase_grid: np.ndarray,
        cfg: CBPAConfig,
        use_unscaled_force: bool = True,
        return_std: bool = False,
) -> np.ndarray | tuple[np.ndarray | None, np.ndarray | None] | None:
    """Load enriched serial frames PER TRIAL and average dynamometer force per phase.

    Loads Task-wise Scaled Force from each trial, phase-normalizes it per trial,
    then averages across all trials from all subjects.

    Parameters
    ----------
    subject_ids : list[int]
        Subject indices to load data from.
    experiment_results_dir : Path
        Root directory containing subject_XX folders.
    phase_grid : np.ndarray
        Phase bins (0-360°) for interpolation.
    cfg : CBPAConfig
        Configuration with trial frequency settings.

    Returns
    -------
    np.ndarray | None
        Averaged dynamometer force per phase bin, or None if data cannot be loaded.
        Shape: (len(phase_grid),)
    """
    def _task_freq_in_span(log_df: pd.DataFrame, t_start: pd.Timestamp, t_end: pd.Timestamp) -> float | None:
        """Return modal task frequency in a trial span."""
        mask = (log_df.index >= t_start) & (log_df.index < t_end)
        col = log_df.loc[mask, "Task Frequency"].dropna()
        if col.empty:
            return None
        return float(col.mode().iloc[0])

    try:
        all_cycles: list[np.ndarray] = []

        for subj_id in tqdm(subject_ids, desc="Load Force Data per Subject"):
            subj_exp_dir = experiment_results_dir / f"subject_{subj_id:02}"

            try:
                # Load enriched serial frame and log for this subject
                serial_frame = data_integration.fetch_enriched_serial_frame(
                    subj_exp_dir, set_time_index=True
                )
                log_df = data_integration.fetch_enriched_log_frame(
                    subj_exp_dir, set_time_index=True, verbose=False
                )
            except (ValueError, FileNotFoundError):
                # Skip if no enriched serial frame or log available
                continue

            force_column = "Unscaled Force [% MVC]" if use_unscaled_force else "Task-wise Scaled Force"
            if force_column not in serial_frame.columns:
                continue

            force_series = pd.to_numeric(serial_frame[force_column], errors="coerce")
            if force_series.notna().sum() < 2:
                continue

            s_idx = force_series.index  # ← hoisted: constant across all trials for this subject

            try:
                task_start_ends = data_integration.get_all_task_start_ends(
                    log_df, output_type='dict'
                )
            except Exception:
                continue

            for _, (trial_start, trial_end) in task_start_ends.items():
                try:
                    task_freq = _task_freq_in_span(log_df, trial_start, trial_end)
                    if task_freq is None or task_freq <= 0:
                        continue

                    # Align trial timestamps to serial index timezone, if needed.
                    t_start = trial_start
                    t_end = trial_end
                    if isinstance(s_idx, pd.DatetimeIndex):
                        if s_idx.tz is None and getattr(t_start, "tzinfo", None) is not None:
                            t_start = t_start.tz_localize(None)
                            t_end = t_end.tz_localize(None)
                        elif s_idx.tz is not None and getattr(t_start, "tzinfo", None) is None:
                            t_start = t_start.tz_localize(s_idx.tz)
                            t_end = t_end.tz_localize(s_idx.tz)

                    trial_force = force_series.loc[
                        (force_series.index >= t_start) & (force_series.index < t_end)
                        ].dropna()
                    if len(trial_force) < 2:
                        continue

                    t_rel = np.array(
                        [(ts - t_start).total_seconds() for ts in trial_force.index],
                        dtype=float,
                    )
                    y_rel = trial_force.to_numpy(dtype=float)
                    trial_dur_sec = (t_end - t_start).total_seconds()

                    # Use cfg.force_phase_start_offset_sec when explicitly set;
                    # otherwise fall back to 1/task_freq, which skips exactly one
                    # cycle and is always cycle-aligned regardless of frequency.
                    force_offset = (
                        float(cfg.force_phase_start_offset_sec)
                        if cfg.force_phase_start_offset_sec is not None
                        else float(1.0 / task_freq)
                    )
                    cycles = data_analysis.phase_normalize_cycles(
                        signal=y_rel,
                        t_rel=t_rel,
                        task_freq=task_freq,
                        trial_dur_sec=trial_dur_sec,
                        phase_grid=phase_grid,
                        min_samples_per_cycle=2,
                        start_offset_sec=force_offset,
                    )
                    all_cycles.extend(cycles)

                except Exception:  # any per-trial failure skips only that trial
                    continue

        if not all_cycles:
            return (None, None) if return_std else None

        # Average across all valid cycles from all valid trials/subjects.
        avg_force = np.nanmean(np.stack(all_cycles, axis=0), axis=0)

        # eventually return standard as well
        if return_std:
            std_force = np.nanstd(np.stack(all_cycles, axis=0), axis=0)
            return avg_force, std_force

        return avg_force

    except Exception as e:
        warnings.warn(f"Failed to load dynamometer force data: {e}")
        return (None, None) if return_std else None


def _target_sine_values(x: np.ndarray, cfg: CBPAConfig) -> np.ndarray:
    """Compute target-force sine values for plotting.

    Parameters
    ----------
    x : np.ndarray
        Domain values. Interpreted as phase (degrees) when
        ``cfg.use_phase_normalization`` is True, otherwise as time in seconds.
    cfg : CBPAConfig
        Configuration containing target sine min/max amplitude and frequency.

    Returns
    -------
    np.ndarray
        Target force values in percent MVC, same shape as ``x``.
    """
    x_arr = np.asarray(x, dtype=float)
    mid = 0.5 * (cfg.target_sine_min_pct_mvc + cfg.target_sine_max_pct_mvc)
    amp = 0.5 * (cfg.target_sine_max_pct_mvc - cfg.target_sine_min_pct_mvc)

    if cfg.use_phase_normalization:
        phase_rad = 2.0 * np.pi * (x_arr / 360.0)
    else:
        phase_rad = 2.0 * np.pi * cfg.target_sine_frequency_hz * x_arr

    # Match experiment logic: start at mean at x=0.
    return mid + amp * np.sin(phase_rad)


def _plot_target_sine_panel(
    ax,
    x: np.ndarray,
    cfg: CBPAConfig,
    x_label: str,
    show_ylabel: bool = True,
        dynamometer_force_y: np.ndarray | None = None,
        is_unscaled_force: bool = True,
        show_legend: bool = True,
        dynamometer_force_std_y: np.ndarray | None = None,
        dynamometer_force_std_factor: float = 0.5,
) -> None:
    """Draw one target-sine reference panel under a main plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into.
    x : np.ndarray
        X-axis values (phase or time).
    cfg : CBPAConfig
        Configuration with target sine settings.
    x_label : str
        Label for the x-axis.
    show_ylabel : bool, optional
        Whether to render the y-axis label. Default is True.
    dynamometer_force_y : np.ndarray | None, optional
        If provided and cfg.include_dynamometer_force is True, overlay the
        averaged per-cycle dynamometer force on the plot. Must have same
        length as x. Default is None.
    """
    x_arr = np.asarray(x, dtype=float)
    y_target = _target_sine_values(x_arr, cfg)

    # Circular wraparound for phase plots: repeat first sample at 360°.
    if cfg.use_phase_normalization and x_arr.size > 1:
        x_plot = np.concatenate([x_arr, [360.0]])
        y_target_plot = np.concatenate([y_target, [y_target[0]]])
    else:
        x_plot = x_arr
        y_target_plot = y_target

    ax.plot(x_plot, y_target_plot, color="dimgray", linewidth=1.2, label="Target" if is_unscaled_force else None)

    pad = 0.1 * max(1e-6, cfg.target_sine_max_pct_mvc - cfg.target_sine_min_pct_mvc)
    ax.set_ylim(cfg.target_sine_min_pct_mvc - pad, cfg.target_sine_max_pct_mvc + pad)
    ax.set_ylabel("Force [% MVC]" if show_ylabel else "")
    ax.set_xlabel(x_label)
    ax.set_title("Target sine", fontsize=12)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)

    if cfg.use_phase_normalization:
        ax.set_xlim(0, 360)

    # Optionally overlay dynamometer force data.
    if dynamometer_force_y is not None and cfg.include_dynamometer_force:
        force_mean = np.asarray(dynamometer_force_y, dtype=float)

        if cfg.use_phase_normalization and force_mean.size > 1:
            force_plot = np.concatenate([force_mean, [force_mean[0]]])
        else:
            force_plot = force_mean

        ax.plot(
            x_plot,
            force_plot,
            color="forestgreen",
            linewidth=1.2,
            alpha=0.9,
            label="Measurement" if is_unscaled_force else None,
        )

        # Optional variability band around force mean.
        if dynamometer_force_std_y is not None and dynamometer_force_std_factor > 0:
            force_std = np.asarray(dynamometer_force_std_y, dtype=float)
            if cfg.use_phase_normalization and force_std.size > 1:
                force_std_plot = np.concatenate([force_std, [force_std[0]]])
            else:
                force_std_plot = force_std

            half_band = dynamometer_force_std_factor * force_std_plot
            ax.fill_between(
                x_plot,
                force_plot - half_band,
                force_plot + half_band,
                color="forestgreen",
                alpha=0.15,
                linewidth=0.0,
                label=f"Measurement ±{dynamometer_force_std_factor:g}x SD" if is_unscaled_force else None,
            )

        # Right axis in native force units.
        if not is_unscaled_force:
            ax_force = ax.twinx()
            ax_force.plot(x_plot, force_plot, color="forestgreen", linewidth=1.2, alpha=0.9)

            if dynamometer_force_std_y is not None and dynamometer_force_std_factor > 0:
                force_std = np.asarray(dynamometer_force_std_y, dtype=float)
                if cfg.use_phase_normalization and force_std.size > 1:
                    force_std_plot = np.concatenate([force_std, [force_std[0]]])
                else:
                    force_std_plot = force_std
                half_band = dynamometer_force_std_factor * force_std_plot
                ax_force.fill_between(
                    x_plot,
                    force_plot - half_band,
                    force_plot + half_band,
                    color="forestgreen",
                    alpha=0.12,
                    linewidth=0.0,
                )

            ax_force.set_ylim(0, 1)
            ax_force.set_ylabel("Dynamometer\nForce [0-1]", color="forestgreen", fontweight="bold")
            ax_force.tick_params(axis="y", labelcolor="forestgreen")
            if cfg.use_phase_normalization:
                ax_force.set_xlim(0, 360)
        else:
            if show_legend:
                ax.legend(loc='center right', bbox_to_anchor=(1.275, 0.5))


def plot_cbpa_results(results: dict, cfg: CBPAConfig, use_unscaled_force: bool = True) -> None:
    """Render heatmap and cluster-summary figures for one CBPA result.

    Parameters
    ----------
    results : dict
        Output dictionary produced by :func:`run_cbpa`.
    cfg : CBPAConfig
        Plotting and output configuration.

    Notes
    -----
    Produces two main panels: a t-statistic heatmap with cluster contours and
    a cluster time-course panel. Optional target-sine panels are appended below.
    """
    t_obs        = results["t_obs"]
    t_thresh     = results["t_thresh"]
    clusters     = results["clusters"]
    cluster_pv   = results["cluster_pv"]
    good_inds    = results["good_cluster_inds"]
    ch_names     = results["ch_names"]
    time_grid    = results["time_grid"]
    n_valid_subjects = results["n_valid_subjects"]

    n_times, n_ch = t_obs.shape
    t_ax = time_grid if time_grid is not None else np.arange(n_times)

    fig, ax, cax, ax2, ax_tgt_left, ax_tgt_right = _create_cbpa_dual_panel_figure(
        show_target_sine=(cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization),
    )

    if cfg.include_suptitle:
        fig.suptitle(
            f"{cfg.hypothesis_label}\n"
            f"Contrast: '{cfg.condition_A}' − '{cfg.condition_B}'  |  "
            f"{cfg.modality} {cfg.freq_band}  |  "
            f"n = {n_valid_subjects} subjects, {cfg.n_permutations} permutations",
            fontsize=10,
        )

    # ── Panel A: heatmap + cluster contours ──────────────────────────────────
    # Use a fixed ±3 baseline for cross-plot comparability;
    # expand only if the observed t-values exceed it
    vlim = max(3.0, np.nanpercentile(np.abs(t_obs), 97))

    # FIX 3: extend right extent to 360.0 so the last bin visually fills to the
    # axis edge, consistent with plot_cmc_accuracy_phase_average and
    # plot_emg_psd_phase_average_plot. In clock-time mode t_ax[-1] is correct.
    extent_right = 360.0 if cfg.use_phase_normalization else t_ax[-1]
    im = ax.imshow(
        t_obs.T, aspect="auto", origin="lower", cmap="RdBu_r",
        vmin=-vlim, vmax=vlim,
        extent=[t_ax[0], extent_right, -0.5, n_ch - 0.5],
    )
    plt.colorbar(im, cax=cax, label="t-statistic")

    for idx, cluster in enumerate(clusters):
        mask = _resolve_cluster_mask(cluster, n_times=n_times, n_ch=n_ch)
        color = "black" if idx in good_inds else "silver"
        lw    = 1.8    if idx in good_inds else 0.8
        # contour needs at least one True and one False cell to draw anything
        if mask.any() and not mask.all():
            # FIX 3 (contour): mirror the imshow extent fix so cluster contour
            # lines align with the heatmap pixels all the way to 360°.
            contour_right = 360.0 if cfg.use_phase_normalization else t_ax[-1]
            ax.contour(
                np.linspace(t_ax[0], contour_right, n_times),
                np.arange(n_ch),
                mask.T.astype(float),
                levels=[0.5], colors=color, linewidths=lw,
            )

    x_label = "Force Cycle Phase (°)" if cfg.use_phase_normalization else "Time within trial (s)"
    if not (cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization):
        ax.set_xlabel(x_label)
    ax.set_ylabel("Channel index")
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels(ch_names, fontsize=6)
    ax.set_title("t-statistic map\n(black contour = significant cluster)")

    # ── Panel B: significant cluster time courses ─────────────────────────────
    if len(good_inds) == 0:
        ax2.text(0.5, 0.5, "No significant clusters", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12, color="grey")
    else:
        for idx in good_inds:
            mask = _resolve_cluster_mask(clusters[idx], n_times=n_times, n_ch=n_ch)

            ch_in_cluster = mask.any(axis=0)   # (n_ch,)   bool
            t_in_cluster  = mask.any(axis=1)   # (n_times,) bool

            if not ch_in_cluster.any():
                continue

            t_course = t_obs[:, ch_in_cluster].mean(axis=1)  # (n_times,)

            # FIX 2: wrap the time-course line and fill mask to close the
            # 350°→360° gap, mirroring the circular wrap used in
            # plot_cmc_accuracy_phase_average. In clock-time mode no wrap is needed.
            if cfg.use_phase_normalization:
                t_ax_plot          = np.concatenate([t_ax,          [360.0]])
                t_course_plot      = np.concatenate([t_course,      [t_course[0]]])
                t_in_cluster_plot  = np.concatenate([t_in_cluster,  [t_in_cluster[0]]])
            else:
                t_ax_plot         = t_ax
                t_course_plot     = t_course
                t_in_cluster_plot = t_in_cluster

            ax2.plot(t_ax_plot, t_course_plot,
                     label=f"Cluster #{idx + 1}  p={cluster_pv[idx]:.3f}")
            ax2.fill_between(t_ax_plot, 0, t_course_plot,
                             where=t_in_cluster_plot, alpha=0.2)

        ax2.axhline(0,         color="k",   linewidth=0.8, linestyle="--")
        ax2.axhline( t_thresh, color="red", linewidth=0.8, linestyle=":",
                     label=f"±t_thresh ({t_thresh:.2f})")
        ax2.axhline(-t_thresh, color="red", linewidth=0.8, linestyle=":")
        ax2.legend(fontsize=8)

    if not (cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization):
        ax2.set_xlabel(x_label)
    ax2.set_ylabel("Mean t-statistic over cluster channels")
    ax2.set_title("Significant cluster time courses")

    if (cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization):
        dyno_force = None
        if cfg.include_dynamometer_force and cfg.use_phase_normalization:
            data_root = cfg.data_root / "data" / "experiment_results"
            valid_subject_ids = list(range(0, 13))  # Fallback to known subjects
            dyno_force = _load_avg_dynamometer_force_per_phase(
                valid_subject_ids, data_root, t_ax, cfg,
                use_unscaled_force=use_unscaled_force,
            )

        _plot_target_sine_panel(
            ax_tgt_left, t_ax, cfg, x_label=x_label, show_ylabel=True,
            dynamometer_force_y=dyno_force, is_unscaled_force=use_unscaled_force,
        )
        _plot_target_sine_panel(
            ax_tgt_right, t_ax, cfg, x_label=x_label, show_ylabel=True,
            dynamometer_force_y=dyno_force, is_unscaled_force=use_unscaled_force,
            show_legend=False,  # legend only for left plot
        )

    if cfg.use_phase_normalization:
        phase_axes = [ax, ax2]
        if (cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization):
            phase_axes.extend([ax_tgt_left, ax_tgt_right])
        _apply_phase_axis_style(
            phase_axes,
            phase_xticks=(0.0, 90.0, 180.0, 270.0, 360.0),
            phase_marker_lines=(90.0, 270.0),
        )

    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.10)
    if cfg.save_plots:
        out = cfg.output_dir / filemgmt.file_title(cfg.hypothesis_label + "_clusters", ".png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {out}")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)

def old_plot_cbpa_results(results: dict, cfg: CBPAConfig, use_unscaled_force: bool = True) -> None:
    """Render heatmap and cluster-summary figures for one CBPA result.

    Parameters
    ----------
    results : dict
        Output dictionary produced by :func:`run_cbpa`.
    cfg : CBPAConfig
        Plotting and output configuration.

    Notes
    -----
    Produces two main panels: a t-statistic heatmap with cluster contours and
    a cluster time-course panel. Optional target-sine panels are appended below.
    """
    t_obs        = results["t_obs"]
    t_thresh     = results["t_thresh"]
    clusters     = results["clusters"]
    cluster_pv   = results["cluster_pv"]
    good_inds    = results["good_cluster_inds"]
    ch_names     = results["ch_names"]
    time_grid    = results["time_grid"]
    n_valid_subjects = results["n_valid_subjects"]

    n_times, n_ch = t_obs.shape
    t_ax = time_grid if time_grid is not None else np.arange(n_times)

    fig, ax, cax, ax2, ax_tgt_left, ax_tgt_right = _create_cbpa_dual_panel_figure(
        show_target_sine=(cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization),
    )

    if cfg.include_suptitle:
        fig.suptitle(
            f"{cfg.hypothesis_label}\n"
            f"Contrast: '{cfg.condition_A}' − '{cfg.condition_B}'  |  "
            f"{cfg.modality} {cfg.freq_band}  |  "
            f"n = {n_valid_subjects} subjects, {cfg.n_permutations} permutations",
            fontsize=10,
        )

    # ── Panel A: heatmap + cluster contours ──────────────────────────────────
    # Use a fixed ±3 baseline for cross-plot comparability;
    # expand only if the observed t-values exceed it
    vlim = max(3.0, np.nanpercentile(np.abs(t_obs), 97))
    im = ax.imshow(
        t_obs.T, aspect="auto", origin="lower", cmap="RdBu_r",
        vmin=-vlim, vmax=vlim,
        extent=[t_ax[0], t_ax[-1], -0.5, n_ch - 0.5],
    )
    plt.colorbar(im, cax=cax, label="t-statistic")

    for idx, cluster in enumerate(clusters):
        mask = _resolve_cluster_mask(cluster, n_times=n_times, n_ch=n_ch)
        color = "black" if idx in good_inds else "silver"
        lw    = 1.8    if idx in good_inds else 0.8
        # contour needs at least one True and one False cell to draw anything
        if mask.any() and not mask.all():
            ax.contour(
                np.linspace(t_ax[0], t_ax[-1], n_times),
                np.arange(n_ch),
                mask.T.astype(float),
                levels=[0.5], colors=color, linewidths=lw,
            )

    x_label = "Force Cycle Phase (°)" if cfg.use_phase_normalization else "Time within trial (s)"
    if not (cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization):
        ax.set_xlabel(x_label)
    ax.set_ylabel("Channel index")
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels(ch_names, fontsize=6)
    ax.set_title("t-statistic map\n(black contour = significant cluster)")

    # ── Panel B: significant cluster time courses ─────────────────────────────
    if len(good_inds) == 0:
        ax2.text(0.5, 0.5, "No significant clusters", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12, color="grey")
    else:
        for idx in good_inds:
            # Use shared resolver — fixes the previously unreshaped 1D mask here
            mask = _resolve_cluster_mask(clusters[idx], n_times=n_times, n_ch=n_ch)

            ch_in_cluster = mask.any(axis=0)   # (n_ch,) bool
            t_in_cluster  = mask.any(axis=1)   # (n_times,) bool

            if not ch_in_cluster.any():
                continue

            t_course = t_obs[:, ch_in_cluster].mean(axis=1)  # (n_times,)
            ax2.plot(t_ax, t_course,
                     label=f"Cluster #{idx + 1}  p={cluster_pv[idx]:.3f}")
            ax2.fill_between(t_ax, 0, t_course, where=t_in_cluster, alpha=0.2)

        ax2.axhline(0,         color="k",   linewidth=0.8, linestyle="--")
        ax2.axhline( t_thresh, color="red", linewidth=0.8, linestyle=":",
                     label=f"±t_thresh ({t_thresh:.2f})")
        ax2.axhline(-t_thresh, color="red", linewidth=0.8, linestyle=":")
        ax2.legend(fontsize=8)

    if not (cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization):
        ax2.set_xlabel(x_label)
    ax2.set_ylabel("Mean t-statistic over cluster channels")
    ax2.set_title("Significant cluster time courses")

    if (cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization):
        # Optionally load and average dynamometer force if requested
        dyno_force = None
        if cfg.include_dynamometer_force and cfg.use_phase_normalization:
            # Load force data from all valid subjects (guessing they're in data root)
            data_root = cfg.data_root / "data" / "experiment_results"
            # Get subject IDs from results (assuming they were used in the CBPA run)
            valid_subject_ids = list(range(0, 13))  # Fallback to known subjects
            dyno_force = _load_avg_dynamometer_force_per_phase(
                valid_subject_ids, data_root, t_ax, cfg, use_unscaled_force=use_unscaled_force,
            )

        _plot_target_sine_panel(ax_tgt_left, t_ax, cfg, x_label=x_label, show_ylabel=True,
                                dynamometer_force_y=dyno_force, is_unscaled_force=use_unscaled_force)
        _plot_target_sine_panel(ax_tgt_right, t_ax, cfg, x_label=x_label, show_ylabel=True,
                                dynamometer_force_y=dyno_force, is_unscaled_force=use_unscaled_force,
                                show_legend=False,  # legend only for left plot
                                )

    if cfg.use_phase_normalization:
        phase_axes = [ax, ax2]
        if (cfg.show_target_sine if cfg.show_target_sine is not None else cfg.use_phase_normalization):
            phase_axes.extend([ax_tgt_left, ax_tgt_right])
        _apply_phase_axis_style(
            phase_axes,
            phase_xticks=(0.0, 90.0, 180.0, 270.0, 360.0),
            phase_marker_lines=(90.0, 270.0),
        )

    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.10)
    if cfg.save_plots:
        out = cfg.output_dir / filemgmt.file_title(cfg.hypothesis_label + "_clusters", ".png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {out}")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)
