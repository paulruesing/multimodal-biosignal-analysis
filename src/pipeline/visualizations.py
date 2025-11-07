import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from typing import Callable, Literal, Tuple
import multiprocessing
import os

import src.utils.file_management as filemgmt



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

##############  PLOTTING FUNCTIONS ##############
def plot_eeg_heatmap(values: dict[str, float] | list[float],
                     include_labels: bool = True,
                     colormap='viridis',
                     color_scale=None,
                     value_label: str = 'Heatmap values',
                     plot_size: (int, int) = (8, 8)):
    """
    Plot heatmap circles at predefined electrode positions for a 64-channel EEG (10-20 system).

    Parameters
    ----------
    values : dict or list of float
        Heatmap values mapped by channel names if dict, or list of values corresponding to predefined electrode positions.
    include_labels : bool, default True
        Whether to include text labels for each electrode.
    colormap : str or matplotlib.colors.Colormap, default 'viridis'
        Colormap used to map values to colors.
    color_scale : tuple of (float, float) or None, optional
        Tuple specifying (vmin, vmax) for normalization of colors. If None, min and max of values are used.
    value_label : str, default 'Heatmap values'
        Label for the colorbar next to the heatmap.
    plot_size : tuple of int, default (8, 8)
        Size of the matplotlib figure in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the heatmap plot.
    """
    fig, ax = plt.subplots(figsize=plot_size)

    # Extract values array for normalization
    vals = list(values.values()) if isinstance(values, dict) else values
    vmin, vmax = color_scale if color_scale is not None else (min(vals), max(vals))

    # Normalize and colormap setup
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps[colormap]

    # Draw circles
    largest_abs_x = 0; largest_abs_y = 0
    for ind, (ch, pos) in enumerate(EEG_POSITIONS.items()):
        val = values.get(ch, vmin) if isinstance(values, dict) else values[ind]
        color = cmap(norm(val))
        circle = patches.Circle(pos, radius=0.05, color=color, ec='black', lw=0.8)
        ax.add_patch(circle)
        if include_labels:
            ax.text(pos[0], pos[1], ch, ha='center', va='center', fontsize=8,
                    color='white' if np.mean(color[:3]) < .5 else 'black',  # use black font for very bright colors (RGB average based)
                    )

    # add elipse around head:
    width, height = 1.256, 1.505
    ellipse = patches.Ellipse([0, 0], width=width, height=height,
                              edgecolor='black', facecolor='none', lw=1.5, ls='-')
    ax.add_patch(ellipse)

    # add nose:
    half_circle = patches.Wedge(center=(0, height/2), r=0.1, theta1=-5, theta2=185,
                                facecolor='none', edgecolor='black', ls='-', lw=1.5)
    ax.add_patch(half_circle)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(vals)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(value_label)

    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()

    return fig

if __name__ == '__main__':
    plot_circles_heatmap(values=range(64))