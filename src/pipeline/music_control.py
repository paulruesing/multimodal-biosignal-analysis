import subprocess  # allows to EXTERNALLY (<-> multiprocessing) run shell commands or other system-level programs
from typing import Literal
import random
from pathlib import Path
import librosa
import numpy as np
import os
import mutagen
import pandas as pd
import time
from tqdm import tqdm
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt

import src.utils.file_management as filemgmt
from src.utils.str_conversion import str_to_float


mpl.use('Qt5Agg')


class SpotifyController:
    """
    Controller class to manage playback of Spotify tracks by category (on Mac).

    Attributes
    ----------
    category_url_dict : dict[str, tuple[tuple[str, int] | str]] | None
        Dictionary mapping category names to lists of track URLs with optional start times.
        Format: {category_name: [(song_url, start_at_second), (song_url2, start_at_second)]}
    randomly_shuffle_category_lists : bool
        If True, randomly shuffle the order of tracks within each category.
    current_category_and_counter : tuple[str, int] | None
        Tracks the current category and index of the next track to play.
    """
    def __init__(self,
                 category_url_dict: dict[str, list[tuple[str, str, int] | tuple[str, str]]] | str | None = None,
             #                       {category_name: [(genre, song_url, start_at_second), (genre, song_url), ...]}
                 randomly_shuffle_category_lists: bool = True,
                 ):
        """
        Initialize SpotifyController instance.

        Parameters
        ----------
        category_url_dict : dict[str, tuple[tuple[str, int] | str]] | str | None, optional
            Either a dictionary describing categories with their associated track URLs and start times,
            or a path (str or Path) to a configuration text file containing this information.
        randomly_shuffle_category_lists : bool, optional
            Whether to shuffle the order of tracks within each category (default is False).
        """
        if isinstance(category_url_dict, (str, Path)):  # read category_url_dict from provided txt file path
            category_url_dict = self.read_category_url_config_txt(category_url_dict)

        # set and eventually shuffle
        self.category_url_dict = category_url_dict
        if category_url_dict is not None and randomly_shuffle_category_lists:  # randomly shuffle songs
            self.category_url_dict = {cat: random.sample(entries, len(entries)) for cat, entries in category_url_dict.items()}

        # initialise counter (for play_next_from)
        self.category_counter_dict: dict[str, int] = {}
        if category_url_dict is not None:
            for category in category_url_dict.keys():
                self.category_counter_dict[category] = - 1  # initial category counter at -1 (because +1 happens before starting first song)
        self.current_category: str | None = None
        # self.current_category_and_counter: tuple[str, int] | None = None  (OLD LOGIC)

        # the following will be updated if a new song is started:
        self.current_genre = None
        self.current_bpm = None
        self.current_file_title = None

    def read_category_url_config_txt(self, txt_file: str | Path) -> dict:
        """
        Read category-track configuration from a text file.

        The text file format expects:
        - Category titles enclosed in single quotes at the start of a line.
        - Track URLs preceded by a Genre label and optionally followed by a start time in seconds. (e.g. R&B --- https://aslöfas --- 10
        - Lines starting with '#' are treated as comments and ignored.
        - Empty lines are ignored.

        Parameters
        ----------
        txt_file : str | Path
            Path to the configuration text file.

        Returns
        -------
        dict
            Dictionary mapping category titles to lists of track URLs or (track URL, start time) tuples.

        Raises
        ------
        ValueError
            If a track entry appears before any category title is defined.
        """
        def detect_song_data(line: str) -> tuple[str, str, float, float, str]:
            """ Detect genre, song_url and start_after_seconds from read line. """
            elements = line.split(" --- ")
            if len(elements) != 5: raise ValueError("Invalid line detected: {}".format(line))

            elements = [element.strip() for element in elements]

            genre, url, start_after_second, bpm, file_title = elements
            bpm = float(bpm); start_after_second = float(start_after_second)
            return genre, url, start_after_second, bpm, file_title

        with open(txt_file, "r") as f:
            result_dict = {}
            for line in f.readlines():
                if line[0] == "'":
                    # if category label (in single quotation marks)
                    current_category_title = line.strip().replace("'", "")
                    result_dict[current_category_title] = []
                    continue
                elif line[0] == "#":
                    continue  # comment
                elif line.strip() == "":
                    continue  # empty line

                try:  # try detect comments
                    line = line.split(" #")[0]  # and separate them
                except IndexError:
                    pass

                try:
                    result_dict[current_category_title].append(detect_song_data(line))
                except KeyError:
                    raise ValueError("Category URL config file needs to start with 'category_name' before first other entry (besides comments or empty lines)!")

        return result_dict

    def play_next_from(self, category: str):
        """
        Play the next track from the specified category.

        Advances the internal track counter for the category and plays the respective track.
        If the end of the category's playlist is reached, starts over from the beginning.

        Parameters
        ----------
        category : str
            The category name from which to play the next track.

        Raises
        ------
        AttributeError
            If the category URL dictionary is not defined.
        """
        if self.category_url_dict.get(category) is None: raise AttributeError(
            f"Instance attribute category_url_dict needs to be defined and contain {category}!")

        # set current_category appropriately:
        if self.current_category is None or self.current_category != category:
            self.current_category = category
        # and increase counter:
        self.category_counter_dict[category] = self.category_counter_dict[category] + 1

        # derive next track and when to start it:
        try:
            song_tuple = self.category_url_dict[category][self.category_counter_dict[category]]
        except IndexError:
            print("No new songs left in category! Starting over.")
            self.category_counter_dict[category] = 0
            song_tuple = self.category_url_dict[category][self.category_counter_dict[category]]

        # try detect start_after:
        start_at = song_tuple[2] if song_tuple[2] != 0.0 else None

        # other properties:
        self.current_genre = song_tuple[0]
        next_track_url = song_tuple[1]
        self.current_bpm = song_tuple[3]
        self.current_file_title = song_tuple[4]

        # play:
        self.play_track(next_track_url); print(f"Playing {next_track_url} (number {self.category_counter_dict[category]} in category {category})");
        if start_at is not None and start_at != 0: print(f"from second {start_at}"); self.skip(start_at)


    # below functions are based on  https://johnculviner.com/automatically-skip-songs-in-spotify-mac-app/, thank you!
    def get_current_track(self, output_type: Literal['str', 'dict'] = 'dict') -> str | dict:
        """
        Retrieve information about the currently playing Spotify track.

        Uses AppleScript to communicate with the Spotify macOS application.

        Parameters
        ----------
        output_type : Literal['str', 'dict'], optional
            If 'str', returns raw string output.
            If 'dict', returns parsed dictionary with track details (default).

        Returns
        -------
        str | dict
            Current track information as a formatted string or a dictionary with keys:
            'Title', 'Artist', 'Album', 'Duration [ms]', 'Position [s]', 'Genre'

        Raises
        ------
        RuntimeError
            If an error occurs when querying Spotify.
        """
        # tell scopes commands to a specific apple-scriptable target
        # command to send via subprocess:
        applescript = '''
        tell application "Spotify"
            if it is running then
                if player state is playing then
                    set trackName to name of current track
                    set trackArtist to artist of current track
                    set trackAlbum to album of current track
                    set trackDuration to duration of current track  -- milliseconds
                    set currentPosition to player position -- seconds
                    return trackName & " | " & trackArtist & " | " & trackAlbum & " | " & trackDuration & " | " & currentPosition
                else
                    return "NOT_PLAYING"
                end if
            else
                return "SPOTIFY_NOT_RUNNING"
            end if
        end tell
        '''
        process = subprocess.Popen(  # starts a new process
            ['osascript', '-e', applescript],  # invoke macOS osascript command-line utility
            stdout=subprocess.PIPE, stderr=subprocess.PIPE  # redirects output and error to python via PIPE
        )
        output, error = process.communicate()  # waits for finished execution and reads all output from stdout and stderr
        if process.returncode == 0:  # indicates successful execution
            if output_type == 'str':
                output_str = output.decode().strip()
                if self.current_genre is not None: output_str += " | " + self.current_genre
                if self.current_bpm is not None: output_str += " | " + f"{self.current_bpm:.2f}"
                return output_str
            else:
                #print(output.decode())
                title, artist, album, duration_ms, position_s = output.decode().strip().split(' | ')
                position_s = str_to_float(position_s, is_ger_format=("." not in position_s))
                duration_ms = str_to_float(duration_ms,  is_ger_format=("." not in duration_ms))
                output_dict = {'Title': title, 'Artist': artist, 'Album': album, 'Duration [ms]': duration_ms,
                               'Position [s]': position_s, 'Genre': self.current_genre, 'BPM': self.current_bpm}
                if self.current_genre is not None: output_dict['Genre'] = self.current_genre
                if self.current_bpm is not None: output_dict['BPM'] = self.current_bpm
                if self.current_file_title is not None: output_dict['File Title'] = self.current_file_title
                return output_dict
        else:
            raise RuntimeError(f'Error while getting current track: {error}')

    @staticmethod
    def skip(seconds):
        """
        Skip forward in the currently playing track by a specified number of seconds.

        Parameters
        ----------
        seconds : int | float
            Number of seconds to skip forward in the track.
        """
        applescript = f'''
        tell application "Spotify"
            set player position to (player position + {seconds})
        end tell
        '''
        subprocess.Popen(['osascript', '-e', applescript],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    @staticmethod
    def skip_track():
        """
        Skip to the next track in the Spotify player.
        """
        applescript = 'tell application "Spotify" to next track'
        process = subprocess.Popen(
            ['osascript', '-e', applescript],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        process.communicate()

    @staticmethod
    def pause():
        """
        Pause the currently playing Spotify track.
        """
        applescript = 'tell application "Spotify" to pause'
        process = subprocess.Popen(
            ['osascript', '-e', applescript],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        process.communicate()

    @staticmethod
    def resume():
        """
        Resume playback of the current Spotify track.
        """
        applescript = 'tell application "Spotify" to play'
        process = subprocess.Popen(
            ['osascript', '-e', applescript],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        process.communicate()

    @staticmethod
    def play_track(spotify_track_url: str):
        """
        Play a Spotify track by its URL or track ID.

        Accepts full or partial Spotify track URLs and extracts the track ID.

        Parameters
        ----------
        spotify_track_url : str
            Spotify track URL or track ID to play.

        Notes
        -----
        After starting playback, the Spotify window is hidden to avoid interrupting user workflow.
        """
        """ Track urls are found in spotify "share song" links as https://open.spotify.com/intl-de/track/TRACKURLHERE?si=7efd60d6a9794e2e but you can also provide the full URL. """
        if "spotify" in spotify_track_url:  # distil track url
            spotify_track_url = spotify_track_url.split("?")[0].split("/")[-1]
        applescript = f'''
        tell application "Spotify"
            if it is running then
                play track "spotify:track:{spotify_track_url}"  -- this brings window to foreground
            end if
        end tell
        tell application "System Events"  -- hide window again
            set visible of process "Spotify" to false
        end tell
        '''
        process = subprocess.Popen(
            ['osascript', '-e', applescript],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, err = process.communicate()
        print(output.decode('utf-8').strip(), err.decode('utf-8').strip())



########### MUSIC CHARACTERISTICS ###########
def load_librosa_file(file_path, duration: float = 120.0) -> tuple:
    """
    Load an audio file and extract its waveform and sample rate.

    Uses mutagen to retrieve the correct sample rate from file metadata,
    then loads the audio using librosa with that sample rate.

    Parameters
    ----------
    file_path : str or Path
        Path to the audio file to load.
    duration : float or None, default=120
        Duration in seconds to load from the audio file.
        If None, loads the entire file.
        Default is 120 seconds, because this is the full listening duration during the experiment.

    Returns
    -------
    y : ndarray
        Audio time series (waveform).
    sr : int
        Sample rate of the audio file.

    """
    # Get sample rate from file metadata
    sr = mutagen.File(file_path).info.sample_rate

    # Load audio with optional duration limit
    y, sr = librosa.load(file_path, sr=sr, duration=duration)

    return y, sr


def compute_bpm_and_beat_times_and_intervals(y, sr: float, verbose: bool = True) -> tuple[float, list, list]:
    """
    Detect beat positions and compute tempo from audio.

    Uses librosa's beat tracking algorithm to identify beat positions in time,
    then derives tempo from the mean inter-beat interval.

    Parameters
    ----------
    y : ndarray
        Audio time series (waveform).
    sr : float
        Sample rate of the audio.
    verbose : bool, default True
        If True, print BPM and first 10 beat positions.

    Returns
    -------
    bpm_from_beats : float
        Tempo in beats per minute, calculated from beat intervals.
    beat_times : ndarray
        Time positions of detected beats in seconds.
    beat_intervals : ndarray
        Time intervals between consecutive beats in seconds.

    """
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Calculate inter-beat intervals and derive BPM from mean interval
    beat_intervals = np.diff(beat_times)
    mean_interval = np.mean(beat_intervals)
    bpm_from_beats = 60 / mean_interval

    if verbose:
        print(f"\nBPM: {bpm_from_beats:.1f}")
        print(f"Beat positions (sec): {beat_times[:10]}")

    return bpm_from_beats, beat_times, beat_intervals


def compute_stft(y, n_fft: int = 2048, hop_length: int = 512):
    """
    Compute the Short-Time Fourier Transform of audio.

    Applies STFT to decompose the audio into frequency components over time,
    returning the magnitude spectrum for spectral analysis.

    Parameters
    ----------
    y : ndarray
        Audio time series (waveform).
    n_fft : int, default 2048
        FFT window size in samples. Larger values provide better frequency
        resolution but coarser time resolution.
    hop_length : int, default 512
        Number of samples between successive frames. Smaller values provide
        better time resolution.

    Returns
    -------
    S : ndarray, shape (n_fft // 2 + 1, n_frames)
        Magnitude spectrogram (absolute value of STFT).

    """
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S = np.abs(D)
    return S


def compute_spectral_flux(S, verbose: bool = True):
    """
    Compute normalized spectral flux from a magnitude spectrogram.

    Spectral flux measures the magnitude of change in the frequency spectrum
    between consecutive frames, indicating rapid spectral changes (attacks,
    transients). Values are normalized to [0, 1].

    Parameters
    ----------
    S : ndarray, shape (n_fft // 2 + 1, n_frames)
        Magnitude spectrogram.
    verbose : bool, default True
        If True, print statistics (min, max, mean, std).

    Returns
    -------
    spectral_flux_normalized : ndarray, shape (n_frames - 1,)
        Normalized spectral flux values in range [0, 1].

    """
    # Compute Euclidean distance (L2 norm) between consecutive spectral frames
    spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))

    # Normalize to [0, 1] range
    spectral_flux_normalized = spectral_flux / np.max(spectral_flux)

    if verbose:
        print(f"\nSpectral flux:")
        print(f"  Min: {spectral_flux_normalized.min():.3f}")
        print(f"  Max: {spectral_flux_normalized.max():.3f}")
        print(f"  Mean: {spectral_flux_normalized.mean():.3f}")
        print(f"  Std: {spectral_flux_normalized.std():.3f}")

    return spectral_flux_normalized


def compute_spectral_centroid(S, sr: float, verbose: bool = True):
    """
    Compute the spectral centroid across time.

    Spectral centroid is the center of mass of the frequency spectrum,
    representing the "brightness" or average frequency of the audio.
    Values increase with higher-frequency content.

    Parameters
    ----------
    S : ndarray, shape (n_fft // 2 + 1, n_frames)
        Magnitude spectrogram.
    sr : float
        Sample rate of the audio.
    verbose : bool, default True
        If True, print mean, min, and max centroid values.

    Returns
    -------
    spectral_centroid : ndarray, shape (n_frames,)
        Centroid frequency in Hz for each time frame.

    """
    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]

    if verbose:
        print(f"\nSpectral Centroid:")
        print(f"  Mean: {spectral_centroid.mean():.1f} Hz")
        print(f"  Min: {spectral_centroid.min():.1f} Hz")
        print(f"  Max: {spectral_centroid.max():.1f} Hz")

    return spectral_centroid


def compute_onset_times(y, sr: float):
    """
    Detect note/sound attack onset times in audio.

    Uses onset detection to identify the start times of percussive or
    transient events in the audio signal.

    Parameters
    ----------
    y : ndarray
        Audio time series (waveform).
    sr : float
        Sample rate of the audio.

    Returns
    -------
    onset_times : ndarray
        Time positions of detected onsets in seconds.

    """
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    return onset_times


def compute_ioi_entropy_and_var_coefficient(onset_times, verbose: bool = True):
    """
    Compute temporal regularity metrics from inter-onset intervals.

    Calculates Shannon entropy of the inter-onset interval (IOI) distribution
    and the coefficient of variation, both measuring timing irregularity.
    Higher values indicate more irregular, less predictable onset patterns.

    Parameters
    ----------
    onset_times : ndarray
        Time positions of detected onsets in seconds.
    verbose : bool, default True
        If True, print detailed IOI statistics and entropy values.

    Returns
    -------
    ioi_entropy : float or None
        Shannon entropy of IOI distribution in bits. Returns None if
        insufficient onsets detected.
    ioi_cv : float or None
        Coefficient of variation of IOI (std / mean). Range [0, inf],
        where 0 indicates perfect regularity. Returns None if insufficient
        onsets detected.

    Notes
    -----
    IOI Entropy uses 20-bin histogram of inter-onset intervals.
    Coefficient of variation (CV) is more robust to outliers than entropy.

    """
    if len(onset_times) > 1:
        # Calculate time intervals between consecutive onsets
        ioi = np.diff(onset_times)

        # Compute histogram-based entropy of IOI distribution
        hist, bin_edges = np.histogram(ioi, bins=20, density=True)
        hist = hist / np.sum(hist)
        ioi_entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))

        # Compute coefficient of variation (dispersion metric)
        ioi_cv = np.std(ioi) / np.mean(ioi) if np.mean(ioi) > 0 else 0

        if verbose:
            print(f"  Number of onsets detected: {len(onset_times)}")
            print(f"  Number of IOIs: {len(ioi)}")
            print(f"  IOI mean: {np.mean(ioi):.4f} s")
            print(f"  IOI std: {np.std(ioi):.4f} s")
            print(f"  IOI range: {ioi.min():.4f} - {ioi.max():.4f} s")
            print(f"  IOI Entropy (Shannon): {ioi_entropy:.3f} bits")
            print(f"  IOI Coefficient of Variation: {ioi_cv:.3f}")
            print(f"    (0 = perfectly regular, higher = more irregular)")
    else:
        print("  Not enough onsets detected for IOI analysis")
        ioi_entropy = None
        ioi_cv = None

    return ioi_entropy, ioi_cv


def compute_syncopation_degree(beat_times, onset_times, beat_intervals, verbose: bool = True):
    """
    Compute syncopation metrics measuring rhythmic displacement from the beat.

    Detects how much onsets deviate from the established beat grid. Syncopation
    occurs when onsets are displaced from expected beat positions, creating
    rhythmic tension or "groove".

    Parameters
    ----------
    beat_times : ndarray
        Time positions of detected beats in seconds.
    onset_times : ndarray
        Time positions of detected onsets in seconds.
    beat_intervals : ndarray
        Time intervals between consecutive beats in seconds.
    verbose : bool, default True
        If True, print detailed syncopation analysis.

    Returns
    -------
    syncopation_degree : float or None
        Mean normalized distance of onsets from nearest beat, scaled to [0, 1].
        0 = strictly on-beat, 1 = highly syncopated. Returns None if
        insufficient beats detected.
    syncopation_ratio : float or None
        Percentage of onsets classified as syncopated (>0.2 beat intervals
        away from nearest beat). Returns None if insufficient beats detected.

    Notes
    -----
    Syncopation degree is computed by normalizing onset-to-beat distances by
    the mean beat interval, then scaling to [0, 1] range.

    """
    if verbose:
        print("\nComputing Degree of Syncopation...")

    if len(beat_times) > 2:
        # For each onset, calculate distance to nearest beat
        onset_to_beat_distances = []
        for onset in onset_times:
            nearest_beat_idx = np.argmin(np.abs(beat_times - onset))
            nearest_beat_time = beat_times[nearest_beat_idx]

            # Normalize distance by mean beat interval
            beat_interval = np.mean(beat_intervals)
            distance = np.abs(onset - nearest_beat_time) / beat_interval
            onset_to_beat_distances.append(distance)

        onset_to_beat_distances = np.array(onset_to_beat_distances)

        # Mean deviation from beat, scaled to [0, 1] range
        syncopation_degree = np.mean(onset_to_beat_distances)

        # Count onsets displaced more than 0.2 beat intervals from nearest beat
        syncopation_threshold = 0.2
        syncopated_onsets = np.sum(onset_to_beat_distances > syncopation_threshold)
        syncopation_percentage = 100 * syncopated_onsets / len(onset_to_beat_distances)

        if verbose:
            print(f"  Number of beats detected: {len(beat_times)}")
            print(f"  Number of onsets detected: {len(onset_times)}")
            print(f"  Mean beat interval: {np.mean(beat_intervals):.4f} s")
            print(f"\n  Syncopation Analysis:")
            print(f"    Mean distance from beat: {syncopation_degree:.3f} beat intervals")
            print(f"    Syncopation degree (0-1): {min(syncopation_degree * 2, 1.0):.3f}")
            print(f"      (0 = strictly on-beat, 1 = highly syncopated)")
            print(f"    Syncopated onsets (>0.2 beat away): {syncopation_percentage:.1f}%")
    else:
        print("  Not enough beats detected for syncopation analysis")
        syncopation_degree = None
        syncopation_percentage = None

    return min(syncopation_degree * 2, 1.0), syncopation_percentage


def compute_all_musical_features(audio_path: str | Path, duration: float = 120.0,
                                 verbose: bool = False):
    """
    Compute comprehensive musical features from an audio file.

    Extracts tempo, spectral properties, and rhythmic characteristics from
    audio data using librosa and custom feature computation functions.

    Parameters
    ----------
    audio_path : str or Path
        File path to the audio file to analyze.
    duration : float or None, default=120
        Duration in seconds to load from the audio file.
        If None, loads the entire file.
        Default is 120 seconds, because this is the full listening duration during the experiment.
    verbose : bool, default False
        If True, print diagnostic information during computation.

    Returns
    -------
    bpm_from_beats : float
        Tempo of the audio in beats per minute.
    spectral_flux_normalized : ndarray
        Normalized spectral flux across time, measuring magnitude of
        spectral changes.
    spectral_centroid : ndarray
        Center of mass of the frequency spectrum across time.
    ioi_cv : float
        Coefficient of variation of inter-onset intervals, measuring
        timing irregularity.
    syncopation_degree : float
        Measure of rhythmic displacement relative to the beat grid.
    syncopation_ratio : float
        Proportion of onsets that are syncopated relative to the beat.

    """
    # Ensure audio_path is a Path object for consistent handling
    if not isinstance(audio_path, Path):
        audio_path = Path(audio_path)

    # Load audio file and extract waveform and sample rate
    y, sr = load_librosa_file(audio_path, duration=duration)

    # Extract tempo and beat-related features
    bpm_from_beats, beat_times, beat_intervals = compute_bpm_and_beat_times_and_intervals(
        y, sr, verbose
    )

    # Compute Short-Time Fourier Transform for spectral analysis
    # Standard window size (2048 samples) with 512-sample hop for overlap
    n_fft = 2048
    S = compute_stft(y, n_fft=n_fft, hop_length=512)

    # Extract spectral flux: magnitude of frame-to-frame spectral changes
    spectral_flux_normalized = compute_spectral_flux(S, verbose)

    # Log frequency distribution if verbose mode enabled
    if verbose:
        freq_distribution = np.mean(S, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        print(f"\nFrequency distribution:")
        print(f"  Freq range: {freqs[0]:.1f} Hz - {freqs[-1]:.1f} Hz")
        print(f"  Max energy at: {freqs[np.argmax(freq_distribution)]:.1f} Hz")

    # Compute spectral centroid: center of mass of the spectrum over time
    spectral_centroid = compute_spectral_centroid(S, sr, verbose)

    # Detect onset times from audio signal
    onset_times = compute_onset_times(y, sr)

    # Compute inter-onset interval entropy and coefficient of variation
    # These measure temporal regularity of note/sound attacks
    ioi_entropy, ioi_cv = compute_ioi_entropy_and_var_coefficient(
        onset_times, verbose=verbose
    )

    # Compute syncopation: rhythmic displacement of onsets relative to beat grid
    # Syncopation degree indicates strength, syncopation ratio indicates prevalence
    syncopation_degree, syncopation_ratio = compute_syncopation_degree(
        beat_times, onset_times, beat_intervals, verbose=verbose
    )

    return bpm_from_beats, spectral_flux_normalized, spectral_centroid, ioi_cv, syncopation_degree, syncopation_ratio


def add_metrics_from_txt(characteristics_df: pd.DataFrame,
                         txt_file_path: Path | str,
                         ) -> pd.DataFrame:
    controller = SpotifyController(txt_file_path)

    # iterate over all categories:
    titles = []; artists = []; bpms = []; genres = []; file_titles = []; categories = []
    for category in controller.category_url_dict.keys():
        # iterate over all songs from categories:
        for song_ind in range(len(controller.category_url_dict[category])):
            controller.play_next_from(category)
            time.sleep(1)  # wait for song to start

            # get and append information:
            characteristics_dict = controller.get_current_track()
            print(characteristics_dict)
            titles.append(characteristics_dict["Title"])
            artists.append(characteristics_dict["Artist"])
            bpms.append(characteristics_dict["BPM"])
            genres.append(characteristics_dict["Genre"])
            file_titles.append(characteristics_dict["File Title"])

            # append category:
            categories.append(category)

    new_df = pd.DataFrame(index=file_titles, data={"Title": titles, "Artist": artists, "BPM": bpms})

    return characteristics_df.join(new_df, how="inner", rsuffix="_manual")







if __name__ == '__main__':
    # load audio:
    ROOT = Path().resolve().parent.parent
    AUDIO_DIR = ROOT / 'data' / 'songs'
    RESULT_DIR = ROOT / 'data' / 'song_characteristics'
    filemgmt.assert_dir(RESULT_DIR)
    AUDIO_CONFIG = ROOT / "config" / "music_selection.txt"
    PLOT_DIR = ROOT / "output" / "plots" / "music_characteristics"  # set to None if no plot saving desired
    filemgmt.assert_dir(PLOT_DIR)

    # which parts to execute:
    ### Step 1:
    compute_metrics: bool = True  # compute music metrics (BPM, Syncopation, Spectral Flux, ...)
    ### Step 2:
    extend_metrics_from_txt: bool = True  # extend computed metrics by Song Title, Artist & Manual BPM



    ### Step 3:
    # single songs to add to existing look up table:
    # below structure [(filepath, Artist, Title, Category, Genre, Spotify URL, Start After), ...]:
    single_files_to_add: list[tuple[str, str, str, str, str, str, float]] = [
        # these two were accidentally played but should be included in the analysis:
        ("Soulsearcher - Can't Get Enough (Jazz N Groove Nu Disco Vocal) - Defected Records.mp3", "Soulsearcher", "Can't Get Enough (Jazz N Groove Nu Disco Vocal)", "Unfamiliar Groovy", "Disco House", "https://open.spotify.com/intl-de/track/13qX3v31O0UBg59v3BC6fU?si=e13034bae8fa4959", 0),
        ("L'autre valse d'Amélie - Yann Tiersen.mp3", "Yann Tiersen",  "L'autre valse d'Amélie", "Unfamiliar Classic", "Film Music", "https://open.spotify.com/intl-de/track/3f0PZlTzwkqS1CJXvEKrHc?si=0e50255c65c34c2c", 0),
    ]


    # optional steps:
    cluster_results: bool = False
    compute_mutual_information: bool = False  # analyse relation music features -> genres / categories
    plot_scatters: bool = False  # plot feature distribution across categories





    if compute_metrics:
        # initialise spotify controller:
        controller = SpotifyController(AUDIO_CONFIG)
        # we'll iterate over controller.category_url_dict to compute the characteristics

        # feature lists:
        categories = []
        genres = []
        spotify_urls = []
        start_afters = []
        file_names = []
        bpm_list = []
        spectral_flux_mins = []
        spectral_flux_maxs = []
        spectral_flux_means = []
        spectral_flux_stds = []
        spectral_centroid_mins = []
        spectral_centroid_maxs = []
        spectral_centroid_means = []
        ioi_var_coeffs = []
        syncopation_degrees = []
        syncopation_ratios = []

        # iterate over all defined songs and compute characteristics:
        for category, song_list in controller.category_url_dict.items():
            for genre, spotify_url, start_after, bpm, audio_file_title in tqdm(song_list, desc=f'Computing Metrics for Category {category}'):
                # sanity checks
                if audio_file_title.startswith('.'): continue
                audio_path = AUDIO_DIR / audio_file_title

                ### compute musical features:
                bpm_from_beats, spectral_flux_normalized, spectral_centroid, ioi_cv, syncopation_degree, syncopation_ratio = compute_all_musical_features(
                    audio_path, verbose=False)

                ### save data:
                # controller data:
                categories.append(category)
                genres.append(genre)
                spotify_urls.append(spotify_url)
                start_afters.append(start_after)
                if abs(bpm_from_beats - bpm) > .5: print(
                    f"Mismatch between calculated bpm ({bpm_from_beats:.2f}) and defined bpm ({bpm:.2f}) larger than .5! (song: {audio_file_title})")

                # computed data:
                file_names.append(audio_file_title)
                bpm_list.append(bpm_from_beats)
                spectral_flux_mins.append(spectral_flux_normalized.min())
                spectral_flux_maxs.append(spectral_flux_normalized.max())
                spectral_flux_means.append(spectral_flux_normalized.mean())
                spectral_flux_stds.append(spectral_flux_normalized.std())
                spectral_centroid_mins.append(spectral_centroid.min())
                spectral_centroid_maxs.append(spectral_centroid.max())
                spectral_centroid_means.append(spectral_centroid.mean())
                ioi_var_coeffs.append(ioi_cv)
                syncopation_degrees.append(syncopation_degree)  # (0 = strictly on-beat, 1 = highly syncopated)
                syncopation_ratios.append(syncopation_ratio)  # syncopated onsets (>0.2 beat away)


        # save as dataframe and export:
        frame = pd.DataFrame(index=file_names, data={
            'Category': categories,
            'Genre': genres,
            'Spotify URL': spotify_urls,
            'Intended Start [sec]': start_afters,
            'BPM': bpm_list,
            'Spectral Flux Min.': spectral_flux_mins,
            'Spectral Flux Max.': spectral_flux_maxs,
            'Spectral Flux Mean': spectral_flux_means,
            'Spectral Flux Std.': spectral_flux_stds,
            'Spectral Centroid Min': spectral_centroid_mins,
            'Spectral Centroid Max': spectral_centroid_maxs,
            'Spectral Centroid Mean': spectral_centroid_means,
            'IOI Variance Coeff': ioi_var_coeffs,
            'Syncopation Degree': syncopation_degrees,
            'Syncopation Ratio': syncopation_ratios,
        })
        print(frame)
        frame.to_csv(RESULT_DIR / filemgmt.file_title("Song Characteristic Lookup Table", ".csv"))

    else:  # load last results
        frame = pd.read_csv(filemgmt.most_recent_file(RESULT_DIR, ".csv", ["Song Characteristic Lookup Table"]))
        print(f"Imported music characteristics for {len(frame)} songs")






    ### EXTEND METRICS
    if extend_metrics_from_txt:
        previous_frame = pd.read_csv(filemgmt.most_recent_file(RESULT_DIR, ".csv", ["Song Characteristic Lookup Table"]))
        previous_frame.set_index("Unnamed: 0", inplace=True)
        print(previous_frame)
        new_frame = add_metrics_from_txt(previous_frame, AUDIO_CONFIG)
        new_frame.to_csv(RESULT_DIR / filemgmt.file_title("Extended Song Characteristic Lookup Table", ".csv"))

        ### ADD NEW SINGLE ENTRIES
        if len(single_files_to_add) > 0:
            current_frame = pd.read_csv(
                filemgmt.most_recent_file(RESULT_DIR, ".csv", ["Extended Song Characteristic Lookup Table"]))
            current_frame.drop(columns=[col for col in current_frame.columns if 'Unnamed' in col], inplace=True)

            # feature lists:
            artists = []
            titles = []
            manual_bpms = []
            categories = []
            genres = []
            spotify_urls = []
            start_afters = []
            file_names = []
            bpm_list = []
            spectral_flux_mins = []
            spectral_flux_maxs = []
            spectral_flux_means = []
            spectral_flux_stds = []
            spectral_centroid_mins = []
            spectral_centroid_maxs = []
            spectral_centroid_means = []
            ioi_var_coeffs = []
            syncopation_degrees = []
            syncopation_ratios = []

            for audio_file_title, artist, title, category, genre, spotify_url, start_after in single_files_to_add:
                bpm_from_beats, spectral_flux_normalized, spectral_centroid, ioi_cv, syncopation_degree, syncopation_ratio = compute_all_musical_features(
                    AUDIO_DIR / audio_file_title, verbose=False)

                artists.append(artist)
                titles.append(title)
                categories.append(category)
                genres.append(genre)
                spotify_urls.append(spotify_url)
                start_afters.append(start_after)

                # computed data:
                file_names.append(audio_file_title)
                bpm_list.append(bpm_from_beats)
                manual_bpms.append(bpm_from_beats)
                spectral_flux_mins.append(spectral_flux_normalized.min())
                spectral_flux_maxs.append(spectral_flux_normalized.max())
                spectral_flux_means.append(spectral_flux_normalized.mean())
                spectral_flux_stds.append(spectral_flux_normalized.std())
                spectral_centroid_mins.append(spectral_centroid.min())
                spectral_centroid_maxs.append(spectral_centroid.max())
                spectral_centroid_means.append(spectral_centroid.mean())
                ioi_var_coeffs.append(ioi_cv)
                syncopation_degrees.append(syncopation_degree)  # (0 = strictly on-beat, 1 = highly syncopated)
                syncopation_ratios.append(syncopation_ratio)  # syncopated onsets (>0.2 beat away)

            # create new rows:
            new_rows = pd.DataFrame(index=file_names, data={
                'Category': categories,
                'Genre': genres,
                'Spotify URL': spotify_urls,
                'Intended Start [sec]': start_afters,
                'BPM': bpm_list,
                'Spectral Flux Min.': spectral_flux_mins,
                'Spectral Flux Max.': spectral_flux_maxs,
                'Spectral Flux Mean': spectral_flux_means,
                'Spectral Flux Std.': spectral_flux_stds,
                'Spectral Centroid Min': spectral_centroid_mins,
                'Spectral Centroid Max': spectral_centroid_maxs,
                'Spectral Centroid Mean': spectral_centroid_means,
                'IOI Variance Coeff': ioi_var_coeffs,
                'Syncopation Degree': syncopation_degrees,
                'Syncopation Ratio': syncopation_ratios,
                'Title': titles,
                'Artist': artists,
                'BPM_manual': manual_bpms,
            })

            # and append
            new_frame = pd.concat([current_frame, new_rows], ignore_index=True)

            new_frame.to_csv(RESULT_DIR / filemgmt.file_title("Extended Song Characteristic Lookup Table", ".csv"),
                             index=False)




    ### CLUSTERING
    if cluster_results:
        # select characteristics:
        feature_df = frame.drop(columns=["Unnamed: 0", "Category", "Genre", "Spotify URL", "Intended Start [sec]", ])
        feature_array = feature_df.to_numpy()
        feature_labels = feature_df.columns
        print("Feature indices: ", list(enumerate(feature_labels)))
        categories = frame['Category'].to_list()
        genres = frame['Genre'].to_list()

        # standardize:
        scaler = StandardScaler()
        standardized_feature_array = scaler.fit_transform(feature_array)

    if cluster_results or compute_mutual_information:
        # parameters:
        visualize_umap: bool = True  # overwrites the below two
        plot_centroids: bool = True
        k = 4
        x_feature_ind: int = 0
        y_feature_ind: int = 3

        # compute k-means:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X=standardized_feature_array)
        cluster_labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        # prepare plots:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_title("K-Means UMAP Visualization")

        # scatterplot:
        if visualize_umap:
            umap_reducer = umap.UMAP(n_components=2, random_state=42)
            feature_embedding = umap_reducer.fit_transform(standardized_feature_array)
            centroid_embedding = umap_reducer.transform(centroids)

        # scatterplot:
        feature_x = feature_embedding[:, 0] if visualize_umap else standardized_feature_array[:, x_feature_ind]
        feature_y = feature_embedding[:, 1] if visualize_umap else standardized_feature_array[:, y_feature_ind]
        sc = ax.scatter(feature_x, feature_y, c=cluster_labels, cmap='Set1', s=15)

        if plot_centroids:
            centroid_x = centroid_embedding[:, 0] if visualize_umap else centroids[:, x_feature_ind]
            centroid_y = centroid_embedding[:, 1] if visualize_umap else centroids[:, x_feature_ind]
            sc_centroid = ax.scatter(centroid_x, centroid_y, c='black', marker='x', s=150, linewidths=3,
                       label='Centroids')

        # legend:
        handles, lab_vals = sc.legend_elements()  # unique values from c
        lab_vals = [f"Cluster {ind}" for ind in lab_vals]
        if plot_centroids:
            handles += [sc_centroid]
            lab_vals += ['Centroids']

            x_min = min(np.min(feature_x), np.min(centroid_x))
            x_max = max(np.max(feature_x), np.max(centroid_x))
            ax.set_xlim(x_min - .05 * (x_max - x_min), x_max + .05 * (x_max - x_min))
            y_min = min(np.min(feature_y), np.min(centroid_y))
            y_max = max(np.max(feature_y), np.max(centroid_y))
            ax.set_ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))


        ax.legend(handles, lab_vals)# title=category_label)#  title="Class")
        ax.set_xlabel('UMAP 1' if visualize_umap else feature_labels[x_feature_ind])
        ax.set_ylabel('UMAP 2' if visualize_umap else feature_labels[y_feature_ind])
        plt.show()


    ### MUTUAL INFORMATION
    if compute_mutual_information:

        target_label: Literal['Genre', 'Category'] = 'Category'

        if target_label == 'Genre': target_array = genres
        elif target_label == 'Category':  # remove familiarity label
            target_array = [cat.replace("Unfamiliar ", "").replace("Familiar ", "") for cat in categories]

        from src.pipeline.signal_features import compute_feature_mi_importance
        _, _ , feature_importance = compute_feature_mi_importance(standardized_feature_array, target_array,
                                                                  feature_labels, target_label, plot_save_dir=PLOT_DIR,)
        print(feature_importance)


    if plot_scatters:
        import src.pipeline.visualizations as visualizations
        # x: spectral flux std, y: spectral flux mean
        x_ind = 4; y_ind = 3
        _ = visualizations.plot_scatter(x=feature_array[:, x_ind], y=feature_array[:, y_ind],
                                    x_label=feature_labels[x_ind], y_label=feature_labels[y_ind],
                                    category_list=target_array, category_label=target_label,
                                    cmap=['pink', 'red', 'green', 'blue'], save_dir=PLOT_DIR);

        # x: Spectral Centroid Mean, y: IOI Variance Coeff
        x_ind = 7; y_ind = 8
        _ = visualizations.plot_scatter(x=feature_array[:, x_ind], y=feature_array[:, y_ind],
                                    x_label=feature_labels[x_ind], y_label=feature_labels[y_ind],
                                    category_list=target_array, category_label=target_label,
                                    cmap=['pink', 'red', 'green', 'blue'], save_dir=PLOT_DIR);


        # x: Spectral Centroid Mean, y: BPM
        x_ind = 7; y_ind = 0
        _ = visualizations.plot_scatter(x=feature_array[:, x_ind], y=feature_array[:, y_ind],
                                    x_label=feature_labels[x_ind], y_label=feature_labels[y_ind],
                                    category_list=target_array, category_label=target_label,
                                    cmap=['pink', 'red', 'green', 'blue'], save_dir=PLOT_DIR);


        # x: Syncopation Ratio, y: IOI Variance Coeff
        x_ind = 10; y_ind = 8
        _ = visualizations.plot_scatter(x=feature_array[:, x_ind], y=feature_array[:, y_ind],
                                    x_label=feature_labels[x_ind], y_label=feature_labels[y_ind],
                                    category_list=target_array, category_label=target_label,
                                    cmap=['pink', 'red', 'green', 'blue'], save_dir=PLOT_DIR);

        # x: Syncopation Ratio, y: Syncopation Degree
        x_ind = 10; y_ind = 9
        _ = visualizations.plot_scatter(x=feature_array[:, x_ind], y=feature_array[:, y_ind],
                                    x_label=feature_labels[x_ind], y_label=feature_labels[y_ind],
                                    category_list=target_array, category_label=target_label,
                                    cmap=['pink', 'red', 'green', 'blue'], save_dir=PLOT_DIR);