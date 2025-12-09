import subprocess  # allows to EXTERNALLY (<-> multiprocessing) run shell commands or other system-level programs
from typing import Literal
import random
from pathlib import Path

from src.utils.str_conversion import str_to_float

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
        self.current_category_and_counter: tuple[str, int] | None = None
        self.current_genre = None  # will be updated if new song is started

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
        def detect_song_data(line: str) -> tuple[str, str] | tuple[str, str, float]:
            """ Detect genre, song_url and start_after_seconds from read line. """
            elements = line.split(" --- ")
            if len(elements) == 2:  # no start_after_provided
                return elements[0].strip(), elements[1].strip()
            elif len(elements) == 3:
                return elements[0].strip(), elements[1].strip(), float(elements[2].strip())
            else: raise ValueError("Invalid line detected: {}".format(line))

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
            "Instance attribute category_url_dict needs to be defined!")
        # reset current_category and counter if first execution or new category:
        if self.current_category_and_counter is None:
            self.current_category_and_counter = (category, 0)
        elif self.current_category_and_counter[0] != category:
            self.current_category_and_counter = (category, 0)
        # else increase counter:
        else:
            self.current_category_and_counter = (category, self.current_category_and_counter[1] + 1)

        # derive next track and when to start it:
        try:
            song_tuple = self.category_url_dict[self.current_category_and_counter[0]][self.current_category_and_counter[1]]
        except IndexError:
            print("No new songs left in category! Starting over.")
            self.current_category_and_counter = (category, 0)
            song_tuple = self.category_url_dict[self.current_category_and_counter[0]][
                self.current_category_and_counter[1]]

        # try detect start_after:
        if len(song_tuple) == 3: start_at = song_tuple[2]
        else: start_at = None
        self.current_genre = song_tuple[0]
        next_track_url = song_tuple[1]

        # play:
        self.play_track(next_track_url); print(f"Playing {next_track_url} (number {self.current_category_and_counter[1]} in category {category})");
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
                return output.decode().strip() + " | " + self.current_genre
            else:
                #print(output.decode())
                title, artist, album, duration_ms, position_s = output.decode().strip().split(' | ')
                position_s = str_to_float(position_s, is_ger_format=("." not in position_s))
                duration_ms = str_to_float(duration_ms,  is_ger_format=("." not in duration_ms))
                return {'Title': title, 'Artist': artist, 'Album': album, 'Duration [ms]': duration_ms,
                        'Position [s]': position_s, 'Genre': self.current_genre}
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

if __name__ == '__main__':
    temp_instance = SpotifyController()
    temp_instance.play_track("https://open.spotify.com/intl-de/track/0n4bITAu0Y0nigrz3MFJMb?si=6ce2542696c644b7")