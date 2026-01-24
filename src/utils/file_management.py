from datetime import datetime
import os
import json
import numpy as np
from pathlib import Path
from typing import Union, Literal


def file_title(title: str, dtype_suffix=".svg", short=False):
    '''
    Creates a file title containing the current time and a data-type suffix.

    Parameters
    ----------
    title: string
            File title to be used
    dtype_suffix: (default is ".svg") string
            Suffix determining the file type.
    short: boolean (default is False)
            If True, doesn't add h_min_sec to title (only Y-m-d).
    Returns
    -------
    file_title: string
            String to be used as the file title.
    '''
    if short:
        return datetime.now().strftime('%Y%m%d') + " " + title + dtype_suffix
    else:
        return datetime.now().strftime('%Y-%m-%d %H_%M_%S') + " " + title + dtype_suffix


def most_recent_file(directory: Path | str,
                     suffix_to_consider: str | None = None,
                     file_title_keywords: list[str] | None = None,
                     search_by: Literal["file-title", "meta-data"] = "file-title",
                     return_type: Literal["dict", "latest_file_path"] = "latest_file_path") -> Path | dict:
    """
    Find most recent file(s) in directory by filename date or modification time.

    search_by='file-title' works only with file-titles starting with YYYY-MM-DD HH_MM_SS

    Args:
        directory: Directory path to search
        suffix_to_consider: File suffix filter (e.g., '.csv')
        file_title_keywords: Keywords that must appear in filename
        search_by: 'file-title' uses date in filename, 'meta-data' uses file modification time
        return_type: 'latest_file_path' returns Path, 'dict' returns sorted files with dates

    Returns:
        - "latest_file_path": Path to most recent file
        - "dict": {"files": [sorted Path objects], "dates": [corresponding dates]} sorted by date descending

    Raises:
        ValueError: If provided path is not a directory or no files match criteria
    """
    # Validate inputs
    if search_by not in ("file-title", "meta-data"):
        raise ValueError(f"search_by must be 'file-title' or 'meta-data', got {search_by}")

    # Convert to Path and validate directory
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Provided path {directory} is not a directory!")

    # Normalize file_title_keywords to list once
    if file_title_keywords is not None:
        if isinstance(file_title_keywords, str):
            file_title_keywords = [file_title_keywords]

    # Collect files and dates
    file_list = []
    date_list = []

    for entry in os.scandir(directory):
        if not entry.is_file():
            continue

        filename = entry.name

        # Check suffix
        if suffix_to_consider is not None:
            if not filename.endswith(suffix_to_consider):
                continue
        elif '.DS_Store' in filename:
            continue

        # Check keywords
        if file_title_keywords is not None:
            if not all(keyword in filename for keyword in file_title_keywords):
                continue

        file_path = directory / filename

        # Extract date based on search mode
        if search_by == "file-title":
            try:
                din_datestring = filename[:10]
                din_timestring = filename[11:19].replace('_', ':')
                date = datetime.fromisoformat(din_datestring + ' ' + din_timestring)
            except (ValueError, IndexError):
                # Skip files with invalid date format
                continue
        else:  # meta-data
            date = entry.stat().st_mtime  # seconds since epoch as float

        file_list.append(file_path)
        date_list.append(date)

    # Validate we found files
    if not file_list:
        raise ValueError("Provided directory doesn't contain files matching the provided criteria!")

    # Sort by date descending
    sorted_indices = np.argsort(date_list)[::-1]
    sorted_files = [file_list[i] for i in sorted_indices]
    sorted_dates = [date_list[i] for i in sorted_indices]

    # Return based on return_type
    if return_type == "latest_file_path":
        return sorted_files[0]
    else:  # dict
        return {
            "files": sorted_files,
            "dates": sorted_dates
        }


def assert_dir(dir_path: str | Path):
    """ Ensure that dir is present, else create. """
    Path(dir_path).mkdir(parents=True, exist_ok=True)

class TxtConfig:
    def __init__(self,
                 txt_file_path: Union[Path, str],
                 read_only_mode: bool = True,):
        self.txt_file_path = txt_file_path
        self.read_only_mode = read_only_mode

    @property
    def settings_dict(self) -> dict:
        temp_dict = {}
        with open(self.txt_file_path, "r") as file:
            for line in file:
                line.strip()

                # skip comments and empty lines:
                if line.startswith("#"): continue
                if line == "\n": continue

                # read entry and sanity check:
                entry = line.split(' --- ')
                if len(entry) != 2:
                    raise ValueError(f"Provided file may only contain lines structured as 'PROPERTY_NAME --- ENTRY'!\nFound line: >> {line} <<")

                # save entry:
                temp_dict[entry[0].strip()] = entry[1].strip()
        return temp_dict

    def change_entry(self, entry, new_entry):
        if self.read_only_mode: raise ValueError("TxtConfig is in read-only mode, hence cannot modify entry!")
        temp_dict = self.settings_dict
        temp_dict[entry] = new_entry
        self._set_dict_to_file(temp_dict)

    def _set_dict_to_file(self, new_dict):
        if self.read_only_mode: raise ValueError("TxtConfig is in read-only mode, hence cannot modify entry!")
        # overwrite txt file with current dictionary content
        with open(self.txt_file_path, "w") as file:
            file.write("# This file was changed during runtime.\n# The structure is 'PROPERTY_NAME --- ENTRY'. Lines starting with '#' are ignored.\n")
            for key, value in new_dict.items():
                if isinstance(value, list):
                    value = ", ".join([str(e) for e in value])  # list formatting (can be retrieved via get_as_type)
                file.write(f"{str(key)} --- {str(value)}\n")

    def get_as_type(self, key, value_type: Literal["int", "float", "float_list", "str_list", "list", "bool", "str"]):
        value = self.settings_dict[key]  # retrieve value

        # format and return value:
        if value_type == "int": return int(value)
        elif value_type == "float": return float(value)
        elif value_type == "bool": return value == 'True' or value == '1'
        elif value_type == "str": return str(value)
        elif value_type == "float_list":
            entries = value.split(', ')
            if len(entries) <= 1: raise ValueError("List entries need to be formatted as 'ENTRY_1, ENTRY_2, ENTRY_3, ...'!")
            return [float(e) for e in entries]
        elif value_type == "str_list" or value_type == "list":
            entries = value.split(', ')
            if len(entries) <= 1: raise ValueError("List entries need to be formatted as 'ENTRY_1, ENTRY_2, ENTRY_3, ...'!")
            return entries
        else:
            raise ValueError(f"Provided value type '{value_type}' is not recognized!")


def fetch_json_recursively(dir: str | Path,
                           file_identifier: str,
                           value_key: str,
                           with_time_from_file_title: bool = False) -> list | dict:
    """Recursively retrieve all values under a key (value_key) from JSON files matching file_identifier in all subdirectories of dir."""
    if not isinstance(dir, Path): dir = Path(dir)

    values = {} if with_time_from_file_title else []  # Collect all matches

    for item in dir.iterdir():
        if item.is_dir():
            # recurse into subdirectory and ADD results:
            if with_time_from_file_title:
                values.update(fetch_json_recursively(item, file_identifier, value_key, with_time_from_file_title))
            else:
                values.extend(fetch_json_recursively(item, file_identifier, value_key, with_time_from_file_title))

        elif item.is_file():
            # check if it's a matching JSON file:
            name_without_ext = item.stem  # e.g. "Trial Summary" from "Trial Summary.json"
            if file_identifier in name_without_ext and item.suffix == '.json':
                try:
                    with open(item, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    value = data[value_key]

                    # either process with or without timestamp
                    if with_time_from_file_title:
                        day = name_without_ext.split(" ")[0]
                        time = name_without_ext.split(" ")[1]
                        time_str = f"{day} {time}"

                        values[time_str] = value  # add to dict

                    else:
                        values.append(value)  # add to list

                except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                    print(f"Warning: Could not read value from {item}: {e}")

    return values