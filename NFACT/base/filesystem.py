import os
import json
from datetime import datetime
import shutil
from NFACT.base.utils import error_and_exit, colours


def delete_folder(outdir: str) -> None:
    """
    Function to delete folder.

    Parameters
    ----------
    outdir: str
        string of directory to delete

    Returns
    -------
    None
    """
    col = colours()
    if os.path.exists(outdir):
        print(
            f"{col['red']}Overwrite flag given. {outdir} directory being overwritten{col['reset']}\n"
        )
        shutil.rmtree(outdir, ignore_errors=True)


def make_directory(
    path: str, overwrite: bool = False, ignore_errors: bool = False
) -> None:
    """
    Function to make a directory.
    If error it will exit

    Parameters
    ----------
    path: str
        string to directory path
    overwrite: bool
        overwrite any previous directories

    Returns
    -------
    None
    """

    try:
        if os.path.exists(path) and overwrite:
            shutil.rmtree(path, ignore_errors=True)
        os.mkdir(path)
    except Exception as e:
        if ignore_errors:
            return None
        error_and_exit(False, f"Unable to create directory due to {e}")


def read_file_to_list(filename: str) -> list:
    """
    Function to dump output of file to
    list format.

    Parameters
    ----------
    filename: str
        path to file

    Returns
    -------
    list: list of subjects
        list of path to subjects directories
    """

    with open(filename, "r") as file:
        lines = file.readlines()
    return [sub.rstrip() for sub in lines]


def load_json(path: str) -> dict:
    """
    Function to load json

    Parameters
    ----------
    path: str
       path to json file

    Returns
    -------
    dict: dictionary file
        json file
    """
    try:
        with open(path) as config:
            return json.load(config)
    except Exception as e:
        error_and_exit(False, f"Unable to load json due to {e}")


def write_to_file(
    file_path: str, name: str, text: str, text_is_list: bool = False
) -> bool:
    """
    Function to write to file.

    Parameters
    ----------
    file_path: str
        abosulte file path to
        where file is created
    name: str
        name of file
    text: str
        string to add to file
    text_is_list: bool
        if text is actually a
        list then will write to file

    Returns
    -------
    bool: boolean
        True if sucessful else
        False
    """
    try:
        with open(f"{file_path}/{name}", "w") as file:
            if text_is_list:
                file.writelines(text)
            if not text_is_list:
                file.write(text)
    except Exception as e:
        print(f"Unable to write to {file_path}/{name} due to :", e)
        return False
    return True


def get_current_date() -> str:
    """
    Function to get the
    date and time in format
    useful for a file name.

    Parameters
    ----------
    None

    Returns
    -------
    str: string
        string of datetime object
    """
    now = datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")
