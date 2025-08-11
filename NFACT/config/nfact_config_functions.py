from NFACT.base.utils import colours, error_and_exit, no_args
from NFACT.base.filesystem import write_to_file
from NFACT.base.config import get_nfact_arguments, process_dictionary_arguments
import inspect
from sklearn.decomposition import FastICA, NMF
import json
import os
import argparse
import glob


def nfact_config_args() -> dict:
    """
    Function to define cmd arguments

    Parameters
    ----------
    None

    Returns
    -------
    dict: dictionary
        dictionary of cmd arguments
    """

    args = argparse.ArgumentParser(
        prog="nfact_config",
        description=print(nfact_config_spalsh()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    args.add_argument(
        "-C",
        "--config",
        dest="config",
        action="store_true",
        help="Creates a config file for NFACT pipeline",
    )
    args.add_argument(
        "-D",
        "--decomp_only",
        dest="decomp_only",
        action="store_true",
        help="Creates a config file for hyperparameters for the NMF/ICA",
    )
    args.add_argument(
        "-s",
        "--subject_list",
        dest="subject_list",
        default=False,
        help="""
        Creates a subject list from a given directory
        Needs path to subjects directory. Depending on file path
        will dictate the subject list. If ran inside
        an nfact_pp directory will make a subject list 
        for decompoisition (adds on omatrix2 to file paths). If ran
        in nfact_dr it will get the G then W files for subjects
        """,
    )
    args.add_argument(
        "-o",
        "--output_dir",
        dest="output_dir",
        default=os.getcwd(),
        help="File path of where to save config file",
    )
    args.add_argument(
        "-f",
        "--file_name",
        dest="file_name",
        default="nfact_config",
        help="""
        Name of the nfact config filename.
        Defaults is nfact_config
        """,
    )
    no_args(args)

    return vars(args.parse_args())


def nfact_config_spalsh() -> str:
    """
    Function to return NFACT config
    splash

    Parameters
    ----------
    None

    Returns
    -------
    str: splash
    """

    col = colours()
    return f"""
{col["pink"]} 
 _   _______ ___  _____ _____                    __ _       
| \ | |  ___/ _ \/  __ \_   _|                  / _(_)      
|  \| | |_ / /_\ \ /  \/ | |     ___ ___  _ __ | |_ _  __ _ 
| . ` |  _||  _  | |     | |    / __/ _ \| '_ \|  _| |/ _` |
| |\  | |  | | | | \__/\ | |   | (_| (_) | | | | | | | (_| |
\_| \_|_|  \_| |_/\____/ \_/    \___\___/|_| |_|_| |_|\__, |
                                                       __/ |
                                                      |___/ 
{col["reset"]} 
"""


def get_arguments(function: object) -> dict:
    """
    Function to get default arguments
    and put them into a dictionary object

    Parameters
    ----------
    function: object
       function to get arguments from

    Returns
    -------
    dict: dictionary object
        dict of functions
    """

    signature = inspect.signature(function)
    return dict(
        zip(
            [param.name for param in signature.parameters.values()],
            [
                param.default
                for param in signature.parameters.values()
                if param.default is not inspect.Parameter.empty
            ],
        )
    )


def process_global_arguments(args: dict) -> dict:
    """
    Function to process global arguments

    Parameters
    ----------
    args: dict
        command line arguments

    Returns
    -------
    args: dict
        dictionary of arguments with
        global arguments
    """
    args["global_input"] = {}
    args["global_input"]["list_of_subjects"] = "Required"
    args["global_input"]["outdir"] = "Required"
    args["global_input"]["seed"] = ["Required unless file_tree specified"]
    args["global_input"]["overwrite"] = False
    args["global_input"]["pp_skip"] = False
    args["global_input"]["dr_skip"] = False
    args["global_input"]["qc_skip"] = False
    return args


def reorganise_args(args: dict) -> dict:
    """
    Function to reorganise
    command line arguments
    removing duplicates
    such as sub list, output dir
    seed etc

    Parameters
    ----------
    args: dict
        command line arguments

    Returns
    -------
    dict: dictionary
        dictionary of arguments with
        reorganised

    """
    return {
        key: {
            top_key: value
            for top_key, value in sub_dict.items()
            if top_key
            not in ["list_of_subjects", "outdir", "seed", "overwrite", "verbose_help"]
            and not (
                (key == "nfact_decomp" and top_key in ["seeds"])
                or (
                    key == "nfact_dr"
                    and top_key in ["algo", "seeds", "nfact_decomp_dir", "decomp_dir"]
                )
            )
        }
        for key, sub_dict in args.items()
    }


def delete_keys(input_dict: dict, keys_to_delete: list) -> dict:
    """
    Function to delete keys
    from a dictionary.

    Parameters
    ----------
    input_dict: dict
        dictionary to change
    keys_to_delete: list
        list of keys

    Returns
    -------
    input_dict: dict
        processed inpur dict
    """
    for key in keys_to_delete:
        del input_dict[key]
    return input_dict


def move_key_to_front(dictionary: dict, dict_key: str) -> dict:
    """
    Function to move dictionary key
    to first place

    Parameters
    -----------
    dictionary: dict
        dictionary to re order
    dict_key: str
        string of key to re-order

    Returns
    -------
    dictionary: dict
        dictionary with
        reordered keys
    """
    if dict_key in dictionary:
        dictionary = {
            dict_key: dictionary[dict_key],
            **{key: value for key, value in dictionary.items() if key != dict_key},
        }
    return dictionary


def process_decomp_dictionary(decomp: dict) -> dict:
    """
    Function to process decomp dictionary.

    Parameters
    -----------
    decomp: dict
        decomp dictionary

    Returns
    -------
    decomp: dict
       decomp dictionary
       re-processed
    """
    decomp["algo"] = "NMF"
    decomp["roi"] = False
    decomp["dim"] = "Required"
    decomp = move_key_to_front(decomp, "roi")
    decomp = move_key_to_front(decomp, "algo")
    decomp = move_key_to_front(decomp, "dim")
    return decomp


def process_config_file(args: dict) -> dict:
    """
    Function to process arguments
    to remove duplicates and indicate
    the required arguments.

    Parameters
    ----------
    args: dict
       dictionary of arguments

    Returns
    -------
    args: dictionary
        processed dict of arguments
    """

    args = reorganise_args(args)
    args = process_global_arguments(args)
    args["global_input"]["folder_name"] = "nfact"
    args["nfact_pp"]["roi"] = []
    args["nfact_pp"]["warps"] = []
    args["nfact_pp"]["n_cores"] = False
    args["nfact_decomp"] = process_decomp_dictionary(args["nfact_decomp"])
    args["nfact_dr"]["n_cores"] = False
    args["nfact_qc"] = delete_keys(args["nfact_qc"], ["nfact_folder", "dim", "algo"])
    return {"global_input": args.pop("global_input"), **args}


def create_config() -> dict:
    """
    Function to create a config file.

    Parameters
    -----------
    None

    Returns
    -------
    dict: dictionary
        dict of all arguments
        needed for a config file

    """
    arguments = get_nfact_arguments()
    arguments = process_dictionary_arguments(arguments)
    return process_config_file(arguments)


def create_combined_algo_dict() -> dict:
    """
    Function to create a combined dictionary
    for nmf and ICA function arguments.

    Parameters
    ----------
    None

    Returns
    -------
    dict: dictionary object
        dict of ICA and NMF
        arguments
    """

    dictionary_to_save = {"ica": get_arguments(FastICA), "nmf": get_arguments(NMF)}
    del dictionary_to_save["nmf"]["n_components"]
    del dictionary_to_save["ica"]["n_components"]
    return dictionary_to_save


def save_to_json(path: str, dictionary_to_save: dict, file_name: str) -> None:
    """
    Function to save nfact config file
    to disk as a json.

    Parameters
    ----------
    path: str
       string of path of where to save the config
       file
    dictionary_to_save: dict
        dictionary of arguements

    Returns
    -------
    None
    """

    config_file = json.dumps(dictionary_to_save, indent=4)
    with open(os.path.join(path, file_name), "w") as json_file:
        json_file.write(config_file)


def check_arguments(args: dict) -> None:
    """
    Function to check arguments

    Parameters
    ----------
    args: dict
        dictionary of command line args

    Returns
    -------
    None
    """

    arg_type = ["decomp_only", "config", "subject_list"]
    matching = [arg for arg in args.keys() if arg in arg_type and args[arg]]
    if len(matching) > 1:
        error_and_exit(
            False,
            f"""Multiple arguments were given {matching}. 
Please specify either --decomp --config or --subject_list only""",
            False,
        )


def check_study_folder(study_folder_path: str) -> None:
    """
    Function to check that study directory
    is a directory with subjects in it it.

    Parameters
    ----------
    study_folder_path: str
        path to study directory

    Returns
    -------
    None
    """

    error_and_exit(
        os.path.exists(study_folder_path), "Study folder provided doesn't exist", False
    )
    error_and_exit(
        os.path.isdir(study_folder_path), "Study directory is not a directory", False
    )
    error_and_exit(
        [
            dirs
            for dirs in os.listdir(study_folder_path)
            if os.path.isdir(os.path.join(study_folder_path, dirs))
        ],
        "Study directory is empty",
        False,
    )


def list_of_subjects_from_directory(
    study_folder_path: str, is_dr: bool = False
) -> list:
    """
    Function to get list of subjects from a directory
    if a list of subjects is not given

    Parameters
    ---------
    study_folder_path: str
       path to study folder

    Returns
    -------
    list: list object
        list of subjects
    """

    list_of_subject = glob.glob(os.path.join(study_folder_path, "*"))
    if is_dr:
        return list_of_subject
    return [f"{direct}\n" for direct in list_of_subject if os.path.isdir(direct)]


def filter_sublist(sub_list: str) -> list:
    """
    Fucntion to filter sub list to
    remove directories that are unlikely to
    be

    Parameters
    ----------
    sub_list: list
        list of subjects

    Returns
    -------
    list: list object
        filtered list of subjects
    """

    remove_dir = [
        "analysis",
        "code",
        "sourcedata",
        "derivatives",
        "nfact_dr",
        ".",
        "nfact_pp",
        "nfact\n",
        "normalised",
    ]
    return [
        dirs
        for dirs in sub_list
        if not any(rem in os.path.basename(dirs).lower() for rem in remove_dir)
    ]


def nfact_study_list(study_folder_path: str) -> list:
    """
    Function to create a list of
    subject directories for nfact

    Parameters
    ----------
    study_folder_path: str
        path to study folder

    Returns
    -------
    sub_list: list
        list of subject filepaths
    """
    sub_list = list_of_subjects_from_directory(study_folder_path)
    if "nfact_pp" in study_folder_path:
        sub_list = [sub.rstrip("\n") + "/omatrix2\n" for sub in sub_list]
    return filter_sublist(sub_list)


def nfact_dr_study_list(study_folder_path: str) -> list:
    """
    Function to create a list of
    subject files for nfact_dr
    organises them into G and W

    Parameters
    ----------
    study_folder_path: str
        path to study folder

    Returns
    -------
    sub_list: list
        list of subject filepaths
    """
    sub_list = list_of_subjects_from_directory(study_folder_path, is_dr=True)
    filtered_list = [
        f"{subs}\n"
        for subs in sub_list
        if os.path.basename(subs)[0].upper() in ["G", "W"]
    ]
    return sorted(
        filtered_list,
        key=lambda ordering: (
            os.path.basename(ordering)[0] != "G",
            os.path.basename(ordering),
        ),
    )


def create_subject_list(study_folder_path: str, ouput_dir: str, filename: str) -> None:
    """
    Function to create subjects
    study list

    Parameters
    ---------
    study_folder_path: str
        path to study folder
    ouput_dir: str
        path to output directory
    filename: str
        filename of sublist

    Returns
    -------
    None
    """
    if study_folder_path == ".":
        study_folder_path = os.getcwd()
    check_study_folder(study_folder_path)
    if "nfact_dr" in study_folder_path:
        sub_list = nfact_dr_study_list(study_folder_path)
    else:
        sub_list = nfact_study_list(study_folder_path)
    write_to_file(ouput_dir, f"{filename}.sublist", sub_list, text_is_list=True)
