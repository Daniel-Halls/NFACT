from NFACT.preprocess.nfactpp_args import nfact_pp_args
from NFACT.decomp.setup.args import nfact_decomp_args
from NFACT.dual_reg.nfact_dr_args import nfactdr_args
from NFACT.qc.nfactQc_args import nfact_qc_args
from NFACT.base.base_args import cluster_args
import inspect
import re


def get_function_output(function: object) -> dict:
    """
    Function to get the argument and its default
    values in dictionary form. If no defaults
    present then it will default to false.

    Parameters
    ----------
    function: object
        function object (nfact arg) to
        read argument from

    Returns
    -------
    dict: dictionary object
        dict of arg and default.
    """

    source_code = inspect.getsource(function)
    pattern = re.compile(
        r'\.add_argument\([^)]*?dest=["\']([a-zA-Z0-9_]+)["\']'
        r"(?:[^)]*?default\s*=\s*([^\s,\)]+))?"
        r"(?:[^)]*?help\s*=\s*([^\)]+))?",
        re.DOTALL,
    )

    matches = pattern.findall(source_code)
    return {
        dest: (
            "Required"
            if "REQUIRED FOR ALL" in help_text or "REQUIRED:" in help_text
            else (default.strip('"') if default else False)
        )
        for dest, default, help_text in matches
    }


def get_nfact_arguments() -> dict:
    """
    Function to get nfact_pp, nfact_decomp
    and nfact_dr args into a dictionary.

    Parameters
    ----------
    None

    Returns
    -------
    dict: dictionary object
        dict of module iwth arg, default pair.
    """
    return {
        "cluster": get_function_output(cluster_args),
        "nfact_pp": get_function_output(nfact_pp_args),
        "nfact_decomp": get_function_output(nfact_decomp_args),
        "nfact_dr": get_function_output(nfactdr_args),
        "nfact_qc": get_function_output(nfact_qc_args),
    }


def convert_str_to_bool(val) -> any:
    """
    Converts string instances of 'True' and 'False'

    Parameters
    ----------
    val: str
       string value

    Returns
    -------
    any: mixed
       either True, False
       or the original string
       type.

    """
    try:
        if val.lower() == "true":
            return True
        if val.lower() == "false":
            return False
    except AttributeError:
        return val
    return val


def process_dictionary_arguments(dictionary_to_process: dict) -> dict:
    """
    Clean str instances of bool to
    actual bool type

    Parameters
    ----------
    dictionary_to_process : dict
        The dictionary to process.

    Returns
    -------
    dict
        The dictionary with strings
        'True' and 'False' converted to bool
        type.
    """
    return {
        key: (
            process_dictionary_arguments(value)
            if isinstance(value, dict)
            else convert_str_to_bool(value)
        )
        for key, value in dictionary_to_process.items()
    }
