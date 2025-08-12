import argparse
from NFACT.base.utils import colours, no_args
from NFACT.base.base_args import algo_arg, nfact_decomp_folder


def nfact_qc_args() -> dict:
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
        prog="nfact_Qc",
        description=print(nfact_Qc_splash()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    nfact_decomp_folder(args)
    algo_arg(args)

    args.add_argument(
        "-t",
        "--threshold",
        dest="threshold",
        default=2,
        help="""
        Threshold value for z scoring the number of times
        a component comes up in a voxel in the image.
        Values below this z score are treated as noise and 
        discarded in the non raw image. 
        """,
    )
    args.add_argument(
        "-O",
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwite previous QC",
    )
    no_args(args)
    return vars(args.parse_args())


def nfact_Qc_splash() -> str:
    """
    Function to return NFACT splash

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
 _   _ ______   ___   _____  _____     ___     ____ 
| \ | ||  ___| / _ \ /  __ \|_   _|   / _ \   / ___|
|  \| || |_   / /_\ \| /  \/  | |    | | | | | | 
| . ` ||  _|  |  _  || |      | |    | | | | | |    
| |\  || |    | | | || \__/\  | |    | |_| | | |___ 
\_| \_/\_|    \_| |_/ \____/  \_/     \__\_\  \____|
{col["reset"]} 
"""
