import argparse
from NFACT.base.utils import colours, no_args, verbose_help_message
from NFACT.base.base_args import (
    algo_arg,
    nfact_decomp_folder,
    set_up_args,
    base_arguments,
)


def nfact_stats_args() -> dict:
    parser = nfact_stats_modules()
    args = parser.parse_args()
    return vars(args)


def nfact_stats_modules() -> dict:
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
        prog="nfact_stats",
        description=print(nfact_stats_splash()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = args.add_subparsers(dest="command")
    comp_loading_args(subparsers)
    stat_map_args(subparsers)
    return args
    
    
def comp_loading_args(args: object):
    col = colours()
    comp_args = args.add_parser("loadings", help="Calculate component loadings")
    comp_args.add_argument(
        "-O",
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=False,
        help="Overwrites previous file structure",
    )
    set_up_args(comp_args, col)
    stats_args = comp_args.add_argument_group(
        f"{col['darker_pink']}Stats args{col['reset']}"
    )
    nfact_decomp_folder(set_up_args)
    algo_arg(set_up_args)
    stats_args.add_argument(
        "-C",
        "--no_csv",
        dest="no_csv",
        action="store_true",
        default=False,
        help="""
        Save Component Loadings as a npy file
        rather than as a csv file
        """,
    )

def stat_map_args(args) -> dict:
    """
    Function to get arguements
    to run NFACT pre-processing

    Parameters
    -----------
    None

    Returns
    -------
    dict: dictionary object
        dict of arguments
    """
    col = colours()
    stats_args = args.add_parser("statsmap", help="Create a statsmap")
    set_up_args(stats_args, col)
    stats_args.add_argument(
        "-f",
        "--folder_path",
        dest="folder_path",
        required=True,
        help="""
        Path to nfact directory. Statsmap only
        works if there is one nfact directory 
        """,
    )
    stats_args.add_argument(
        "-c",
        "--components",
        dest="components",
        required=True,
        type=int,
        nargs="+",
        help="""
        Components to merge
        """,
    )

def nfact_stats_base_args(args):


    #stats_args.add_argument(
    #    "-o",
    #    "--save_path",
    #    dest="save_path",
    #    required=True,
    #    help="""
    #    Path to save output as
    #    """,
    #)
    #stats_args.add_argument(
    #    "-l",
    #    "--list_of_subjects",
    #    dest="list_of_subjects",
    #    required=True,
    #    help="""
    #    List of Subjects to give order to 
    #    """,
#
    #)




def nfact_stats_splash() -> str:
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
 _   _ ______   ___   _____  _____   _____  _____   ___   _____  _____ 
| \ | ||  ___| / _ \ /  __ \|_   _| /  ___||_   _| / _ \ |_   _|/  ___|
|  \| || |_   / /_\ \| /  \/  | |   \ `--.   | |  / /_\ \  | |  \ `--. 
| . ` ||  _|  |  _  || |      | |    `--. \  | |  |  _  |  | |   `--. \\
| |\  || |    | | | || \__/\  | |   /\__/ /  | |  | | | |  | |  /\__/ /
\_| \_/\_|    \_| |_/ \____/  \_/   \____/   \_/  \_| |_/  \_/  \____/ 
{col["reset"]} 
"""