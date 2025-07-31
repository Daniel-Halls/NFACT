import argparse
from NFACT.base.utils import colours, no_args
from NFACT.base.base_args import algo_arg,

def nfact_dr_args() -> dict:
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
        prog="nfact_PD",
        description=print(nfact_pd_splash()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    col = colours()
    args.add_argument(
        "-n",
        "--nfact_folder",
        dest="nfact_folder",
        help=f"""{col["red"]}REQUIRED:{col["reset"]} 
        Absolute path to nfact_decomp output folder.
        nfact_Qc folder is also saved within this
        folder.
        """,
    )
    algo_arg(args)
    no_args(args)
    return vars(args.parse_args())



def nfact_pd_splash() -> str:
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
 _   _ ______   ___   _____  _____  ______ ______ 
| \ | ||  ___| / _ \ /  __ \|_   _| | ___ \|  _  \\
|  \| || |_   / /_\ \| /  \/  | |   | |_/ /| | | |
| . ` ||  _|  |  _  || |      | |   |  __/ | | | |
| |\  || |    | | | || \__/\  | |   | |    | |/ / 
\_| \_/\_|    \_| |_/ \____/  \_/   \_|    |___/  
{col["reset"]} 
"""