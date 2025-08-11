import argparse
from NFACT.base.utils import colours, no_args
from NFACT.base.base_args import algo_arg, nfact_decomp_folder


def nfact_stats_args() -> dict:
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
    # col = colours()
    nfact_decomp_folder(args)
    algo_arg(args)
    no_args(args)
    return vars(args.parse_args())


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
