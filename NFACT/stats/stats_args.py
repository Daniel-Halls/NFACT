import argparse
from NFACT.base.utils import colours, no_args, verbose_help_message
from NFACT.base.base_args import (
    algo_arg,
    nfact_decomp_folder,
    set_up_args,
    base_arguments,
)


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
    col = colours()
    base_arguments(args)
    set_up_args(args, col)
    stats_args = args.add_argument_group(
        f"{col['darker_pink']}Stats args{col['reset']}"
    )
    nfact_decomp_folder(stats_args)
    algo_arg(stats_args)
    no_args(args)
    options = args.parse_args()
    if options.verbose_help:
        verbose_help_message(args, "")
    return vars(options)


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
