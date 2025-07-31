import argparse
from NFACT.base.utils import colours


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
        prog="nfact_Qc",
        description=print(),
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
