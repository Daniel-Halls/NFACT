from NFACT.base.setup import creat_subfolder_setup
from NFACT.base.utils import error_and_exit
import os


def create_nfact_dr_folder_set_up(nfact_path: str) -> None:
    """
    Function to create nfact dr folder set up

    Parameters
    ----------
    nfact_pat: str
        string to nfact directory

    Returns
    -------
    None
    """
    error_string = "Please provide a directory with --outdir"
    error_and_exit(nfact_path, f"No output directory given. {error_string}")
    error_and_exit(
        os.path.exists(nfact_path), f"Output directory does not exist. {error_string}"
    )
    subfolders = ["logs", "ICA", "NMF", "ICA/normalised", "NMF/normalised"]
    nfactdr_directory = os.path.join(nfact_path, "nfact_dr")
    creat_subfolder_setup(nfactdr_directory, subfolders)
