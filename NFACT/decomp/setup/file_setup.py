from NFACT.base.setup import creat_subfolder_setup
from NFACT.base.utils import error_and_exit
import os
import shutil


def create_folder_set_up(directory: str) -> None:
    """
    Function to create NFACT directory set up

    Parameters
    ----------
    directory: str
        path to directory to save output
        in.

    Returns
    -------
    None
    """
    error_and_exit(
        os.path.exists(directory),
        "Output directory does not exist. Please provide actual directory",
    )

    nfact_directory = os.path.join(directory, "nfact_decomp")

    sub_folders = [
        "group_averages",
        "logs",
        "components",
        "files",
        "components/NMF",
        "components/ICA",
        "components/NMF/decomp",
        "components/ICA/decomp",
        "components/ICA/normalised",
        "components/NMF/normalised",
        "components/ICA/WTA",
        "components/NMF/WTA",
    ]
    creat_subfolder_setup(nfact_directory, sub_folders)


def get_group_average_files(file_directory: str, nfact_directory: str) -> None:
    """
    Function to move lookup_tractspace_fdt_matrix2 and
    coords_for_fdt_matrix2 of a subject to a the group
    average folder.

    Parameters
    ----------
    file_directory: str
        file directory of a omatrix2 file
    nfact_direcotry: str
        string of path to group_averages nfact
        directory

    Returns
    -------
    None
    """
    lookup_space = os.path.join(file_directory, "lookup_tractspace_fdt_matrix2.nii.gz")
    coords_for_fdt_matrix2 = os.path.join(file_directory, "coords_for_fdt_matrix2")
    tract_space_for_fdt_matrix2 = os.path.join(
        file_directory, "tract_space_coords_for_fdt_matrix2"
    )

    [
        error_and_exit(
            os.path.exists(file), f"{file} does not exist. Please check pre-processing"
        )
        for file in [lookup_space, coords_for_fdt_matrix2]
    ]
    shutil.copyfile(
        lookup_space,
        os.path.join(nfact_directory, "lookup_tractspace_fdt_matrix2.nii.gz"),
    )
    shutil.copyfile(
        coords_for_fdt_matrix2, os.path.join(nfact_directory, "coords_for_fdt_matrix2")
    )
    shutil.copyfile(
        tract_space_for_fdt_matrix2,
        os.path.join(nfact_directory, "tract_space_coords_for_fdt_matrix2"),
    )
