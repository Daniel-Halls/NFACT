from NFACT.base.imagehandling import (
    save_grey_matter_components,
    save_white_matter,
)
from NFACT.base.utils import colours, nprint, error_and_exit
import numpy as np
import os
import nibabel as nb
from glob import glob
import re


def get_key_to_organise_list(seed_path: str) -> str:
    """
    Function to get key to organise
    Function to get key to organise
    grey matter file names.

    Parameters
    ----------
    seed_path: str
        path to first seed
        path to first seed
        in the list

    Returns
    -------
    str: string
        string of key to organise
        grey matter list by
    """
    seed_name = os.path.basename(seed_path).lower()
    file_split = re.split(r"[.-]", seed_name)
    return next(
        (
            side
            for keys, side in {
                ("l", "left"),
                ("left", "left"),
                ("r", "right"),
                ("right", "right"),
            }
            if keys in file_split
        ),
        seed_name,
    )


def sort_grey_matter_order(grey_matter_list: list, keyword: str) -> list:
    """
    Function to organise grey matter
    Function to organise grey matter
    seed by a keyword. Does so on
    a partial match.

    Parameters
    ----------
    grey_matter_list: list
        list of grey matter
        list of grey matter
    keyword: str
        str of key to organise
        grey matter by


    Returns
    -------
    list: list object
        sorted grey_matter_list
        by keyword
        by keyword
    """
    return sorted(grey_matter_list, key=lambda sk: (keyword not in sk, sk))


def vol2mat(matvol: np.ndarray, lut_vol: np.ndarray) -> np.ndarray:
    """
    Function to reshape a volume back into
    the original matrix format.

    Parameters
    ----------
    matvol: np.ndarray
        array reformatted as a volume
    lut_vol: np.ndarray
        np.ndarray containing
        lookup volume data

    Returns
    -------
    matrix: np.ndarray
        array of volume data converted back
        to original matrix form
    """
    mask = lut_vol > 0
    num_rows = matvol.shape[-1]
    matrix = np.zeros((num_rows, np.max(lut_vol)))

    for row in range(num_rows):
        matrix[row, lut_vol[mask] - 1] = matvol.reshape(-1, num_rows)[
            mask.flatten(), row
        ]

    return matrix


def save_dual_regression_images(
    components: dict,
    nfact_path: str,
    seeds: list,
    algo: str,
    dim: int,
    sub: str,
    ptx_directory: str,
    roi: list,
) -> None:
    """
    Function to save regression images

    Parameters
    ----------
    components: dict
        dictionary of components
    nfact_path: str
        str to nfact directory
    seeds: list
        list of seeds
    algo: str
        str of algo
    dim: int
        number of dimensions
        used for naming output
    sub: str
        Subject id in string format
    ptx_dir: str
        needed to obtain coords/lookup
        tractspace
    roi: list
        list of roi. Needed
        for surfaces.

    Returns
    -------
    None
    """

    col = colours()
    for comp, _ in components.items():
        algo_path = algo
        w_file_name = f"W_{sub}_dim{dim}"
        grey_prefix = f"G_{sub}"

        if "normalised" in comp:
            algo_path = os.path.join(algo, "normalised")
            w_file_name = f"W_{sub}_norm_dim{dim}"
            grey_prefix = f"G_{sub}_norm"

        if "grey" in comp:
            nprint(f"{col['pink']}Image:{col['reset']} {comp}")
            save_grey_matter_components(
                components[comp],
                nfact_path,
                seeds,
                algo_path,
                dim,
                os.path.join(ptx_directory, "coords_for_fdt_matrix2"),
                roi,
                grey_prefix,
            )
        if "white" in comp:
            nprint(f"{col['pink']}Image:{col['reset']} {comp}")
            save_white_matter(
                components[comp],
                os.path.join(ptx_directory, "lookup_tractspace_fdt_matrix2.nii.gz"),
                os.path.join(nfact_path, algo_path, w_file_name),
            )


def white_component(component_dir: str, group_averages_dir: str) -> np.ndarray:
    """
    Function to get the group level
    white matter component for dual regression.

    Parameters
    ----------
    component_dir: str
        path to the saved components
    algo: str
        The algorithm for dual regression

    Returns
    -------
    np.darray: np.array
        array of white matter component
        from the volume
    """
    lookup_vol = nb.load(
        os.path.join(group_averages_dir, "lookup_tractspace_fdt_matrix2.nii.gz")
    )
    white_matter = nb.load(glob(os.path.join(component_dir, "W_*_dim*"))[0])
    return vol2mat(
        white_matter.get_fdata().astype(np.int32),
        lookup_vol.get_fdata().astype(np.int32),
    )


def load_grey_matter_volume(nifti_file: str, x_y_z_coordinates: np.array) -> np.array:
    """
    Function to load a grey matter NIfTI file and convert it
    back into a grey matter component matrix.

    Parameters
    ----------
    nifti_file: str
        Path to the grey matter NIfTI file
    x_y_z_coordinates: np.array
        Array of x, y, z coordinates

    Returns
    -------
    np.array
        Grey matter component matrix
    """
    img = nb.load(nifti_file)
    data = img.get_fdata()
    vol_shape = data.shape[:3]
    xyz_idx = np.ravel_multi_index(x_y_z_coordinates.T, vol_shape)
    ncols = data.shape[3] if len(data.shape) > 3 else 1
    flattened_data = data.reshape(-1, ncols)
    return flattened_data[xyz_idx, :]


def load_grey_matter_gifti_seed(file_name: str, roi: str) -> np.array:
    """
    Load grey matter component from a GIFTI file.

    Parameters
    ----------
    file_name: str
        Path to the GIFTI file.
    roi: str
        str to roi path

    Returns
    -------
    grey_matter_component: np.array
        Reconstructed grey matter component.
    """

    m_wall = nb.load(roi).darrays[0].data != 0
    gifti_img = nb.load(file_name)
    grey_component = np.column_stack([darray.data for darray in gifti_img.darrays])
    grey_component = grey_component[m_wall == 1, :]
    return grey_component


def grey_components(
    seeds: list, decomp_dir: str, group_averages: str, mw: list
) -> np.ndarray:
    """
    Function to get grey components.

    Parameters
    ----------
    seeds: list
        list of seeds paths
    decomp_dir: str
        str of absolute path to nfact directory
    group_averages_dir: str
        str to group averages directory
    mw: list
        list of wedial wall files

    Returns
    -------
    np.ndarray: np.array
        grey matter components array
    """
    grey_matter = glob(os.path.join(decomp_dir, "G_*dim*"))
    seed_key_word = get_key_to_organise_list(seeds[0])
    sorted_components = sort_grey_matter_order(grey_matter, seed_key_word)
    save_type = "gii" if "gii" in sorted_components[0] else "nii"

    if save_type == "nii":
        coord_file = np.loadtxt(
            os.path.join(group_averages, "coords_for_fdt_matrix2"),
            dtype=int,
        )
        return np.vstack(
            [
                load_grey_matter_volume(
                    seed, coord_file[coord_file[:, 3] == idx][:, :3]
                )
                for idx, seed in enumerate(sorted_components)
            ]
        )
    if save_type == "gii":
        return np.vstack(
            [
                load_grey_matter_gifti_seed(seed, mw[idx])
                for idx, seed in enumerate(sorted_components)
            ]
        )


def get_group_level_components(
    component_dir: str, group_averages_dir: str, seeds: list, mw: list
):
    """
    Function to get group level components

    Parameters
    ----------
    component_dir: str
        path to the component_dir
    group_averages_dir: str
        path to group averages directory
    seeds: list
        A list of seeds
    mw: list
        list of wedial wall files

    Returns
    -------
    dict: dictionary
        dict of components
    """
    return {
        "white_components": white_component(component_dir, group_averages_dir),
        "grey_components": grey_components(
            seeds, component_dir, group_averages_dir, mw
        ),
    }


def get_paths(args: dict) -> dict:
    """
    Function to return components
    path.

    Parameters
    ----------
    args: dict
        dictionary of command line
        arguments

    Returns
    -------
    str: string of component path.
    """
    if args["nfact_decomp_dir"]:
        return {
            "component_path": os.path.join(
                args["nfact_decomp_dir"], "components", args["algo"].upper(), "decomp"
            ),
            "group_average_path": os.path.join(
                args["nfact_decomp_dir"], "group_averages"
            ),
        }
    if args["decomp_dir"]:
        return {
            "component_path": args["decomp_dir"],
            "group_average_path": args["decomp_dir"],
        }

    error_and_exit(
        False,
        "Directory to components not given. Please specify with --nfact_decomp_dir or --decomp_dir",
    )


def get_subject_id(path: str, number: int) -> str:
    """
    Function to assign a subjects Id

    Parameters
    ----------
    path: str
        string of path to subjects
    number: int
        subject number

    Returns
    ------
    str: string
        subject id either taken from file path
        or assigned number in the list
    """
    try:
        stripped_path = re.sub(r"subjects", "", path)
        return re.findall(r"sub_[a-zA-Z0-9]*", stripped_path)[0]
    except IndexError:
        sub_name = os.path.basename(os.path.dirname(path))
        if "MR" in sub_name:
            try:
                return sub_name.split("_")[0]
            except IndexError:
                pass
        return f"sub-{number}"
