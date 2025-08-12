from NFACT.base.imagehandling import (
    save_grey_matter_components,
    save_white_matter,
    imaging_type,
    get_cifti_data,
)
from NFACT.base.utils import colours, nprint
from NFACT.base.setup import creat_subfolder_setup
from NFACT.base.utils import error_and_exit
import numpy as np
import os
import nibabel as nb
from glob import glob
import re


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
    full_name = next(
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
    abbreviation = next(
        (
            side
            for keys, side in {
                ("l", "l"),
                ("left", "l"),
                ("r", "r"),
                ("right", "r"),
            }
            if keys in file_split
        ),
        seed_name,
    )
    return [full_name, abbreviation]


def keyword_sort_key(pattern: object, sk: str) -> tuple:
    """
    Function to sort key word
    by regex pattern.

    Parameters
    ----------
    pattern: object
        regex pattern
    sk: str
        str pattern
        to match

    Returns
    --------
    tuple: tuple
        tuple of 0 or 1
        depending on success
        with lower string
    """
    if pattern.search(sk):
        return (0, sk.lower())
    else:
        return (1, sk.lower())


def reorder_lists_order(list_to_organise: list, keywords: list) -> list:
    """
    Function to reorder
    a list based on a list
    of keywords

    Parameters
    ----------
    list_to_organise: list
        list to organise
    keywords: list
        list of keys to match on

    Returns
    -------
    list: list object
        sorted list by given
        keywords

    """
    pattern = re.compile(
        r"(?<![a-zA-Z0-9])("
        + "|".join(re.escape(k) for k in keywords)
        + r")(?![a-zA-Z0-9])",
        re.IGNORECASE,
    )
    return sorted(list_to_organise, key=lambda sk: keyword_sort_key(pattern, sk))


def reorder_file_inputs(organise_list: list, list_to_organise: list) -> list:
    """
    Function to reorder a list
    by a set of keywords

    Parameters
    ----------
    organise_list: list
        list to extract keywords from
    list_to_organise: list
        list to organise

    Returns
    -------
    list: list object
        list or reorganised files
    """
    key_word = get_key_to_organise_list(organise_list)
    return reorder_lists_order(list_to_organise, key_word)


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


def convert_volume_to_component_matrix(
    img_data: np.ndarray, x_y_z_coordinates: np.ndarray
) -> np.ndarray:
    """
    Function to convert volume to
    a component matrix

    Parameters
    ----------
    img_data: np.ndarray
        imaging data as an array
    x_y_z_coordinates: np.ndarray
        coords as an array

    Returns
    --------
    np.ndarray np.array
        flattened numpy array
        in the same of components
    """
    xyz_idx = np.ravel_multi_index(x_y_z_coordinates.T, img_data.shape[:3])
    ncols = img_data.shape[3] if len(img_data.shape) > 3 else 1
    flattened_data = img_data.reshape(-1, ncols)
    return flattened_data[xyz_idx, :]


def load_grey_matter_volume(
    nifti_file: str, x_y_z_coordinates: np.ndarray
) -> np.ndarray:
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
    np.ndarray
        Grey matter component matrix
    """
    img_data = nb.load(nifti_file).get_fdata()
    return convert_volume_to_component_matrix(img_data, x_y_z_coordinates)


def remove_medial_wall(grey_component: np.ndarray, roi: str) -> np.ndarray:
    """
    Function to remove medial wall
    from grey component

    Parameters
    ----------
    grey_component: np.ndarray
        grey matter component
    roi: str
        str of roi path

    Returns
    --------
    grey_component: np.ndarray
        grey matter component with
        removed medial wall
    """
    m_wall = nb.load(roi).darrays[0].data != 0
    return grey_component[m_wall == 1, :]


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

    gifti_img = nb.load(file_name)
    grey_component = np.column_stack([darray.data for darray in gifti_img.darrays])
    return remove_medial_wall(grey_component, roi)


def load_grey_cifit(
    cifti_file: str, roi: list, x_y_z_coordinates: np.ndarray
) -> np.ndarray:
    """
    Function to load grey cifti file

    Parameters
    ----------
    cifti_file: str
        string path of cifti file
    roi: list
        list of rois
    x_y_z_coordinates: np.ndarray
        np.ndarry of coords for
        subcortical seeds. Can
        be None

    Returns
    -------
    grey_comp: np.ndarray
        grey matter component
    """
    grey_cifti = get_cifti_data(cifti_file)
    left_surface = remove_medial_wall(grey_cifti["L_surf"], roi[0])
    right_surface = remove_medial_wall(grey_cifti["R_surf"], roi[1])
    grey_comp = np.vstack([left_surface, right_surface])

    if "vol" in grey_cifti.keys():
        # Remove left and righ seeds coords
        cifit_coords = np.vstack(list(x_y_z_coordinates.values())[2:])
        subcortical = convert_volume_to_component_matrix(
            grey_cifti["vol"].get_fdata(), cifit_coords
        )
        grey_comp = np.vstack([grey_comp, subcortical])

    return grey_comp


def load_type(coords_by_idx: dict, mw: list) -> dict:
    """
    Functions to either nifti or giti
    load function

    Parameters
    ----------
    coords_by_idx: dict
        dictionary of coordinates
        by seed
    mw: list
        list of rois

    Returns
    -------
    loaders: dict
        dict of load functions
    """
    loaders = {
        "nifti": lambda seed, idx: load_grey_matter_volume(seed, coords_by_idx[idx])
    }

    if mw:
        loaders["gifti"] = lambda seed, idx: load_grey_matter_gifti_seed(seed, mw[idx])

    return loaders


def sort_coord_file(group_averages: str) -> dict:
    """
    Function to sort out coords
    file by seed.

    Parameters
    ----------
    group_averages: str
        path to group averages file
        containing coords_for_fdt_matrix2

    Returns
    -------
    dict: dictionary
        dict of coords
        organised seeds
    """
    coord_file = np.loadtxt(
        os.path.join(group_averages, "coords_for_fdt_matrix2"),
        dtype=int,
    )
    return {
        idx: coord_file[coord_file[:, 3] == idx][:, :3]
        for idx in range(coord_file[:, 3].max() + 1)
    }


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
    sorted_components = reorder_file_inputs(seeds[0], grey_matter)
    # Needed if mw are not given
    try:
        mw = reorder_file_inputs(seeds[0], mw)
    except TypeError:
        pass
    coords_by_idx = sort_coord_file(group_averages)

    if imaging_type(sorted_components[0]) == "cifti":
        return load_grey_cifit(sorted_components[0], mw, coords_by_idx)

    loaders = load_type(coords_by_idx, mw)
    grey_component = []
    for idx, seed in enumerate(sorted_components):
        grey_comp = loaders[imaging_type(seed)](seed, idx)
        grey_component.append(grey_comp)

    return np.vstack(grey_component)


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


def save_dual_regression_images(
    components: dict,
    nfact_path: str,
    seeds: list,
    algo: str,
    dim: int,
    sub: str,
    ptx_directory: str,
    roi: list,
    cifti_save: bool = False,
) -> None:
    """
    Wrapper function to save regression images

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
    cifti_save: bool
        save gm as dscalar.
        Default is False

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
                cifti_save,
            )
        if "white" in comp:
            nprint(f"{col['pink']}Image:{col['reset']} {comp}")
            save_white_matter(
                components[comp],
                os.path.join(ptx_directory, "lookup_tractspace_fdt_matrix2.nii.gz"),
                os.path.join(ptx_directory, "tract_space_coords_for_fdt_matrix2"),
                os.path.join(nfact_path, algo_path, w_file_name),
            )
