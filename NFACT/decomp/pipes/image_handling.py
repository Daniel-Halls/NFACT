import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from NFACT.base.imagehandling import (
    save_white_matter,
    save_grey_matter_components,
)
from NFACT.base.matrix_handling import comp_disk_save
from NFACT.base.utils import colours, nprint


def save_images(
    components: dict,
    nfact_path: str,
    seeds: list,
    algo: str,
    dim: int,
    roi: list,
) -> None:
    """
    Function to save  grey and white
    components.

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
    roi: list
        rois. Needed
        for surface

    Returns
    -------
    None
    """

    col = colours()
    nprint("SAVING IMAGES")
    nprint("-" * 100)
    for comp, _ in components.items():
        algo_path = os.path.join("components", algo, "decomp")
        w_file_name = f"W_{algo.upper()}_dim{dim}"
        grey_prefix = f"G_{algo.upper()}"

        if "normalised" in comp:
            algo_path = os.path.join("components", algo, "normalised")
            w_file_name = f"W_{algo.upper()}_norm_dim{dim}"
            grey_prefix = f"G_{algo.upper()}_norm"

        if "grey" in comp:
            nprint(f"{col['pink']}Image:{col['reset']} {comp}")
            try:
                save_grey_matter_components(
                    components[comp],
                    nfact_path,
                    seeds,
                    algo_path,
                    dim,
                    os.path.join(
                        nfact_path, "group_averages", "coords_for_fdt_matrix2"
                    ),
                    roi,
                    grey_prefix,
                )
            except Exception as e:
                comp_disk_save(
                    components[comp],
                    os.path.join(nfact_path, algo_path),
                    f"{col['red']}Unable to save GM component due to: {e}{col['reset']}",
                    grey_prefix,
                )

        if "white" in comp:
            nprint(f"{col['pink']}Image:{col['reset']} {comp}")
            try:
                save_white_matter(
                    components[comp],
                    os.path.join(
                        nfact_path,
                        "group_averages",
                        "lookup_tractspace_fdt_matrix2.nii.gz",
                    ),
                    os.path.join(
                        nfact_path,
                        "group_averages",
                        "tract_space_coords_for_fdt_matrix2",
                    ),
                    os.path.join(nfact_path, algo_path, w_file_name),
                )
            except Exception as e:
                comp_disk_save(
                    components[comp],
                    os.path.join(nfact_path, algo_path),
                    f"{col['red']}Unable to save WM component due to: {e}{col['reset']}",
                    w_file_name,
                    to_exit=True,
                )


def save_components_to_disk(
    component_dict: dict, nfact_path: str, algo: str, dim: str
) -> None:
    """
    Function to save components
    to disk when requested

    Parameters
    ----------
    components: dict
        dictionary of components
    nfact_path: str
        str to nfact directory
    algo: str
        str of algo
    dim: int
        number of dimensions
        used for naming output

    Returns
    -------
    None
    """
    base_path = os.path.join(nfact_path, "components", algo)
    prefix_map = {"grey": "G", "white": "W"}

    for comp_name, comp_data in component_dict.items():
        prefix = next(
            (value for key, value in prefix_map.items() if key in comp_name), None
        )
        name = f"{prefix}_{algo.upper()}_dim{dim}"
        subfolder = "normalised" if "normalised" in comp_name else "decomp"
        if subfolder == "normalised":
            name += "_norm"
        save_path = os.path.join(base_path, subfolder)
        comp_disk_save(comp_data, save_path, "", name)


def winner_takes_all(
    components: dict,
    z_thr: float,
    algo: str,
    nfact_path: str,
    seeds: list,
    dim: str,
    roi: str,
) -> None:
    """
    Wrapper function around creating WTA and saving
    the components

    Parameters
    ---------
    components: dict
        dictionary of components
    z_thr: float
        threshold the map at
    algo: str
        ICA or NFM
    nfact_path: str
        Path to NFACT directory
    save_type: str
        Should grey components be gifti/nifit
    seeds: list
        list of seeds
    dim: str
        number of dimensions (for saving files)

    Returns
    -------
    None
    """

    white_wta_map = create_wta_map(components["white_components"], 0, z_thr)
    grey_wta_map = create_wta_map(components["grey_components"], 1, z_thr)
    save_white_matter(
        white_wta_map,
        os.path.join(
            nfact_path, "group_averages", "lookup_tractspace_fdt_matrix2.nii.gz"
        ),
        os.path.join(
            nfact_path, "group_averages", "tract_space_coords_for_fdt_matrix2"
        ),
        os.path.join(
            nfact_path, "components", algo, "WTA", f"W_{algo.upper()}_WTA_dim{dim}"
        ),
    )
    save_grey_matter_components(
        grey_wta_map,
        nfact_path,
        seeds,
        os.path.join(nfact_path, "components", algo, "WTA"),
        dim,
        os.path.join(nfact_path, "group_averages", "coords_for_fdt_matrix2"),
        roi,
        f"G_{algo.upper()}_WTA",
    )


def create_wta_map(
    component: np.ndarray,
    axis: int,
    z_thr: float,
) -> np.ndarray:
    """
    Function to create a winner takes all
    map from a component

    Parameters
    ----------
    component: np.ndarray
        component to create a
    axis: int
        axis to get max values
        from
    z_thr: float
        threshold the map at

    Returns
    -------
    np.array: array
        array of thresholded component
    """

    component_zscored = StandardScaler().fit_transform(component)
    component_max = np.max(component_zscored, axis=axis, keepdims=True)
    component_wta = np.argmax(component_zscored, axis=axis, keepdims=True) + 1
    component_wta[component_max < z_thr] = 0.0
    return np.array(component_wta, dtype=int)
