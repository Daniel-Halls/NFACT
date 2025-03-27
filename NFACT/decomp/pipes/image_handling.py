import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from NFACT.base.imagehandling import (
    save_white_matter,
    save_grey_matter_components,
)
from NFACT.base.matrix_handling import save_matrix
from NFACT.base.utils import colours, nprint, error_and_exit


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
            try:
                nprint(f"{col['pink']}Image:{col['reset']} {comp}")
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
                save_path = os.path.join(nfact_path, algo_path)
                nprint(f"{col['red']}Unable to save Grey component due to: {e}")
                nprint(
                    f"Continuing however dual regression not possbile.{col['reset']}"
                )
                nprint(f"Saving Component as np array {save_path}")
                save_matrix(components[comp], save_path, grey_prefix)
        if "white" in comp:
            try:
                nprint(f"{col['pink']}Image:{col['reset']} {comp}")
                save_white_matter(
                    components[comp],
                    os.path.join(
                        nfact_path,
                        "group_averages",
                        "lookup_tractspace_fdt_matrix2.nii.gz",
                    ),
                    os.path.join(nfact_path, algo_path, w_file_name),
                )
            except Exception as e:
                save_path = os.path.join(nfact_path, algo_path)
                nprint(f"{col['red']}Unable to save White component due to: {e}")
                nprint(f"Saving Component as np array {save_path}")
                save_matrix(components[comp], save_path, w_file_name)
                error_and_exit(False)


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
