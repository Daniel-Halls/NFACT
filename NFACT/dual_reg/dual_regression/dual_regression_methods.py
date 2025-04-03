from NFACT.base.utils import error_and_exit, nprint, colours, Timer
import numpy as np
from scipy.optimize import nnls
from joblib import Parallel, delayed
from tqdm import tqdm


def run_decomp(
    decomp: object,
    components: dict,
    connectivity_matrix: np.ndarray,
    parallel: int = None,
) -> dict:
    """
    Function to

    Parameters
    ----------
    components: dict
        dictionary of components
    connectivity_matrix: np.ndarray
        subjects loaded connectivity matrix
    parallel: int=None
        How many cores to run the decomp
        with. For ICA this just prints
        error message

    Returns
    -------
    dict: dictionary
        dictionary of components
    """
    try:
        components = decomp(components, connectivity_matrix, parallel)
    except ValueError as e:
        error_and_exit(
            False,
            f"Components have incompatable size with connectivity Matrix {e}",
        )
    except Exception as e:
        error_and_exit(False, f"Unable to perform dual regression due to {e}")
    return components


def ica_dual_regression(
    components: dict, connectivity_matrix: np.ndarray, parallel: int = None
) -> dict:
    """
    Dual regression function for ICA.
    Regresses the invidiual connectivity matrix
    onto the group components.
    If white matter component then regresses
    grey matter map onto connectivity matrix and vice versa.

    Parameters
    ----------
    components: dict
        dictionary of components
    connectivity_matrix: np.ndarray
        subjects loaded connectivity matrix
    parallel: int=None
        This is a parameters added for
        running run_decomp function.
        If given prints out error
        message.

    Returns
    -------
    dict: dictionary
        dictionary of components
    """
    if parallel:
        print("ICA cannot be run in parallel")

    wm_component_grey_map = (
        np.linalg.pinv(components["white_components"].T) @ connectivity_matrix.T
    ).T
    wm_component_white_map = np.linalg.pinv(wm_component_grey_map) @ connectivity_matrix
    gm_component_grey = (
        np.linalg.pinv(components["grey_components"]) @ connectivity_matrix
    )
    gm_component_grey_map = (
        np.linalg.pinv(gm_component_grey.T) @ connectivity_matrix.T
    ).T

    return {
        "grey_components": gm_component_grey_map,
        "white_components": wm_component_white_map,
    }


def nmf_dual_regression(
    components: dict, connectivity_matrix: np.ndarray, n_jobs: int = -1
) -> dict:
    """
    Dual regression function for NMF.

    Parameters
    ----------
    components: dict
        Dictionary of components.
    connectivity_matrix: np.ndarray
        Subjects' loaded connectivity matrix.
    n_jobs: int
        Number of parallel jobs for computation.
        Default is -1 (all available CPUs).

    Returns
    -------
    dict
        Dictionary of components.
    """

    if int(n_jobs) <= 1 or not n_jobs:
        return nnls_non_parallel(components, connectivity_matrix)
    return nnls_parallel(components, connectivity_matrix, n_jobs)


def nnls_non_parallel(components: dict, connectivity_matrix: np.ndarray):
    """
    Dual regression method for NMF.

    Parameters
    ----------
    components: dict
        Dictionary of components.
    connectivity_matrix: np.ndarray
        Subjects' loaded connectivity matrix.

    Returns
    -------
    dict
        Dictionary of components.
    """
    col = colours()
    nprint(f"{col['pink']}Regression:{col['reset']} White Matter")
    wm_component_white_map = np.array(
        [
            nnls(components["grey_components"], connectivity_matrix[:, col])[0]
            for col in tqdm(
                range(connectivity_matrix.shape[1]),
                colour="magenta",
                unit=" voxel",
                position=0,
                dynamic_ncols=True,
            )
        ]
    ).T
    nprint(f"{col['pink']}Regression:{col['reset']} Grey Matter")
    gm_component_grey_map = np.array(
        [
            nnls(wm_component_white_map.T, connectivity_matrix.T[:, col])[0]
            for col in tqdm(
                range(connectivity_matrix.shape[0]),
                colour="magenta",
                unit=" vertex",
                position=0,
                dynamic_ncols=True,
            )
        ]
    )
    return {
        "grey_components": gm_component_grey_map,
        "white_components": wm_component_white_map,
    }


def nnls_for_parallel(
    col: int, component: np.ndarray, connectivity_matrix: np.ndarray
) -> np.ndarray:
    """
    nnls function wrapper around the nnls function.
    Needed for parallelization

    Parameters
    ----------
    col: int
        column index
    component: np.ndarray
        components to run nnls on
    connectivity_matrix: np.ndarray
        connectivity matrix

    Returns
    -------
    np.ndarray: np.ndarray
        output of nnls function
    """
    return nnls(component, connectivity_matrix[:, col])[0]


def nnls_parallel(components: dict, connectivity_matrix: np.ndarray, n_jobs: int = -1):
    """
    Dual regression function for NMF with parallelization

    Parameters
    ----------
    components: dict
        Dictionary of components.
    connectivity_matrix: np.ndarray
        Subjects' loaded connectivity matrix.
    n_jobs: int
        Number of parallel jobs for computation.
        Default is -1 (all available CPUs).

    Returns
    -------
    dict
        Dictionary of components.
    """
    time = Timer()
    time.tic()

    grey_components = components["grey_components"]
    col = colours()
    nprint(f"{col['pink']}Regression:{col['reset']} White Matter")
    wm_component_white_map = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(nnls_for_parallel)(col, grey_components, connectivity_matrix)
            for col in range(connectivity_matrix.shape[1])
        )
    ).T

    nprint(f"{col['pink']}Regression:{col['reset']} Grey Matter")
    time.tic()
    wm_component_white_map_T = wm_component_white_map.T
    gm_component_grey_map = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(nnls_for_parallel)(
                col, wm_component_white_map_T, connectivity_matrix.T
            )
            for col in range(connectivity_matrix.shape[0])
        )
    )
    nprint(f"Dual regression took {time.how_long()}")

    return {
        "grey_components": gm_component_grey_map,
        "white_components": wm_component_white_map,
    }
