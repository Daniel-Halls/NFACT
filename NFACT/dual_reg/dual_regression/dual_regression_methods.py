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
                desc="WM",
                colour="magenta",
                unit=" WM seed unit",
                position=0,
                dynamic_ncols=True,
            )
        ]
    )
    nprint(f"{col['pink']}Regression:{col['reset']} Grey Matter")
    gm_component_grey_map = np.array(
        [
            nnls(wm_component_white_map, connectivity_matrix.T[:, col])[0]
            for col in tqdm(
                range(connectivity_matrix.shape[0]),
                desc="GM",
                colour="magenta",
                unit=" GM seed unit",
                position=0,
                dynamic_ncols=True,
            )
        ]
    )
    return {
        "grey_components": gm_component_grey_map,
        "white_components": wm_component_white_map.T,
    }


class NNLS:
    """
    Class to perform NNLS
    for batch parallelization

    Usage
    -----
    gm = NNLS(comp, conn, 10, "gm")
    gm_component = gm.run()
    """

    def __init__(
        self,
        components: np.ndarray,
        conn_matrix: np.ndarray,
        n_jobs: int,
        data_type: str,
        batch_size: int = None,
    ) -> None:
        """
        Method to init class.

        Parameters
        ----------
        components: np.ndarray
            array of components
        conn_matrix: np.ndarray
            connectivity matrix
        n_jobs: int
            number of jobs to parallelize
        data_type: str
            str of what type of data is being
            modelled
        batch_size: int=None


        Returns
        -------
        None
        """
        self.components = components
        self.conn_matrix = conn_matrix
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.data_type = data_type
        if self.batch_size is None:
            self.batch_size = self.get_batch_size(conn_matrix.shape[0])

    def get_batch_size(
        self, n_cols: int, min_size: int = 10, max_size: int = 500, fraction: int = 0.02
    ) -> int:
        """
        Method to determine optimal batch size given
        a number of columns. Has a min and max size

        Parameters
        ----------
        n_cols: int
            number of colums
            the matrix has
        min_size: int=10
            the minimum the
            batch size can be
        max_size: int=500
            the maximum the batch size
            can be
        fraction: int=0.02
            fraction to
            calculate minimum
            size

        Returns
        --------
        int: int
            batch size
        """
        size = max(min_size, int(n_cols * fraction))
        return min(size, max_size)

    def run_nnls_batch(self, cols: np.ndarray) -> list:
        """
        Method to run nnls

        Parameters
        ----------
        cols: np.ndarray
            number of columns

        Returns
        -------
        list: list object
            list of nnls
            output
        """
        return [nnls(self.components, self.conn_matrix[:, col])[0] for col in cols]

    def run(self) -> np.ndarray:
        """
        Run method for class.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            nnls output
        """
        n_cols = self.conn_matrix.shape[1]
        col_batches = [
            range(batch, min(batch + self.batch_size, n_cols))
            for batch in range(0, n_cols, self.batch_size)
        ]

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.run_nnls_batch)(batch)
            for batch in tqdm(
                col_batches,
                desc=self.data_type,
                colour="magenta",
                unit=" batches",
                position=0,
                dynamic_ncols=True,
            )
        )

        flat_results = [item for sublist in results for item in sublist]
        return np.array(flat_results)


def nnls_parallel(
    components: dict,
    connectivity_matrix: np.ndarray,
    n_jobs: int,
    n_batches: int = None,
):
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

    Returns
    -------
    dict
        Dictionary of components.
    """
    time = Timer()
    time.tic()
    col = colours()
    nprint(f"{col['pink']}Regression:{col['reset']} White Matter")
    wm_parallel = NNLS(
        components["grey_components"], connectivity_matrix, n_jobs, n_batches
    )
    wm_component_white_map = wm_parallel.run()
    nprint(f"{col['pink']}Regression:{col['reset']} Grey Matter")
    gm_parallel = NNLS(wm_component_white_map, connectivity_matrix.T, n_jobs, n_batches)
    gm_component_grey_map = gm_parallel.run()
    nprint(f"Dual regression took {time.how_long()}")

    return {
        "grey_components": gm_component_grey_map,
        "white_components": wm_component_white_map.T,
    }
