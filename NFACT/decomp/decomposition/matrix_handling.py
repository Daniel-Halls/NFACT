import numpy as np
from tqdm import tqdm
from scipy.sparse.linalg import eigsh
import os
from NFACT.base.utils import Timer, error_and_exit, colours, nprint
from NFACT.base.matrix_handling import load_fdt_matrix, load_single_matrix
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def process_fdt_matrix2(list_of_ptx_folds: list, group_mode: bool) -> np.ndarray:
    """
    Function to get group average matrix

    Parameters
    ----------
    list_of_ptx_folds: list
        list of probtrackx folders

    Returns
    -------
    fdt_matrix2: np.array
       np.ndarray of fdt2 matrix either averaged
       across subjects or single subjects
    """
    list_of_fdt = [
        os.path.join(sub_folder, fname)
        for sub_folder in list_of_ptx_folds
        for fname in ("fdt_matrix2.dot", "fdt_matrix2.dot.lz4")
        if os.path.exists(os.path.join(sub_folder, fname))
    ]

    if group_mode:
        try:
            fdt_matrix2 = avg_fdt(list_of_fdt)
        except Exception as e:
            error_and_exit(False, f"Unable to load fdt_matrix2 due to {e}")
    if not group_mode:
        try:
            fdt_matrix2 = load_fdt_matrix(list_of_fdt[0])
        except Exception as e:
            error_and_exit(False, f"Unable to load fdt_matrix2 due to {e}")
    return fdt_matrix2


def load_previous_matrix(path: str) -> np.ndarray:
    """
    Function to load previous matrix.

    Parameters
    ----------
    path: str
       path to matrix

    Returns
    -------
    array: np.array
        fdt2 matrix.
    """

    try:
        fdt = np.load(os.path.join(path))
        return fdt
    except Exception:
        col = colours()
        nprint(
            f"{col['pink']}Error:{col['reset']} Unable to read previous matrix. Averaging"
        )
        return None


def avg_fdt(list_of_matfiles: list) -> np.ndarray:
    """
    Function to create and create
    an average group matrix.

    Parameters
    ----------
    list_of_matfiles: list
        list of matricies
        for the group.
    waytotal_path: str
        waytotal path

    Returns
    -------
    sparse_matrix: np.array
        np.array of sparse matrix.
    """
    sparse_matrix = None
    for matrix in tqdm(list_of_matfiles, colour="magenta", unit=" Per matrix"):
        mat = load_single_matrix(matrix)
        if sparse_matrix is None:
            sparse_matrix = mat.copy()
        else:
            sparse_matrix += mat
        del mat
    sparse_matrix /= len(list_of_matfiles)
    return sparse_matrix.toarray().astype(np.float32)


def demean(matrix: np.array, axis: int = 0) -> np.ndarray:
    """
    Function to demean a matrix

    Parameters
    ----------
    matrix: np.array
        matrix
    axis: int
        which axis to calculate
        the mean from
    """
    return matrix - np.mean(matrix, axis=axis, keepdims=True)


def melodic_incremental_group_pca(
    fdt_matrix: np.array, n_dim: int = 1000, d_pca: int = 1000, keep_mean: bool = False
) -> np.ndarray:
    """
    Function wrapper around matrix_MIGP.

    Parameters
    ----------
    fdt_matrix: np.array,
        Should be wide i.e. nxN where N bigger than n t
        fdt_matrix is made of column blocks, each block is
        one 'subject', and 'time' is the column dimension
    n_dim: int
        number of dimensions that fdt_matrix is split into.
        Default is 1000.
    d_pca: int
        maximum number of prinicple components kept
        (set to n_dim if larger than n_dim) Default is 1000.


    Returns
    -------
    pca_matrix: np.array
        matrix that has been reduced.

    """

    migpa_timer = Timer()
    migpa_timer.tic()
    col = colours()
    nprint(f"{col['purple']}\nPerforming PCA (MIGP){col['reset']}")
    nprint("WARNING THIS CAN TAKE A VERY LONG TIME")
    if keep_mean:
        matrix_mean = np.mean(fdt_matrix, axis=1, keepdims=True)

    if d_pca > n_dim:
        d_pca = n_dim

    pca_matrix = matrix_migp(fdt_matrix, n_dim, d_pca)

    if keep_mean:
        pca_matrix = pca_matrix + matrix_mean

    nprint(f"Old matrix size {fdt_matrix.shape[0]}x{fdt_matrix.shape[1]}")
    nprint(f"New matrix size now {pca_matrix.shape[0]}x{pca_matrix.shape[1]}")
    nprint(f"MIGP finished in {migpa_timer.toc()} secs.\n")
    return pca_matrix


def matrix_migp(
    fdt_matrix: np.ndarray,
    n_dim: int = 1000,
    d_pca: int = 1000,
) -> np.ndarray:
    """
    Function to apply
    MELODIC's Incremental Group-PCA dimensionality to
    a given matrix.

    Parameters
    ----------
    fdt_matrix: np.array,
        Should be wide i.e. nxN where N bigger than n t
        fdt_matrix is made of column blocks, each block is
        one 'subject', and 'time' is the column dimension
    n_dim: int
        number of dimensions that fdt_matrix is split into.
        Default is 1000.
    d_pca: int
        maximum number of prinicple components kept
        (set to n_dim if larger than n_dim) Default is 1000.


    Returns
    -------
    pca_matrix: np.array
        matrix that has been reduced.
    """

    random_idx = np.random.permutation(fdt_matrix.shape[1])
    shuffled_column = fdt_matrix[:, random_idx]
    intermediary_matrix = None

    for matrix_index in tqdm(range(0, fdt_matrix.shape[1], n_dim), colour="magenta"):
        pca_matrix = shuffled_column[
            :, matrix_index : min(matrix_index + n_dim, fdt_matrix.shape[1] + 1)
        ].T

        if intermediary_matrix is not None:
            intermediary_matrix = np.concatenate(
                (intermediary_matrix, demean(pca_matrix)), axis=0
            )
        else:
            intermediary_matrix = demean(pca_matrix)

        k_to_compute = min(d_pca, n_dim)
        _, k_eignvectors = eigsh(
            intermediary_matrix @ intermediary_matrix.T, k_to_compute
        )

        intermediary_matrix = k_eignvectors.T @ intermediary_matrix

    pca_matrix = intermediary_matrix[: min(intermediary_matrix.shape[0], d_pca), :].T

    return pca_matrix
