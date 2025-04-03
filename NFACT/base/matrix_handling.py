from NFACT.base.utils import colours, nprint, error_and_exit
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sps
import numpy as np
import os


def img_save_failed(
    component: np.ndarray,
    save_path: str,
    fail_message: str,
    matrix_name: str,
    to_exit: bool = False,
) -> None:
    """
    Function to save component to
    disk when image fail.

    Parameters
    ----------
    component: np.ndarray
        numpy array of component
    save_path: str
        Path to save matrix to
    fail_message: str
        message to print on fail
    matrix_name: str
        name of matrix
    to_exit: bool
        to exit on fail

    Returns
    -------
    None
    """
    col = colours()
    nprint(fail_message)
    nprint(f"{col['pink']}Saving Component to disk:{col['reset']} {save_path}")
    save_matrix(component, save_path, matrix_name)
    if to_exit:
        error_and_exit(False)


def normalise_components(grey_matter: np.array, white_matter: np.array) -> dict:
    """
    Normalise components.
    Useful for visulaization

    Parameters
    ----------
    grey_matter: np.array
        grey matter component
    white_matter: np.array
        white matter component

    Returns
    -------
    dict: dictionary.
        dictionary of normalised components
    """
    col = colours()
    nprint(f"{col['pink']}Normalising:{col['reset']} Components")

    return {
        "grey_matter": StandardScaler().fit_transform(grey_matter),
        "white_matter": StandardScaler().fit_transform(white_matter.T).T,
    }


def load_fdt_matrix(matfile: str) -> np.ndarray:
    """
    Function to load a single fdt matrix
    as a ptx sparse matrix format.

    Parameters
    ----------
    matfile: str
       path to file

    Returns
    -------
    sparse_matrix: np.array
       sparse matrix in numpy array
       form.
    """
    mat = np.loadtxt(matfile)
    data = mat[:-1, -1]
    rows = np.array(mat[:-1, 0] - 1, dtype=int)
    cols = np.array(mat[:-1, 1] - 1, dtype=int)
    nrows = int(mat[-1, 0])
    ncols = int(mat[-1, 1])
    return sps.csc_matrix((data, (rows, cols)), shape=(nrows, ncols)).toarray()


def save_matrix(matrix: np.array, directory: str, matrix_name: str) -> None:
    """
    Function to save average matrix as npz file

    Parameters
    ----------
    matrix: np.array
        matrix to save
    directory: str
        directory to save matrix to
    matrix_name: str
        name of matrix to save as

    Returns
    -------
    None
    """
    try:
        np.save(os.path.join(directory, matrix_name), matrix)
    except Exception as e:
        error_and_exit(False, f"Unable to save matrix due to {e}")
