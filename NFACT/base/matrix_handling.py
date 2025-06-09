from NFACT.base.utils import colours, nprint, error_and_exit
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sps
import numpy as np
import os


def comp_disk_save(
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


def normalise_matrix(waytotal: str, matrix: object) -> np.ndarray:
    """
    Function to normalise matrix by waytotal

    Parameters
    ----------
    waytotal: str
        path to waytotal file
    matrix: object
        scipy csc_matrix

    Returns
    --------
    sparse matrix: object
        sparse matrix normalised by waytotal


    """
    waytotal = float(np.loadtxt(waytotal))
    return matrix.multiply(1e8 / waytotal)


def load_single_matrix(matfile: str) -> object:
    """
    Function to load a single fdt matrix
    as a ptx sparse matrix format.

    Parameters
    ----------
    matfile: str
       path to file

    Returns
    -------
    sparse_matrix: csc_matrix
       sparse matrix in csc_matrix
    """
    mat = np.loadtxt(matfile)
    data = mat[:-1, -1]
    rows = np.array(mat[:-1, 0] - 1, dtype=int)
    cols = np.array(mat[:-1, 1] - 1, dtype=int)
    nrows = int(mat[-1, 0])
    ncols = int(mat[-1, 1])
    sparse_mat = sps.csc_matrix((data, (rows, cols)), shape=(nrows, ncols))
    return normalise_matrix(
        os.path.join(os.path.dirname(matfile), "waytotal"), sparse_mat
    )


def load_fdt_matrix(matfile: str) -> np.ndarray:
    """
    Function to a load fdt matrix

    Parameters
    ----------
    matfile: str
       path to file

    Returns
    -------
    np.ndarray: np.array
        fdt_matrix2 matrix in numpy array form.

    """
    return load_single_matrix(matfile).toarray().astype(np.float32)


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


def threshold_grey_components(
    components: np.ndarray, coord_path: str, seeds: list, zscore_val: int
) -> np.ndarray:
    """
    Function to threshold grey matter components

    Parameters
    ----------
    components: np.ndarray
        grey mattter component
    coord_path: str
        path to coords
    seeds: list
        list of seeds
    zscore_val: int
        zscore value

    Returns
    -------
    components: np.ndarray
      components thresholded by seed
      and component

    """
    coord_mat2 = np.loadtxt(coord_path, dtype=int)
    seeds_id = coord_mat2[:, -2]
    for idx, _ in enumerate(seeds):
        mask_to_get_seed = seeds_id == idx
        grey_matter_seed = components[mask_to_get_seed, :].T
        thresholding(grey_matter_seed, zscore_val)
        components[mask_to_get_seed, :] = grey_matter_seed.T
    return components


def thresholding(component: np.ndarray, zscore_val: int) -> np.ndarray:
    """
    Function to threshold components
    to remove noise. This is done in
    place.

    Parameteres
    -----------
    component: np.ndarray
        component ot threshold

    Returns
    -------
    component: np.ndarray
        thresholded component
    """
    for comp in range(component.shape[0]):
        tract = component[comp, :]
        threshold = tract.mean() + (zscore_val * tract.std())
        tract[tract < threshold] = 0.0
    return component
