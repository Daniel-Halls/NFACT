import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def rownorm(nmf_mat: np.ndarray):
    """
    Normalise rows for clustering

    Parameters
    ----------
    nmf_mat: np.ndarray
        nmf matrix to normalise
        on

    Returns
    -------
    nmf_mat: np.ndarray
        nmf matrix normalised
    """
    norms = np.linalg.norm(nmf_mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return nmf_mat / norms


def compute_similairty_matrix(components: np.ndarray) -> np.ndarray:
    """
    Function to compute similarity across NMF components
    from multiple runs.

    Parameters
    ----------
    components: np.ndarray
        out from NMF sso

    Returns
    -------
    sim: np.ndarray
        similarity matrix
    """

    normalised_mat = rownorm(components)
    sim = np.corrcoef(normalised_mat)
    np.clip(sim, 0, 1, out=sim)
    return sim


def sim2dis(sim: np.ndarray) -> np.ndarray:
    """
    Function to calculate the dissimlairty matrix

    Parameters
    ----------
    sim: np.ndarray
        similarity matrix

    Returns
    -------
    np.ndarry: array
        disimilarity matrix
    """
    return 1 - sim


def Z2partition(link_mat: np.ndarray) -> np.ndarray:
    """
    Function to take a  hierarchical clustering linkage matrix
    and turn it into partition vectors (all merges).

    Parameters
    ----------
    link_mat: ndarray
        Linkage matrix from scipy.cluster.hierarchy.linkage.

    Returns
    -------
    partitions : ndarray
        Array of shape (N-1, N), where each row contains cluster assignments
        after the idx-th merge. Cluster labels are consecutive integers.
    """
    linkage_mat_shape = link_mat.shape[0] + 1
    partitions_matrix = np.zeros((linkage_mat_shape, linkage_mat_shape), dtype=int)
    partitions_matrix[0, :] = np.arange(1, linkage_mat_shape + 1)
    for idx in range(1, linkage_mat_shape - 1):
        partitions_matrix[idx, :] = partitions_matrix[i - 1, :]
        # Convert SciPy indices (0-based) to 1-based like MATLAB
        cluster1 = int(link_mat[idx - 1, 0]) + 1
        cluster2 = int(link_mat[idx - 1, 1]) + 1
        mask = (partitions_matrix[idx, :] == cluster1) | (
            partitions_matrix[idx, :] == cluster2
        )
        partitions_matrix[idx, mask] = max(partitions_matrix[idx, :]) + 1

    # Recode cluster IDs to consecutive integers for each row
    for row in range(partitions_matrix.shape[0]):
        _, new_index = np.unique(partitions_matrix[row, :], return_inverse=True)
        partitions_matrix[row, :] = new_index + 1

    return partitions_matrix[::-1, :]


def clustering_components(dis: np.ndarray):
    dis_flateened = squareform(dis, checks=False)
    z_link = linkage(dis_flateened, method="average")
    partitions = Z2partition(z_link)
    rindex = compute_r_index(dis, partitions)
