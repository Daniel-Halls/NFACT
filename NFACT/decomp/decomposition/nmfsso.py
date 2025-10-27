import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from NFACT.base.utils import error_and_exit


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
        partitions_matrix[idx, :] = partitions_matrix[idx - 1, :]
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


def create_stats_dict(n_clusters: int) -> dict:
    """
    Function to create stats dictionary

    Parameters
    ----------
    n_clusters: int
        number of clusters

    Returns
    --------
    dict: dictionary object
        dictionary of cluster stats
        array
    """
    return {
        "N": np.zeros(n_clusters, dtype=int),
        "internal": {
            "sum": np.full(n_clusters, np.nan),
            "min": np.full(n_clusters, np.nan),
            "avg": np.full(n_clusters, np.nan),
            "max": np.full(n_clusters, np.nan),
        },
        "external": {
            "sum": np.full(n_clusters, np.nan),
            "min": np.full(n_clusters, np.nan),
            "avg": np.full(n_clusters, np.nan),
            "max": np.full(n_clusters, np.nan),
        },
    }


def between_cluster_stats(
    stat_dict: dict, sim: np.ndarray, partition: np.ndarray
) -> dict:
    """
    Function to calculate between cluster stats

    between (stats between two nodes in two different clusters)
        - min: minimum edge weight between all nodes in cluster 1 and 2
        - avg: avg edge weight between all nodes in cluster 1 and 2
        - max: maximum edge weight between all nodes in cluster 1 and 2

    Parameters
    ----------
    stat_dict: dict
        stats dictionary
    sim: np.ndarray
        similairty matrix
    partition: np.ndarray
        partition matrix

    Returns
    -------
    stats_dict: dict
        stats dictionary
    """
    n_clusters = stat_dict["N"].shape[0]
    stat_dict["between"] = {
        "min": np.zeros((n_clusters, n_clusters)),
        "max": np.zeros((n_clusters, n_clusters)),
        "avg": np.zeros((n_clusters, n_clusters)),
    }

    # Iterate over all unique pairs (i, j) where i < j (to avoid redundant calculations)
    for cluster1 in range(1, n_clusters + 1):
        partition_cluster1_mask = partition == cluster1
        cluster1_idx = cluster1 - 1
        for cluster2 in range(cluster1 + 1, n_clusters + 1):
            partition_cluster2_mask = partition == cluster2
            cluster2_idx = cluster2 - 1
            # Similarity between members of cluster i and members of cluster j (S(Pi, Pj))
            sim_cluster1_2 = sim[partition_cluster1_mask][
                :, partition_cluster2_mask
            ].flatten()

            # Calculate stats only if there are connecting edges
            if sim_cluster1_2.size > 0:
                stat_dict["between"]["min"][cluster1_idx, cluster2_idx] = np.min(
                    sim_cluster1_2
                )
                stat_dict["between"]["avg"][cluster1_idx, cluster2_idx] = np.mean(
                    sim_cluster1_2
                )
                stat_dict["between"]["max"][cluster1_idx, cluster2_idx] = np.max(
                    sim_cluster1_2
                )

    # Symmetrize the matrices (Stat.between.min = Stat.between.min + Stat.between.min')
    stat_dict["between"]["min"] = (
        stat_dict["between"]["min"] + stat_dict["between"]["min"].T
    )
    stat_dict["between"]["max"] = (
        stat_dict["between"]["max"] + stat_dict["between"]["max"].T
    )
    stat_dict["between"]["avg"] = (
        stat_dict["between"]["avg"] + stat_dict["between"]["avg"].T
    )
    return stat_dict


def calculate_cluster_stats(
    sim: np.ndarray, partition: np.ndarray, between: bool = False
) -> dict:
    """
    Function to calculate cluster stats:

    Internal (stats on all unique pairs of distinct nodes in a cluster)
        - sum: sum of the total edge weight in cluster
        - min: minimium edge weight
        - avg: avergae edge weight
        - max: maximum edge weight
    External
        -


    Args:
        S: A square NumPy array (similarity/adjacency matrix).
        partition: A 1D NumPy array where each element indicates the cluster ID (1-based)
                   for the corresponding data point (must match rows/cols of S).
        between: If True, calculates between-cluster statistics.

    Returns:
        A dictionary containing the calculated statistics. NumPy arrays are used for
        all internal data structures.
    """
    n_clusters = int(np.max(partition))
    check_clustering(n_clusters, partition, sim)
    stat = create_stats_dict(n_clusters)

    # Iterate over clusters (using 1-based indexing for matching the partition labels)
    for cluster in range(1, n_clusters + 1):
        k = cluster - 1

        # Boolean mask for current cluster members: (partition == cluster)
        this_partition_mask = partition == cluster

        # --- 1. Internal Statistics (S(thisPartition,thisPartition)) ---
        sim_internal = sim[this_partition_mask][:, this_partition_mask]
        stat["N"][k] = sim_internal.shape[0]

        N_k = stat["N"][k]

        # Only calculate internal stats if cluster size is > 1 (required for off-diagonal values)
        if N_k > 1:
            # Create a copy to avoid modification warning/issue
            S_off_diag = np.copy(sim_internal)

            # Remove diagonal elements (self-similarity) by setting them to NaN
            # Equivalent to MATLAB's S_(eye(size(S_))==1)=[] on the flattened array
            np.fill_diagonal(S_off_diag, np.nan)

            # Flatten, then filter out the NaNs (the diagonal elements)
            S_values = S_off_diag.flatten()
            S_values_clean = S_values[~np.isnan(S_values)]

            # Calculate and store internal statistics
            if S_values_clean.size > 0:
                stat["internal"]["sum"][k] = np.sum(S_values_clean)
                stat["internal"]["min"][k] = np.min(S_values_clean)
                stat["internal"]["avg"][k] = np.mean(S_values_clean)
                stat["internal"]["max"][k] = np.max(S_values_clean)

        # --- 2. External Statistics (S(thisPartition, ~thisPartition)) ---
        if n_clusters > 1:
            # Boolean mask for non-members
            not_this_partition_mask = ~this_partition_mask

            # External similarity matrix
            sim_external = sim[this_partition_mask][:, not_this_partition_mask]

            # Flatten the external matrix for calculation (S_(:))
            S_external_flat = sim_external.flatten()

            # Calculate and store external statistics only if the resulting array is not empty
            if S_external_flat.size > 0:
                stat["external"]["sum"][k] = np.sum(S_external_flat)
                stat["external"]["min"][k] = np.min(S_external_flat)
                stat["external"]["avg"][k] = np.mean(S_external_flat)
                stat["external"]["max"][k] = np.max(S_external_flat)

    if between and n_clusters > 1:
        stat = between_cluster_stats(stat, sim, partition)

    return stat


def compute_r_index(dist, partitions):
    """
    Compute R-index for multiple partitions, equivalent to the MATLAB loop.

    Parameters
    ----------
    dist : np.ndarray
        Square dissimilarity matrix (N x N)
    partitions : list or 2D array-like
        Each element (or row) is a partition vector (length N)

    Returns
    -------
    ri : np.ndarray
        R-index for each partition
    """
    ri = []

    # Loop over each partition (row in MATLAB)
    for part in partitions:
        part = np.array(part)
        clusters, counts = np.unique(part, return_counts=True)
        Ncluster = len(clusters)

        # Skip partitions with singleton clusters or degenerate partition
        if np.any(counts == 1) or Ncluster == 1:
            ri.append(np.nan)
            continue

        # Compute cluster statistics
        Stat = calculate_cluster_stats(dist, part, between=True)

        # Set diagonal of between.avg to Inf (ignore self-distances)
        between_avg = Stat["between"]["avg"].copy()
        np.fill_diagonal(between_avg, np.inf)

        # Compute R-index: mean(internal.avg / min(between_avg))
        r_index = np.mean(Stat["internal"]["avg"] / np.min(between_avg, axis=1))
        ri.append(r_index)

    return np.array(ri)


def clustering_components(dis: np.ndarray):
    """
    Function to cluster components

    Parameters
    ----------
    """
    dis_flatened = squareform(dis, checks=False)
    z_link = linkage(dis_flatened, method="average")
    return Z2partition(z_link)


def check_clustering(n_clusters, partition, sim):
    error_and_exit(partition.size != 0, "Clustering Failed. Please check Num of dims")
    error_and_exit(
        sim.size != 0, "Failed to Calculate similairty matrix. Please check Num of dims"
    )
    error_and_exit(n_clusters > 0, "Clustering Failed. Please check Num of dims")
