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


def calculate_internal_stats(
    sim_internal: np.ndarray, cluster: int, stat: dict
) -> dict:
    """
    Function to calculate the internal statistics
    of a given cluster

    Parameters
    ----------
    sim_internal: np.ndarray
        array of similarity measures within
        a cluster
    cluster: int
        cluster working on
    stat: dict
        stats dictionary

    Returns
    -------
    stats: dictionary
        stats dictionary
        with sum, min, max & avg
        within cluster stats
    """
    sim_off_diag = np.copy(sim_internal)
    np.fill_diagonal(sim_off_diag, np.nan)
    sim_values = sim_off_diag.flatten()
    sim_values_clean = sim_values[~np.isnan(sim_values)]

    # Calculate and store internal statistics
    if sim_values_clean.size > 0:
        stat["internal"]["sum"][cluster] = np.sum(sim_values_clean)
        stat["internal"]["min"][cluster] = np.min(sim_values_clean)
        stat["internal"]["avg"][cluster] = np.mean(sim_values_clean)
        stat["internal"]["max"][cluster] = np.max(sim_values_clean)
    return stat


def calculate_external_cluster_stats(
    stat: dict, cluster: int, sim_external: np.ndarray
) -> dict:
    """
    Function to calculate external cluster
    stats

    Parameters
    ----------
    stat: dict
        dictionary to store stats in
    cluster: int
        cluster number
    sim_external: np.ndarray
        external similairty matrix

    Returns
    -------
    stats: dictionary
        stats dictionary
        with sum, min, max & avg
        external cluster stats
    """
    stat["external"]["sum"][cluster] = np.sum(sim_external)
    stat["external"]["min"][cluster] = np.min(sim_external)
    stat["external"]["avg"][cluster] = np.mean(sim_external)
    stat["external"]["max"][cluster] = np.max(sim_external)
    return stat


def calculate_cluster_stats(
    sim: np.ndarray, partition: np.ndarray, between: bool = False
) -> dict:
    """
    Function to calculate cluster stats:

    Internal (stats on all unique pairs of distinct
              nodes in a cluster)
        - sum: sum of the total edge weight in cluster
        - min: minimium edge weight
        - avg: avergae edge weight
        - max: maximum edge weight
    External (current cluster nad all nodes in all other clusters
              combined)
        - sum: total edge weight of cluster to rest of graph
        - min: min edge weight of cluster to rest of graph
        - avg: avg edge weight of cluster to rest of graph
        - max: max edge weight of cluster to rest of graph


    Parameters
    ----------
    sim: np.ndarray
        similairty matrix
    partition: np.ndarray
        array of cluster labels for all partitions
    between: bool
        Calculate between cluster stats.
         Default is False

    Returns
    -------
        A dictionary containing the calculated statistics. NumPy arrays are used for
        all internal data structures.
    """
    n_clusters = int(np.max(partition))
    check_clustering(n_clusters, partition, sim)
    stat = create_stats_dict(n_clusters)

    # Iterate over clusters (using 1-based indexing for matching the partition labels)
    for cluster in range(1, n_clusters + 1):
        working_cluster = cluster - 1

        # Boolean mask for current cluster members: (partition == cluster)
        this_partition_mask = partition == cluster

        # Internal Statistics (S(thisPartition,thisPartition)) ---
        sim_internal = sim[this_partition_mask][:, this_partition_mask]
        stat["N"][working_cluster] = sim_internal.shape[0]

        # Only calculate internal stats if cluster size is > 1 (required for off-diagonal values)
        if stat["N"][working_cluster] > 1:
            stat = calculate_internal_stats(sim_internal, working_cluster, stat)

        # External Statistics (S(thisPartition, ~thisPartition)) ---
        not_this_partition_mask = ~this_partition_mask
        sim_external = sim[this_partition_mask][:, not_this_partition_mask].flatten()

        # Calculate and store external statistics only if the resulting array is not empty
        if sim_external.size > 0:
            stat = calculate_external_cluster_stats(stat, working_cluster, sim_external)

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

        # Skip partitions with singleton clusters or degenerate partition
        if np.any(counts == 1) or len(clusters) == 1:
            ri.append(np.nan)
            continue

        # Compute cluster statistics
        stat = calculate_cluster_stats(dist, part, between=True)

        # Set diagonal of between.avg to Inf (ignore self-distances)
        between_avg = stat["between"]["avg"].copy()
        np.fill_diagonal(between_avg, np.inf)

        # Compute R-index: mean(internal.avg / min(between_avg))
        r_index = np.mean(stat["internal"]["avg"] / np.min(between_avg, axis=1))
        ri.append(r_index)

    return np.array(ri)


def clustering_components(dis: np.ndarray) -> np.ndarray:
    """
    Function to cluster components

    Parameters
    ----------
    dis: np.ndarray
       dis-similairty matrix

    Returns
    --------
    np.ndarray: array
       a partition array
    """
    dis_flatened = squareform(dis, checks=False)
    z_link = linkage(dis_flatened, method="average")
    return Z2partition(z_link)


def check_clustering(n_clusters: int, partition: np.ndarray, sim: np.ndarray) -> None:
    """
    Function wrapper to check that clustering
    and similairty calculations worked

    Parameters
    -----------
    n_clusters: int
        how many clusters
    partition: np.ndarray
        paritition array
    sim: np.ndarray
        similairty matrix

    Returns
    -------
    None
    """
    error_and_exit(partition.size != 0, "Clustering Failed. Please check Num of dims")
    error_and_exit(
        sim.size != 0, "Failed to Calculate similairty matrix. Please check Num of dims"
    )
    error_and_exit(n_clusters > 0, "Clustering Failed. Please check Num of dims")


def centrotype(sim: np.ndarray) -> np.ndarray:
    """
    Find the centrotype (most central element)
    of a similarity matrix.

    Parameters
    ----------
    sim : ndarray
        Similarity matrix.

    Returns
    -------
    idx : int
        Index of the centrotype within
        similairty matrix
    """
    col_sums = np.sum(sim, axis=0)
    idx = np.argmax(col_sums)
    return idx


def idx2centrotype(sim: np.ndarray, partition: np.ndarray) -> np.ndarray:
    """
    Function to compute centrotype(s)
    given a similarity matrix and partitions.

    Parameters
    ----------
    sim : ndarray
        Similarity matrix.
    partition : ndarray
        partition array

    Returns
    -------
    index2centrotype : ndarray
        Index/indices of the centriods of the cluster
    """

    n_cluster = partition.max()
    index2centrotype = np.zeros(n_cluster, dtype=int)

    for cluster in range(1, n_cluster + 1):
        indices = np.where(partition == cluster)[0]
        sim_cluster_sub = sim[np.ix_(indices, indices)]
        centroid_idx = centrotype(sim_cluster_sub)
        index2centrotype[cluster - 1] = indices[centroid_idx]

    return index2centrotype
