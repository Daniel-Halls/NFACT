import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import networkx as nx
import numpy as np
from itertools import combinations
import seaborn as sns


def threshold_for_graph(sim: np.ndarray) -> float:
    """
    Function to define threshold for graph.
    Takes the 95% centile as threshold
    value

    Parameters
    -----------
    sim: np.ndarray
        similiarity matrix

    Returns
    --------
    float: float value
        threshold value
    """
    lower_idx = np.tril_indices_from(sim, k=-1)
    return np.percentile(sim[lower_idx], 95)


def build_graph(n_points: int, coordinates: np.ndarray) -> object:
    """
    Function to build out a graph

    Parameters
    -----------
    n_points: int
        n_points in the graph
    coordinates: np.ndarray
        co-ordinates of points

    Returns
    -------
    graph: nx.Graph object
        graph object from networkx
    """

    graph = nx.Graph()
    for node in range(n_points):
        graph.add_node(node, pos=coordinates[node])
    return graph


def build_edges(n_points: int, similarity: np.ndarray, threshold: float) -> dict:
    """
    Function to calssify edges of similairty matrix as either
    greater or less than a given threshold

    Parameters
    ----------
    n_points: int
        number of points in the graph
    similarity: np.ndarray
        similairty matrix
    threshold: float
        threshold value

    Returns
    -------
    dit: dictionary object
        dictionary of edges above threshold
        and below threshold

    """
    edges_above = []
    edges_below = []

    for node1, node2 in combinations(range(n_points), 2):
        if similarity[node1, node2] > threshold:
            edges_above.append((node1, node2))
        else:
            edges_below.append((node1, node2))

    return {"edges_above": edges_above, "edges_below": edges_below}


def plot_network(
    coordinates: np.ndarray,
    partition: np.ndarray,
    similarity: np.ndarray,
    centers: np.ndarray,
    filepath: str,
    expand_factor=1.1,
) -> None:
    """
    Function to plot a network with cluster outlines

    Parameters
    ----------
    coordinates : np.ndarray
        Node coordinates (n_points, 2)
    partition : np.ndarray
        Cluster assignment for each node
    similarity : np.ndarray
        Similarity matrix
    centers : list or np.ndarray
        Indices of centroid nodes
    filepath: str
        where to save graph to
    threshold : float, optional
        Similarity threshold for edges
    expand_factor : float, optional
        Factor to expand cluster outlines (1.0 = no expansion)

    Returns
    --------
    None
    """
    n_points = coordinates.shape[0]
    threshold = threshold_for_graph(similarity)
    graph = build_graph(n_points, coordinates)
    edges = build_edges(n_points, similarity, threshold)

    pos = {i: coordinates[i] for i in range(n_points)}

    cluster_nodes = [node for node in range(n_points) if node not in centers]
    centroid_nodes = centers

    plt.figure(figsize=(8, 8))
    cluster_color = "#2487ea"
    centroid_color = "#f86d6d"
    # Draw nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=cluster_nodes,
        node_color=cluster_color,
        edgecolors="black",
        linewidths=1.5,
        node_size=150,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=centroid_nodes,
        node_color=centroid_color,
        edgecolors="black",
        linewidths=2,
        node_size=200,
    )

    # Draw edges
    nx.draw_networkx_edges(
        graph, pos, edgelist=edges["edges_below"], edge_color="grey", alpha=0.3, width=1
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=edges["edges_above"],
        edge_color="black",
        alpha=0.7,
        width=1.5,
    )

    # Label centroid nodes
    for idx, center in enumerate(centroid_nodes, start=1):
        x_position, y_position = pos[center]
        plt.text(
            x_position,
            y_position + 0.02,
            f"{idx}",
            fontsize=12,
            fontweight="bold",
            color="black",
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
        )

    # Draw expanded red outlines around clusters
    unique_clusters = np.unique(partition)
    for cluster_id in unique_clusters:
        cluster_points = coordinates[partition == cluster_id]
        if len(cluster_points) >= 3:
            hull = ConvexHull(cluster_points)
            hull_pts = cluster_points[hull.vertices]

            # Expand outward from cluster centroid
            centroid = cluster_points.mean(axis=0)
            expanded_pts = centroid + expand_factor * (hull_pts - centroid)

            plt.plot(
                np.append(expanded_pts[:, 0], expanded_pts[0, 0]),
                np.append(expanded_pts[:, 1], expanded_pts[0, 1]),
                color="red",
                linewidth=2.5,
                alpha=0.9,
            )

    plt.axis("off")
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    plt.savefig(filepath, format="tiff", dpi=300, bbox_inches="tight")
    plt.close()


def plot_matrix(file_path: str, mat: np.ndarray, title: str) -> None:
    """
    Function to plot matrix and save
    to file

    Parameters
    ----------
    file_path: str
        file path to save graph to.
        Must include file name
    mat: np.ndarray
        matrix to plot
    title: str
        title of graph

    Returns
    -------
    None
    """
    ax = sns.heatmap(mat, xticklabels="", yticklabels="")
    ax.set_title(title)
    plt.savefig(file_path, format="tiff", dpi=300, bbox_inches="tight")
    plt.close()


def plot_rindex(rindex: np.ndarray, file_path: str) -> None:
    """
    Function to plot r index of clusters

    Parameters
    ----------
    rindex: np.ndarray
        array of r index
    file_path: str
        file path to save graph to.
        Must include file name

    Returns
    -------
    None

    """
    plt.plot(rindex, "o:", color="k", markersize=8, markerfacecolor="k")
    plt.grid(True)
    plt.xlabel(r"Cluster number")
    plt.title("R-index by cluster")
    plt.ylabel("R-index")
    plt.savefig(file_path, format="tiff", dpi=300, bbox_inches="tight")
    plt.close()


def plot_cluster_stats(
    cluster_stat_number: np.ndarray,
    clusternumber: np.ndarray,
    cluster_score: np.ndarray,
    estimate_order: np.ndarray,
    filepath: str,
) -> None:
    """
    Function to plot cluster stats

    Parameters
    ----------
    cluster_stat_number: np.ndarray
        array of cluster numbers
    clusternumber: np.ndarray
        an array of cluster labels
    cluster_score: np.ndarray
        cluster score for each cluster
    estimate_order: np.ndarray
        the order of cluster from
        high to low
    filepath: str
        file path to save graph to.
        Must include file name

    Returns
    --------
    None
    """
    plt.figure(2)
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(cluster_score, "o-")
    plt.title("Iq")
    plt.xlabel("Number of clusters")
    plt.ylabel("Iq")
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(clusternumber))
    plt.barh(y_pos, np.array(cluster_stat_number)[estimate_order], align="center")
    plt.gca().invert_yaxis()
    plt.xlim(0, max(cluster_stat_number) * 1.05)
    plt.title("Cluster labels ranked by Iq and components")
    plt.xlabel("Number of components")
    plt.ylabel("Label")
    plt.gcf().canvas.manager.set_window_title("NMF-sso: Estimate Quality")
    plt.tight_layout()
    plt.savefig(filepath, format="tiff", dpi=300, bbox_inches="tight")
    plt.close()
