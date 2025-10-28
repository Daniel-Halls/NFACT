import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import networkx as nx
import numpy as np
from itertools import combinations


def build_graph(n_points, coordinates):
    graph = nx.Graph()
    for node in range(n_points):
        graph.add_node(node, pos=coordinates[node])
    return graph


def build_edges(n_points: int, similarity: np.ndarray, threshold: float) -> dict:
    """
    Function to calssify edges of similairty matrix as either
    greater or less than a given threshold
    """
    edges_above = []
    edges_below = []

    for node1, node2 in combinations(range(n_points), 2):
        if similarity[node1, node2] > threshold:
            edges_above.append((node1, node2))
        else:
            edges_below.append((node1, node2))

    return {"edge_above": edges_above, "edges_below": edges_below}


def plot_network(
    coordinates,
    partition,
    similarity,
    centers,
    filepath,
    threshold=0.5,
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
    threshold : float, optional
        Similarity threshold for edges
    expand_factor : float, optional
        Factor to expand cluster outlines (1.0 = no expansion)

    Returns
    --------
    None
    """
    n_points = coordinates.shape[0]
    graph = build_graph(n_points, coordinates)

    # Add edges and classify them based on threshold
    edges_above, edges_below = [], []
    for i in range(n_points):
        for j in range(i + 1, n_points):
            if similarity[i, j] > threshold:
                edges_above.append((i, j))
            else:
                edges_below.append((i, j))

    pos = {i: coordinates[i] for i in range(n_points)}

    cluster_nodes = [node for node in range(n_points) if node not in centers]
    centroid_nodes = centers

    plt.figure(figsize=(8, 8))
    cluster_color = "#2487ea"  # light blue
    centroid_color = "#f86d6d"
    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=cluster_nodes,
        node_color=cluster_color,
        edgecolors="black",
        linewidths=1.5,
        node_size=150,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=centroid_nodes,
        node_color=centroid_color,
        edgecolors="black",
        linewidths=2,
        node_size=200,
    )

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, edgelist=edges_below, edge_color="grey", alpha=0.3, width=1
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=edges_above, edge_color="black", alpha=0.7, width=1.5
    )

    # Label centroid nodes
    for idx, center in enumerate(centroid_nodes, start=1):
        x, y = pos[center]
        plt.text(
            x,
            y + 0.02,
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
    plt.savefig(
        f"{filepath}/cluster_network.tiff", format="tiff", dpi=300, bbox_inches="tight"
    )
    plt.close()
