import matplotlib.colors
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Wedge
from scipy.spatial.distance import pdist


def FlowSOM_colors():
    """Colormap of default FlowSOM colors"""
    cmap = matplotlib.colors.ListedColormap(
        ["#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"]
    )
    return cmap


def gg_color_hue():
    """Colormap of default ggplot colors"""
    cmap = matplotlib.colors.ListedColormap(
        ["#F8766D", "#D89000", "#A3A500", "#39B600", "#00BF7D", "#00BFC4", "#00B0F6", "#9590FF", "#E76BF3", "#FF62BC"]
    )
    return cmap


def scale_star_heights(median_values, node_sizes):
    max_all_nodes = median_values[~np.isnan(median_values)].max()
    min_all_nodes = median_values[~np.isnan(median_values)].min()
    scaled_row = [
        np.divide(np.multiply(np.subtract(row, min_all_nodes), node_sizes[i]), max_all_nodes - min_all_nodes)
        for i, row in enumerate(median_values)
    ]
    return np.vstack(scaled_row)


def parse_node_sizes(fsom, view="MST", node_sizes=None, max_node_size=1, ref_node_size=None, equal_node_size=False):
    node_sizes = fsom.get_cluster_data().obs["percentages"] if node_sizes == None else node_sizes
    ref_node_size = max(node_sizes) if ref_node_size == None else ref_node_size
    layout = fsom.get_cluster_data().obsm["layout"] if view == "MST" else fsom.get_cluster_data().obsm["grid"]
    auto_node_size = auto_max_node_size(layout, 1 if view == "MST" else -0.3)  # overlap
    max_node_size = auto_node_size * max_node_size

    if equal_node_size:
        node_sizes = np.repeat(max_node_size, len(node_sizes))
    n_nodes = len(node_sizes)
    if len(np.unique(node_sizes)) == 1:
        return np.repeat(max_node_size, n_nodes)
    scaled_node_size = np.sqrt(np.multiply((np.divide(node_sizes, ref_node_size)), np.square(max_node_size)))
    return scaled_node_size


def auto_max_node_size(layout, overlap):
    overlap = 1 + overlap
    min_distance = min(pdist(layout))
    return min_distance / 2 * overlap


def add_text(ax, layout, text, text_size=20, text_color="black", horizontal_alignment=["right"]):
    if len(horizontal_alignment) == 1:
        horizontal_alignment = np.repeat(horizontal_alignment, len(text))
    for i, row in enumerate(layout):
        ax.text(row[0], row[1], text[i], size=text_size, horizontalalignment=horizontal_alignment[i], clip_on=False)
    return ax


def add_MST(fsom):
    edges = parse_edges(fsom)
    lines = [[(row[0], row[1]), (row[2], row[3])] for row in edges]
    return lines


def parse_edges(fsom):
    edge_list = fsom.get_cluster_data().uns["graph"].get_edgelist()
    coords = fsom.get_cluster_data().obsm["layout"]
    segment_plot = [
        (coords[nodeID[0], 0], coords[nodeID[0], 1], coords[nodeID[1], 0], coords[nodeID[1], 1]) for nodeID in edge_list
    ]
    return np.asarray(segment_plot, dtype=np.float32)


def add_nodes(layout, heights):
    patches = [Circle((row[0], row[1]), heights[i]) for i, row in enumerate(layout)]
    return patches


def add_stars(layout, heights):
    patches = np.hstack([add_wedges((row[0], row[1]), heights[i, :]) for i, row in enumerate(layout)])
    return patches


def add_wedges(coord, heights, angles=None):
    if angles is None:
        part = 360 / len(heights)
        angles = np.arange(0, 360.01, part)
    stars = [Wedge(coord, heights[i], angles[i], angles[i + 1], edgecolor="black") for i in range(len(angles) - 1)]
    return stars
