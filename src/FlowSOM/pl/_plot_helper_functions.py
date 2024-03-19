from __future__ import annotations

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import collections as mc
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge
from scipy.spatial.distance import pdist


def FlowSOM_colors():
    """Colormap of default FlowSOM colors."""
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "FlowSOM_colors",
        ["#00007F", "#0000E1", "#007FFF", "#00E1E1", "#7FFF7F", "#E1E100", "#FF7F00", "#E10000", "#7F0000"],
    )
    return cmap


def gg_color_hue():
    """Colormap of default ggplot colors."""
    cmap = matplotlib.colors.ListedColormap(
        ["#F8766D", "#D89000", "#A3A500", "#39B600", "#00BF7D", "#00BFC4", "#00B0F6", "#9590FF", "#E76BF3", "#FF62BC"]
    )
    return cmap


def add_legend(fig, ax, data, title, cmap, location="best", orientation="horizontal", bbox_to_anchor=None):
    if data.dtype == np.float64 or data.dtype == np.int64:
        norm = matplotlib.colors.Normalize(vmin=min(data), vmax=max(data))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(data)
        fig.colorbar(sm, ax=ax, orientation=orientation, shrink=0.4, label=title)
    else:
        unique_data = sorted(np.unique(data))
        colors = cmap(np.linspace(0, 1, len(unique_data)))
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", label=unique_data[i], markerfacecolor=colors[i], markersize=5)
            for i in range(len(unique_data))
        ]
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        legend = plt.legend(
            handles=legend_elements,
            loc=location,
            frameon=False,
            title=title,
            bbox_to_anchor=bbox_to_anchor,  # (1, 0.5),
            fontsize=5,
            title_fontsize=6,
        )
        plt.gca().add_artist(legend)
    return ax, fig


def plot_FlowSOM(
    fsom,
    view: str = "MST",
    background_values: np.array = None,
    background_cmap=gg_color_hue(),
    background_size=1.5,
    equal_background_size=False,
    node_sizes: np.array = None,
    max_node_size: int = 1,
    ref_node_size: int = None,
    equal_node_size: bool = False,
):
    """Plots the base layer of a FlowSOM result.

    :param fsom: A FlowSOM object
    :type fsom: A object from the class FlowSOM
    :param view: The view you want to plot, can be either "grid" for
    a grid view or "MST" for a minimum spanning tree view
    :type view: str
    :param background_values: The background values to be plotted
    :type background_values: np.array
    :param background_cmap: A colormap for the background colors
    :type background_cmap: Colormap
    :param background_size: The size of the background nodes relative to the nodes
    :type background_size: float
    :param equal_background_size: If True the background nodes will be equally sized
    :type equal_background_size: boolean
    :param node_sizes: An array with the node sizes. Will be scaled between 0
    and max_node_size and transformed with a sqrt. Default is the percentages
    :type node_sizes: np.array
    :param max_node_size: The maximum node size
    :type max_node_size: float
    :param ref_node_size: Reference for node size against which the node sizes
    will be scaled. Default is the maximum of the node sizes
    :type ref_node_size: float
    :param equal_node_size: If True the all the nodes will be equally sized to
    max_node_size
    :type equal_node_size: boolean
    :param title: Title for the plot
    :type title: str
    """
    # Initialization
    nNodes = fsom.get_cell_data().uns["n_nodes"]
    isEmpty = fsom.get_cluster_data().obs["percentages"] == 0

    # Warnings
    if node_sizes is not None:
        assert nNodes == len(
            node_sizes
        ), 'Length of "node_sizes" should be equal to number of clusters in FlowSOM object'

    if background_values is not None:
        assert (
            background_values.shape[0] == fsom.mudata["cell_data"].uns["n_nodes"]
        ), "Length of background_values should be equal to number of clusters in FlowSOM object"

    # Node sizes
    node_sizes = parse_node_sizes(
        fsom,
        view=view,
        node_sizes=node_sizes,
        max_node_size=max_node_size,
        ref_node_size=ref_node_size,
        equal_node_size=equal_node_size,
    )
    node_sizes[isEmpty] = min([0.05, node_sizes.max()])

    # Layout
    layout = fsom.get_cluster_data().obsm["layout"] if view == "MST" else fsom.get_cluster_data().obsm["grid"]

    # Start plot
    fig, ax = plt.subplots()

    # Add background
    if background_values is not None:
        if equal_background_size:
            background_size = np.repeat(np.max(node_sizes) * background_size, len(background_values))
        else:
            background_size = (
                parse_node_sizes(
                    fsom,
                    view=view,
                    node_sizes=None,
                    max_node_size=max_node_size,
                    ref_node_size=ref_node_size,
                    equal_node_size=False,
                )
                * background_size
            )
        background = add_nodes(layout, background_size)
        b = mc.PatchCollection(background, cmap=background_cmap)
        if background_values.dtype == np.float64 or background_values.dtype == np.int64:
            b.set_array(background_values)
        else:
            b.set_array(pd.get_dummies(background_values).values.argmax(1))
        b.set_alpha(0.5)
        b.set_zorder(1)
        ax.add_collection(b)
        ax, fig = add_legend(
            fig=fig,
            ax=ax,
            data=background_values,
            title="Background",
            cmap=background_cmap,
            location="lower left",
            bbox_to_anchor=(1.04, 0),
        )

    # Add MST
    if view == "MST":
        e = add_MST(fsom)
        MST = mc.LineCollection(e)
        MST.set_edgecolor("black")
        MST.set_linewidth(0.2)
        MST.set_zorder(0)
        ax.add_collection(MST)

    # Add nodes
    nodes = add_nodes(layout, node_sizes)
    n = mc.PatchCollection(nodes)
    n.set_facecolor(["#C7C7C7" if tf else "#FFFFFF" for tf in isEmpty])  # "white")
    n.set_edgecolor("black")
    n.set_linewidth(0.1)
    n.set_zorder(2)
    ax.add_collection(n)

    return fig, ax, layout, node_sizes


def plot_star_legend(fig, ax, markers, coords=(0, 0), cmap=FlowSOM_colors(), max_star_height=1, star_height=1):
    """Function makes the legend of the FlowSOM star plot.

    :param markers:
    :type markers:
    :param cmap:
    :type cmap:
    :param star_height:
    :type star_height:
    """
    n_markers = len(markers)
    if isinstance(star_height, int) | isinstance(star_height, float):
        star_height = np.repeat(star_height, len(markers)).tolist()
    else:
        assert len(star_height) == n_markers, "Make sure star_height is an array with the same length as markers"
    star_height = np.divide(star_height, max(star_height)) * max_star_height
    x = 2 * np.pi / (n_markers * 2)
    y = 2 * np.pi / n_markers
    circular_coords = np.linspace(start=x, stop=x + (n_markers - 1) * y, num=n_markers)
    segments = np.column_stack(
        (
            markers,
            [np.cos(x) * max_star_height for x in circular_coords],
            [np.sin(x) * max_star_height for x in circular_coords],
            [1.1 if i >= 0 else -1.1 for i in np.cos(circular_coords)],
            np.repeat(None, len(markers)),
            range(len(markers)),
        )
    )
    n_left_right = segments[:, 1] >= 0
    n_left_right = pd.crosstab(n_left_right, columns="x")
    if n_left_right.shape[0] != 1:
        by = 1 if len(markers) <= 8 else 0.65
        left = np.linspace(start=0, stop=(n_left_right.x.iloc[0] - 1) * by, num=n_left_right.x.iloc[0])
        right = np.multiply(
            -1, np.linspace(start=0, stop=(n_left_right.x.iloc[1] - 1) * by, num=n_left_right.x.iloc[1])
        )
        segments_left = segments[segments[:, 1] < 0, :]
        segments_left = segments_left[segments_left[:, 2].argsort()]
        segments_right = segments[segments[:, 1] >= 0]
        segments_right = segments_right[segments_right[:, 2].argsort()[::-1]]
        segments = np.concatenate((segments_right, segments_left))
        segments[segments[:, 1] < 0, 4] = left - sum(left) / len(left)
        segments[segments[:, 1] >= 0, 4] = right - sum(right) / len(right)
        segments = segments[segments[:, 5].argsort()]
        segments = np.delete(segments, 5, axis=1)
    else:
        segments[:, 4] = -1
        segments[:, 1] = segments[:, 1] * -1
        segments[:, 3] = segments[:, 3] * -1
    horizontal_lines = np.column_stack(
        (
            segments[:, 0],
            segments[:, 3],
            segments[:, 4],
            np.add(segments[:, 3], [0.5 if i >= 0 else -0.5 for i in segments[:, 3]]),
            segments[:, 4],
        )
    )
    segments = np.concatenate((segments, horizontal_lines))
    x = np.add(horizontal_lines[:, 3], [0.3 if i >= 0 else -0.3 for i in horizontal_lines[:, 3]])
    y = np.asarray(horizontal_lines[:, 4])
    x_coord = coords[0] - min(x) + 0.2 * len(max(markers, key=len))
    dfLabels = np.column_stack((x + x_coord, y + coords[1], ["left" if i >= 0 else "right" for i in x]))
    lines = []
    for row in segments:
        lines += [[(row[1] + x_coord, row[2] + coords[1]), (row[3] + x_coord, row[4] + coords[1])]]
    e = mc.LineCollection(lines, cmap=cmap, capstyle="round", joinstyle="round")
    e.set_array(range(n_markers))
    e.set_linewidth(1)
    e.set_zorder(0)
    ax.add_collection(e)
    ax = add_text(ax, dfLabels, markers, ha=dfLabels[:, 2], text_size=5)
    l = mc.PatchCollection(add_wedges(np.array((x_coord, coords[1])), star_height), cmap=cmap)
    l.set_array(range(n_markers))
    l.set_edgecolor("black")
    l.set_linewidth(0.1)
    ax.add_collection(l)
    ax.axis("equal")

    return fig, ax


def scale_star_heights(median_values, node_sizes):
    if isinstance(node_sizes, pd.Series):
        node_sizes = node_sizes.to_numpy()
    max_all_nodes = median_values[~np.isnan(median_values)].max()
    min_all_nodes = median_values[~np.isnan(median_values)].min()
    scaled_row = [
        np.divide(np.multiply(np.subtract(row, min_all_nodes), node_sizes[i]), max_all_nodes - min_all_nodes)
        for i, row in enumerate(median_values)
    ]
    return np.vstack(scaled_row)


def parse_node_sizes(fsom, view="MST", node_sizes=None, max_node_size=1, ref_node_size=None, equal_node_size=False):
    node_sizes = fsom.get_cluster_data().obs["percentages"] if node_sizes is None else node_sizes
    ref_node_size = max(node_sizes) if ref_node_size is None else ref_node_size
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


def add_text(ax, layout, text, text_size=25, text_color="black", ha=None, va=None):
    if isinstance(text, pd.Series):
        text = text.to_numpy()
    if va is None:
        va = ["center"]
    if ha is None:
        ha = ["right"]
    if len(ha) == 1:
        ha = np.repeat(ha, len(text))
    if len(va) == 1:
        va = np.repeat(va, len(text))
    for i, row in enumerate(layout):
        ax.text(row[0], row[1], text[i], size=text_size, ha=ha[i], va=va[i], c=text_color, clip_on=False)
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
    if isinstance(heights, pd.Series):
        heights = heights.to_numpy()
    patches = [Circle((row[0], row[1]), heights[i]) for i, row in enumerate(layout)]
    return patches


def add_stars(layout, heights):
    if isinstance(heights, pd.Series):
        heights = heights.to_numpy()
    patches = np.hstack([add_wedges((row[0], row[1]), heights[i, :]) for i, row in enumerate(layout)])
    return patches


def add_wedges(coord, heights, angles=None):
    if isinstance(heights, pd.Series):
        heights = heights.to_numpy()
    if angles is None:
        part = 360 / len(heights)
        angles = np.arange(0, 360.01, part)
    stars = [Wedge(coord, heights[i], angles[i], angles[i + 1], edgecolor="black") for i in range(len(angles) - 1)]
    return stars
