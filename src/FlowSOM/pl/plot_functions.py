import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.colors
import random

from FlowSOM.main import get_channels, get_markers
from matplotlib import collections as mc
from matplotlib import gridspec
from FlowSOM.pl.plot_helper_functions import *


def plot_FlowSOM(
    fsom,
    view: str = "MST",
    background_values: np.array = None,
    background_cmap=gg_color_hue(),
    node_sizes: np.array = None,
    max_node_size: int = 1,
    ref_node_size: int = None,
    equal_node_size: bool = False,
):
    """Plots the base layer of a FlowSOM result

    :param fsom: A FlowSOM object
    :type fsom: A object from the class FlowSOM
    :param view: The view you want to plot, can be either "grid" for
    a grid view or "MST" for a minimum spanning tree view
    :type view: str
    :param background_values: The background values to be plotted
    :type background_values: np.array
    :param background_cmap: A colormap for the background colors
    :type background_cmap: Colormap
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
    # assert nNodes == len(node_sizes), f"Length of \"node_sizes\" should be equal to number of clusters in FlowSOM object"
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
        background = add_nodes(layout, node_sizes * 1.5)
        b = mc.PatchCollection(background, cmap=background_cmap)
        b.set_array(background_values)
        b.set_alpha(0.5)
        b.set_zorder(1)
        ax.add_collection(b)

    # Add MST
    if view == "MST":
        e = add_MST(fsom)
        MST = mc.LineCollection(e)
        MST.set_edgecolor("black")
        MST.set_linewidth(0.5)
        MST.set_zorder(0)
        ax.add_collection(MST)

    # Add nodes
    nodes = add_nodes(layout, node_sizes)
    n = mc.PatchCollection(nodes)
    n.set_facecolor(["#C7C7C7" if tf else "#FFFFFF" for tf in isEmpty])  # "white")
    n.set_edgecolor("black")
    n.set_linewidth(0.5)
    n.set_zorder(2)
    ax.add_collection(n)

    return fig, ax, layout, node_sizes


def plot_2D_scatters(
    fsom,
    channelpairs: np.array,
    clusters: np.array = None,
    metaclusters: np.array = None,
    max_background_points: int = 3000,
    size_background_points: float = 0.5,
    max_points: int = 1000,
    size_points: float = 0.5,
    x_lim: np.array = None,
    y_lim: np.array = None,
    xy_labels: list = ["marker"],
    density: bool = True,
    centers: bool = True,
    colors: np.array = None,
    plot_file: str = "2DScatterPlots.png",
    title=None,
):
    """Function to draw 2D scatter plots of FlowSOM (meta)clusters

    :param fsom:
    :type fsom:
    :param channelpairs:
    :type channelpairs:
    :param clusters:
    :type clusters:
    :param metaclusters:
    :type metaclusters:
    :param max_background_points:
    :type max_background_points:
    :param size_background_points:
    :type size_background_points:
    :param max_points:
    :type max_points:
    :param size_points:
    :type size_points:
    :param x_lim:
    :type x_lim:
    :param y_lim:
    :type y_lim:
    :param xy_labels:
    :type xy_labels:
    :param density:
    :type density: boolean
    :param centers:
    :type centers:
    :param colors:
    :type colors:
    :param plot_file:
    :type plot_file:
    """
    assert metaclusters is not None or clusters is not None, f"Please add clusters or metaclusters to plot."
    assert (
        "marker" in xy_labels or "channel" in xy_labels
    ), f'xy_labels should be a list containing "marker" and/or "channel".'
    metacluster = fsom.get_cell_data().obs["metaclustering"]

    cell_cluster = fsom.get_cell_data().obs["clustering"]

    bgI = random.sample(
        range(fsom.get_cell_data().X.shape[0]), min([fsom.get_cell_data().X.shape[0], max_background_points])
    )

    if clusters is None:
        clusters = []
    if metaclusters is None:
        metaclusters = []
    # fig, axs = plt.subplots(len(channelpairs), len(clusters) + len(metaclusters))
    fig = plt.figure()
    spec = gridspec.GridSpec(ncols=len(channelpairs), nrows=len(clusters) + len(metaclusters))
    subsets = {"Cluster": np.array(clusters), "Metacluster": np.array(metaclusters)}
    for i, group in enumerate(subsets.keys()):
        if len(subsets[group] == 0):
            next
        for j, subset in enumerate(subsets[group]):
            """
            color += 1
            """
            rowI = i + j
            n = int(subset)

            for k, channelpair in enumerate(channelpairs):
                channelpair = list(get_channels(fsom, channelpair).keys())
                df_bg = np.array(fsom.get_cell_data().X[bgI, :])
                indices_markers = (np.asarray(fsom.get_cell_data().var_names)[:, None] == channelpair).argmax(axis=0)
                df_bg = np.take(df_bg, indices_markers, axis=1)
                if group == "Cluster":
                    clusters_OI = np.isin(cell_cluster, n)
                    df_ss = fsom.get_cell_data().X[clusters_OI, :]
                    df_ss = df_ss[:, indices_markers]
                    df_ss = np.c_[df_ss, cell_cluster[clusters_OI]]

                else:
                    df_ss = fsom.get_cell_data().X[np.isin(metacluster.astype(int), n), :]
                    df_ss = df_ss[:, indices_markers]

                if len(xy_labels) == 1 and xy_labels[0] == "channel":
                    xy_label = list(get_channels(fsom, channelpair).keys())
                elif len(xy_labels) == 1 and xy_labels[0] == "marker":
                    xy_label = list(get_markers(fsom, channelpair).keys())
                else:
                    channel_label = list(get_channels(fsom, channelpair).keys())
                    marker_label = list(get_markers(fsom, channelpair).keys())
                    xy_label = [marker_label[i] + " (" + channel_label[i] + ")" for i in range(len(channel_label))]
                ssI = random.sample(range(df_ss.shape[0]), min([df_ss.shape[0], max_points]))
                df_ss = df_ss[ssI, :]
                ax = fig.add_subplot(spec[rowI + (rowI + k)])
                ax.scatter(df_bg[:, 0], df_bg[:, 1], c="grey")
                ax.scatter(df_ss[:, 0], df_ss[:, 1], c="red")
                ax.set(xlabel=xy_label[0], ylabel=xy_label[1])


def plot_numbers(fsom, level="clusters", max_node_size=0, **kwargs):
    """Plot cluster ids for each cluster

    :param level:
    :type level:
    :param max_node_size:
    :type max_node_size: int
    """
    assert level in ["clusters", "metaclusters"], f"level should be clusters or metaclusters"
    if level == "clusters":
        numbers = np.arange(1, fsom.get_cell_data().uns["n_nodes"] + 1)
    elif level == "metaclusters":
        numbers = np.asarray(fsom.get_cluster_data().obs["metaclustering"], dtype=int)
    plot_labels(fsom=fsom, labels=numbers, max_node_size=max_node_size, **kwargs)


def plot_labels(
    fsom,
    cell_types,
    cmap=matplotlib.colors.ListedColormap(
        ["white", "#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"]
    ),
    title=None,
    **kwargs,
):
    """Plot FlowSOM grid or tree, with pies indicating another clustering or
    manual gating result

    :param fsom:
    :type fsom:
    :param cell_types:
    :type cell_types:
    :param cmap:
    :type cmap:
    """
    fig, ax, layout, scaledNodeSize = plot_FlowSOM(fsom, **kwargs)

    cell_types["cl"] = fsom.GetClusters()
    cell_types.columns = ["gatingResult", "cl"]
    grouped_df = cell_types.groupby("cl")
    i = 0
    uniquecell_types = np.unique(cell_types["gatingResult"])
    color_dict = dict(zip(uniquecell_types, cmap(np.linspace(0, 1, len(uniquecell_types)))))
    for _, group in grouped_df:
        table = pd.crosstab(index=group["gatingResult"], columns="count")
        table["part"] = np.multiply(np.divide(table["count"], sum(table["count"])), 360)
        angles = np.asarray(np.cumsum(table["part"]))
        if 0 not in angles:
            angles = np.insert(angles, 0, 0)
        row = layout[i, :]
        patches = add_wedges(tuple(row), heights=np.repeat(scaledNodeSize[i], len(angles)), angles=angles)
        p = mc.PatchCollection(patches)
        p.set_facecolor([color_dict.get(key) for key in table.index.values])
        p.set_edgecolor("black")
        p.set_linewidth(0.5)
        p.set_zorder(3)
        ax.add_collection(p)
        i += 1
    ax.axis("equal")
    if title is not None:
        plt.title(title)
    plt.show()
    return (fig, ax)


def plot_variable(fsom, variable, cmap=FlowSOM_colors(), lim=None, title=None, **kwargs):
    """Plot FlowSOM grid or tree, colored by node values given in variable

    :param fsom:
    :type fsom:
    :param variable:
    :type variable:
    :param cmap:
    :type cmap:
    :param lim:
    :type lim:
    """
    # assert isinstance(variable, list) and len(variable) == fsom.nClusters(), f"Length of variable should be the same as the number of nodes in your FlowSOM object"
    fig, ax, layout, scaled_node_size = plot_FlowSOM(fsom, **kwargs)
    nodes = add_nodes(layout, scaled_node_size)
    n = mc.PatchCollection(nodes, cmap=cmap)
    n.set_array(variable)
    if lim is not None:
        n.set_clim(lim)
    n.set_edgecolor("black")
    n.set_linewidth(0.5)
    n.set_zorder(2)
    ax.add_collection(n)
    ax.axis("equal")
    if title is not None:
        plt.title(title)
    plt.show()


def plot_marker(fsom, marker, ref_markers=None, lim=None, cmap=FlowSOM_colors(), **kwargs):
    """Plot FlowSOM grid or tree, colored by node values for a specific marker

    :param fsom:
    :type fsom:
    :param marker:
    :type marker:
    :param ref_markers:
    :type ref_markers:
    :param lim:
    :type lim:
    :param cmap:
    :type cmap:
    """
    if ref_markers is None:
        ref_markers = fsom.get_cell_data().uns["cols_used"]
    mfis = fsom.get_cluster_data().X
    ref_markers = list(get_channels(fsom, ref_markers).keys())
    indices_markers = (np.asarray(fsom.get_cell_data().var_names)[:, None] == ref_markers).argmax(axis=0)
    if lim is None:
        lim = (mfis[:, indices_markers].min(), mfis[:, indices_markers].max())
    marker = list(get_channels(fsom, marker).keys())[0]
    marker_index = np.where(fsom.get_cell_data().var_names == marker)[0][0]
    plot_variable(fsom, variable=mfis[:, marker_index], cmap=cmap, lim=lim, **kwargs)


def plot_labels(fsom, labels, max_node_size=0, text_size=20, text_color="black", title=None, **kwargs):
    """Plot labels for each cluster

    :param fsom:
    :type fsom:
    :param labels:
    :type labels:
    :param max_node_size:
    :type max_node_size:
    :param text_size:
    :type text_size: int
    :param text_color:
    :type text_color:
    """
    fig, ax, layout, _ = plot_FlowSOM(fsom=fsom, max_node_size=max_node_size, **kwargs)
    ax = add_text(ax, layout, labels, text_size, text_color)
    ax.axis("equal")
    if title is not None:
        plt.title(title)
    plt.show()


def plot_star_legend(markers, cmap=FlowSOM_colors(), star_height=1):
    """Function makes the legend of the FlowSOM star plot

    :param markers:
    :type markers:
    :param cmap:
    :type cmap:
    :param star_height:
    :type star_height:
    """
    n_markers = len(markers)
    if isinstance(star_height, int):
        star_height = np.repeat(star_height, len(markers)).tolist()
    else:
        assert len(star_height) == n_markers, f"Make sure star_height is an array with the same length as markers"
    x = 2 * np.pi / (n_markers * 2)
    y = 2 * np.pi / n_markers
    circular_coords = np.linspace(start=x, stop=x + (n_markers - 1) * y, num=n_markers)
    segments = np.column_stack(
        (
            markers,
            [np.cos(x) for x in circular_coords],
            [np.sin(x) for x in circular_coords],
            [1.1 if i >= 0 else -1.1 for i in np.cos(circular_coords)],
            np.repeat(None, len(markers)),
            range(len(markers)),
        )
    )
    n_left_right = segments[:, 1] >= 0
    n_left_right = pd.crosstab(n_left_right, columns="x")
    if n_left_right.shape[0] != 1:
        by = 1 if len(markers) <= 8 else 0.65
        left = np.linspace(start=0, stop=(n_left_right.x[0] - 1) * by, num=n_left_right.x[0])
        right = np.multiply(-1, np.linspace(start=0, stop=(n_left_right.x[1] - 1) * by, num=n_left_right.x[1]))
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
    dfLabels = np.column_stack((x, y, ["left" if i >= 0 else "right" for i in x]))
    fig, ax = plt.subplots()
    lines = []
    for row in segments:
        lines += [[(row[1], row[2]), (row[3], row[4])]]
    e = mc.LineCollection(lines, cmap=cmap)
    e.set_array(range(n_markers))
    e.set_linewidth(2)
    e.set_zorder(0)
    ax.add_collection(e)
    ax = add_text(ax, dfLabels, markers, horizontal_alignment=dfLabels[:, 2])
    coords = np.array((0, 0))
    l = mc.PatchCollection(add_wedges(coords, star_height), cmap=cmap)
    l.set_array(range(n_markers))
    l.set_edgecolor("black")
    l.set_linewidth(0.5)
    ax.add_collection(l)
    ax.axis("equal")
    plt.show()


def plot_stars(fsom, markers=None, cmap=FlowSOM_colors(), title=None, **kwargs):
    """Plot star charts

    :param fsom:
    :type fsom:
    :param markers:
    :type markers:
    :param cmap:
    :type cmap:
    """
    if markers is None:
        markers_bool = fsom.get_cell_data().uns["cols_used"]
        markers = fsom.get_cell_data().var_names[markers_bool]
    fig, ax, layout, scaled_node_size = plot_FlowSOM(fsom, **kwargs)
    data = fsom.get_cluster_data()[:, markers].X
    heights = scale_star_heights(data, scaled_node_size)
    s = mc.PatchCollection(add_stars(layout, heights), cmap=cmap)
    s.set_array(range(data.shape[1]))
    s.set_edgecolor("black")
    s.set_linewidth(0.5)
    s.set_zorder(3)
    ax.add_collection(s)
    ax.axis("equal")
    if title is not None:
        plt.title(title)
    plt.savefig("star_chart.png", dpi=300)
    # plt.show()


def plot_pies(
    fsom,
    cell_types,
    cmap=matplotlib.colors.ListedColormap(
        ["white", "#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"]
    ),
    title=None,
    **kwargs,
):
    """Plot FlowSOM grid or tree, with pies indicating another clustering or
    manual gating result

    :param fsom:
    :type fsom:
    :param cell_types:
    :type cell_types:
    :param cmap:
    :type cmap:
    """

    fig, ax, layout, scaled_node_size = plot_FlowSOM(fsom, **kwargs)
    unique_cell_types = np.unique(cell_types)
    color_dict = dict(zip(unique_cell_types, cmap(np.linspace(0, 1, len(unique_cell_types)))))

    for cl in range(fsom.get_cell_data().uns["n_nodes"]):
        node_cell_types = cell_types[fsom.get_cell_data().obs["clustering"] == (cl + 1)]
        table = pd.crosstab(node_cell_types, columns="count")
        table["part"] = np.multiply(np.divide(table["count"], sum(table["count"])), 360)
        angles = np.asarray(np.cumsum(table["part"]))
        if 0 not in angles:
            angles = np.insert(angles, 0, 0)
        row = layout[cl, :]
        patches = add_wedges(tuple(row), heights=np.repeat(scaled_node_size[cl], len(angles)), angles=angles)
        p = mc.PatchCollection(patches)
        p.set_facecolor([color_dict.get(key) for key in table.index.values])
        p.set_edgecolor("black")
        p.set_linewidth(0.5)
        p.set_zorder(3)
        ax.add_collection(p)

    ax.axis("equal")
    if title is not None:
        plt.title(title)
    plt.show()
