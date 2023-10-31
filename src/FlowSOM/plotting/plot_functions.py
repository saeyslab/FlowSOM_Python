import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import anndata as ad
import matplotlib.colors
import matplotlib.backends.backend_pdf

import matplotlib.colors

from ..tools import get_channels, get_markers
from ..preprocessing import aggregate_flowframes
from ..read_write import read_FCS
from matplotlib import collections as mc
from matplotlib import gridspec
from ._plot_helper_functions import *
from scipy.stats import gaussian_kde
from scanpy.tools import umap
from scanpy.preprocessing import neighbors


def plot_2D_scatters(
    fsom,
    channelpairs: list,
    clusters: list = None,
    metaclusters: list = None,
    max_background_points: int = 3000,
    size_background_points: float = 0.5,
    max_points: int = 1000,
    size_points: float = 0.5,
    x_lim: tuple = None,
    y_lim: tuple = None,
    xy_labels: list = ["marker"],
    density: bool = True,
    centers: bool = True,
    colors: np.array = None,
    plot_file: str = "2DScatterPlots.png",
    title=None,
):
    """Function to draw 2D scatter plots of FlowSOM (meta)clusters

    :param fsom: A FlowSOM object
    :type fsom: FlowSOM
    :param channelpairs: A list of list with channels/markers to plot
    :type channelpairs: list
    :param clusters: A list of lists with cluster ids to plot
    :type clusters: list
    :param metaclusters: A list of lists with metacluster ids to plot
    :type metaclusters: list
    :param max_background_points: Maximum number of background points to plot. Default is 3000.
    :type max_background_points: int
    :param size_background_points: Size of background points. Default is 0.5.
    :type size_background_points: float
    :param max_points: Maximum number of points to plot. Default is 1000.
    :type max_points: int
    :param size_points: Size of points. Default is 0.5.
    :type size_points: float
    :param x_lim: Limits for the x-axis. Default is None.
    :type x_lim: tuple
    :param y_lim: Limits for the y-axis. Default is None.
    :type y_lim: tuple
    :param xy_labels: Labels for the x and y axis. Default is ["marker"]. Can be 'marker' and/or 'channel'.
    :type xy_labels: list
    :param density: Whether to plot density or not. Default is True.
    :type density: boolean
    :param centers: Whether to plot centers or not. Default is True.
    :type centers: boolean
    :param colors:
    :type colors:
    :param plot_file:
    :type plot_file:
    """
    assert metaclusters is not None or clusters is not None, f"Please add clusters or metaclusters to plot."
    assert (
        "marker" in xy_labels or "channel" in xy_labels
    ), f'xy_labels should be a list containing "marker" and/or "channel".'
    if clusters is not None:
        assert isinstance(clusters[0], list), f"clusters should be a list of lists."
    if metaclusters is not None:
        assert isinstance(metaclusters[0], list), f"metaclusters should be a list of lists."
    assert isinstance(channelpairs[0], list), f"channelpairs should be a list of lists."

    cell_metacluster = fsom.get_cell_data().obs["metaclustering"]
    n_metaclusters = fsom.get_cell_data().uns["n_metaclusters"]
    cell_cluster = fsom.get_cell_data().obs["clustering"]
    metacluster = fsom.get_cluster_data().obs["metaclustering"]
    bgI = np.random.choice(
        range(fsom.get_cell_data().X.shape[0]),
        min([fsom.get_cell_data().X.shape[0], max_background_points]),
        replace=False,
    )
    if clusters is None:
        clusters = []
    if metaclusters is None:
        metaclusters = []
    fig = plt.figure(figsize=(len(channelpairs) * 2, (len(clusters) + len(metaclusters)) * 2))
    rowI = 0
    spec = gridspec.GridSpec(ncols=len(channelpairs), nrows=len(clusters) + len(metaclusters))
    subsets = {"Cluster": np.array(clusters, dtype="object"), "Metacluster": np.array(metaclusters, dtype="object")}
    for group in subsets.keys():
        for subset in subsets[group]:
            n = [int(x) for x in subset]
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
                    col = np.asarray([gg_color_hue()(i) for i in range(n_metaclusters)])[
                        np.array([int(i) for i in metacluster])[n]
                    ]
                    col_dict = {i: j for i, j in zip(np.unique(df_ss[:, 2]), col)}
                    cols = [col_dict[i] for i in df_ss[:, 2]]

                    df_c = np.c_[fsom.get_cluster_data()[n, indices_markers].X, n]
                else:
                    metaclusters_OI = np.isin(cell_metacluster.astype(int), n)
                    df_ss = fsom.get_cell_data().X[metaclusters_OI, :]
                    df_ss = df_ss[:, indices_markers]
                    df_ss = np.c_[df_ss, cell_metacluster[metaclusters_OI]]
                    col = np.asarray([gg_color_hue()(i) for i in range(n_metaclusters)])[n]
                    col_dict = {i: j for i, j in zip(np.unique(df_ss[:, 2]), col)}
                    cols = [col_dict[i] for i in df_ss[:, 2]]
                    cl_in_mcl = np.where(np.isin(metacluster.astype(int), n))[0]
                    df_c = np.c_[fsom.get_cluster_data()[cl_in_mcl, indices_markers].X, cl_in_mcl]
                plural = "s" if len(subset) > 1 else ""
                cl_or_mcl = group + plural + ": "
                subset_names = (
                    ", ".join([str(i) for i in subset]) if group == "Cluster" else ", ".join([str(i) for i in subset])
                )
                title = cl_or_mcl + subset_names
                if len(xy_labels) == 1 and xy_labels[0] == "channel":
                    xy_label = list(get_channels(fsom, channelpair).keys())
                elif len(xy_labels) == 1 and xy_labels[0] == "marker":
                    xy_label = list(get_markers(fsom, channelpair).keys())
                else:
                    channel_label = list(get_channels(fsom, channelpair).keys())
                    marker_label = list(get_markers(fsom, channelpair).keys())
                    xy_label = [marker_label[i] + " (" + channel_label[i] + ")" for i in range(len(channel_label))]
                ssI = np.random.choice(range(df_ss.shape[0]), min([df_ss.shape[0], max_points]), replace=False)
                df_ss = df_ss[ssI, :]
                cols = np.array(cols)[ssI, :]
                ax = fig.add_subplot(spec[rowI, k])
                ax.scatter(df_bg[:, 0], df_bg[:, 1], c="grey", s=size_background_points)
                cmap = None
                if density:
                    cols = gaussian_kde(df_ss[:, [0, 1]].T)(df_ss[:, [0, 1]].T)
                    cmap = "jet"
                ax.scatter(df_ss[:, 0], df_ss[:, 1], c=cols, s=size_points, cmap=cmap)
                if centers:
                    ax.scatter(df_c[:, 0], df_c[:, 1], c="black", s=10)
                ax.set(xlabel=xy_label[0], ylabel=xy_label[1])
                if x_lim is not None:
                    ax.set_xlim(x_lim)
                if y_lim is not None:
                    ax.set_ylim(y_lim)
                ax.set_title(title)
            rowI += 1

    return fig


def plot_file_scatters(
    input,
    file_id="File",
    channels=None,
    y_lim=None,
    y_label="marker",
    quantiles=None,
    names=None,
    groups=None,
    color=None,
    legend=False,
    max_points=50000,
    n_col=None,
    nrow=None,
    width=None,
    height=None,
    silent=None,
    plot_file="file_scatters.png",
):
    if color is not None and groups is not None and len(np.unique(groups)) != len(color):
        raise ValueError("Number of colors should be equal to the number of groups")

    if color is not None and groups is None and len(color) != 1:
        raise ValueError("Number of colors should be equal to 1")

    if isinstance(input, ad.AnnData):
        if channels is None:
            channels = input.var.channel.to_numpy()
        ff = input
        data = input[:, [channels, file_id]].to_df()
        file_values = data[:, file_id]
        input = np.unique(file_values)
    else:
        if channels is None:
            channels = read_FCS(input[0]).var.channel.to_numpy()
        ff = aggregate_flowframes(input, c_total=max_points, channels=channels)
        data = ff.to_df()
        file_values = ff.obs[file_id]

    subset = np.random.choice(range(data.shape[0]), min([max_points, data.shape[0]]), replace=False)
    data = data.iloc[subset, :]
    file_values = file_values[subset]

    if names is not None and len(np.unique(file_values)) != len(names):
        raise ValueError("Number of names should be equal to the number of files")
    if groups is not None and len(np.unique(file_values)) != len(groups):
        raise ValueError("Number of groups should be equal to the number of files")

    if names is None:
        names = range(len(input))

    if groups is None:
        groups = np.repeat("0", len(np.unique(file_values)))

    spec = gridspec.GridSpec(ncols=1, nrows=len(channels))
    fig = plt.figure(figsize=(len(np.unique(file_values)) * 2, len(channels) * 1))
    for i, channel in enumerate(channels):
        if len(y_label) == 1 and y_label[0] == "channel":
            y_label = list(get_channels(ff, [channel]).keys())
        elif len(y_label) == 1 and y_label[0] == "marker":
            y_label = list(get_markers(ff, [channel]).keys())
        else:
            channel_label = list(get_channels(ff, [channel]).keys())
            marker_label = list(get_markers(ff, [channel]).keys())
            y_label = [marker_label[i] + " (" + channel_label[i] + ")" for i in range(len(channel_label))]
        ax = fig.add_subplot(spec[i, 0])
        marker = list(get_markers(ff, [channel]).keys())[0]
        ax.scatter(np.random.normal(file_values, 0.1, data.shape[0]), data.loc[:, marker], s=0.5)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        ax.set(ylabel=y_label)
    return fig


def plot_labels(fsom, labels, max_node_size=0, text_size=20, text_color="black", title=None, **kwargs):
    """Plot labels for each cluster

    :param fsom: A FlowSOM object
    :type fsom: FlowSOM
    :param labels: An array of labels for every node
    :type labels: np.array
    :param max_node_size: Determines the maximum node size. Default = 0.
    :type max_node_size: float
    :param text_size: Determines the size of the text. Default = 20.
    :type text_size: float
    :param text_color: Determines the color of the text. Default = "black".
    :type text_color: str
    :param title: Title of the plot
    :type title: str
    """
    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)
    assert (
        labels.shape[0] == fsom.get_cell_data().uns["n_nodes"]
    ), f"Length of labels should be the same as the number of nodes in your FlowSOM object"
    fig, ax, layout, _ = plot_FlowSOM(fsom=fsom, max_node_size=max_node_size, **kwargs)
    ax = add_text(ax, layout, labels, text_size, text_color)
    ax.axis("equal")
    if title is not None:
        plt.title(title)
    plt.axis("off")
    return fig


def plot_numbers(fsom, level="clusters", max_node_size=0, **kwargs):
    """Plot cluster ids for each cluster
    :param fsom: A FlowSOM object
    :type fsom: FlowSOM
    :param level: Should be either "clusters" (default) or "metaclusters".
    :type level: str
    :param max_node_size: Determines the maximum node size. Default is 0.
    :type max_node_size: float
    """
    assert level in ["clusters", "metaclusters"], f"level should be clusters or metaclusters"
    if level == "clusters":
        numbers = np.arange(1, fsom.get_cell_data().uns["n_nodes"] + 1)
    elif level == "metaclusters":
        numbers = np.asarray(fsom.get_cluster_data().obs["metaclustering"], dtype=int)
    plot_labels(fsom=fsom, labels=numbers, max_node_size=max_node_size, **kwargs)


def plot_variable(
    fsom, variable, cmap=FlowSOM_colors(), labels=None, text_size=5, text_color="black", lim=None, title=None, **kwargs
):
    """Plot FlowSOM grid or tree, colored by node values given in variable

    :param fsom: A FlowSOM object
    :type fsom: FlowSOM
    :param variable: An array containing a value for every cluster
    :type variable: np.array
    :param cmap: A colormap to use
    :type cmap:
    :param lim: Limits for the color scale
    :type lim: tuple
    :param title: Title of the plot
    :type title: str
    """
    if not isinstance(variable, np.ndarray):
        variable = np.asarray(variable)
    assert (
        variable.shape[0] == fsom.get_cell_data().uns["n_nodes"]
    ), f"Length of variable should be the same as the number of nodes in your FlowSOM object"
    if variable.dtype == "object":
        string_to_number = {string: index for index, string in enumerate(np.unique(variable))}
        variable = np.asarray([string_to_number[string] for string in variable])
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
    if labels is not None:
        ax = add_text(ax, layout, labels, text_size=text_size, text_color=text_color, ha=["center"], va=["center"])
    ax, fig = add_legend(
        fig=fig, ax=ax, data=variable, title="Marker", cmap=cmap, location="upper left", bbox_to_anchor=(1.04, 1)
    )
    ax.axis("equal")
    if title is not None:
        plt.title(title)
    plt.axis("off")
    return fig


def plot_marker(fsom, marker, ref_markers=None, lim=None, cmap=FlowSOM_colors(), **kwargs):
    """Plot FlowSOM grid or tree, colored by node values for a specific marker

    :param fsom: A FlowSOM object
    :type fsom: FlowSOM
    :param marker: A marker to plot. Can be a channel/marker/index.
    :type marker: str, int
    :param ref_markers: Is used to determine relative scale of the marker that will be plotted. Default are all markers used in the clustering.
    :type ref_markers: np.array,  list
    :param lim: Limits for the color scale
    :type lim: tuple
    :param cmap: A colormap to use
    :type cmap:
    """
    if ref_markers is None:
        ref_markers_bool = fsom.get_cell_data().var["cols_used"]
        ref_markers = fsom.get_cell_data().var_names[ref_markers_bool]

    mfis = fsom.get_cluster_data().X
    ref_markers = list(get_channels(fsom, ref_markers).keys())
    indices_markers = (np.asarray(fsom.get_cell_data().var_names)[:, None] == ref_markers).argmax(axis=0)
    if lim is None:
        lim = (mfis[:, indices_markers].min(), mfis[:, indices_markers].max())
    marker = list(get_channels(fsom, marker).keys())[0]
    marker_index = np.where(fsom.get_cell_data().var_names == marker)[0][0]
    fig = plot_variable(fsom, variable=mfis[:, marker_index], cmap=cmap, lim=lim, **kwargs)
    return fig


def plot_stars(fsom, markers=None, cmap=FlowSOM_colors(), title=None, **kwargs):
    """Plot star charts

    :param fsom: A FlowSOM object
    :type fsom: FlowSOM
    :param markers: Markers, channels or indices to plot
    :type markers: np.array
    :param cmap: A colormap to use
    :type cmap:
    :param title: Title of the plot
    :type title: str
    """
    if markers is None:
        markers_bool = fsom.get_cell_data().var["cols_used"]
        markers = fsom.get_cell_data().var_names[markers_bool]
    if not isinstance(markers, np.ndarray):
        markers = np.asarray(markers)
    pretty_markers = fsom.get_cell_data()[:, markers].var["pretty_colnames"]
    fig, ax, layout, scaled_node_size = plot_FlowSOM(fsom, **kwargs)
    max_x, max_y = np.max(layout, axis=0)
    fig, ax = plot_star_legend(
        fig,
        ax,
        pretty_markers,
        coords=(max_x, max_y),
        cmap=cmap,
        max_star_height=max(scaled_node_size) * 3,
        star_height=1,
    )
    data = fsom.get_cluster_data()[:, markers].X
    heights = scale_star_heights(data, scaled_node_size)
    s = mc.PatchCollection(add_stars(layout, heights), cmap=cmap)
    s.set_array(range(data.shape[1]))
    s.set_edgecolor("black")
    s.set_linewidth(0.1)
    s.set_zorder(3)
    ax.add_collection(s)
    ax.axis("equal")
    plt.axis("off")
    if title is not None:
        plt.title(title)
    return fig


def plot_pies(
    fsom,
    cell_types,
    cmap=matplotlib.colors.ListedColormap(
        ["white", "#00007F", "#0000E1", "#007FFF", "#00E1E1", "#7FFF7F", "#E1E100", "#FF7F00", "#E10000", "#7F0000"],
    ),
    title=None,
    **kwargs,
):
    """Plot FlowSOM grid or tree, with pies indicating another clustering or
    manual gating result

    :param fsom: A FlowSOM object
    :type fsom: FlowSOM
    :param cell_types:  An array containing a value for every cell
    :type cell_types: np.array
    :param cmap: A colormap to use
    :type cmap:
    :param title: Title of the plot
    :type title: str
    """
    if not isinstance(cell_types, np.ndarray):
        cell_types = np.asarray(cell_types)
    assert (
        cell_types.shape[0] == fsom.get_cell_data().shape[0]
    ), f"Length of cell_types should be the same as the number of cells in your FlowSOM object"
    fig, ax, layout, scaled_node_size = plot_FlowSOM(fsom, **kwargs)
    unique_cell_types = np.unique(cell_types)
    color_dict = dict(zip(unique_cell_types, cmap(np.linspace(0, 1, len(unique_cell_types)))))

    for cl in range(fsom.get_cell_data().uns["n_nodes"]):
        node_cell_types = cell_types[fsom.get_cell_data().obs["clustering"] == cl]
        if len(node_cell_types) != 0:
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
            p.set_linewidth(0.1)
            p.set_zorder(3)
            ax.add_collection(p)

    ax.axis("equal")
    ax, fig = add_legend(
        fig=fig,
        ax=ax,
        data=cell_types,
        title="",
        cmap=cmap,
        location="upper left",
        orientation="vertical",
        bbox_to_anchor=(1.04, 1),
    )
    if title is not None:
        plt.title(title)
    plt.axis("off")
    return fig


def FlowSOMmary(fsom, plot_file="./FlowSOMmary.pdf"):
    """Makes a visual FlowSOMmary of a FlowSOM object

    :param fsom: A FlowSOM object
    :type fsom: FlowSOM
    :param plot_file: File name of the plot
    :type plot_file: str
    """
    # Initializing
    metacluster_present = "cluster_data" in fsom.mudata.mod.keys()
    if metacluster_present:
        mfis = fsom.get_cluster_data().uns["metacluster_MFIs"]
        metaclusters = fsom.get_cell_data().obs["metaclustering"]
        n_metaclusters = fsom.get_cell_data().uns["n_metaclusters"]
    clusters = fsom.get_cell_data().obs["clustering"]
    n_clusters = np.arange(fsom.get_cell_data().uns["n_nodes"])
    file_present = False
    plot_dict = dict()

    # Plot fsom trees and grids
    for view in ["MST", "grid"]:
        if metacluster_present:
            plot_dict["stars_" + view] = plot_stars(
                fsom,
                background_values=fsom.get_cluster_data().obs.metaclustering,
                title="FlowSOM " + view,
                view=view,
            )

            plot_dict["mcl_labels_" + view] = plot_variable(
                fsom,
                variable=fsom.get_cluster_data().obs.metaclustering,
                equal_node_size=True,
                view=view,
                labels=fsom.get_cluster_data().obs.metaclustering,
                cmap=gg_color_hue(),
                title="FlowSOM Metaclusters",
            )

            plot_dict["cl_labels_" + view] = plot_variable(
                fsom,
                variable=fsom.get_cluster_data().obs.metaclustering,
                equal_node_size=True,
                view=view,
                labels=np.arange(fsom.get_cell_data().uns["n_nodes"]),
                cmap=gg_color_hue(),
                title="FlowSOM Clusters",
            )
        else:
            plot_dict["stars_" + view] = plot_stars(fsom, title="FlowSOM" + view, view=view)
            plot_dict["labels_" + view] = plot_numbers(fsom, view=view, title="FlowSOM Clusters", equal_node_size=True)

    # Plot Markers
    ref_markers_bool = fsom.get_cell_data().var["cols_used"]
    ref_markers = fsom.get_cell_data().var_names[ref_markers_bool]
    for marker in ref_markers:
        plot_dict["marker_" + marker] = plot_marker(
            fsom,
            [marker],
            ref_markers=ref_markers,
            equal_node_size=True,
            title=list(get_markers(fsom, [marker]).keys())[0],
        )

    # File distribution
    if file_present:
        pass

    # t-SNE
    subset_fsom = fsom.get_cell_data()[
        np.random.choice(range(fsom.get_cell_data().shape[0]), 5000, replace=False),
        fsom.get_cell_data().var_names[ref_markers_bool],
    ]
    subset_fsom
    neighbors(subset_fsom)
    umap(subset_fsom)

    for marker in ref_markers:
        fig, ax = plt.subplots()
        ax.scatter(
            x=subset_fsom.obsm["X_umap"][:, 0],
            y=subset_fsom.obsm["X_umap"][:, 1],
            c=subset_fsom.to_df().loc[:, marker],
        )
        ax.set_title("UMAP: " + list(get_markers(fsom, [marker]).keys())[0])
        plot_dict["umap_" + marker] = fig
    if metacluster_present:
        subset_fsom.obs["metaclustering"] = subset_fsom.obs["metaclustering"].astype(str)
        colors = {
            i: j
            for i, j in zip(
                np.unique(subset_fsom.obs["metaclustering"]),
                gg_color_hue()(np.linspace(0, 1, n_metaclusters)),
            )
        }
        fig, ax = plt.subplots()
        ax.scatter(
            x=subset_fsom.obsm["X_umap"][:, 0],
            y=subset_fsom.obsm["X_umap"][:, 1],
            c=subset_fsom.obs["metaclustering"].map(colors),
        )
        ax.set_title("UMAP: metaclustering")
        plot_dict["umap_metaclustering"] = fig

    # Heatmap
    data_heatmap = fsom.get_cluster_data().uns["metacluster_MFIs"]
    data_heatmap.columns = fsom.get_cluster_data().var_names
    data_heatmap_subset = data_heatmap.loc[:, fsom.get_cell_data().var.cols_used.to_list()]
    plot_dict["heatmap"] = sns.clustermap(data_heatmap_subset, z_score=1, annot=data_heatmap_subset, cbar=True).fig

    # Tables
    table1 = pd.DataFrame(
        {
            "Total number of cells": [fsom.get_cell_data().shape[0]],
            "Total number of metaclusters": [n_metaclusters],
            "Total number of clusters": len(n_clusters),
            "Markers used for FlowSOM": ", ".join(
                fsom.get_cell_data().var["pretty_colnames"][ref_markers_bool].to_list()
            ),
        }
    ).transpose()
    fig, ax = plt.subplots(1, 1)
    ax.axis("tight")
    ax.axis("off")
    ax.table(
        cellText=table1.values,
        rowLabels=table1.index,
        colLabels=["FlowSOMmary"],
        loc="center",
    )
    plot_dict["table1"] = fig

    # Plot
    pdf = matplotlib.backends.backend_pdf.PdfPages(plot_file)
    for fig in plot_dict.keys():
        pdf.savefig(plot_dict[fig])
    pdf.close()
