import numpy as np
import pandas as pd
import anndata as ad

import re


def get_channels(obj, markers, exact=True):
    """Gets the channels of the provided markers based on a FlowSOM object or an FCS file

    :param obj: A FlowSOM object or a FCS AnnData object
    :type obj: FlowSOM / AnnData
    :param markers: An array of markers
    :type markers: np.array
    :param exact: If True, a strict search is performed. If False, regexps can be used.
    :type exact: boolean
    """
    assert obj.__class__.__name__ == "FlowSOM" or isinstance(
        obj, ad.AnnData
    ), f"Please provide an FCS file or a FlowSOM object"
    if obj.__class__.__name__ == "FlowSOM":
        object_markers = np.asarray(
            [re.sub(" <.*", "", pretty_colname) for pretty_colname in obj.mudata["cell_data"].var["pretty_colnames"]]
        )
        object_channels = np.asarray(
            [
                re.sub(r".*<(.*)>.*", r"\1", pretty_colname)
                for pretty_colname in obj.mudata["cell_data"].var["pretty_colnames"]
            ]
        )
    else:
        object_markers = np.asarray(obj.uns["meta"]["channels"]["$PnS"])
        object_channels = np.asarray(obj.uns["meta"]["channels"]["$PnN"])

    channelnames = dict()
    for marker in markers:
        if isinstance(marker, int):
            i_channel = [marker]
        else:
            if exact:
                marker = r"^" + marker + r"$"
            i_channel = np.asarray([i for i, m in enumerate(object_markers) if re.search(marker, m) is not None])
        if len(i_channel) != 0:
            for i in i_channel:
                channelnames[object_channels[i]] = object_markers[i]
        else:
            i_channel = np.asarray([i for i, c in enumerate(object_channels) if re.search(marker, c) is not None])
            if len(i_channel) != 0:
                for i in i_channel:
                    channelnames[object_channels[i]] = object_channels[i]
            else:
                raise Exception("Marker {} could not be found!".format(marker))
    return channelnames


def get_markers(obj, channels, exact=True):
    """Gets the markers of the provided channels based on a FlowSOM object or an FCS file

    :param obj: A FlowSOM object or a FCS AnnData object
    :type obj: FlowSOM / AnnData
    :param channels: An array of channels
    :type channels: np.array
    :param exact: If True, a strict search is performed. If False, regexps can be used.
    :type exact: boolean
    """
    assert obj.__class__.__name__ == "FlowSOM" or isinstance(
        obj, ad.AnnData
    ), f"Please provide an FCS file or a FlowSOM object"
    if obj.__class__.__name__ == "FlowSOM":
        object_markers = np.asarray(
            [re.sub(" <.*", "", pretty_colname) for pretty_colname in obj.mudata["cell_data"].var["pretty_colnames"]]
        )
        object_channels = np.asarray(
            [
                re.sub(r".*<(.*)>.*", r"\1", pretty_colname)
                for pretty_colname in obj.mudata["cell_data"].var["pretty_colnames"]
            ]
        )
    else:
        object_markers = np.asarray(obj.uns["meta"]["channels"]["$PnS"])
        object_channels = np.asarray(obj.uns["meta"]["channels"]["$PnN"])

    markernames = dict()
    for channel in channels:
        if isinstance(channel, int):
            i_marker = [channel]
        else:
            if exact:
                channel = r"^" + channel + r"$"
            i_marker = np.asarray([i for i, c in enumerate(object_channels) if re.search(channel, c) is not None])
        if len(i_marker) != 0:
            for i in i_marker:
                markernames[object_markers[i]] = object_channels[i]
        else:
            i_marker = np.asarray([i for i, m in enumerate(object_markers) if re.search(channel, m) is not None])
            if len(i_marker) != 0:
                for i in i_marker:
                    markernames[object_markers[i]] = object_markers[i]
            else:
                raise Exception("Channel {} could not be found!".format(channel))
    return markernames


def get_counts(fsom, level="metaclusters"):
    """Get counts of number of cells in clusters or metaclusters

    :param fsom: A FlowSOM object
    :type fsom: FlowSOM
    :param level: The level to get counts for. Should be 'metaclusters' or 'clusters'
    :type level: str
    """
    assert level in ["metaclusters", "clusters"], f"Level should be 'metaclusters' or 'clusters'"
    if level == "metaclusters":
        counts = {
            "C" + str(i): (fsom.get_cell_data().obs["metaclustering"] == i).sum()
            for i in range(fsom.get_cell_data().uns["n_metaclusters"])
        }
    elif level == "clusters":
        counts = {
            "C" + str(i): (fsom.get_cell_data().obs["clustering"] == i).sum()
            for i in range(fsom.get_cell_data().uns["n_nodes"])
        }
    return pd.DataFrame(counts, index=["counts"]).T


def get_percentages(fsom, level="metaclusters"):
    """Get percentages of number of cells in clusters or metaclusters

    :param fsom: A FlowSOM object
    :type fsom: FlowSOM
    :param level: The level to get counts for. Should be 'metaclusters' or 'clusters'
    :type level: str
    """
    assert level in ["metaclusters", "clusters"], f"Level should be 'metaclusters' or 'clusters'"
    counts = get_counts(fsom, level=level)
    percentages = counts / counts.sum()
    return percentages


def get_cluster_percentages_positive(fsom, cutoffs, cols_used=False, pretty_colnames=False):
    """Get percentage-positive values for all clusters

    :param fsom: A FlowSOM object
    :type fsom: FlowSOM
    :param cutoffs: A dictionary with channels/markers as keys and cutoffs as values
    :type cutoffs: dict
    :param cols_used: If True, only the columns that were used for clustering are used
    :type cols_used: boolean
    :param pretty_colnames: If True, the pretty column names are used
    :type pretty_colnames: boolean
    """
    cl_per_cell = fsom.get_cell_data().obs["clustering"]
    clusters = np.arange(0, fsom.get_cell_data().uns["n_nodes"])
    if "cols_used" not in fsom.get_cell_data().var.columns:
        cols_used = False
    if "pretty_colnames" not in fsom.get_cell_data().var.columns:
        pretty_colnames = False
    channels = get_channels(fsom, list(cutoffs.keys()))
    if cols_used and not pretty_colnames:
        markers_bool = fsom.get_cell_data().var["cols_used"]
        markers = fsom.get_cell_data().var_names[markers_bool]
        channels = {key: value for key, value in channels.items() if key in markers}

    perc_pos = np.empty((len(clusters), len(channels)))
    perc_pos.fill(np.NaN)
    for i, cluster in enumerate(clusters):
        data_per_cluster = fsom.get_cell_data().to_df().loc[cl_per_cell == cluster, list(channels.keys())]
        if data_per_cluster.shape[0] != 0:
            for j, column in enumerate(data_per_cluster.columns):
                cutoff = cutoffs[channels[column]]
                num_above_cutoff = data_per_cluster[column].gt(cutoff).sum()
                percentage_above_cutoff = num_above_cutoff / data_per_cluster[column].shape[0]
                perc_pos[i, j] = percentage_above_cutoff
    if pretty_colnames:
        column_names = fsom.get_cell_data().var["pretty_colnames"].loc[list(channels.keys())]
    else:
        column_names = channels.values()
    return pd.DataFrame(perc_pos, columns=column_names)


def get_metacluster_percentages_positive(fsom, cutoffs, cols_used=False, pretty_colnames=False):
    """Get percentage-positive values for all metaclusters

    :param fsom: A FlowSOM object
    :type fsom: FlowSOM
    :param cutoffs: A dictionary with channels/markers as keys and cutoffs as values
    :type cutoffs: dict
    :param cols_used: If True, only the columns that were used for clustering are used
    :type cols_used: boolean
    :param pretty_colnames: If True, the pretty column names are used
    :type pretty_colnames: boolean
    """
    mcl_per_cell = fsom.get_cell_data().obs["metaclustering"]
    metaclusters = np.arange(0, fsom.get_cell_data().uns["n_metaclusters"])
    if "cols_used" not in fsom.get_cell_data().var.columns:
        cols_used = False
    if "pretty_colnames" not in fsom.get_cell_data().var.columns:
        pretty_colnames = False
    channels = get_channels(fsom, list(cutoffs.keys()))
    if cols_used and not pretty_colnames:
        markers_bool = fsom.get_cell_data().var["cols_used"]
        markers = fsom.get_cell_data().var_names[markers_bool]
        channels = {key: value for key, value in channels.items() if key in markers}

    perc_pos = np.empty((len(metaclusters), len(channels)))
    perc_pos.fill(np.NaN)
    for i, cluster in enumerate(metaclusters):
        data_per_metacluster = fsom.get_cell_data().to_df().loc[mcl_per_cell == cluster, list(channels.keys())]
        if data_per_metacluster.shape[0] != 0:
            for j, column in enumerate(data_per_metacluster.columns):
                cutoff = cutoffs[channels[column]]
                num_above_cutoff = data_per_metacluster[column].gt(cutoff).sum()
                percentage_above_cutoff = num_above_cutoff / data_per_metacluster[column].shape[0]
                perc_pos[i, j] = percentage_above_cutoff
    if pretty_colnames:
        column_names = fsom.get_cell_data().var["pretty_colnames"].loc[list(channels.keys())]
    else:
        column_names = channels.values()
    return pd.DataFrame(perc_pos, columns=column_names)


def get_features(
    fsom,
    files,
    level=np.array(["clusters", "metaclusters"]),
    type=np.array(["counts"]),
    MFI=None,
    positive_cutoffs=None,
    filenames=None,
):
    """Map FCS files on an existing FlowSOM object

    :param fsom: A FlowSOM object
    :type fsom: FlowSOM
    :param files: A list of FCS files
    :type files: list
    :param level: The level(s) to get features for. Should be 'metaclusters' and/or 'clusters'
    :type level: np.array
    :param type: The type(s) of features to get. Should be 'counts', 'percentages', 'MFIs' and/or 'percentages_positive'
    :type type: np.array
    :param MFI: A array of markers to get MFIs for
    :type MFI: np.array
    :param positive_cutoffs: A dictionary with channels/markers as keys and cutoffs as values
    :type positive_cutoffs: dict
    :param filenames: A list of file names
    """
    n_clus = fsom.get_cell_data().uns["n_nodes"]
    n_mcl = fsom.get_cell_data().uns["n_metaclusters"]
    nfiles = len(files)
    i = 0
    if filenames is not None:
        assert len(filenames) != nfiles, f"The number of file names should be equal to the number of files"
    assert all([i in ["metaclusters", "clusters"] for i in level]), f"Level should be 'metaclusters' or 'clusters'"
    assert all(
        [i in ["counts", "percentages", "MFIs", "percentages_positive"] for i in type]
    ), f"Type should be 'counts', 'percentages','MFI' or 'percentages_positive'"
    if "MFIs" in type:
        assert MFI is not None, f"If type is 'MFIs', MFI should be provided"
        MFI = list(get_channels(fsom, MFI).keys())
    if "percentages_positive" in type:
        assert positive_cutoffs is not None, f"If type is 'percentages_positive', positive_cutoffs should be provided"
        assert isinstance(positive_cutoffs, dict), f"positive_cutoffs should be a dictionary"

    matrices = dict()

    # Prepare matrices
    if filenames is None:
        if all([isinstance(i, str) for i in files]):
            filenames = files
        else:
            filenames = [str(i) for i in range(nfiles)]

    C_counts = np.zeros((nfiles, n_clus))
    C_outliers = np.zeros((nfiles, n_clus))

    if "MFIs" in type:
        n_marker = len(MFI)
        C_MFIs = np.zeros((nfiles, n_marker * n_clus))
        MC_MFIs = np.zeros((nfiles, n_marker * n_mcl))

    if "percentages_positive" in type:
        n_mcl = fsom.get_cell_data().uns["n_metaclusters"]
        n_marker = len(positive_cutoffs)
        C_perc_pos = np.zeros((nfiles, n_marker * n_clus))
        MC_perc_pos = np.zeros((nfiles, n_marker * n_mcl))

    # Loop over every file
    for i, file in enumerate(files):
        fsom_tmp = fsom.new_data(file)

        counts_t = fsom_tmp.get_cell_data().obs["clustering"].value_counts()
        C_counts[i, counts_t.index.astype(int)] = counts_t.values
        outliers_t = fsom_tmp.get_cluster_data().uns["outliers"]["number_of_outliers"]
        if outliers_t.shape[0] != 0:
            C_outliers[i, outliers_t.index.astype(int)] = outliers_t.values

        if "MFIs" in type:
            if "clusters" in level:
                C_MFIs[i,] = fsom_tmp.get_cluster_data().to_df().loc[:, MFI].to_numpy().flatten()
            if "metaclusters" in level:
                MFI_i = [i for i, x in enumerate(fsom_tmp.get_cluster_data().var_names) if x in MFI]
                MC_MFIs[i,] = fsom_tmp.get_cluster_data().uns["metacluster_MFIs"].loc[:, MFI_i].to_numpy().flatten()

        if "percentages_positive" in type:
            if "clusters" in level:
                C_perc_pos[i,] = get_cluster_percentages_positive(fsom_tmp, positive_cutoffs).to_numpy().flatten()
            if "metaclusters" in level:
                MC_perc_pos[i,] = get_metacluster_percentages_positive(fsom_tmp, positive_cutoffs).to_numpy().flatten()

    # Add matrices to dictionary
    if "clusters" in level:
        cluster_names = ["C" + str(i) for i in np.arange(0, n_clus)]
        if "counts" in type:
            matrices["cluster_counts"] = pd.DataFrame(C_counts, index=filenames, columns=cluster_names)
        if "percentages" in type:
            C_percentages = np.true_divide(C_counts, C_counts.sum(axis=1, keepdims=True))
            matrices["cluster_percentages"] = pd.DataFrame(C_percentages, index=filenames, columns=cluster_names)
        if "MFIs" in type:
            MFI_names = [i + " " + j for i in cluster_names for j in MFI]
            matrices["cluster_MFIs"] = pd.DataFrame(C_MFIs, index=filenames, columns=MFI_names)
        if "percentages_positive" in type:
            pretty_colnames = (
                fsom.get_cell_data()
                .var["pretty_colnames"]
                .loc[get_channels(fsom, list(positive_cutoffs.keys())).keys()]
            )
            perc_pos_names = [i + " " + j for i in cluster_names for j in pretty_colnames]
            matrices["cluster_percantages_pos"] = pd.DataFrame(C_perc_pos, columns=perc_pos_names)
    if "metaclusters" in level:
        MC_counts = (
            pd.concat(
                [pd.DataFrame(C_counts).T, fsom.get_cluster_data().obs.metaclustering.reset_index(drop=True)], axis=1
            )
            .groupby("metaclustering")
            .sum()
            .to_numpy()
            .T
        )
        if "counts" in type:
            MC_names = ["MC" + str(i) for i in np.arange(0, n_mcl)]
            matrices["metacluster_counts"] = pd.DataFrame(MC_counts, index=filenames, columns=MC_names)
        if "percentages" in type:
            MC_percentages = np.true_divide(MC_counts, MC_counts.sum(axis=1, keepdims=True))
            matrices["metacluster_percentages"] = pd.DataFrame(MC_percentages, index=filenames, columns=MC_names)
        if "MFIs" in type:
            MFI_names = [i + " " + j for i in MC_names for j in MFI]
            matrices["metacluster_MFIs"] = pd.DataFrame(MC_MFIs, index=filenames, columns=MFI_names)
        if "percentages_positive" in type:
            pretty_colnames = (
                fsom.get_cell_data()
                .var["pretty_colnames"]
                .loc[get_channels(fsom, list(positive_cutoffs.keys())).keys()]
            )
            perc_pos_names = [i + " " + j for i in MC_names for j in pretty_colnames]
            matrices["metacluster_percantages_pos"] = pd.DataFrame(MC_perc_pos, columns=perc_pos_names)

    return matrices
