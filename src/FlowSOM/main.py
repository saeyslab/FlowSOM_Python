import numpy as np
import pandas as pd
import anndata as ad
import pytometry as pm
import igraph as ig

import re
import random
import collections

from pyFlowSOM import map_data_to_nodes, som
from scipy.stats import median_abs_deviation
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering


class FlowSOM:
    """
    A class that contains all the FlowSOM data using AnnData objects.

    """

    def __init__(self, inp, cols_to_use: np.array = None, n_clus=10, max_meta=None, seed: int = None, **kwargs):
        """Initialize the FlowSOM AnnData object

        :param inp: A file path to an FCS file or a AnnData FCS file to cluster
        :type inp: str / ad.AnnData
        :param cols_to_use:  An array of the columns to use for clustering
        :type cols_to_use: np.array
        :param n_clus: The number of metacluster
        :type n_clus: int
        :param max_meta: To be adapted
        :type max_meta: int
        :param seed: A fixed seed
        :type seed: int
        """
        if seed is not None:
            random.seed(seed)
        self.adata = self.read_input(inp)
        self.build_SOM(cols_to_use, **kwargs)
        self.build_MST()
        self.metacluster(n_clus)
        self.update_derived_values()

    def read_input(self, inp):
        """Converts input to a FlowSOM AnnData object

        :param inp: A file path to an FCS file or a AnnData FCS file to cluster
        :type inp: str / ad.AnnData
        """
        if isinstance(inp, ad.AnnData):
            data = inp.X
        elif isinstance(inp, str):
            inp = read_FCS(inp)
            data = inp.X
        channels = np.asarray(inp.var["channel"])
        markers = np.asarray(inp.var["marker"])
        isnan_markers = [str(marker) == "nan" for marker in markers]
        markers[isnan_markers] = channels[isnan_markers]
        pretty_colnames = [markers[i] + " <" + channels[i] + ">" for i in range(len(markers))]
        fsom = np.array(data, dtype=np.float32)
        fsom = ad.AnnData(fsom)
        fsom.uns["pretty_colnames"] = np.asarray(pretty_colnames, dtype=str)
        fsom.var_names = np.asarray(inp.uns["meta"]["channels"]["$PnN"])
        return fsom

    def build_SOM(self, cols_to_use: np.array = None, outlier_MAD=4, **kwargs):
        """Initialize the SOM clustering and update FlowSOM object

        :param cols_to_use:  An array of the columns to use for clustering
        :type cols_to_use: np.array
        :param outlier_MAD: To be adapted
        :type outlier_MAD: int
        """
        if cols_to_use is None:
            cols_to_use = self.adata.var_names
        cols_to_use = list(get_channels(self, cols_to_use).keys())
        nodes, clusters, dists, xdim, ydim = self.SOM(inp=self.adata[:, cols_to_use].X, **kwargs)
        self.adata.obs["clustering"] = np.array(clusters)
        self.adata.obs["mapping"] = np.array(dists)
        self.adata.uns["cols_used"] = cols_to_use
        self.adata.uns["xdim"] = xdim
        self.adata.uns["ydim"] = ydim
        self.adata.uns["n_nodes"] = xdim * ydim
        self = self.update_derived_values()
        self.adata.uns["cluster_adata"].obsm["codes"] = np.array(nodes)
        self.adata.uns["cluster_adata"].obsm["grid"] = np.array([(x, y) for x in range(xdim) for y in range(ydim)])
        self.adata.uns["cluster_adata"].uns["outliers"] = self.test_outliers(mad_allowed=outlier_MAD).reset_index()
        return self

    def SOM(self, inp, xdim=10, ydim=10, rlen=10, importance=None):
        """Perform SOM clustering

        :param inp:  An array of the columns to use for clustering
        :type inp: np.array
        :param xdim: x dimension of SOM
        :type xdim: int
        :param ydim: y dimension of SOM
        :type ydim: int
        :param rlen: Number of times to loop over the training data for each MST
        :type rlen: int
        :param importance: Array with numeric values. Parameters will be scaled
        according to importance
        :type importance: np.array
        """
        data = np.asarray(inp, dtype=np.float64)
        if importance is not None:
            data = np.stack([data[:, i] * importance[i] for i in range(len(importance))], axis=1)
        nodes = som(data, xdim=xdim, ydim=ydim, rlen=rlen * 5)
        clusters, dists = map_data_to_nodes(nodes, data)
        return nodes, clusters, dists, xdim, ydim

    def update_derived_values(self):
        """Update the derived values such as median values and CV values"""
        df = self.adata.X  # [self.adata.X[:, 0].argsort()]
        df = np.c_[self.adata.obs["clustering"], df]
        n_nodes = self.adata.uns["n_nodes"]
        cluster_median_values = np.vstack(
            [
                np.median(df[df[:, 0] == cl + 1], axis=0)
                if df[df[:, 0] == cl + 1].shape[0] != 0
                else np.repeat(np.nan, df[df[:, 0] == cl + 1].shape[1])
                for cl in range(n_nodes)
            ]
        )
        if "cluster_adata" in self.adata.uns_keys():
            cluster_adata = self.adata.uns["cluster_adata"]
            cluster_adata.X = np.delete(cluster_median_values, 0, axis=1)
        else:
            cluster_adata = ad.AnnData(np.delete(cluster_median_values, 0, axis=1))
        cluster_adata.var_names = self.adata.var_names
        sd_values = list()
        cv_values = list()
        mad_values = list()
        pctgs = dict()
        for cl in range(n_nodes):
            cluster_data = df[df[:, 0] == cl + 1, :]
            cv_values.append(np.divide(np.std(cluster_data, axis=0), np.mean(cluster_data, axis=0)))
            sd_values.append(np.std(cluster_data, axis=0))
            mad_values.append(median_abs_deviation(cluster_data, axis=0))
            pctgs[cl] = cluster_data.shape[0]

        cluster_adata.obsm["cv_values"] = np.vstack(cv_values)
        cluster_adata.obsm["sd_values"] = np.vstack(sd_values)
        cluster_adata.obsm["mad_values"] = np.vstack(mad_values)
        pctgs = np.divide(list(pctgs.values()), np.sum(list(pctgs.values())))
        cluster_adata.obs["percentages"] = pctgs

        self.adata.uns["cluster_adata"] = cluster_adata
        if "metaclustering" in self.adata.obs_keys():
            df = self.adata.X[self.adata.X[:, 0].argsort()]
            df = np.c_[self.adata.obs["metaclustering"], df]
            metacluster_median_values = np.vstack(
                [
                    np.median(df[df[:, 0] == cl + 1], axis=0)
                    if df[df[:, 0] == cl + 1].shape[0] != 0
                    else np.repeat(np.nan, df[df[:, 0] == cl + 1].shape[1])
                    for cl in range(self.adata.uns["n_metaclusters"])
                ]
            )
            self.adata.uns["metacluster_MFIs"] = np.vstack(metacluster_median_values)

        return self

    def build_MST(self):
        """Make a minimum spanning tree"""
        adjacency = cdist(
            self.adata.uns["cluster_adata"].obsm["codes"],
            self.adata.uns["cluster_adata"].obsm["codes"],
            metric="euclidean",
        )
        full_graph = ig.Graph.Weighted_Adjacency(adjacency, mode="undirected", loops=False)
        MST_graph = ig.Graph.spanning_tree(full_graph, weights=full_graph.es["weight"])
        MST_graph.es["weight"] /= np.mean(MST_graph.es["weight"])
        layout = MST_graph.layout_kamada_kawai(
            seed=MST_graph.layout_grid(), maxiter=50 * MST_graph.vcount(), kkconst=max([MST_graph.vcount(), 1])
        ).coords
        self.adata.uns["cluster_adata"].obsm["layout"] = np.array(layout)
        self.adata.uns["cluster_adata"].uns["graph"] = MST_graph
        return self

    def metacluster(self, n_clus):
        """Perform a hierarchical clustering

        :param n_clus: The number of metaclusters
        :type n_clus: int
        """
        average = AgglomerativeClustering(n_clusters=n_clus, linkage="average")
        metaclustering = average.fit(self.adata.uns["cluster_adata"].obsm["codes"])
        metaclusters = metaclustering.labels_
        self.adata.uns["n_metaclusters"] = n_clus
        self.adata.uns["cluster_adata"].obs["metaclustering"] = metaclusters
        metaclustering = np.array(metaclusters)
        self.adata.obs["metaclustering"] = np.asarray(
            [np.array(metaclusters)[int(i) - 1] for i in np.asarray(self.adata.obs["clustering"])]
        )
        return self

    def test_outliers(self, mad_allowed: int = 4, fsom_reference=None, plot_file=None, channels=None):
        """Test if any cells are too far from their cluster centers

        :param mad_allowed:
        :type mad_allowed: int
        :param fsom_reference:
        :type fsom_reference:
        :param plot_file:
        :type plot_file:
        :param channels:
        :type channels:
        """
        if fsom_reference is None:
            fsom_ref = self
        cell_cl = fsom_ref.adata.obs["clustering"]
        mapping = fsom_ref.adata.obs["mapping"]
        distances_median = [
            np.median(mapping[cell_cl == cl + 1]) if len(mapping[cell_cl == cl + 1]) > 0 else 0
            for cl in range(fsom_ref.adata.uns["n_nodes"])
        ]

        distances_mad = [
            median_abs_deviation(mapping[cell_cl == cl + 1]) if len(mapping[cell_cl == cl + 1]) > 0 else 0
            for cl in range(fsom_ref.adata.uns["n_nodes"])
        ]
        thresholds = np.add(distances_median, np.multiply(mad_allowed, distances_mad))

        max_distances_new = [
            np.max(self.adata.obs["mapping"][self.adata.obs["clustering"] == cl + 1])
            if len(self.adata.obs["mapping"][self.adata.obs["clustering"] == cl + 1]) > 0
            else 0
            for cl in range(self.adata.uns["n_nodes"])
        ]
        distances = [
            self.adata.obs["mapping"][self.adata.obs["clustering"] == cl + 1] for cl in range(self.adata.uns["n_nodes"])
        ]
        outliers = [sum(distances[i] > thresholds[i]) for i in range(len(distances))]

        result = pd.DataFrame(
            {
                "median_dist": distances_median,
                "median_absolute_deviation": distances_mad,
                "threshold": thresholds,
                "number_of_outliers": outliers,
                "maximum_outlier_distance": max_distances_new,
            }
        )

        if channels is not None:
            outliers_dict = dict()
            codes = fsom_ref.get_cluster_adata().obsm["codes"]
            data = fsom_ref.adata.X
            channels = list(get_channels(fsom_ref, channels).keys())
            for channel in channels:
                channel_i = np.where(fsom_ref.adata.var_names == channel)[0][0]
                distances_median_channel = [
                    np.median(np.abs(np.subtract(data[cell_cl == cl + 1, channel_i], codes[cl, channel_i])))
                    if len(data[cell_cl == cl + 1, channel_i]) > 0
                    else 0
                    for cl in range(fsom_ref.adata.uns["n_nodes"])
                ]
                distances_mad_channel = [
                    median_abs_deviation(np.abs(np.subtract(data[cell_cl == cl + 1, channel_i], codes[cl, channel_i])))
                    if len(data[cell_cl == cl + 1, channel_i]) > 0
                    else 0
                    for cl in range(fsom_ref.n_adata.uns["n_nodes"])
                ]
                thresholds_channel = np.add(distances_median_channel, np.multiply(mad_allowed, distances_mad_channel))

                distances_channel = [
                    np.abs(
                        np.subtract(
                            self.adata.X[self.adata.obs["clustering"] == cl + 1, channel_i],
                            fsom_ref.adata.uns["n_nodes"][cl, channel_i],
                        )
                    )
                    for cl in range(self.adata.uns["n_nodes"])
                ]
                outliers_channel = [
                    sum(distances_channel[i] > thresholds_channel[i]) for i in range(len(distances_channel))
                ]
                outliers_dict[list(get_markers(self, [channel]).keys())[0]] = outliers_channel
            result_channels = pd.DataFrame(outliers_dict)
            result = result.join(result_channels)
        return result

    def new_data(self, inp, mad_allowed=4):
        fsom_new = FlowSOM(inp)
        fsom_new.adata.uns["pretty_colnames"] = self.adata.uns["pretty_colnames"]
        fsom_new.adata.uns["cols_used"] = self.adata.uns["cols_used"]
        fsom_new.adata.uns["xdim"] = self.adata.uns["xdim"]
        fsom_new.adata.uns["ydim"] = self.adata.uns["ydim"]
        fsom_new.adata.uns["n_nodes"] = self.adata.uns["n_nodes"]
        fsom_new.adata.uns["n_metaclusters"] = self.adata.uns["n_metaclusters"]
        fsom_new.adata.uns["cluster_adata"] = self.adata.uns["cluster_adata"]

        clusters, dists = map_data_to_nodes(
            np.array(self.get_cluster_adata().obsm["codes"], dtype=np.float64),
            np.array(fsom_new.adata.X, dtype=np.float64),
        )
        fsom_new.adata.obsm["mapping"] = np.array(dists)
        fsom_new.adata.obs["clustering"] = np.array(clusters)
        fsom_new = fsom_new.update_derived_values()
        metaclusters = self.get_cluster_adata().obs["metaclustering"]
        fsom_new.adata.obs["metaclustering"] = np.asarray(
            [np.array(metaclusters)[int(i) - 1] for i in np.asarray(fsom_new.adata.obs["clustering"])]
        )
        # test_outliers = fsom_new.test_outliers(mad_allowed = mad_allowed, fsom_reference = self)
        return fsom_new

    def get_cluster_adata(self):
        return self.adata.uns["cluster_adata"]


def get_channels(obj, markers, exact=True):
    """Gets the channels of the provided markers based on a FlowSOM object or an FCS file

    :param obj: A FlowSOM object or a FCS AnnData object
    :type obj: FlowSOM / AnnData
    :param markers: An array of markers
    :type markers: np.array
    :param exact: If True, a strict search is performed. If False, regexps can be used.
    :type exact: boolean
    """
    assert isinstance(obj, FlowSOM) or isinstance(obj, ad.AnnData), f"Please provide an FCS file or a FlowSOM object"
    if isinstance(obj, FlowSOM):
        object_markers = np.asarray(
            [re.sub(" <.*", "", pretty_colname) for pretty_colname in obj.adata.uns["pretty_colnames"]]
        )
        object_channels = np.asarray(
            [re.sub(r".*<(.*)>.*", r"\1", pretty_colname) for pretty_colname in obj.adata.uns["pretty_colnames"]]
        )
    elif isinstance(obj, ad.AnnData):
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
    assert isinstance(obj, FlowSOM) or isinstance(obj, ad.AnnData), f"Please provide an FCS file or a FlowSOM object"
    if isinstance(obj, FlowSOM):
        object_markers = np.asarray(
            [re.sub(" <.*", "", pretty_colname) for pretty_colname in obj.adata.uns["pretty_colnames"]]
        )
        object_channels = np.asarray(
            [re.sub(r".*<(.*)>.*", r"\1", pretty_colname) for pretty_colname in obj.adata.uns["pretty_colnames"]]
        )
    if isinstance(obj, ad.AnnData):
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


def aggregate_flowframes(filenames, c_total, channels=None, keep_order=False, silent=False):
    """Aggregate multiple FCS files together
    :param filenames: An array containing full paths to the FCS files
    :type filenames: np.array
    :param c_total: Total number of cells to write to the output file
    :type c_total: int
    :param channels: Channels/markers to keep in the aggregate. Default None
    takes all channels of the first file
    :type channels: np.array
    :param keep_order: If True, the random subsample will be ordered in the same
    way as they were originally ordered in the file. Default=False.
    :type keep_order: boolean
    :param silent: If False, prints an update every time it starts processing a
    new file. Default = False.
    :type silent: boolean
    """
    nFiles = len(filenames)
    cFile = int(np.ceil(c_total / nFiles))

    flow_frame = []
    for i, file_path in enumerate(filenames):
        f = read_FCS(file_path)

        cPerFile = min([f.X.shape[0], cFile])

        # Random sampling
        ids = random.sample(range(f.X.shape[0]), cPerFile)
        if keep_order:
            ids = sorted(ids)

        file_ids = np.repeat(i, cPerFile)

        f = f[
            ids,
        ]
        f.obs["Original_ID"] = np.array(ids, dtype=np.float32)
        f.obs["File"] = np.array(file_ids, dtype=np.float32)
        f.obs["File_scattered"] = np.array(
            np.add(file_ids, np.random.normal(loc=0.0, scale=0.1, size=len(file_ids))), dtype=np.float32
        )
        flow_frame.append(f)
    flow_frame = ad.AnnData.concatenate(*flow_frame, join="outer", uns_merge="first")
    return flow_frame


def read_FCS(filepath, truncate_max_range=False):
    """Reads in an FCS file
    :param filepath: An array containing a full path to the FCS file
    :type filepath: str
    """
    try:
        f = pm.io.read_fcs(filepath)
    except:
        f = pm.io.read_fcs(filepath, reindex=False)
        markers = dict(
            (str(re.sub("S$", "", re.sub("^P", "", string))), f.uns["meta"][string])
            for string in f.uns["meta"].keys()
            if re.match("^P[0-9]+S$", string)
        )
        fluo_channels = [i for i in markers.keys()]
        non_fluo_channels = dict(
            (i, f.uns["meta"]["channels"]["$PnN"][i]) for i in f.uns["meta"]["channels"].index if i not in fluo_channels
        )
        index_markers = dict(markers, **non_fluo_channels)
        f.var.rename(index=index_markers, inplace=True)
        f.uns["meta"]["channels"]["$PnS"] = [index_markers[key] for key in f.uns["meta"]["channels"].index]
    return f
