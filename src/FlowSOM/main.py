import numpy as np
import pandas as pd
import anndata as ad
import pytometry as pm
import igraph as ig
import FlowSOM.consensus_clustering as cc

import re
import random

from scipy.stats import median_abs_deviation
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from mudata import MuData
from FlowSOM.som import SOM, map_data_to_codes


class FlowSOM:
    """
    A class that contains all the FlowSOM data using MuData objects.

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
        self.mudata = self.read_input(inp)
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
        isnan_markers = [str(marker) == "nan" or len(marker) == 0 for marker in markers]
        markers[isnan_markers] = channels[isnan_markers]
        pretty_colnames = [markers[i] + " <" + channels[i] + ">" for i in range(len(markers))]
        fsom = np.array(data, dtype=np.float32)
        fsom = MuData({"cell_data": ad.AnnData(fsom)})
        fsom.mod["cell_data"].var["pretty_colnames"] = np.asarray(pretty_colnames, dtype=str)
        fsom.mod["cell_data"].var_names = np.asarray(channels)
        fsom.mod["cell_data"].var["markers"] = np.asarray(markers)
        fsom.mod["cell_data"].var["channels"] = np.asarray(channels)
        return fsom

    def build_SOM(self, cols_to_use: np.array = None, outlier_MAD=4, **kwargs):
        """Initialize the SOM clustering and update FlowSOM object

        :param cols_to_use:  An array of the columns to use for clustering
        :type cols_to_use: np.array
        :param outlier_MAD: To be adapted
        :type outlier_MAD: int
        """
        if cols_to_use is None:
            cols_to_use = self.mudata["cell_data"].var_names
        cols_to_use = list(get_channels(self, cols_to_use).keys())
        codes, clusters, dists, xdim, ydim = self.SOM(inp=self.mudata["cell_data"][:, cols_to_use].X, **kwargs)
        self.mudata["cell_data"].obs["clustering"] = np.array(clusters)
        self.mudata["cell_data"].obs["distance_to_bmu"] = np.array(dists)
        self.mudata["cell_data"].var["cols_used"] = [x in cols_to_use for x in self.mudata["cell_data"].var_names]

        self.mudata["cell_data"].uns["n_nodes"] = xdim * ydim
        self = self.update_derived_values()
        self.mudata["cluster_data"].uns["xdim"] = xdim
        self.mudata["cluster_data"].uns["ydim"] = ydim
        self.mudata["cluster_data"].obsm["codes"] = np.array(codes)
        self.mudata["cluster_data"].obsm["grid"] = np.array([(x, y) for x in range(xdim) for y in range(ydim)])
        self.mudata["cluster_data"].uns["outliers"] = self.test_outliers(mad_allowed=outlier_MAD).reset_index()
        return self

    def SOM(
        self,
        inp,
        xdim=10,
        ydim=10,
        rlen=10,
        mst=1,
        alpha=(0.05, 0.01),
        init=False,
        initf=None,
        map=True,
        codes=None,
        importance=None,
        seed=None,
    ):
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
        if codes is not None:
            assert (codes.shape[1] == data.shape[1]) or (
                codes.shape[0] != xdim * ydim
            ), f"If codes is not NULL, it should have the same number of columns as the data and the number of rows should correspond with xdim*ydim"

        if importance is not None:
            data = np.stack([data[:, i] * importance[i] for i in range(len(importance))], axis=1)

        # Initialize the grid
        grid = [(x, y) for x in range(xdim) for y in range(ydim)]
        n_codes = len(grid)
        if codes is None:
            if init:
                codes = initf(data, xdim, ydim)
            else:
                codes = data[np.random.choice(data.shape[0], n_codes, replace=False), :]

        # Initialize the neighbourhood
        nhbrdist = squareform(pdist(grid, metric="chebyshev"))

        # Initialize the radius
        radius = (np.quantile(nhbrdist, 0.67), 0)
        if mst == 1:
            radius = [radius]
            alpha = [alpha]
        else:
            radius = np.linspace(radius[0], radius[1], num=mst + 1)
            radius = [tuple(radius[i : i + 2]) for i in range(mst)]
            alpha = np.linspace(alpha[0], alpha[1], num=mst + 1)
            alpha = [tuple(alpha[i : i + 2]) for i in range(mst)]

        # Compute the SOM
        for i in range(mst):
            codes = SOM(
                data,
                codes,
                nhbrdist,
                alphas=alpha[i],
                radii=radius[i],
                ncodes=n_codes,
                rlen=rlen,
                seed=seed,
            )
            if mst != 1:
                nhbrdist = self.dist_mst(codes)

        if map:
            clusters, dists = map_data_to_codes(data=data, codes=codes)
        else:
            clusters = dists = np.zeros((n_codes))

        return codes, clusters, dists, xdim, ydim

    def update_derived_values(self):
        """Update the derived values such as median values and CV values"""
        df = self.mudata["cell_data"].X  # [self.adata.X[:, 0].argsort()]
        df = np.c_[self.mudata["cell_data"].obs["clustering"], df]
        n_nodes = self.mudata["cell_data"].uns["n_nodes"]
        cluster_median_values = np.vstack(
            [
                np.median(df[df[:, 0] == cl], axis=0)  # cl + 1 if cluster starts with 1
                if df[df[:, 0] == cl].shape[0] != 0  # cl + 1 if cluster starts with 1
                else np.repeat(np.nan, df[df[:, 0] == cl].shape[1])  # cl + 1 if cluster starts with 1
                for cl in range(n_nodes)
            ]
        )
        if "cluster_data" in self.mudata.mod.keys():
            cluster_mudata = self.mudata.mod["cluster_data"]
            cluster_mudata.X = np.delete(cluster_median_values, 0, axis=1)
        else:
            cluster_mudata = ad.AnnData(np.delete(cluster_median_values, 0, axis=1))
        cluster_mudata.var_names = self.mudata["cell_data"].var_names
        sd_values = list()
        cv_values = list()
        mad_values = list()
        pctgs = dict()
        for cl in range(n_nodes):
            cluster_data = df[df[:, 0] == cl, :]  # +1 if cluster starts at 1
            cv_values.append(np.divide(np.std(cluster_data, axis=0), np.mean(cluster_data, axis=0)))
            sd_values.append(np.std(cluster_data, axis=0))
            mad_values.append(median_abs_deviation(cluster_data, axis=0))
            pctgs[cl] = cluster_data.shape[0]

        cluster_mudata.obsm["cv_values"] = np.vstack(cv_values)
        cluster_mudata.obsm["sd_values"] = np.vstack(sd_values)
        cluster_mudata.obsm["mad_values"] = np.vstack(mad_values)
        pctgs = np.divide(list(pctgs.values()), np.sum(list(pctgs.values())))
        cluster_mudata.obs["percentages"] = pctgs

        self.mudata.mod["cluster_data"] = cluster_mudata
        if "metaclustering" in self.mudata["cell_data"].obs_keys():
            df = self.mudata["cell_data"].X[self.mudata["cell_data"].X[:, 0].argsort()]
            df = np.c_[self.mudata["cell_data"].obs["metaclustering"], df]
            metacluster_median_values = pd.DataFrame(df).groupby(0).median()
            self.mudata["cluster_data"].uns["metacluster_MFIs"] = metacluster_median_values

        return self

    def build_MST(self):
        """Make a minimum spanning tree"""
        adjacency = cdist(
            self.mudata["cluster_data"].obsm["codes"],
            self.mudata["cluster_data"].obsm["codes"],
            metric="euclidean",
        )
        full_graph = ig.Graph.Weighted_Adjacency(adjacency, mode="undirected", loops=False)
        MST_graph = ig.Graph.spanning_tree(full_graph, weights=full_graph.es["weight"])
        MST_graph.es["weight"] /= np.mean(MST_graph.es["weight"])
        layout = MST_graph.layout_kamada_kawai(
            seed=MST_graph.layout_grid(), maxiter=50 * MST_graph.vcount(), kkconst=max([MST_graph.vcount(), 1])
        ).coords
        self.mudata["cluster_data"].obsm["layout"] = np.array(layout)
        self.mudata["cluster_data"].uns["graph"] = MST_graph
        return self

    def dist_mst(self, codes):
        adjacency = cdist(
            codes,
            codes,
            metric="euclidean",
        )
        full_graph = ig.Graph.Weighted_Adjacency(adjacency, mode="undirected", loops=False)
        MST_graph = ig.Graph.spanning_tree(full_graph, weights=full_graph.es["weight"])
        codes = [
            [len(x) - 1 for x in MST_graph.get_shortest_paths(v=i, to=MST_graph.vs.indices, weights=None)]
            for i in MST_graph.vs.indices
        ]
        return codes

    def metacluster(self, n_clus):
        """Perform a (consensus) hierarchical clustering

        :param n_clus: The number of metaclusters
        :type n_clus: int
        """
        # metaclusters = (self.hierarchical_clustering(self.mudata["cluster_data"].obsm["codes"], n_clus))
        metaclusters = self.consensus_hierarchical_clustering(self.mudata["cluster_data"].obsm["codes"], n_clus)
        self.mudata["cell_data"].uns["n_metaclusters"] = n_clus
        self.mudata["cluster_data"].obs["metaclustering"] = metaclusters.astype(str)
        self.mudata["cell_data"].obs["metaclustering"] = np.asarray(
            [np.array(metaclusters)[int(i)] for i in np.asarray(self.mudata["cell_data"].obs["clustering"])]
        )
        return self

    def hierarchical_clustering(self, data, n_clus, linkage="average"):
        average = AgglomerativeClustering(n_clusters=n_clus, linkage=linkage)
        metaclustering = average.fit(data)
        metaclusters = metaclustering.labels_
        return metaclusters

    def consensus_hierarchical_clustering(
        self, data, n_clus, n_subsamples=100, linkage="average", resample_proportion=0.9
    ):
        average = cc.ConsensusCluster(
            AgglomerativeClustering, K=n_clus, H=n_subsamples, resample_proportion=resample_proportion, linkage=linkage
        )
        average.fit(data)
        metaclusters = average.predict(n_clus=n_clus)
        return metaclusters

    def test_outliers(self, mad_allowed: int = 4, fsom_reference=None, plot_file=None, channels=None):
        """Test if any cells are too far from their cluster centers

        :param mad_allowed: Number of median absolute deviations allowed. Default = 4.
        :type mad_allowed: int
        :param fsom_reference: FlowSOM object to use as reference. If NULL (default), the original fsom object is used.
        :type fsom_reference: FlowSOM
        :param plot_file:
        :type plot_file:
        :param channels:If channels are given, the number of outliers in the original space for those channels will be calculated and added to the final results table.
        :type channels: np.array
        """
        if fsom_reference is None:
            fsom_reference = self
        cell_cl = fsom_reference.mudata["cell_data"].obs["clustering"]
        distance_to_bmu = fsom_reference.mudata["cell_data"].obs["distance_to_bmu"]
        distances_median = [
            np.median(distance_to_bmu[cell_cl == cl + 1]) if len(distance_to_bmu[cell_cl == cl + 1]) > 0 else 0
            for cl in range(fsom_reference.mudata["cell_data"].uns["n_nodes"])
        ]

        distances_mad = [
            median_abs_deviation(distance_to_bmu[cell_cl == cl + 1])
            if len(distance_to_bmu[cell_cl == cl + 1]) > 0
            else 0
            for cl in range(fsom_reference.mudata["cell_data"].uns["n_nodes"])
        ]
        thresholds = np.add(distances_median, np.multiply(mad_allowed, distances_mad))

        max_distances_new = [
            np.max(
                self.mudata["cell_data"].obs["distance_to_bmu"][self.mudata["cell_data"].obs["clustering"] == cl + 1]
            )
            if len(
                self.mudata["cell_data"].obs["distance_to_bmu"][self.mudata["cell_data"].obs["clustering"] == cl + 1]
            )
            > 0
            else 0
            for cl in range(self.mudata["cell_data"].uns["n_nodes"])
        ]
        distances = [
            self.mudata["cell_data"].obs["distance_to_bmu"][self.mudata["cell_data"].obs["clustering"] == cl + 1]
            for cl in range(self.mudata["cell_data"].uns["n_nodes"])
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
            codes = fsom_reference.mudata["cluster_data"]().obsm["codes"]
            data = fsom_reference.mudata["cell_data"].X
            channels = list(get_channels(fsom_reference, channels).keys())
            for channel in channels:
                channel_i = np.where(fsom_reference.mudata["cell_data"].var_names == channel)[0][0]
                distances_median_channel = [
                    np.median(np.abs(np.subtract(data[cell_cl == cl + 1, channel_i], codes[cl, channel_i])))
                    if len(data[cell_cl == cl + 1, channel_i]) > 0
                    else 0
                    for cl in range(fsom_reference.mudata["cell_data"].uns["n_nodes"])
                ]
                distances_mad_channel = [
                    median_abs_deviation(np.abs(np.subtract(data[cell_cl == cl + 1, channel_i], codes[cl, channel_i])))
                    if len(data[cell_cl == cl + 1, channel_i]) > 0
                    else 0
                    for cl in range(fsom_reference.mudata["cell_data"].uns["n_nodes"])
                ]
                thresholds_channel = np.add(distances_median_channel, np.multiply(mad_allowed, distances_mad_channel))

                distances_channel = [
                    np.abs(
                        np.subtract(
                            self.mudata["cell_data"].X[self.mudata["cell_data"].obs["clustering"] == cl + 1, channel_i],
                            fsom_reference.mudata["cell_data"].uns["n_nodes"][cl, channel_i],
                        )
                    )
                    for cl in range(self.mudata["cell_data"].uns["n_nodes"])
                ]
                outliers_channel = [
                    sum(distances_channel[i] > thresholds_channel[i]) for i in range(len(distances_channel))
                ]
                outliers_dict[list(get_markers(self, [channel]).keys())[0]] = outliers_channel
            result_channels = pd.DataFrame(outliers_dict)
            result = result.join(result_channels)
        return result

    def new_data(self, inp, mad_allowed=4):
        """Map new data to a FlowSOM grid

        :param inp: An anndata or filepath to an FCS file
        :type inp: ad.AnnData / str
        :param mad_allowed: A warning is generated if the distance of the new
        data points to their closest cluster center is too big. This is computed
        based on the typical distance of the points from the original dataset
        assigned to that cluster, the threshold being set to median +
        madAllowed * MAD. Default is 4.
        :type mad_allowed: int
        """
        fsom_new = FlowSOM(inp)
        fsom_new.get_cell_data().var["pretty_colnames"] = self.get_cell_data().var["pretty_colnames"]
        fsom_new.get_cell_data().var["cols_used"] = self.get_cell_data().var["cols_used"]
        fsom_new.get_cluster_data().uns["xdim"] = self.get_cluster_data().uns["xdim"]
        fsom_new.get_cluster_data().uns["ydim"] = self.get_cluster_data().uns["ydim"]
        fsom_new.get_cell_data().uns["n_nodes"] = self.get_cell_data().uns["n_nodes"]
        fsom_new.get_cell_data().uns["n_metaclusters"] = self.get_cell_data().uns["n_metaclusters"]
        fsom_new.mudata.mod["cluster_data"] = self.get_cluster_data()

        markers_bool = self.get_cell_data().var["cols_used"]
        markers = self.get_cell_data().var_names[markers_bool]
        data = fsom_new.get_cell_data()[:, markers].X
        clusters, dists = map_data_to_codes(data=data, codes=self.get_cluster_data().obsm["codes"])
        fsom_new.get_cell_data().obsm["distance_to_bmu"] = np.array(dists)
        fsom_new.get_cell_data().obs["clustering"] = np.array(clusters)
        fsom_new = fsom_new.update_derived_values()
        metaclusters = self.get_cluster_data().obs["metaclustering"]
        fsom_new.get_cell_data().obs["metaclustering"] = np.asarray(
            [np.array(metaclusters)[int(i)] for i in np.asarray(fsom_new.get_cell_data().obs["clustering"])]
        )
        test_outliers = fsom_new.test_outliers(mad_allowed=mad_allowed, fsom_reference=self)
        fsom_new.mudata["cluster_data"].uns["outliers"] = test_outliers
        return fsom_new

    def subset(self, ids):
        """Take a subset from a FlowSOM object

        :param ids: An array of ids to subset
        :type ids: np.array
        """
        fsom_subset = self
        fsom_subset.mudata.mod["cell_data"] = fsom_subset.mudata["cell_data"][ids, :]
        fsom_subset = fsom_subset.update_derived_values()
        return fsom_subset

    def get_cell_data(self):
        return self.mudata["cell_data"]

    def get_cluster_data(self):
        return self.mudata["cluster_data"]


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
            [re.sub(" <.*", "", pretty_colname) for pretty_colname in obj.mudata["cell_data"].var["pretty_colnames"]]
        )
        object_channels = np.asarray(
            [
                re.sub(r".*<(.*)>.*", r"\1", pretty_colname)
                for pretty_colname in obj.mudata["cell_data"].var["pretty_colnames"]
            ]
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
            [re.sub(" <.*", "", pretty_colname) for pretty_colname in obj.mudata["cell_data"].var["pretty_colnames"]]
        )
        object_channels = np.asarray(
            [
                re.sub(r".*<(.*)>.*", r"\1", pretty_colname)
                for pretty_colname in obj.mudata["cell_data"].var["pretty_colnames"]
            ]
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
        n_mcl = fsom.get_cell_data().uns["n_metaclusters"]
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


def aggregate_flowframes(filenames, c_total, channels=None, keep_order=False):
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
        if channels is not None:
            f = f[:, list(get_markers(f, channels).keys())]
        cPerFile = min([f.X.shape[0], cFile])

        # Random sampling
        ids = random.sample(range(f.X.shape[0]), cPerFile)
        if keep_order:
            ids = sorted(ids)

        file_ids = np.repeat(i, cPerFile)

        f = f[ids,]
        f.obs["Original_ID"] = np.array(ids, dtype=np.float32)
        f.obs["File"] = np.array(file_ids, dtype=np.float32)
        f.obs["File_scattered"] = np.array(
            np.add(file_ids, np.random.normal(loc=0.0, scale=0.1, size=len(file_ids))), dtype=np.float32
        )
        flow_frame.append(f)
    flow_frame = ad.AnnData.concatenate(*flow_frame, join="outer", uns_merge="first")
    return flow_frame


def read_FCS(filepath):
    """Reads in an FCS file
    :param filepath: An array containing a full path to the FCS file
    :type filepath: str
    """
    try:
        f = pm.io.read_fcs(filepath)
        f.var.n = f.var.n.astype(int)
        f.var = f.var.sort_values(by="n")
        f.uns["meta"]["channels"].index = f.uns["meta"]["channels"].index.astype(int)
        f.uns["meta"]["channels"] = f.uns["meta"]["channels"].sort_index()
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
