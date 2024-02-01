from __future__ import annotations

import copy
import random

import anndata as ad
import igraph as ig
import numpy as np
import pandas as pd
from mudata import MuData
from scipy.sparse import issparse
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import median_abs_deviation
from sklearn.cluster import AgglomerativeClustering

from .io import read_csv, read_FCS
from .tl import SOM, ConsensusCluster, get_channels, get_markers, map_data_to_codes


class FlowSOM:
    """A class that contains all the FlowSOM data using MuData objects."""

    def __init__(self, inp=None, cols_to_use: np.ndarray | None = None, n_clus=10, seed: int | None = None, **kwargs):
        """Initialize the FlowSOM AnnData object

        :param inp: A file path to an FCS file or a AnnData FCS file to cluster
        :type inp: str / ad.AnnData
        :param cols_to_use:  An array of the columns to use for clustering
        :type cols_to_use: np.array
        :param n_clus: The number of metacluster
        :type n_clus: int
        :param seed: A fixed seed
        :type seed: int
        """
        if seed is not None:
            random.seed(seed)
        if inp is not None:
            self.mudata = self.read_input(inp)
            self.build_SOM(cols_to_use, **kwargs)
            self.build_MST()
            self.metacluster(n_clus)
            self._update_derived_values()

    def read_input(self, inp):
        """Converts input to a FlowSOM AnnData object

        :param inp: A file path to an FCS file or a AnnData FCS file to cluster
        :type inp: str / ad.AnnData
        """
        if isinstance(inp, str):
            if inp.endswith(".csv"):
                adata = read_csv(inp)
            elif inp.endswith(".fcs"):
                adata = read_FCS(inp)
        else:
            adata = inp
        if issparse(adata.X):
            # sparse matrices are not supported
            adata.X = adata.X.todense()
        if "channel" not in adata.var.keys():
            adata.var["channel"] = np.asarray(adata.var_names)
        channels = np.asarray(adata.var["channel"])
        if "marker" not in adata.var.keys():
            adata.var["marker"] = np.asarray(adata.var_names)
        markers = np.asarray(adata.var["marker"])
        isnan_markers = [str(marker) == "nan" or len(marker) == 0 for marker in markers]
        markers[isnan_markers] = channels[isnan_markers]
        pretty_colnames = [markers[i] + " <" + channels[i] + ">" for i in range(len(markers))]
        adata.var["pretty_colnames"] = np.asarray(pretty_colnames, dtype=str)
        adata.var_names = np.asarray(channels)
        adata.var["markers"] = np.asarray(markers)
        adata.var["channels"] = np.asarray(channels)
        self.mudata = MuData({"cell_data": adata})
        return self.mudata

    def build_SOM(self, cols_to_use: np.ndarray | None = None, outlier_MAD=4, **kwargs):
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
        self.mudata["cell_data"].obs["clustering"] = np.array(clusters, dtype=int)
        self.mudata["cell_data"].obs["distance_to_bmu"] = np.array(dists)
        self.mudata["cell_data"].var["cols_used"] = [x in cols_to_use for x in self.mudata["cell_data"].var_names]

        self.mudata["cell_data"].uns["n_nodes"] = xdim * ydim
        self._update_derived_values()
        self.mudata["cluster_data"].uns["xdim"] = xdim
        self.mudata["cluster_data"].uns["ydim"] = ydim
        self.mudata["cluster_data"].obsm["codes"] = np.array(codes)
        self.mudata["cluster_data"].obsm["grid"] = np.array([(x, y) for x in range(xdim) for y in range(ydim)])
        self.mudata["cluster_data"].uns["outliers"] = self.test_outliers(mad_allowed=outlier_MAD).reset_index()

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
            assert (
                (codes.shape[1] == data.shape[1]) or (codes.shape[0] != xdim * ydim)
            ), "If codes is not NULL, it should have the same number of columns as the data and the number of rows should correspond with xdim*ydim"

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
                nhbrdist = self._dist_mst(codes)

        if map:
            clusters, dists = map_data_to_codes(data=data, codes=codes)
        else:
            clusters = dists = np.zeros(n_codes)

        return codes, clusters, dists, xdim, ydim

    def _update_derived_values(self):
        """Update the derived values such as median values and CV values"""
        # get dataframe of intensities and cluster labels on cell level
        df = self.mudata["cell_data"].to_df()  # [self.adata.X[:, 0].argsort()]
        df = pd.concat([self.mudata["cell_data"].obs["clustering"], df], axis=1)
        n_nodes = self.mudata["cell_data"].uns["n_nodes"]

        # get median values per cluster on cell level
        cluster_median_values = df.groupby("clustering").median()
        # make sure cluster_median_values is of length n_nodes
        # some clusters might be empty when fitting on new data
        missing_clusters = set(range(n_nodes)) - set(cluster_median_values.index)
        if len(missing_clusters) > 0:
            cluster_median_values = cluster_median_values.reindex(
                list(cluster_median_values.index) + list(missing_clusters)
            )
        cluster_median_values.sort_index(inplace=True)
        # create values for cluster_data
        if "cluster_data" in self.mudata.mod.keys():
            cluster_mudata = self.mudata.mod["cluster_data"]
            cluster_mudata.X = cluster_median_values.values
        else:
            cluster_mudata = ad.AnnData(cluster_median_values.values)
        cluster_mudata.var_names = self.mudata["cell_data"].var_names
        # standard deviation of cells per cluster
        sd_values = []
        # coefficient of variation of cells per cluster
        cv_values = []
        # median absolute deviation of cells per cluster
        mad_values = []
        # percentages of cells of cells per cluster
        pctgs = {}
        for cl in range(n_nodes):
            cluster_data = df[df["clustering"] == cl]
            # if cluster is empty, set values to nan for all markers
            if cluster_data.shape[0] == 0:
                cluster_mudata.X[cl, :] = np.nan
                cv_values.append([np.nan] * cluster_data.shape[1])
                sd_values.append([np.nan] * cluster_data.shape[1])
                mad_values.append([np.nan] * cluster_data.shape[1])
                pctgs[cl] = 0
                continue
            means = np.nanmean(cluster_data, axis=0)
            means[means == 0] = np.nan
            cv_values.append(np.divide(np.nanstd(cluster_data, axis=0), means))
            sd_values.append(np.nanstd(cluster_data, axis=0))
            mad_values.append(median_abs_deviation(cluster_data, axis=0, nan_policy="omit"))
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

    def _dist_mst(self, codes):
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
        metaclusters = self.consensus_hierarchical_clustering(self.mudata["cluster_data"].obsm["codes"], n_clus)
        self.mudata["cell_data"].uns["n_metaclusters"] = n_clus
        self.mudata["cluster_data"].obs["metaclustering"] = metaclusters.astype(str)
        self.mudata["cell_data"].obs["metaclustering"] = np.asarray(
            [np.array(metaclusters)[int(i)] for i in np.asarray(self.mudata["cell_data"].obs["clustering"])]
        )
        return self

    def consensus_hierarchical_clustering(
        self, data, n_clus, n_subsamples=100, linkage="average", resample_proportion=0.9
    ):
        average = ConsensusCluster(
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
            outliers_dict = {}
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
        fsom_new = copy.deepcopy(FlowSOM(inp))
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
        fsom_new._update_derived_values()
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
        fsom_subset = copy.deepcopy(self)
        fsom_subset.mudata.mod["cell_data"] = fsom_subset.mudata["cell_data"][ids, :]
        fsom_subset._update_derived_values()
        return fsom_subset

    def get_cell_data(self):
        return self.mudata["cell_data"]

    def get_cluster_data(self):
        return self.mudata["cluster_data"]


def flowsom_clustering(inp, cols_to_use=None, n_clus=10, xdim=10, ydim=10, seed=None, **kwargs):
    """Perform FlowSOM clustering on an anndata object and returns the anndata
       object with the FlowSOM clusters and metaclusters added as variable

    :param inp: An anndata or filepath to an FCS file
    :type inp: ad.AnnData / str
    """
    fsom = FlowSOM(inp, cols_to_use=cols_to_use, n_clus=n_clus, xdim=xdim, ydim=ydim, seed=seed, **kwargs)
    inp.obs["FlowSOM_clusters"] = fsom.mudata["cell_data"].obs["clustering"]
    inp.obs["FlowSOM_metaclusters"] = fsom.mudata["cell_data"].obs["metaclustering"]
    inp.uns["FlowSOM"] = {"cols_to_use": cols_to_use, "n_clus": n_clus, "xdim": xdim, "ydim": ydim, "seed": seed}
    return inp
