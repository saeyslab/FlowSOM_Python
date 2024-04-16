from __future__ import annotations

import copy

import anndata as ad
import igraph as ig
import numpy as np
import pandas as pd
from loguru import logger
from mudata import MuData
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from scipy.stats import median_abs_deviation
from sklearn.base import check_is_fitted

from flowsom.io import read_csv, read_FCS
from flowsom.models.flowsom_estimator import FlowSOMEstimator
from flowsom.tl import get_channels, get_markers


class FlowSOM:
    """A class that contains all the FlowSOM data using MuData objects."""

    def __init__(
        self,
        inp,
        n_clusters,
        cols_to_use=None,
        model=FlowSOMEstimator,
        xdim=10,
        ydim=10,
        rlen=10,
        mst=1,
        alpha=(0.05, 0.01),
        mad_allowed=4,
        **kwargs,
    ):
        """Initialize the FlowSOM AnnData object.

        :param n_clusters: The number of clusters
        :type n_clusters: int
        :param xdim: The x dimension of the SOM
        :type xdim: int
        :param ydim: The y dimension of the SOM
        :type ydim: int
        :param rlen: Number of times to loop over the training data for each MST
        :type rlen: int
        :param mst: Number of times to loop over the training data for each MST
        :type mst: int
        :param alpha: The learning rate
        :type alpha: tuple
        :param cols_to_use: The columns to use for clustering
        :type cols_to_use: np.array
        :param mad_allowed: Number of median absolute deviations allowed
        :type mad_allowed: int
        :param model: The model to use
        :type model: FlowSOMEstimator
        :param kwargs: Additional keyword arguments. See documentation of the cluster_model and metacluster_model for more information.
        :type kwargs: dict
        """
        self.cols_to_use = cols_to_use
        self.mad_allowed = mad_allowed
        # cluster model params
        self.xdim = xdim
        self.ydim = ydim
        self.rlen = rlen
        self.mst = mst
        self.alpha = alpha
        # metacluster model params
        self.n_clusters = n_clusters

        self.model = model(
            xdim=xdim,
            ydim=ydim,
            rlen=rlen,
            mst=mst,
            alpha=alpha,
            n_clusters=n_clusters,
            **kwargs,
        )
        self.mudata = MuData(
            {
                "cell_data": ad.AnnData(),
                "cluster_data": ad.AnnData(),
            }
        )
        logger.debug("Reading input.")
        self.read_input(inp)
        logger.debug("Fitting model: clustering and metaclustering.")
        self.run_model()
        logger.debug("Updating derived values.")
        self._update_derived_values()

    @property
    def labels_(self):
        """Get the labels."""
        if "cell_data" in self.mudata.mod.keys():
            if "clustering" in self.mudata["cell_data"].obs_keys():
                return self.mudata["cell_data"].obs["clustering"]
        return None

    @labels_.setter
    def labels_(self, value):
        """Set the labels."""
        if "cell_data" in self.mudata.mod.keys():
            self.mudata["cell_data"].obs["clustering"] = value
        else:
            raise ValueError("No cell data found in the MuData object.")

    def read_input(
        self,
        inp=None,
        cols_to_use=None,
    ):
        """Converts input to a FlowSOM AnnData object.

        :param inp: A file path to an FCS file or a AnnData FCS file to cluster
        :type inp: str / ad.AnnData
        """
        if cols_to_use is not None:
            self.cols_to_use = cols_to_use
        if isinstance(inp, str):
            if inp.endswith(".csv"):
                adata = read_csv(inp)
            elif inp.endswith(".fcs"):
                adata = read_FCS(inp)
        elif isinstance(inp, ad.AnnData):
            adata = inp
        else:
            adata = ad.AnnData(inp)
        self.mudata.mod["cell_data"] = adata
        self.clean_anndata()
        if self.cols_to_use is not None:
            self.cols_to_use = list(get_channels(self, self.cols_to_use).keys())
        if self.cols_to_use is None:
            self.cols_to_use = self.mudata["cell_data"].var_names.values

    def clean_anndata(self):
        """Cleans marker and channel names."""
        adata = self.get_cell_data()
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
        self.mudata.mod["cell_data"] = adata
        return self.mudata

    def run_model(self):
        """Run the model on the input data."""
        X = self.mudata["cell_data"][:, self.cols_to_use].X
        self.model.fit_predict(X)

    def _update_derived_values(self):
        """Update the derived values such as median values and CV values."""
        self.mudata["cell_data"].obs["clustering"] = self.model.cluster_labels
        self.mudata["cell_data"].obs["distance_to_bmu"] = self.model.distances
        self.mudata["cell_data"].uns["n_nodes"] = self.xdim * self.ydim
        self.mudata["cell_data"].var["cols_used"] = np.array(
            col in self.cols_to_use for col in self.mudata["cell_data"].var_names
        )
        self.mudata["cell_data"].uns["n_metaclusters"] = self.n_clusters
        self.mudata["cell_data"].obs["metaclustering"] = self.model.metacluster_labels
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
        cluster_mudata.obs["metaclustering"] = self.model._y_codes
        cluster_mudata.uns["xdim"] = self.xdim
        cluster_mudata.uns["ydim"] = self.ydim
        cluster_mudata.obsm["codes"] = self.model.codes
        cluster_mudata.obsm["grid"] = np.array([(x, y) for x in range(self.xdim) for y in range(self.ydim)])
        cluster_mudata.uns["outliers"] = self.test_outliers(mad_allowed=self.mad_allowed).reset_index()
        # update metacluster values

        self.mudata.mod["cluster_data"] = cluster_mudata
        df = self.mudata["cell_data"].X[self.mudata["cell_data"].X[:, 0].argsort()]
        df = np.c_[self.mudata["cell_data"].obs["metaclustering"], df]
        metacluster_median_values: pd.DataFrame = pd.DataFrame(df).groupby(0).median()
        self.mudata["cluster_data"].uns["metacluster_MFIs"] = metacluster_median_values
        self.build_MST()

    def build_MST(self):
        """Make a minimum spanning tree."""
        check_is_fitted(self.model)
        adjacency = cdist(
            self.model.codes,
            self.model.codes,
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

    def metacluster(self, n_clusters=None):
        """Perform a (consensus) hierarchical clustering.

        :param n_clusters: The number of metaclusters
        :type n_clusters: int
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
            self.model.set_n_clusters(n_clusters)
        self.model.metacluster_model.fit_predict(self.model.codes)
        return self

    def test_outliers(self, mad_allowed: int = 4, fsom_reference=None, plot_file=None, channels=None):
        """Test if any cells are too far from their cluster centers.

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
        """Map new data to a FlowSOM grid.

        :param inp: An anndata or filepath to an FCS file
        :type inp: ad.AnnData / str
        :param mad_allowed: A warning is generated if the distance of the new
        data points to their closest cluster center is too big. This is computed
        based on the typical distance of the points from the original dataset
        assigned to that cluster, the threshold being set to median +
        madAllowed * MAD. Default is 4.
        :type mad_allowed: int
        """
        fsom_new = copy.deepcopy(self)
        fsom_new.read_input(inp)
        fsom_new.mad_allowed = mad_allowed
        X = fsom_new.get_cell_data()[:, self.cols_to_use].X
        fsom_new.model.predict(X)
        fsom_new._update_derived_values()
        return fsom_new

    def subset(self, ids):
        """Take a subset from a FlowSOM object.

        :param ids: An array of ids to subset
        :type ids: np.array
        """
        fsom_subset = copy.deepcopy(self)
        fsom_subset.mudata.mod["cell_data"] = fsom_subset.mudata["cell_data"][ids, :].copy()
        fsom_subset.model.subset(ids)
        fsom_subset._update_derived_values()
        return fsom_subset

    def get_cell_data(self):
        """Get the cell data."""
        return self.mudata["cell_data"]

    def get_cluster_data(self):
        """Get the cluster data."""
        return self.mudata["cluster_data"]


def flowsom_clustering(inp: ad.AnnData, cols_to_use=None, n_clusters=10, xdim=10, ydim=10, **kwargs):
    """Perform FlowSOM clustering on an anndata object and returns the anndata object.

    The FlowSOM clusters and metaclusters are added as variable.

    :param inp: An anndata or filepath to an FCS file
    :type inp: ad.AnnData / str
    """
    fsom = FlowSOM(inp.copy(), cols_to_use=cols_to_use, n_clusters=n_clusters, xdim=xdim, ydim=ydim, **kwargs)
    inp.obs["FlowSOM_clusters"] = fsom.mudata["cell_data"].obs["clustering"]
    inp.obs["FlowSOM_metaclusters"] = fsom.mudata["cell_data"].obs["metaclustering"]
    d = kwargs
    d["cols_to_use"] = cols_to_use
    d["n_clusters"] = n_clusters
    d["xdim"] = xdim
    d["ydim"] = ydim
    inp.uns["FlowSOM"] = d
    return inp
