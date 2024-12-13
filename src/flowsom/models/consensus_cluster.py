###############################################
# Copyright Žiga Sajovic, XLAB 2019           #
# Distributed under the MIT License           #
#                                             #
# github.com/ZigaSajovic/Consensus_Clustering #
#                                             #
###############################################

from __future__ import annotations

import bisect
from itertools import combinations

import numpy as np
from scipy.stats import zscore
from sklearn.cluster import AgglomerativeClustering

from . import BaseClusterEstimator


class ConsensusCluster(BaseClusterEstimator):
    """
    Implementation of Consensus clustering.

    This follows the paper
    https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf
    https://github.com/ZigaSajovic/Consensus_Clustering/blob/master/consensusClustering.py
      * cluster -> clustering class
      * NOTE: the class is to be instantiated with parameter `n_clusters`,
        and possess a `fit_predict` method, which is invoked on data.
      * L -> smallest number of clusters to try
      * K -> biggest number of clusters to try
      * H -> number of resamplings for each cluster number
      * resample_proportion -> percentage to sample.
    """

    def __init__(
        self, n_clusters, K=None, H=100, resample_proportion=0.9, linkage="average", cluster=AgglomerativeClustering
    ):
        super().__init__()
        assert 0 <= resample_proportion <= 1, "proportion has to be between 0 and 1"
        self.n_clusters = n_clusters
        self.K = K if K else n_clusters
        self.H = H
        self.resample_proportion = resample_proportion
        self.cluster = cluster
        self.linkage = linkage

    def _internal_resample(self, data, proportion):
        """Resamples the data.

        Args:
          * data -> (examples,attributes) format
          * proportion -> percentage to sample.
        """
        resampled_indices = np.random.choice(range(data.shape[0]), size=int(data.shape[0] * proportion), replace=False)
        return resampled_indices, data[resampled_indices, :]

    def fit(self, data):
        """
        Fits a consensus matrix for each number of clusters.

        Args:
          * data -> (examples,attributes) format
        """
        # zscore and clip
        data = zscore(data, axis=0)
        data = np.clip(data, a_min=-3, a_max=3)
        Mk = np.zeros((data.shape[0], data.shape[0]))
        Is = np.zeros((data.shape[0],) * 2)
        for _ in range(self.H):
            resampled_indices, resample_data = self._internal_resample(data, self.resample_proportion)
            Mh = self.cluster(n_clusters=self.K, linkage=self.linkage).fit_predict(resample_data)
            index_mapping = np.array((Mh, resampled_indices)).T
            index_mapping = index_mapping[index_mapping[:, 0].argsort()]
            sorted_ = index_mapping[:, 0]
            id_clusts = index_mapping[:, 1]
            for i in range(self.K):
                ia = bisect.bisect_left(sorted_, i)
                ib = bisect.bisect_right(sorted_, i)
                is_ = id_clusts[ia:ib]
                ids_ = np.array(list(combinations(is_, 2))).T
                if ids_.size != 0:
                    Mk[ids_[0], ids_[1]] += 1
            ids_2 = np.array(list(combinations(resampled_indices, 2))).T
            Is[ids_2[0], ids_2[1]] += 1
        Mk /= Is + 1e-8
        Mk += Mk.T
        Mk[range(data.shape[0]), range(data.shape[0])] = 1
        self.Mk = Mk
        self._is_fitted = True
        return self

    def fit_predict(self, data):
        """Predicts on the consensus matrix, for best found cluster number."""
        data = zscore(data, axis=0)
        data = np.clip(data, a_min=-3, a_max=3)
        return self.cluster(n_clusters=self.n_clusters, linkage=self.linkage).fit_predict(data)
