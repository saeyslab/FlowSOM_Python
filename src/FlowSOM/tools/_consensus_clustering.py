###############################################
# Copyright Žiga Sajovic, XLAB 2019           #
# Distributed under the MIT License           #
#                                             #
# github.com/ZigaSajovic/Consensus_Clustering #
#                                             #
###############################################

import bisect
from itertools import combinations

import numpy as np


class ConsensusCluster:
    """
    Implementation of Consensus clustering, following the paper
    https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf
    https://github.com/ZigaSajovic/Consensus_Clustering/blob/master/consensusClustering.py
      * cluster -> clustering class
      * NOTE: the class is to be instantiated with parameter `n_clusters`,
        and possess a `fit_predict` method, which is invoked on data.
      * L -> smallest number of clusters to try
      * K -> biggest number of clusters to try
      * H -> number of resamplings for each cluster number
      * resample_proportion -> percentage to sample
    """

    def __init__(self, cluster, K, H, resample_proportion=0.9, linkage="average"):
        assert 0 <= resample_proportion <= 1, "proportion has to be between 0 and 1"
        self.cluster_ = cluster
        self.resample_proportion_ = resample_proportion
        self.linkage = linkage
        self.K_ = K
        self.H_ = H

    def _internal_resample(self, data, proportion):
        """
        Args:
          * data -> (examples,attributes) format
          * proportion -> percentage to sample
        """
        resampled_indices = np.random.choice(range(data.shape[0]), size=int(data.shape[0] * proportion), replace=False)
        return resampled_indices, data[resampled_indices, :]

    def fit(self, data):
        """
        Fits a consensus matrix for each number of clusters

        Args:
          * data -> (examples,attributes) format
        """
        Mk = np.zeros((data.shape[0], data.shape[0]))
        Is = np.zeros((data.shape[0],) * 2)
        for _ in range(self.H_):
            resampled_indices, resample_data = self._internal_resample(data, self.resample_proportion_)
            Mh = self.cluster_(n_clusters=self.K_, linkage=self.linkage).fit_predict(resample_data)
            index_mapping = np.array((Mh, resampled_indices)).T
            index_mapping = index_mapping[index_mapping[:, 0].argsort()]
            sorted_ = index_mapping[:, 0]
            id_clusts = index_mapping[:, 1]
            for i in range(self.K_):
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

    def predict(self, n_clus=10):
        """
        Predicts on the consensus matrix, for best found cluster number
        """
        return self.cluster_(n_clusters=n_clus, linkage=self.linkage, metric="precomputed").fit_predict(1 - self.Mk)
