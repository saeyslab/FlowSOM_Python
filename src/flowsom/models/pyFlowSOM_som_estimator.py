import numpy as np
from sklearn.utils.validation import check_is_fitted

from . import BaseClusterEstimator


class PyFlowSOM_SOMEstimator(BaseClusterEstimator):
    """Estimate a Self-Organizing Map (SOM) clustering model using pyFlowSOM SOM implementation."""

    def __init__(
        self,
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
        super().__init__()
        self.xdim = xdim
        self.ydim = ydim
        self.rlen = rlen
        self.mst = mst
        self.alpha = alpha
        self.init = init
        self.initf = initf
        self.map = map
        self.codes = codes
        self.importance = importance
        self.seed = seed

    def fit(
        self,
        X,
        y=None,
    ):
        """Perform SOM clustering.

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
        from pyFlowSOM import map_data_to_nodes, som

        alpha = self.alpha
        X = X.astype("double")

        if self.seed is not None:
            np.random.seed(self.seed)

        codes = som(
            X,
            xdim=self.xdim,
            ydim=self.ydim,
            rlen=self.rlen,
            alpha_range=alpha,
            importance=self.importance,
            seed=self.seed,
        )

        clusters, dists = map_data_to_nodes(codes, X)
        self.codes, self.labels_, self.distances = codes.copy(), clusters.astype(int) - 1, dists
        self._is_fitted = True
        return self

    def predict(self, X, y=None):
        """Predict labels using the model."""
        from pyFlowSOM import map_data_to_nodes

        check_is_fitted(self)

        X = X.astype("double")
        clusters, dists = map_data_to_nodes(self.codes, X)
        self.labels_ = clusters.astype(int) - 1
        self.distances = dists
        return self.labels_

    def fit_predict(self, X, y=None):
        """Fit the model and predict labels."""
        self.fit(X)
        return self.predict(X)
