from sklearn.utils.validation import check_is_fitted

from . import BaseClusterEstimator


class BaseFlowSOMEstimator(BaseClusterEstimator):
    """Base class for all FlowSOM estimators in FlowSOM."""

    def __init__(
        self,
        cluster_kwargs,
        metacluster_kwargs,
        cluster_model,
        metacluster_model,
    ):
        """Initialize the FlowSOMEstimator object."""
        super().__init__()
        self.cluster_kwargs = cluster_kwargs
        self.metacluster_kwargs = metacluster_kwargs
        self.cluster_model = cluster_model(**cluster_kwargs)
        self.metacluster_model = metacluster_model(**metacluster_kwargs)

    @property
    def codes(self):
        """Return the codes, shaped: (n_clusters, n_features)."""
        check_is_fitted(self, "_is_fitted")
        return self.cluster_model.codes

    @property
    def distances(self):
        """Return the distances."""
        check_is_fitted(self, "_is_fitted")
        return self.cluster_model.distances

    @property
    def cluster_labels(self):
        """Return the cluster labels."""
        check_is_fitted(self, "_is_fitted")
        return self.cluster_labels_

    @property
    def metacluster_labels(self):
        """Return the metacluster labels."""
        check_is_fitted(self, "_is_fitted")
        return self.labels_

    def fit(self, X, y=None):
        """Fit the model."""
        self.cluster_model.fit(X)
        y_codes = self.metacluster_model.fit_predict(self.cluster_model.codes)
        self._y_codes = y_codes
        self._is_fitted = True
        return self

    def fit_predict(self, X):
        """Fit the model and predict the clusters."""
        # overcluster
        y_clusters = self.cluster_model.fit_predict(X)
        self.cluster_labels_ = y_clusters
        # metacluster the overclustered data
        X_codes = self.cluster_model.codes
        y_codes = self.metacluster_model.fit_predict(X_codes)
        # assign the metacluster labels to the original data via the overcluster labels
        y = y_codes[y_clusters]
        self._y_codes = y_codes
        self.labels_ = y
        self._is_fitted = True
        return y

    def predict(self, X):
        """Predict the clusters."""
        check_is_fitted(self, "_is_fitted")
        y_clusters = self.cluster_model.predict(X)
        self.cluster_labels_ = y_clusters
        # skip the metaclustering step
        # assign the metacluster labels to the original data via the overcluster labels
        y = self._y_codes[y_clusters]
        self.labels_ = y
        return y

    def subset(self, indices):
        """Subset the model."""
        self.labels_ = self.labels_[indices]
        self.cluster_labels_ = self.cluster_labels_[indices]
        self.cluster_model.distances = self.cluster_model.distances[indices]
        return self

    def set_n_clusters(self, n_clusters):
        """Set the number of clusters."""
        self.metacluster_kwargs["n_clusters"] = n_clusters
        self.metacluster_model.n_clusters = n_clusters
        return self
