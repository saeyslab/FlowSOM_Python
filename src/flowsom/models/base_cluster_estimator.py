from sklearn.base import BaseEstimator, ClusterMixin


class BaseClusterEstimator(BaseEstimator, ClusterMixin):
    """Base class for all cluster estimators in FlowSOM."""

    def __init__(self):
        super().__init__()
        self._is_fitted = False

    def __sklearn_is_fitted__(self):
        """Check fitted status and return a Boolean value."""
        return hasattr(self, "_is_fitted") and self._is_fitted
