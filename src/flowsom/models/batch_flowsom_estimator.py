from . import BaseFlowSOMEstimator, ConsensusCluster  # isort:skip
from .batch import BatchSOMEstimator  # isort:skip


class BatchFlowSOMEstimator(BaseFlowSOMEstimator):
    """A class that implements the FlowSOM model."""

    def __init__(
        self,
        cluster_model=BatchSOMEstimator,
        metacluster_model=ConsensusCluster,
        **kwargs,
    ):
        """Initialize the FlowSOMEstimator object."""
        super().__init__(
            cluster_model=cluster_model,
            metacluster_model=metacluster_model,
            **kwargs,
        )
