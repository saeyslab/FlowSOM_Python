from . import BaseFlowSOMEstimator, ConsensusCluster # isort:skip
from .batch import SOMEstimator_batch_init # isort:skip


class BatchFlowSOMEstimator(BaseFlowSOMEstimator):
    """A class that implements the FlowSOM model."""

    def __init__(
        self,
        cluster_model=SOMEstimator_batch_init,
        metacluster_model=ConsensusCluster,
        **kwargs,
    ):
        """Initialize the FlowSOMEstimator object."""
        super().__init__(
            cluster_model=cluster_model,
            metacluster_model=metacluster_model,
            **kwargs,
        )
