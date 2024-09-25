from . import PyFlowSOM_SOMEstimator
from . import BaseFlowSOMEstimator, ConsensusCluster


class PyFlowSOMEstimator(BaseFlowSOMEstimator):
    """A class that implements the FlowSOM model."""

    def __init__(
        self,
        cluster_model=PyFlowSOM_SOMEstimator,
        metacluster_model=ConsensusCluster,
        **kwargs,
    ):
        """Initialize the FlowSOMEstimator object."""
        super().__init__(
            cluster_model=cluster_model,
            metacluster_model=metacluster_model,
            **kwargs,
        )
