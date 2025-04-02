from . import BaseFlowSOMEstimator, ConsensusCluster
from .pyFlowSOM_som_estimator import PyFlowSOM_SOMEstimator


class PyFlowSOMEstimator(BaseFlowSOMEstimator):
    """A class that implements the FlowSOM model."""

    def __init__(
        self,
        cluster_model=PyFlowSOM_SOMEstimator,
        metacluster_model=ConsensusCluster,
        **kwargs,
    ):
        super().__init__(
            cluster_model=cluster_model,
            metacluster_model=metacluster_model,
            **kwargs,
        )
