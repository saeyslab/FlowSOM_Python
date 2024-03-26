from flowsom.models import ConsensusCluster, SOMEstimator
from flowsom.models.base_flowsom_estimator import BaseFlowSOMEstimator


class FlowSOMEstimator(BaseFlowSOMEstimator):
    """A class that implements the FlowSOM model."""

    def __init__(
        self,
        cluster_kwargs,
        metacluster_kwargs,
        cluster_model=SOMEstimator,
        metacluster_model=ConsensusCluster,
    ):
        """Initialize the FlowSOMEstimator object."""
        super().__init__(
            cluster_kwargs=cluster_kwargs,
            metacluster_kwargs=metacluster_kwargs,
            cluster_model=cluster_model,
            metacluster_model=metacluster_model,
        )
