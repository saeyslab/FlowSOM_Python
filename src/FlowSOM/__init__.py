from importlib.metadata import version

from . import io, pl, pp, tl
from .main import FlowSOM, flowsom_clustering

__all__ = ["FlowSOM", "flowsom_clustering", "io", "pl", "pp", "tl"]

__version__ = version("flowsom")
