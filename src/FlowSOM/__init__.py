from importlib.metadata import version

from . import pl 
from . import pp
from . import io
from . import tl
from .main import FlowSOM, flowsom_clustering

__all__ = ["FlowSOM", "flowsom_clustering", "io", "pl", "pp", "tl"]

__version__ = version("FlowSOM")
