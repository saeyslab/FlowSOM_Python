from importlib.metadata import version

from . import pl
from . import FlowSOM

__all__ = ["pl", "FlowSOM"]

__version__ = version("FlowSOM")
