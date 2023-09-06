from importlib.metadata import version

from . import pl
from . import main

__all__ = ["pl", "main"]

__version__ = version("FlowSOM")
