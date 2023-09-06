from importlib.metadata import version
from . import tools as tl
from . import read_write as io
from . import preprocessing as pp
from . import plotting as pl
from .main import FlowSOM, flowsom_clustering

__all__ = ["plotting", "main"]

__version__ = version("FlowSOM")
