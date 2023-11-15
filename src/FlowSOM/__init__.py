from importlib.metadata import version

from . import plotting as pl
from . import preprocessing as pp
from . import read_write as io
from . import tools as tl
from .main import FlowSOM, flowsom_clustering

__all__ = [
    "pl",
    "pp",
    "io",
    "tl",
    "main",
    "FlowSOM",
    "flowsom_clustering",
]

__version__ = version("FlowSOM")
