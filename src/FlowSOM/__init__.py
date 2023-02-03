from importlib.metadata import version

from . import pl
from . import main

# import pooch

__all__ = ["pl", "main"]

__version__ = version("FlowSOM")

"""
example_dataset = pooch.create(
    path=pooch.os_cache("example_data_set"),
    base_url="https://dl01.irc.ugent.be/",
    version="0.1.0",
    # overwriting data path with environment variable
    env="FlowSOM",
    registry={
        "flow/FlowRepository_FR-FCM-ZZPH/Levine_13dim.fcs": None,
        "flow/FlowRepository_FR-FCM-ZZPH_zarr/Levine_13dim.zarr": None,
        "flow/pixie/fov0.feather": None,
    }
    # TODO: create registry file with hashes
    # registry=None,
)

"""
