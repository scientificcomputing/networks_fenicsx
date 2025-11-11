"""Top-level package for 1D networks simulations with FEniCSx."""

from importlib.metadata import metadata

meta = metadata("networks_fenicsx")
__version__ = meta["Version"]
__author__ = meta.get("Author", "")
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]

import networks_fenicsx.network_generation as network_generation
import networks_fenicsx.post_processing as post_processing

from .assembly import HydraulicNetworkAssembler
from .mesh import NetworkMesh
from .solver import Solver

__all__ = [
    "HydraulicNetworkAssembler",
    "NetworkMesh",
    "post_processing",
    "Solver",
    "network_generation",
]
