"""Top-level package for 1D networks simulations with FEniCSx."""

from importlib.metadata import metadata

meta = metadata("networks_fenicsx")
__version__ = meta["Version"]
__author__ = meta.get("Author", "")
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]

from .mesh.mesh import NetworkMesh
from .assembly import HydraulicNetworkAssembler
from .solver import Solver
from .config import Config
import networks_fenicsx.post_processing as post_processing

__all__ = ["HydraulicNetworkAssembler", "NetworkMesh", "post_processing", "Solver", "Config"]
