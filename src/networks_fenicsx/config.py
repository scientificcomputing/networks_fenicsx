from dataclasses import dataclass
from pathlib import Path
from mpi4py import MPI
import shutil


@dataclass
class Config:
    _outdir: Path = Path("results")
    lm_space: bool = True
    lcar: float = 1.0
    flux_degree: int = 1
    pressure_degree: int = 0
    export: bool = False
    clean: bool = True
    graph_coloring: bool = False
    color_strategy: str = "largest_first"

    def __post_init__(self):
        if self.graph_coloring and not self.lm_space:
            raise ValueError("Graph coloring can only be used with lm_space=True")

    @property
    def outdir(self):
        return self._outdir

    @outdir.setter
    def outdir(self, value):
        self._outdir = Path(value)

    def clean_dir(self):
        if self.clean and MPI.COMM_WORLD.rank == 0:
            dirpath = Path(self.outdir)
            if dirpath.exists() and dirpath.is_dir():
                shutil.rmtree(dirpath, ignore_errors=True)
        MPI.COMM_WORLD.barrier()

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def default_parameters():
    return {k: v for k, v in Config.__dict__.items() if not k.startswith("_")}
