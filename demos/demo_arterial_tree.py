import os
import numpy as np
from pathlib import Path
from mpi4py import MPI

from networks_fenicsx import NetworkMesh
from networks_fenicsx.mesh import arterial_tree
from networks_fenicsx.solver import assembly, solver
from networks_fenicsx.config import Config

# from networks_fenicsx.utils.timers import timing_dict, timing_table
from networks_fenicsx.utils.post_processing import export  # , perf_plot

cfg = Config()
cfg.outdir = "demo_arterial_tree"
cfg.export = True
cfg.lm_spaces = False
cfg.flux_degree = 1
cfg.pressure_degree = 0


class p_bc_expr:
    def eval(self, x):
        return np.full(x.shape[1], x[1])


# One element per segment
cfg.lcar = 0.0025

# Cleaning directory only once
cfg.clean_dir()
cfg.clean = False

p = Path(cfg.outdir)
p.mkdir(exist_ok=True)

n = 3

G = arterial_tree.make_arterial_tree(N=n)

network_mesh = NetworkMesh(G, cfg)
assembler = assembly.Assembler(cfg, network_mesh)
# Compute forms
assembler.compute_forms(p_bc_ex=p_bc_expr())
# Assemble
assembler.assemble()
# Solve

solver_ = solver.Solver(cfg, network_mesh, assembler)
sol = solver_.solve()
(fluxes, global_flux, pressure) = export(
    cfg, network_mesh, assembler.function_spaces, sol, export_dir="n" + str(n)
)
