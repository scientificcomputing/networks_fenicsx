import os
import numpy as np
from pathlib import Path
from mpi4py import MPI

from networks_fenicsx.mesh import arterial_tree
from networks_fenicsx.solver import assembly, solver
from networks_fenicsx.config import Config

# from networks_fenicsx.utils.timers import timing_dict, timing_table
from networks_fenicsx.utils.post_processing import export  # , perf_plot

cfg = Config()
cfg.outdir = "demo_arterial_tree"
cfg.export = True

cfg.flux_degree = 2
cfg.pressure_degree = 1


class p_bc_expr:
    def eval(self, x):
        return np.full(x.shape[1], x[1])


# One element per segment
cfg.lcar = 0.1

# Cleaning directory only once
cfg.clean_dir()
cfg.clean = False

p = Path(cfg.outdir)
p.mkdir(exist_ok=True)

n = 3
if MPI.COMM_WORLD.rank == 0:
    print("Clearing cache")
    os.system("rm -rf $HOME/.cache/fenics/")

G = arterial_tree.make_arterial_tree(N=n, cfg=cfg)
exit()
assembler = assembly.Assembler(cfg, G)
# Compute forms
assembler.compute_forms(p_bc_ex=p_bc_expr())
# Assemble
assembler.assemble()
# Solve
solver_ = solver.Solver(cfg, G, assembler)
sol = solver_.solve()
(fluxes, global_flux, pressure) = export(
    cfg, G, assembler.function_spaces, sol, export_dir="n" + str(n)
)
