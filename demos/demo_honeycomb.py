import os
import numpy as np
from pathlib import Path
from mpi4py import MPI

from networks_fenicsx.mesh import mesh_generation
from networks_fenicsx.solver import assembly, solver
from networks_fenicsx.config import Config
from networks_fenicsx.utils.timers import timing_table
# from networks_fenicsx.utils.post_processing import export

# Clear fenics cache
print('Clearing cache')
os.system('dijitso clean')

cfg = Config()
cfg.outdir = "demo_honeycomb"
cfg.export = True

# cfg.lcar = 0.25
cfg.lm_space = True

# Cleaning directory only once
cfg.clean_dir()
cfg.clean = False

# cfg.outdir = cfg.outdir + "_cache0"
p = Path(cfg.outdir)
p.mkdir(exist_ok=True)

n = 1
if MPI.COMM_WORLD.rank == 0:
    print('Clearing cache')
    os.system('rm -rf $HOME/.cache/fenics/')

    with (p / 'profiling.txt').open('a') as f:
        f.write("n: " + str(n) + "\n")

# Create Y bifurcation graph
G = mesh_generation.make_honeycomb(6, 6, cfg)


class p_bc_expr:
    def eval(self, x):
        return np.full(x.shape[1], x[0] + x[1])


assembler = assembly.Assembler(cfg, G)
assembler.compute_forms(p_bc_ex=p_bc_expr())

solver = solver.Solver(cfg, G, assembler)
sol = solver.solve()

# FIXME (parallel)
# (fluxes, global_flux, pressure) = export(cfg, G, assembler.function_spaces, sol)

t_dict = timing_table(cfg)

if MPI.COMM_WORLD.rank == 0:
    print("n = ", t_dict["n"])
    print("compute_forms time = ", t_dict["compute_forms"])
    print("assembly time = ", t_dict["assemble"])
    print("solve time = ", t_dict["solve"])
