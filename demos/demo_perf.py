import os
import numpy as np
from pathlib import Path
from mpi4py import MPI

from networks_fenicsx import NetworkMesh
from networks_fenicsx.mesh import mesh_generation
from networks_fenicsx.solver import assembly, solver
from networks_fenicsx.config import Config
from networks_fenicsx.utils.timers import timing_table
from networks_fenicsx.utils.post_processing import export

cfg = Config()
cfg.outdir = "demo_perf_lm_spaces"
cfg.export = True

cfg.flux_degree = 3
cfg.pressure_degree = 2

cfg.lm_spaces = True
cfg.lm_jump_vectors = False
# cfg.lm_spaces = False
# cfg.lm_jump_vectors = True


class p_bc_expr:
    def eval(self, x):
        return np.full(x.shape[1], x[1])


# One element per segment
cfg.lcar = 2.0

# Cleaning directory only once
cfg.clean_dir()
cfg.clean = False

# cfg.outdir = cfg.outdir + "_cache0"
p = Path(cfg.outdir)
p.mkdir(exist_ok=True)

for n in range(2, 7):
        with (p / "profiling.txt").open("a") as f:
            f.write("n: " + str(n) + "\n")

    # Create tree
    G = mesh_generation.make_tree(n=n, H=n, W=n)
    network_mesh = NetworkMesh(G, cfg)

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

t_dict = timing_table(cfg)

if MPI.COMM_WORLD.rank == 0:
    print("n = ", t_dict["n"])
    print("compute_forms time = ", t_dict["compute_forms"])
    print("assembly time = ", t_dict["assemble"])
    print("solve time = ", t_dict["solve"])


# Run again without clearing the cache
cfg.outdir = cfg.outdir + "_cache1"
p = Path(cfg.outdir)
p.mkdir(exist_ok=True)

for n in range(2, 7):
    if MPI.COMM_WORLD.rank == 0:
        with (p / "profiling.txt").open("a") as f:
            f.write("n: " + str(n) + "\n")

    # Create tree
    G = mesh_generation.make_tree(n=n, H=n, W=n, cfg=cfg)

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

t_dict = timing_table(cfg)

if MPI.COMM_WORLD.rank == 0:
    print("n = ", t_dict["n"])
    print("compute_forms time = ", t_dict["compute_forms"])
    print("assembly time = ", t_dict["assemble"])
    print("solve time = ", t_dict["solve"])
