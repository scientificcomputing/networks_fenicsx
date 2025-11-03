import numpy as np
from pathlib import Path
from mpi4py import MPI
import shutil
from networks_fenicsx import NetworkMesh
from networks_fenicsx.mesh import mesh_generation
from networks_fenicsx.solver import assembly, solver
from networks_fenicsx.config import Config
from networks_fenicsx.utils.timers import timing_table
from networks_fenicsx.utils.post_processing import export

cfg = Config()
cfg.outdir = "demo_perf_lm_space"
cfg.export = True

cfg.flux_degree = 1
cfg.pressure_degree = 0

cfg.lm_space = True


class p_bc_expr:
    def eval(self, x):
        return np.full(x.shape[1], x[1])




# One element per segment
cfg.lcar = 2.0

# Cleaning directory only once
cfg.clean_dir()
cfg.clean = False

cfg.outdir.mkdir(exist_ok=True,parents=True)
cache_dir = cfg.outdir / f".cache"
if cache_dir.exists():
    shutil.rmtree(cache_dir, ignore_errors=True)

jit_options = {"cache_dir": cache_dir}
ns = [7]
for n in ns:
    with (cfg.outdir / "profiling.txt").open("a") as f:
        f.write("n: " + str(n) + "\n")

    # Create tree
    G = mesh_generation.make_tree(n=n, H=n, W=n)
    network_mesh = NetworkMesh(G, cfg)

    assembler = assembly.Assembler(cfg, network_mesh)
    # Compute forms
    assembler.compute_forms(p_bc_ex=p_bc_expr(), jit_options=jit_options)

    # Solve
    solver_ = solver.Solver(cfg, network_mesh, assembler)
    sol = solver_.solve()

    (fluxes, global_flux, pressure) = export(
        network_mesh, assembler.function_spaces, sol, outpath=cfg.outdir / f"n{n}"
                )
t_dict = timing_table(cfg)

if MPI.COMM_WORLD.rank == 0:
    print("n = ", t_dict["n"])
    print("compute_forms time = ", t_dict["compute_forms"])
    print("assembly time = ", t_dict["assemble"])
    print("solve time = ", t_dict["solve"])


for n in ns:
    if MPI.COMM_WORLD.rank == 0:
        with (cfg.outdir / "profiling.txt").open("a") as f:
            f.write("n: " + str(n) + "\n")

    # Create tree
    G = mesh_generation.make_tree(n=n, H=n, W=n)
    network_mesh = NetworkMesh(G, cfg)

    assembler = assembly.Assembler(cfg, network_mesh)
    # Compute forms
    assembler.compute_forms(p_bc_ex=p_bc_expr(), jit_options=jit_options)

    # Solve
    solver_ = solver.Solver(cfg, network_mesh, assembler)
    sol = solver_.solve()
    (fluxes, global_flux, pressure) = export(
        network_mesh, assembler.function_spaces, sol, outpath=cfg.outdir / f"n{n}"
    )

t_dict = timing_table(cfg)

if MPI.COMM_WORLD.rank == 0:
    print("n = ", t_dict["n"])
    print("compute_forms time = ", t_dict["compute_forms"])
    print("assembly time = ", t_dict["assemble"])
    print("solve time = ", t_dict["solve"])
