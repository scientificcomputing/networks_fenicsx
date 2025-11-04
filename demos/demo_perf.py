import time
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
cfg.export = True

cfg.flux_degree = 1
cfg.pressure_degree = 0
cfg.graph_coloring = True
cfg.color_strategy = "smallest_last"
cfg.outdir = "demo_perf"


class p_bc_expr:
    def eval(self, x):
        return x[1]


# One element per segment
cfg.lcar = 1
# Cleaning directory only once
# cfg.clean_dir()
cfg.clean = False

cfg.outdir.mkdir(exist_ok=True, parents=True)
cache_dir = cfg.outdir / f".cache"
if cache_dir.exists():
    shutil.rmtree(cache_dir, ignore_errors=True)

cffi_options = ["-Ofast", "-march=native"]
jit_options = {"cache_dir": cache_dir, "cffi_extra_compile_args": cffi_options}
ns = [3, 6, 12, 24]
for n in ns:
    if MPI.COMM_WORLD.rank == 0:
        with (cfg.outdir / "profiling.txt").open("a") as f:
            f.write("n: " + str(n) + "\n")

    # Create tree
    if MPI.COMM_WORLD.rank == 0:
        G = mesh_generation.make_tree(n=n, H=n, W=n)
    else:
        G = None
    start2 = time.perf_counter()
    network_mesh = NetworkMesh(G, cfg)
    del G
    network_mesh.export_tangent()

    for i in range(network_mesh._num_edge_colors):
        import dolfinx

        network_mesh.submeshes[i].topology.create_connectivity(0, 1)
        with dolfinx.io.XDMFFile(
            network_mesh.submeshes[i].comm, cfg.outdir / f"submesh_{i}.xdmf", "w"
        ) as xdmf:
            xdmf.write_mesh(network_mesh.submeshes[i])
            xdmf.write_meshtags(
                network_mesh.submesh_facet_markers[i], network_mesh.submeshes[i].geometry
            )

    end2 = time.perf_counter()
    print(f"generate mesh: {end2 - start2}")
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
exit()
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
