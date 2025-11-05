from pathlib import Path

from mpi4py import MPI

import matplotlib.pyplot as plt
import numpy as np

import dolfinx
import ufl
from dolfinx import fem
from networks_fenicsx import (
    Config,
    HydraulicNetworkAssembler,
    NetworkMesh,
    Solver,
    network_generation,
)
from networks_fenicsx.post_processing import export_functions, extract_global_flux

cfg = Config()
cfg.outdir = "demo_tree"
cfg.export = True
cfg.clean = False
cfg.flux_degree = 1
cfg.pressure_degree = 0
cfg.outdir = "demo_tree"


class p_bc_expr:
    def eval(self, x):
        return np.full(x.shape[1], x[1])


lcars, min_q, max_q, mean_q = [], [], [], []
lcar = 1.0

# Create tree
G = network_generation.make_tree(n=2, H=1, W=1)
for i in range(10):
    lcar /= 2.0
    cfg.lcar = lcar
    lcars.append(lcar)

    network_mesh = NetworkMesh(G, cfg)
    assembler = HydraulicNetworkAssembler(cfg, network_mesh)

    assembler.compute_forms(p_bc_ex=p_bc_expr())

    solver = Solver(
        assembler,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        kind="mpi",
    )
    solver.timing_dir = cfg.outdir
    solver.assemble()
    sol = solver.solve()

    global_flux = extract_global_flux(network_mesh, sol)
    export_functions(sol, outpath=Path(cfg.outdir) / f"lcar_{lcar:.5e}")
    with dolfinx.io.VTXWriter(
        global_flux.function_space.mesh.comm,
        Path(cfg.outdir) / f"lcar_{lcar:.5e}" / "global_flux.bp",
        [global_flux],
    ) as vtx:
        vtx.write(0.0)

    max_global_flux = network_mesh.comm.allreduce(np.max(global_flux.x.array), op=MPI.MAX)
    min_global_flux = network_mesh.comm.allreduce(np.min(global_flux.x.array), op=MPI.MIN)

    mean_flux = fem.form(global_flux * ufl.dx)
    area = fem.form(fem.Constant(network_mesh.mesh, 1.0) * ufl.dx)
    mean_global_flux = network_mesh.comm.allreduce(
        fem.assemble_scalar(mean_flux), op=MPI.SUM
    ) / network_mesh.comm.allreduce(fem.assemble_scalar(area), op=MPI.SUM)

    min_q.append(min_global_flux)
    max_q.append(max_global_flux)
    mean_q.append(mean_global_flux)


if network_mesh.comm.rank == 0:
    fig, ax = plt.subplots()
    ax.plot(lcars, mean_q, "-ro", label="mean flux")
    ax.plot(lcars, max_q, "-gs", label="max flux")
    ax.plot(lcars, min_q, "-bx", label="min flux")
    ax.legend()
    ax.grid()
    plt.savefig(Path(cfg.outdir) / "convergence_flux_tree.png")
