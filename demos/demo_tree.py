from pathlib import Path

from mpi4py import MPI

import matplotlib.pyplot as plt
import numpy as np

import dolfinx
import ufl
from dolfinx import fem
from networks_fenicsx import (
    HydraulicNetworkAssembler,
    NetworkMesh,
    Solver,
    network_generation,
)
from networks_fenicsx.post_processing import export_functions, extract_global_flux

outdir = Path("results_tree")
outdir.mkdir(exist_ok=True, parents=True)


class p_bc_expr:
    def eval(self, x):
        return np.full(x.shape[1], x[1])


min_q, max_q, mean_q = [], [], []

# Create tree
G = network_generation.make_tree(n=2, H=1, W=1)

N = 1
lcars: list[float] = []
for i in range(10):
    N *= 2
    lcars.append(1.0 / N)

    network_mesh = NetworkMesh(G, N=N)
    assembler = HydraulicNetworkAssembler(network_mesh)

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
    solver.assemble()
    sol = solver.solve()

    global_flux = extract_global_flux(network_mesh, sol)
    export_functions(sol, outpath=outdir / f"N_{N:d}")
    with dolfinx.io.VTXWriter(
        global_flux.function_space.mesh.comm,
        outdir / f"N_{N:d}" / "global_flux.bp",
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
    plt.savefig(outdir / "convergence_flux_tree.png")
