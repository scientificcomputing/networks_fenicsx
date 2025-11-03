from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from dolfinx import fem
import ufl
from pathlib import Path
from networks_fenicsx import NetworkMesh
from networks_fenicsx.utils.post_processing import export
from networks_fenicsx.mesh import mesh_generation
from networks_fenicsx.solver import assembly, solver
from networks_fenicsx.config import Config


cfg = Config()
cfg.outdir = "demo_tree"
cfg.export = True
cfg.clean = False
cfg.lm_space = True
cfg.flux_degree = 1
cfg.pressure_degree = 0

if cfg.lm_space:
    cfg.outdir = "demo_tree_lm"
else:
    cfg.outdir = "demo_tree"


class p_bc_expr:
    def eval(self, x):
        return np.full(x.shape[1], x[1])


lcars, min_q, max_q, mean_q = [], [], [], []
lcar = 1.0

# Create tree
G = mesh_generation.make_tree(n=2, H=1, W=1)
for i in range(10):
    lcar /= 2.0
    cfg.lcar = lcar
    lcars.append(lcar)

    network_mesh = NetworkMesh(G, cfg)
    assembler = assembly.Assembler(cfg, network_mesh)
    
    # Workaround until: https://github.com/FEniCS/dolfinx/pull/3974 is merged
    network_mesh.lm_mesh.topology.create_entity_permutations()
    assembler.compute_forms(p_bc_ex=p_bc_expr())

    solver_ = solver.Solver(cfg, network_mesh, assembler)
    sol = solver_.solve()
    
    (fluxes, global_flux, pressure) = export(network_mesh, assembler.function_spaces, sol, outpath=Path(cfg.outdir) / f"lcar_{lcar:.5e}")

    max_global_flux = network_mesh.comm.allreduce(np.max(global_flux.x.array), op=MPI.MAX)
    min_global_flux = network_mesh.comm.allreduce(np.min(global_flux.x.array), op=MPI.MIN)

    mean_flux = fem.form(global_flux*ufl.dx)
    area = fem.form(fem.Constant(network_mesh.mesh, 1.0)*ufl.dx)
    mean_global_flux = network_mesh.comm.allreduce(fem.assemble_scalar(mean_flux), op=MPI.SUM) / network_mesh.comm.allreduce(fem.assemble_scalar(area), op=MPI.SUM)


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
