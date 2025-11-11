from pathlib import Path

import networkx as nx
import numpy as np

import dolfinx.io
from networks_fenicsx import HydraulicNetworkAssembler, NetworkMesh, Solver
from networks_fenicsx.network_generation import make_arterial_tree
from networks_fenicsx.post_processing import export_functions, extract_global_flux


class p_bc_expr:
    def eval(self, x):
        return x[1]


n = 5
G = make_arterial_tree(N=n, direction=np.array([0.1, 1, 0]))

network_mesh = NetworkMesh(G, N=40, color_strategy=nx.coloring.strategy_largest_first)
assembler = HydraulicNetworkAssembler(network_mesh, flux_degree=1, pressure_degree=0)
# Compute forms
assembler.compute_forms(p_bc_ex=p_bc_expr())
# Solve

solver = Solver(assembler, kind="nest")
solver.assemble()
sol = solver.solve()
global_flux = extract_global_flux(network_mesh, sol)

# Export results
outdir = Path("results_arterial_tree")
outdir.mkdir(exist_ok=True)

with dolfinx.io.VTXWriter(
    global_flux.function_space.mesh.comm,
    outdir / f"n{n}" / "global_flux.bp",
    [global_flux],
) as vtx:
    vtx.write(0.0)
export_functions(
    functions=sol,
    outpath=outdir / f"n{n}",
)
