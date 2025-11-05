from pathlib import Path

import networkx as ntx
import numpy as np

import dolfinx.io
from networks_fenicsx import Config, HydraulicNetworkAssembler, NetworkMesh, Solver
from networks_fenicsx.network_generation import make_arterial_tree
from networks_fenicsx.post_processing import export_functions, extract_global_flux

cfg = Config()
cfg.export = True
cfg.graph_coloring = True
cfg.color_strategy = ntx.coloring.strategy_largest_first
cfg.outdir = "demo_arterial_tree"
cfg.flux_degree = 1
cfg.pressure_degree = 0


class p_bc_expr:
    def eval(self, x):
        return x[1]


# One element per segment
cfg.lcar = 0.025

# Cleaning directory only once
cfg.clean_dir()
cfg.clean = False

p = Path(cfg.outdir)
p.mkdir(exist_ok=True)

n = 5

G = make_arterial_tree(N=n, direction=np.array([0.1, 1, 0]))

network_mesh = NetworkMesh(G, cfg)
assembler = HydraulicNetworkAssembler(cfg, network_mesh)
# Compute forms
assembler.compute_forms(p_bc_ex=p_bc_expr())
# Solve

solver = Solver(assembler, kind="nest")
solver.timing_dir = cfg.outdir
solver.assemble()
sol = solver.solve()
global_flux = extract_global_flux(network_mesh, sol)

# Export results
with dolfinx.io.VTXWriter(
    global_flux.function_space.mesh.comm,
    Path(cfg.outdir) / f"n{n}" / "global_flux.bp",
    [global_flux],
) as vtx:
    vtx.write(0.0)
export_functions(
    functions=sol,
    outpath=Path(cfg.outdir) / f"n{n}",
)
