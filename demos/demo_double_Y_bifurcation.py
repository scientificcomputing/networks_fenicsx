import numpy as np

import dolfinx.io
from networks_fenicsx import HydraulicNetworkAssembler, NetworkMesh, Solver, network_generation
from networks_fenicsx.config import Config
from networks_fenicsx.post_processing import export_functions, extract_global_flux

cfg = Config()
cfg.outdir = "demo_double_Y_bifurcation"
cfg.export = True
cfg.clean = True

cfg.lcar = 0.2

# Create Y bifurcation graph
G = network_generation.make_tree(2, 3.1, 7.3)
network_mesh = NetworkMesh(G, cfg)


class p_bc_expr:
    def eval(self, x):
        return np.full(x.shape[1], x[0])


assembler = HydraulicNetworkAssembler(cfg, network_mesh)
assembler.compute_forms(p_bc_ex=p_bc_expr())

solver = Solver(assembler)
solver.assemble()
sol = solver.solve()


global_flux = extract_global_flux(network_mesh, sol)
export_functions(sol, outpath=cfg.outdir)
with dolfinx.io.VTXWriter(
    global_flux.function_space.mesh.comm,
    cfg.outdir / "global_flux.bp",
    [global_flux],
) as vtx:
    vtx.write(0.0)
