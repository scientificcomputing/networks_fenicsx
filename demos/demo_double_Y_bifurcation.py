import numpy as np

from networks_fenicsx import HydraulicNetworkAssembler, NetworkMesh
from networks_fenicsx.config import Config
from networks_fenicsx.mesh import network_generation
from networks_fenicsx.post_processing import export
from networks_fenicsx.solver import solver

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


assembler = HydraulicNetworkAssembler(cfg, G)
assembler.compute_forms(p_bc_ex=p_bc_expr())

solver = solver.Solver(cfg, G, assembler)
sol = solver.solve()
(fluxes, global_flux, pressure) = export(cfg, G, assembler.function_spaces, sol)
