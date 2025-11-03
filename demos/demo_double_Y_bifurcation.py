import os
import numpy as np

from networks_fenicsx.mesh import mesh_generation
from networks_fenicsx.solver import assembly, solver
from networks_fenicsx.config import Config
from networks_fenicsx.utils.post_processing import export

# Clear fenics cache
print('Clearing cache')
os.system('dijitso clean')

cfg = Config()
cfg.outdir = "demo_double_Y_bifurcation"
cfg.export = True
cfg.clean = True

cfg.lcar = 0.2

# Create Y bifurcation graph
G = mesh_generation.make_double_Y_bifurcation(cfg)


class p_bc_expr:
    def eval(self, x):
        return np.full(x.shape[1], x[0])


assembler = assembly.Assembler(cfg, G)
assembler.compute_forms(p_bc_ex=p_bc_expr())

solver = solver.Solver(cfg, G, assembler)
sol = solver.solve()
(fluxes, global_flux, pressure) = export(cfg, G, assembler.function_spaces, sol)
