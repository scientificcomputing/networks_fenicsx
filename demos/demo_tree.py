import numpy as np
import matplotlib.pyplot as plt

from networks_fenicsx import NetworkMesh
from networks_fenicsx.mesh import mesh_generation
from networks_fenicsx.solver import assembly, solver
from networks_fenicsx.config import Config


cfg = Config()
cfg.outdir = "demo_tree"
cfg.export = True
cfg.clean = True
cfg.lm_spaces = True

if cfg.lm_spaces:
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
for i in range(6):
    lcar /= 2.0
    cfg.lcar = lcar
    lcars.append(lcar)

    network_mesh = NetworkMesh(G, cfg)
    assembler = assembly.Assembler(cfg, network_mesh)
    assembler.compute_forms(p_bc_ex=p_bc_expr())
    assembler.assemble()

    solver_ = solver.Solver(cfg, network_mesh, assembler)
    sol = solver_.solve()

    # print("global flux min = ", min(global_flux.x.array))
    # print("global flux max = ", max(global_q.flux.array))
    # print("global flux mean = ", np.mean(global_q.flux.array))

    # min_q.append(min(global_flux.x.array))
    # max_q.append(max(global_flux.x.array))
    # mean_q.append(np.mean(global_flux.x.array))

    # print("pressure min = ", min(pressure.x.array))
    # print("pressure max = ", max(pressure.x.array))

# fig, ax = plt.subplots()
# ax.plot(lcars, mean_q)
# ax.plot(lcars, max_q)
# ax.plot(lcars, min_q)
# plt.show()
