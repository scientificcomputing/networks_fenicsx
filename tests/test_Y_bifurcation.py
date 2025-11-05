from networks_fenicsx import network_generation
from networks_fenicsx.config import Config

cfg = Config()
cfg.outdir = "demo_Y_bifurcation"
cfg.export = True
cfg.clean = True
cfg.lcar = 0.2

# Create Y bifurcation graph
G = network_generation.make_Y_bifurcation(cfg)


domain = G.mesh
tdim = domain.topology.dim
assert tdim == 1
num_cells_global = domain.topology.index_map(tdim).size_global
assert num_cells_global == 3 * 1 / cfg.lcar
num_vertices_global = domain.topology.index_map(0).size_global
assert num_vertices_global == 3 * 1 / cfg.lcar + 1

print("HELLO")
