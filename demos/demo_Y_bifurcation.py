from pathlib import Path

import dolfinx.io
from networks_fenicsx import Config, HydraulicNetworkAssembler, NetworkMesh, Solver
from networks_fenicsx.mesh import mesh_generation
from networks_fenicsx.post_processing import export_functions, extract_global_flux

cfg = Config()
cfg.outdir = "demo_Y_bifurcation"
cfg.export = True
cfg.clean = True
cfg.lcar = 0.25

# Create Y bifurcation graph
G = mesh_generation.make_tree(2, 1, 3)

network_mesh = NetworkMesh(G, cfg)


class p_bc_expr:
    def eval(self, x):
        return x[1]


assembler = HydraulicNetworkAssembler(cfg, network_mesh)
assembler.compute_forms(p_bc_ex=p_bc_expr())


solver = Solver(assembler)
solver.assemble()
solver.timing_dir = cfg.outdir
sol = solver.solve()

global_flux = extract_global_flux(network_mesh, sol)

# Export results
with dolfinx.io.VTXWriter(
    global_flux.function_space.mesh.comm,
    Path(cfg.outdir) / "global_flux.bp",
    [global_flux],
) as vtx:
    vtx.write(0.0)
export_functions(functions=sol, outpath=Path(cfg.outdir))
