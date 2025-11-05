
import dolfinx.io
from pathlib import Path
from networks_fenicsx.mesh import mesh_generation
from networks_fenicsx import Solver, HydraulicNetworkAssembler, NetworkMesh, Config
from networks_fenicsx.post_processing import extract_global_flux, export_functions

cfg = Config()
cfg.outdir = "demo_Y_bifurcation"
cfg.export = True
cfg.clean = True
cfg.lcar = 0.25

# Create Y bifurcation graph
G = mesh_generation.make_Y_bifurcation(cfg)

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
export_functions(
    functions=sol,
    outpath=Path(cfg.outdir)
)
