from pathlib import Path

import dolfinx.io
import ufl
from networks_fenicsx import HydraulicNetworkAssembler, NetworkMesh, Solver, network_generation
from networks_fenicsx.post_processing import export_functions, extract_global_flux

# Create Y bifurcation graph

G = network_generation.make_tree(2, 3.1, 7.3)
network_mesh = NetworkMesh(G, N=5)


x = ufl.SpatialCoordinate(network_mesh.mesh)

assembler = HydraulicNetworkAssembler(network_mesh)
assembler.compute_forms(p_bc_ex=x[0])

solver = Solver(assembler)
solver.assemble()
sol = solver.solve()


outdir = Path("results_double_Y_bifurcation")
global_flux = extract_global_flux(network_mesh, sol)
export_functions(sol, outpath=outdir)
with dolfinx.io.VTXWriter(
    global_flux.function_space.mesh.comm,
    outdir / "global_flux.bp",
    [global_flux],
) as vtx:
    vtx.write(0.0)
