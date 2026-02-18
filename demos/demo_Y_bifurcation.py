from pathlib import Path

import dolfinx.io
import ufl
from networks_fenicsx import (
    HydraulicNetworkAssembler,
    NetworkMesh,
    Solver,
    network_generation,
)
from networks_fenicsx.post_processing import export_functions, extract_global_flux

outdir = Path(__file__).parent / "results_Y_bifurcation"
outdir.mkdir(exist_ok=True, parents=True)

# Create Y bifurcation graph
G = network_generation.make_tree(2, 1, 3)

network_mesh = NetworkMesh(G, N=4)

x = ufl.SpatialCoordinate(network_mesh.mesh)
assembler = HydraulicNetworkAssembler(network_mesh)
assembler.compute_forms(p_bc_ex=x[1])


solver = Solver(assembler)
solver.assemble()
sol = solver.solve()

global_flux = extract_global_flux(network_mesh, sol)

# Export results
with dolfinx.io.VTXWriter(
    global_flux.function_space.mesh.comm,
    outdir / "global_flux.bp",
    [global_flux],
) as vtx:
    vtx.write(0.0)
export_functions(functions=sol, outpath=outdir)
