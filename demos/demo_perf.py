import shutil
from pathlib import Path

from mpi4py import MPI

from dolfinx.io import VTXWriter
from networks_fenicsx import (
    HydraulicNetworkAssembler,
    NetworkMesh,
    Solver,
    network_generation,
)
from networks_fenicsx.post_processing import export_functions, export_submeshes, extract_global_flux


class p_bc_expr:
    def eval(self, x):
        return x[1]


outdir = Path("results_perf")
outdir.mkdir(exist_ok=True, parents=True)
cache_dir = outdir / ".cache"
if cache_dir.exists():
    shutil.rmtree(cache_dir, ignore_errors=True)

cffi_options = ["-Ofast", "-march=native"]
jit_options = {"cache_dir": cache_dir, "cffi_extra_compile_args": cffi_options}
ns = [3, 6, 12]
for n in ns:
    # Create tree
    if MPI.COMM_WORLD.rank == 0:
        G = network_generation.make_tree(n=n, H=n, W=n)
    else:
        G = None
    network_mesh = NetworkMesh(G, N=1, color_strategy="smallest_last")
    del G
    # network_mesh.export()
    export_submeshes(network_mesh, outdir / f"n{n}")
    assembler = HydraulicNetworkAssembler(network_mesh, flux_degree=1, pressure_degree=0)
    # Compute forms
    assembler.compute_forms(p_bc_ex=p_bc_expr(), jit_options=jit_options)

    # Solve
    solver = Solver(assembler)
    solver.assemble()
    sol = solver.solve()

    export_functions(sol, outpath=outdir / f"n{n}")
    global_flux = extract_global_flux(network_mesh, sol)
    with VTXWriter(
        global_flux.function_space.mesh.comm, outdir / f"n{n}" / "global_flux.bp", [global_flux]
    ) as vtx:
        vtx.write(0.0)
    del assembler
    del solver
    del network_mesh

# Rerun with cache
for n in ns:
    if MPI.COMM_WORLD.rank == 0:
        with (outdir / "profiling.txt").open("a") as f:
            f.write("n: " + str(n) + "\n")

    # Create tree
    G = network_generation.make_tree(n=n, H=n, W=n)
    network_mesh = NetworkMesh(G, N=1, color_strategy="smallest_last")

    assembler = HydraulicNetworkAssembler(network_mesh, flux_degree=1, pressure_degree=0)

    # Compute forms
    assembler.compute_forms(p_bc_ex=p_bc_expr(), jit_options=jit_options)

    # Solve
    solver = Solver(assembler)
    solver.assemble()
    sol = solver.solve()
    del assembler
    del solver
    del network_mesh
