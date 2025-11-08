import shutil

from mpi4py import MPI

from dolfinx.io import VTXWriter
from networks_fenicsx import (
    Config,
    HydraulicNetworkAssembler,
    NetworkMesh,
    Solver,
    network_generation,
)
from networks_fenicsx.post_processing import export_functions, export_submeshes, extract_global_flux

cfg = Config()
cfg.export = True

cfg.flux_degree = 1
cfg.pressure_degree = 0
cfg.graph_coloring = True
cfg.color_strategy = "smallest_last"
cfg.outdir = "demo_perf"


class p_bc_expr:
    def eval(self, x):
        return x[1]


# One element per segment
cfg.lcar = 2
# Cleaning directory only once
# cfg.clean_dir()
cfg.clean = False

cfg.outdir.mkdir(exist_ok=True, parents=True)
cache_dir = cfg.outdir / ".cache"
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
    network_mesh = NetworkMesh(G, cfg)
    del G
    network_mesh.export_orientation()
    export_submeshes(network_mesh, cfg.outdir / f"n{n}")
    assembler = HydraulicNetworkAssembler(cfg, network_mesh)
    # Compute forms
    assembler.compute_forms(p_bc_ex=p_bc_expr(), jit_options=jit_options)

    # Solve
    solver = Solver(assembler)
    solver.assemble()
    sol = solver.solve()

    export_functions(sol, outpath=cfg.outdir / f"n{n}")
    global_flux = extract_global_flux(network_mesh, sol)
    with VTXWriter(
        global_flux.function_space.mesh.comm, cfg.outdir / f"n{n}" / "global_flux.bp", [global_flux]
    ) as vtx:
        vtx.write(0.0)
    del assembler
    del solver
    del network_mesh

# Rerun with cache
for n in ns:
    if MPI.COMM_WORLD.rank == 0:
        with (cfg.outdir / "profiling.txt").open("a") as f:
            f.write("n: " + str(n) + "\n")

    # Create tree
    G = network_generation.make_tree(n=n, H=n, W=n)
    network_mesh = NetworkMesh(G, cfg)

    assembler = HydraulicNetworkAssembler(cfg, network_mesh)
    # Compute forms
    assembler.compute_forms(p_bc_ex=p_bc_expr(), jit_options=jit_options)

    # Solve
    solver = Solver(assembler)
    solver.assemble()
    sol = solver.solve()
    del assembler
    del solver
    del network_mesh
