# # Performance testing for network assembly
#
# This script check the scalability of the network assembly, given an increasing number
# of bifurcations.
# We start by importing the necessary modules

# +
import datetime
import shutil
from pathlib import Path

from mpi4py import MPI

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dolfinx.common import timing
from dolfinx.io import VTXWriter
from networks_fenicsx import (
    HydraulicNetworkAssembler,
    NetworkMesh,
    Solver,
    network_generation,
)
from networks_fenicsx.post_processing import export_functions, extract_global_flux

# -

# Next, we define the boundary condition that we will prescribe for the pressure.
# Here, we let $p=y$.


def p_bc(x):
    return x[1]


# Next, as we will evaluate the performance of the assemble, we will check it with and
# without a cache of the compiled C-kernels.

cache_dir = Path(".benchmarking_cache").absolute()
cache_dir.mkdir(exist_ok=True, parents=True)

# We also provide some compilation arguments for the C-kernels.

cffi_options = ["-Ofast", "-march=native"]
jit_options = {"cache_dir": cache_dir, "cffi_extra_compile_args": cffi_options}

# Next, we will loop over a tree that has `n` generations, where each generation doubles the number
# of branches.

ns = [3, 6, 12, 16]
tracked_calls = [
    "nxfx:HydraulicNetworkAssembler:__init__",
    "nxfx:HydraulicNetworkAssembler:compute_forms",
    "nxfx:HydraulicNetworkAssembler:assemble",
    "nxfx:NetworkMesh:build_mesh",
    "nxfx:NetworkMesh:build_network_submeshes",
    "nxfx:NetworkMesh:create_lm_submesh",
    "nxfx:Solver:solve",
]
timings: dict[str, dict[int, float]] = {
    "BuildMesh": {},
    "BuildSubMeshes": {},
    "ComputeIntegrationData": {},
    "CreateLMSubmesh": {},
    "Compile": {},
    "CompileCached": {},
    "Assemble": {},
    "Solve": {},
}
previous_timing = {call: datetime.timedelta(0) for call in tracked_calls}
for n in ns:
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    # Create tree on root node
    if MPI.COMM_WORLD.rank == 0:
        G = network_generation.make_tree(n=n, H=n, W=n)
    else:
        G = None
    network_mesh = NetworkMesh(G, N=1, color_strategy="smallest_last")
    del G

    _, build_mesh_timing = timing("nxfx:NetworkMesh:build_mesh")
    timings["BuildMesh"][n] = (
        build_mesh_timing.total_seconds()
        - previous_timing["nxfx:NetworkMesh:build_mesh"].total_seconds()
    )
    previous_timing["nxfx:NetworkMesh:build_mesh"] = build_mesh_timing

    _, build_submeshes_timing = timing("nxfx:NetworkMesh:build_network_submeshes")
    timings["BuildSubMeshes"][n] = (
        build_submeshes_timing.total_seconds()
        - previous_timing["nxfx:NetworkMesh:build_network_submeshes"].total_seconds()
    )
    previous_timing["nxfx:NetworkMesh:build_network_submeshes"] = build_submeshes_timing

    _, create_lm_submesh_timing = timing("nxfx:NetworkMesh:create_lm_submesh")
    timings["CreateLMSubmesh"][n] = (
        create_lm_submesh_timing.total_seconds()
        - previous_timing["nxfx:NetworkMesh:create_lm_submesh"].total_seconds()
    )
    previous_timing["nxfx:NetworkMesh:create_lm_submesh"] = create_lm_submesh_timing

    # Setup assembler
    assembler = HydraulicNetworkAssembler(network_mesh, flux_degree=1, pressure_degree=0)

    _, compute_integration_data_timing = timing("nxfx:HydraulicNetworkAssembler:__init__")
    timings["ComputeIntegrationData"][n] = (
        compute_integration_data_timing.total_seconds()
        - previous_timing["nxfx:HydraulicNetworkAssembler:__init__"].total_seconds()
    )
    previous_timing["nxfx:HydraulicNetworkAssembler:__init__"] = compute_integration_data_timing

    # Compute forms (without cache)
    assembler.compute_forms(p_bc_ex=p_bc, jit_options=jit_options)

    _, compile_forms_timing = timing("nxfx:HydraulicNetworkAssembler:compute_forms")
    timings["Compile"][n] = (
        compile_forms_timing.total_seconds()
        - previous_timing["nxfx:HydraulicNetworkAssembler:compute_forms"].total_seconds()
    )
    previous_timing["nxfx:HydraulicNetworkAssembler:compute_forms"] = compile_forms_timing

    # Compute forms (with cache)
    assembler.compute_forms(p_bc_ex=p_bc, jit_options=jit_options)
    _, compile_forms_timing = timing("nxfx:HydraulicNetworkAssembler:compute_forms")
    timings["CompileCached"][n] = (
        compile_forms_timing.total_seconds()
        - previous_timing["nxfx:HydraulicNetworkAssembler:compute_forms"].total_seconds()
    )

    # Assemble system
    solver = Solver(assembler)
    solver.assemble()
    _, assemble_timing = timing("nxfx:HydraulicNetworkAssembler:assemble")
    timings["Assemble"][n] = (
        assemble_timing.total_seconds()
        - previous_timing["nxfx:HydraulicNetworkAssembler:assemble"].total_seconds()
    )
    previous_timing["nxfx:HydraulicNetworkAssembler:assemble"] = assemble_timing

    if n < 20:
        sol = solver.solve()
        _, solve_timing = timing("nxfx:Solver:solve")
        timings["Solve"][n] = (
            solve_timing.total_seconds() - previous_timing["nxfx:Solver:solve"].total_seconds()
        )
        previous_timing["nxfx:Solver:solve"] = solve_timing

        # Export results
        outdir = Path("demo_perf_output").absolute()
        outdir.mkdir(exist_ok=True, parents=True)
        export_functions(sol, outpath=outdir / f"n{n}")
        global_flux = extract_global_flux(network_mesh, sol)
        with VTXWriter(
            global_flux.function_space.mesh.comm, outdir / f"n{n}" / "global_flux.bp", [global_flux]
        ) as vtx:
            vtx.write(0.0)
    del assembler
    del solver
    del network_mesh

# Finally, we plot the results.

flattened_data = []
for operation in timings.keys():
    for n in ns:
        flattened_data.append(
            [operation, sum(2**i for i in range(n)), timings[operation].get(n, None)]
        )
dataframe = pd.DataFrame(flattened_data, columns=["Operation", "NumSegments", "Time"])

fig, ax = plt.subplots()
plot = sns.lineplot(data=dataframe, x="NumSegments", y="Time", hue="Operation", ax=ax)
ax.set(xscale="log", yscale="log")
ax.grid(True)
fig.savefig("demo_perf.png", bbox_inches="tight")
plt.show()
