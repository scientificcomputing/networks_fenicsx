from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io
from pathlib import Path
import dolfinx.fem.petsc as _petsc
import matplotlib.pyplot as plt
import numpy as np

from networks_fenicsx.mesh import mesh
from networks_fenicsx import config


def export(
    graph_mesh: mesh.NetworkMesh,
    function_spaces: list[fem.FunctionSpace],
    sol: PETSc.Vec,
    outpath: Path | str | None = None,
) -> tuple[list[fem.Function], fem.Function, fem.Function]:
    """Export solution to files.
    
    Args:
        graph_mesh: The network mesh
        function_spaces: The function spaces corresponding to the solution
        sol: The PETSc solution vectior
        outpath: The output directory to save the files to.
            If not supplied, no files are saved.
    Returns:
        The flux functions per submesh, the flux function represented
        on the network mesh and the pressure function.    
    """
    outputs = [fem.Function(fs) for fs in function_spaces]
    _petsc.assign(sol, outputs)
    flux_functions = outputs[:-2]
    
    # Extract info regarding global flux space
    q_space = flux_functions[0].function_space
    q_degree = q_space.element.basix_element.degree
    global_q_space = fem.functionspace(graph_mesh.mesh, ("DG", q_degree))
    global_q = fem.Function(global_q_space, name="Global_Flux")

    # Recover solution
    # Flux spaces are the first M ones
    for i, (flux, entity_map) in enumerate(zip(flux_functions, graph_mesh.entity_maps)):
        # Interpolated to DG space
        flux.name = f"Flux_{i}"
        submesh = flux.function_space.mesh
        num_cells_local = (
        submesh.topology.index_map(submesh.topology.dim).size_local
            + submesh.topology.index_map(submesh.topology.dim).num_ghosts
        )
        submesh_to_parent = entity_map.sub_topology_to_topology(
            np.arange(num_cells_local, dtype=np.int32), inverse=False
        )
        global_q.interpolate(
            flux,
            cells0=np.arange(num_cells_local, dtype=np.int32),
            cells1=submesh_to_parent,
        )
    outputs[-2].name = "Pressure"

    if outpath is not None:
        export_path = Path(outpath)
        for i, q in enumerate(flux_functions):
            with io.VTXWriter(
                MPI.COMM_WORLD,
                export_path / f"flux_{i}.bp",
                q,
            ) as f:
                f.write(0.0)
        with io.VTXWriter(MPI.COMM_WORLD, export_path / "flux.bp", global_q) as f:
            f.write(0.0)
        with io.VTXWriter(
            MPI.COMM_WORLD, export_path / "pressure.bp", outputs[-2]
        ) as f:
            f.write(0.0)

    return (flux_functions, global_q, outputs[-2])


def perf_plot(timing_dict):
    # set width of bar
    barWidth = 0.1
    # fig = plt.subplots(figsize =(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(timing_dict["n"]))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(
        br1,
        timing_dict["compute_forms"],
        color="r",
        width=barWidth,
        edgecolor="grey",
        label="forms",
    )
    plt.bar(
        br2,
        timing_dict["assemble"],
        color="g",
        width=barWidth,
        edgecolor="grey",
        label="assembly",
    )
    plt.bar(
        br3,
        timing_dict["solve"],
        color="b",
        width=barWidth,
        edgecolor="grey",
        label="solve",
    )

    # Adding Xticks
    plt.xlabel("Number of generations", fontweight="bold", fontsize=15)
    plt.ylabel("Time [s]", fontweight="bold", fontsize=15)
    plt.xticks([r + barWidth for r in range(len(timing_dict["n"]))], timing_dict["n"])

    plt.legend()
    plt.show()
