from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io

import matplotlib.pyplot as plt
import numpy as np

from networks_fenicsx.mesh import mesh
from networks_fenicsx import config


def export(
    config: config.Config,
    graph: mesh.NetworkMesh,
    function_spaces: list,
    sol: PETSc.Vec,
    export_dir="results",
):
    breakpoint()

    q_space = function_spaces[0]
    p_space = function_spaces[-2]
    q_degree = q_space.element.basix_element.degree
    global_q_space = fem.functionspace(graph.mesh, ("DG", q_degree))
    global_q = fem.Function(global_q_space)

    # Recover solution
    fluxes = []
    start = 0
    # Flux spaces are the first M ones
    for i, (submesh, entity_map) in enumerate(zip(graph.submeshes, graph.entity_maps)):
        q_space = function_spaces[i]
        q = fem.Function(q_space)
        offset = q_space.dofmap.index_map.size_local * q_space.dofmap.index_map_bs
        q.x.array[:offset] = sol.array_r[start : start + offset]
        q.x.scatter_forward()
        start += offset
        fluxes.append(q)

        # Interpolated to DG space
        num_cells_local = (
            submesh.topology.index_map(submesh.topology.dim).size_local
            + submesh.topology.index_map(submesh.topology.dim).num_ghosts
        )
        submesh_to_parent = entity_map.sub_topology_to_topology(
            np.arange(num_cells_local, dtype=np.int32), inverse=False
        )
        global_q.interpolate(
            q,
            cells0=np.arange(num_cells_local, dtype=np.int32),
            cells1=submesh_to_parent,
        )
    # Then we have the pressure space
    offset = p_space.dofmap.index_map.size_local * p_space.dofmap.index_map_bs
    pressure = fem.Function(p_space)
    pressure.x.array[:offset] = sol.array_r[start : start + offset]
    pressure.x.scatter_forward()

    # Last we have LM
    # NOTE: ADD processing here
    # Write to file
    for i, q in enumerate(fluxes):
        with io.VTXWriter(
            MPI.COMM_WORLD,
            config.outdir + "/" + export_dir + "/flux_" + str(i) + ".bp",
            q,
        ) as f:
            f.write(0.0)
    with io.VTXWriter(MPI.COMM_WORLD, config.outdir + "/" + export_dir + "/flux.bp", global_q) as f:
        f.write(0.0)
    with io.VTXWriter(
        MPI.COMM_WORLD, config.outdir + "/" + export_dir + "/pressure.bp", pressure
    ) as f:
        f.write(0.0)

    return (fluxes, global_q, pressure)


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
