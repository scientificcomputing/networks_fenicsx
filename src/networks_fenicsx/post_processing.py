# Copyright (C) Simula Research Laboratory and Jørgen S. Dokken
# SPDX-License-Identifier:    MIT
"""
Convenience functions for post-processing.
"""

from pathlib import Path

from mpi4py import MPI

import numpy as np

from dolfinx import fem, io
from networks_fenicsx.mesh import NetworkMesh

__all__ = ["extract_global_flux", "export_functions"]


def extract_global_flux(graph_mesh: NetworkMesh, functions: list[fem.Function]) -> fem.Function:
    """Extract global flux function from submesh solutions

    Args:
        graph_mesh: The network mesh
        functions: The list of functions `[flux_1, flux_2, ..., flux_M, pressure, lm]`
    """
    flux_functions = functions[:-2]

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
    return global_q


def export_functions(
    functions: list[fem.Function],
    outpath: Path | str,
):
    """Export global flux function from submesh solutions.
    Args:
        functions: The list of functions `[flux_1, flux_2, ..., flux_M, pressure, lm]`
        outpath: The output directory to save the files to.

    """
    export_path = Path(outpath)
    flux_functions = functions[:-2]
    for i, q in enumerate(flux_functions):
        with io.VTXWriter(
            MPI.COMM_WORLD,
            export_path / f"flux_{i}.bp",
            q,
        ) as f:
            f.write(0.0)

    with io.VTXWriter(MPI.COMM_WORLD, export_path / "pressure.bp", functions[-2]) as f:
        f.write(0.0)
    with io.VTXWriter(MPI.COMM_WORLD, export_path / "lm.bp", functions[-1]) as f:
        f.write(0.0)
