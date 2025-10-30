import networkx as nx
import numpy as np
from typing import List
import copy
import basix.ufl
from mpi4py import MPI
from dolfinx import fem, io as _io, mesh, graph
import ufl

from networks_fenicsx.utils.timers import timeit
from networks_fenicsx import config

"""
This file is based on the graphnics project (https://arxiv.org/abs/2212.02916), https://github.com/IngeborgGjerde/fenics-networks - forked on August 2022
Copyright (C) 2022-2023 by Ingeborg Gjerde

You can freely redistribute it and/or modify it under the terms of the GNU General Public License, version 3.0, provided that the above copyright notice is kept intact and that the source code is made available under an open-source license.

Modified by Cécile Daversin-Catty - 2023
Modified by Joseph P. Dean - 2023
Modified by Jørgen S. Dokken - 2025
"""


class NetworkGraph(nx.DiGraph):
    """
    Make FEniCSx mesh from networkx directed graph
    """

    _msh: mesh.Mesh | None
    _subdomains: mesh.MeshTags | None
    _facet_markers: mesh.MeshTags | None

    def __init__(self, config: config.Config, graph: nx.DiGraph = None):
        nx.DiGraph.__init__(self, graph)

        self.comm = MPI.COMM_WORLD
        self.cfg = config
        self.cfg.clean_dir()

        self.bifurcation_ixs: List[int] = []  # noqa: F821
        self.boundary_ixs: List[int] = []  # noqa: F821

        self._msh = None
        self.lm_smsh = None
        self._subdomains = None
        self._facet_markers = None
        self.global_tangent = None

        self.BIF_IN = 1
        self.BIF_OUT = 2
        self.BOUN_IN = 3
        self.BOUN_OUT = 4

    @timeit
    def build_mesh(self):
        """Convert the networkx graph into a {py:class}`dolfinx.mesh.Mesh`.

        The element size is controlled by `self.cfg.lcar`.
        Each segment in the networkx graph gets a unique subdomain marker.
        Each bifurcation and boundary node is marked on the facets with a unique integer.

        Note:
            This function attaches data to `self.mesh`, `self.subdomains` and
            `self.boundaries`.
        """

        self.geom_dim = len(self.nodes[1]["pos"])
        self.num_edges = len(self.edges)

        vertex_coords = np.asarray([self.nodes[v]["pos"] for v in self.nodes()])
        cells_array = np.asarray([[u, v] for u, v in self.edges()])

        line_weights = np.linspace(
            0, 1, int(np.ceil(1 / self.cfg.lcar)), endpoint=False
        )[1:][:, None]

        # Create mesh segments
        if MPI.COMM_WORLD.rank == 0:
            mesh_nodes = vertex_coords.copy()
            cells = []
            cell_markers = []
            for i, segment in enumerate(cells_array):
                if len(line_weights) == 0:
                    cells.append([segment[0], segment[1]])
                    cell_markers.append(i)
                else:
                    start_coord_pos = mesh_nodes.shape[0]
                    start = vertex_coords[segment[0]]
                    end = vertex_coords[segment[1]]

                    internal_line_coords = (
                        start * (1 - line_weights) + end * line_weights
                    )
                    mesh_nodes = np.vstack((mesh_nodes, internal_line_coords))
                    cells.append([segment[0], start_coord_pos])
                    segment_connectivity = (
                        np.repeat(np.arange(internal_line_coords.shape[0]), 2)[
                            1:-1
                        ].reshape(internal_line_coords.shape[0] - 1, 2)
                        + start_coord_pos
                    )
                    cells.append(segment_connectivity)
                    cells.append(
                        [
                            start_coord_pos + internal_line_coords.shape[0] - 1,
                            segment[1],
                        ]
                    )
                    cell_markers.extend(
                        np.full(internal_line_coords.shape[0] + 1, i, dtype=np.int32)
                    )
            cells = np.vstack(cells).astype(np.int64)
            cell_markers = np.array(cell_markers, dtype=np.int32)
        else:
            cells = np.empty((0, 2), dtype=np.int64)
            mesh_nodes = np.empty((0, self.geom_dim), dtype=np.float64)
            cell_markers = np.empty((0,), dtype=np.int32)
        bifurcations, num_connections = np.unique(
            cells_array.flatten(), return_counts=True
        )
        partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
        graph_mesh = mesh.create_mesh(
            MPI.COMM_WORLD,
            x=mesh_nodes,
            cells=cells,
            e=ufl.Mesh(
                basix.ufl.element("Lagrange", "interval", 1, shape=(self.geom_dim,))
            ),
            partitioner=partitioner,
            max_facet_to_cell_links=np.max(num_connections),
        )
        self._msh = graph_mesh

        local_entities, local_values = _io.distribute_entity_data(
            self.mesh,
            self.mesh.topology.dim,
            cells,
            cell_markers,
        )
        self._subdomains = mesh.meshtags_from_entities(
            self.mesh,
            self.mesh.topology.dim,
            graph.adjacencylist(local_entities),
            local_values,
        )

        if MPI.COMM_WORLD.rank == 0:
            lb = bifurcations.astype(np.int64).reshape((-1, 1))
            lbv = np.arange(len(bifurcations), dtype=np.int32)
        else:
            lb = np.empty((0, 1), dtype=np.int64)
            lbv = np.empty((0,), dtype=np.int32)
        self.mesh.topology.create_connectivity(0, 1)
        local_bifurcations, local_bifurcation_values = _io.distribute_entity_data(
            self.mesh, 0, lb, lbv
        )
        self._facet_markers = mesh.meshtags_from_entities(
            self.mesh,
            0,
            graph.adjacencylist(local_bifurcations),
            local_bifurcation_values,
        )
        self.subdomains.name = "subdomains"
        self.boundaries.name = "bifurcations"
        if self.cfg.export:
            with _io.XDMFFile(
                self.comm, self.cfg.outdir + "/mesh/mesh.xdmf", "w"
            ) as file:
                file.write_mesh(self.mesh)
                file.write_meshtags(self.subdomains, self.mesh.geometry)
                file.write_meshtags(self.boundaries, self.mesh.geometry)

        # Submesh for the Lagrange multiplier
        # self.lm_smsh = mesh.create_submesh(self.mesh, self.mesh.topology.dim, [0])[0]

    @timeit
    def build_network_submeshes(self):
        for i, (u, v) in enumerate(self.edges):
            edge_subdomain = self.subdomains.find(i)

            self.edges[u, v]["submesh"], self.edges[u, v]["entity_map"] = (
                mesh.create_submesh(self.mesh, self.mesh.topology.dim, edge_subdomain)[
                    0:2
                ]
            )
            self.edges[u, v]["tag"] = i

            self.edges[u, v]["entities"] = []
            self.edges[u, v]["b_values"] = []

    @timeit
    def compute_tangent(self):
        """Compute tangent vector for all cells.

        Tangent is oriented according to positive y-axis.
        If perpendicular to y-axis, align with x-axis.

        Note:
            Assuming that the mesh is affine.
        """
        cell_map = self.mesh.topology.index_map(self.mesh.topology.dim)
        geom_indices = mesh.entities_to_geometry(
            self.mesh,
            self.mesh.topology.dim,
            np.arange(cell_map.size_local + cell_map.num_ghosts, dtype=np.int32),
        )
        geom_coordinates = self.mesh.geometry.x[geom_indices]
        tangent = geom_coordinates[:, 0, :] - geom_coordinates[:, 1, :]
        global_orientation = np.sign(np.dot(tangent, [0, 1, 0]))
        is_x_aligned = np.isclose(global_orientation, 0)
        global_orientation[is_x_aligned] = np.sign(
            np.dot(tangent[is_x_aligned], [1, 0, 0])
        )
        tangent *= global_orientation[:, None]
        assert np.all(np.linalg.norm(tangent, axis=1) > 0), (
            "Zero-length tangent vector detected"
        )
        gdim = self.mesh.geometry.dim
        DG0 = fem.functionspace(self.mesh, ("DG", 0, (gdim,)))
        self.global_tangent = fem.Function(DG0)
        self.global_tangent.x.array[:] = tangent.flatten()

    @property
    def mesh(self):
        if self._msh is None:
            raise RuntimeError("Mesh has not been built yet. Call build_mesh() first.")
        return self._msh

    @property
    def subdomains(self):
        if self._subdomains is None:
            raise RuntimeError("Mesh has no subdomains")
        return self._subdomains

    @property
    def boundaries(self):
        if self._facet_markers is None:
            raise RuntimeError("Mesh has no boundaries/facet markers")
        return self._facet_markers

    def submeshes(self):
        return list(nx.get_edge_attributes(self, "submesh").values())

    def tangent(self):
        if self.cfg.export:
            self.global_tangent.x.scatter_forward()
            with _io.XDMFFile(
                self.comm, self.cfg.outdir + "/mesh/tangent.xdmf", "w"
            ) as file:
                file.write_mesh(self.mesh)
                file.write_function(self.global_tangent)

        return self.global_tangent
