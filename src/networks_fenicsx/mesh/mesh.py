"""
Interface for converting a networkx graph into a {py:class}`dolfinx.mesh.Mesh`.

This idea stems from the Graphnics project (https://arxiv.org/abs/2212.02916), https://github.com/IngeborgGjerde/fenics-networks
by Ingeborg Gjerde.

Modified by Cécile Daversin-Catty - 2023
Modified by Joseph P. Dean - 2023
Modified by Jørgen S. Dokken - 2025
"""

import networkx as nx
import numpy as np
import basix.ufl
import numpy.typing as npt
from mpi4py import MPI
from dolfinx import fem, io as _io, mesh, graph as _graph
import scifem
import ufl

from networks_fenicsx.utils.timers import timeit
from networks_fenicsx import config

__all__ = ["NetworkMesh", "compute_tangent"]


class NetworkMesh:
    """A representation of a Networkx graph in DOLFINx.

    Stores the resulting mesh, subdomains, and facet markers for bifurcations and boundary nodes.
    Has a globally oriented tangent vector field.
    Has a submesh for each edge in the Networkx graph.
    """

    # Configuration
    _cfg: config.Config

    # Graph properties
    _geom_dim: int
    _num_segments: int
    _bifurcation_in_edges: dict[int, list[int]]
    _bifurcation_out_edges: dict[int, list[int]]

    # Mesh properties
    _msh: mesh.Mesh | None
    _subdomains: mesh.MeshTags | None
    _facet_markers: mesh.MeshTags | None
    _submesh_facet_markers: list[mesh.MeshTags]
    _edge_meshes: list[mesh.Mesh]
    _edge_entity_maps: list[mesh.EntityMap]
    _tangent: fem.Function
    _bifurcation_values: npt.NDArray[np.int32]
    _boundary_values: npt.NDArray[np.int32]
    _lm_mesh: mesh.Mesh | None
    _lm_map: mesh.EntityMap | None

    def __init__(self, graph: nx.DiGraph, config: config.Config):
        self._cfg = config
        self._cfg.clean_dir()
        self._build_mesh(graph)
        self._build_network_submeshes()
        self._tangent = compute_tangent(self.mesh)
        self._create_lm_submesh()

    @property
    def lm_mesh(self) -> mesh.Mesh:
        if self._lm_mesh is None:
            raise RuntimeError("Lagrange multiplier submesh has not been created.")
        return self._lm_mesh

    @property
    def lm_map(self) -> mesh.EntityMap:
        if self._lm_map is None:
            raise RuntimeError("Lagrange multiplier entity map has not been created.")
        return self._lm_map

    @property
    def cfg(self) -> config.Config:
        return self._cfg

    @timeit
    def _create_lm_submesh(self):
        """Create a submesh for the Lagrange multipliers at the bifurcations."""
        assert self._msh is not None
        assert self._facet_markers is not None
        bifurcation_indices = self._facet_markers.indices[
            np.isin(self._facet_markers.values, self._bifurcation_values)
        ]
        self._lm_mesh, self._lm_map = mesh.create_submesh(
            self.mesh,
            self.mesh.topology.dim - 1,
            bifurcation_indices,
        )[:2]

    @timeit
    def _build_mesh(self, graph: nx.DiGraph):
        """Convert the networkx graph into a {py:class}`dolfinx.mesh.Mesh`.

        The element size is controlled by `self.cfg.lcar`.
        Each segment in the networkx graph gets a unique subdomain marker.
        Each bifurcation and boundary node is marked on the facets with a unique integer.

        Note:
            This function attaches data to `self.mesh`, `self.subdomains` and
            `self.boundaries`.
        """

        # Extract all the data required form the graph
        self._geom_dim = len(graph.nodes[1]["pos"])

        vertex_coords = np.asarray([graph.nodes[v]["pos"] for v in graph.nodes()])
        cells_array = np.asarray([[u, v] for u, v in graph.edges()])

        line_weights = np.linspace(0, 1, int(np.ceil(1 / self.cfg.lcar)), endpoint=False)[1:][
            :, None
        ]

        self._num_segments = cells_array.shape[0]

        graph_nodes, num_connections = np.unique(cells_array.flatten(), return_counts=True)
        self._bifurcation_values = np.flatnonzero(num_connections > 1)
        self._boundary_values = np.flatnonzero(num_connections == 1)

        # Create lookup of in and out edges for each bifurcation
        self._bifurcation_in_edges = {}
        self._bifurcation_out_edges = {}
        self._boundary_in_nodes = []
        self._boundary_out_nodes = []
        for bifurcation in self._bifurcation_values:
            in_edges = graph.in_edges(bifurcation)
            self._bifurcation_in_edges[int(bifurcation)] = []
            for edge in in_edges:
                self._bifurcation_in_edges[int(bifurcation)].append(
                    np.flatnonzero(np.all(cells_array == edge, axis=1))[0]
                )
            out_edges = graph.out_edges(bifurcation)
            self._bifurcation_out_edges[int(bifurcation)] = []
            for edge in out_edges:
                self._bifurcation_out_edges[int(bifurcation)].append(
                    np.flatnonzero(np.all(cells_array == edge, axis=1))[0]
                )

        # Map boundary_values to inlet and outlet data from graph
        for boundary in self._boundary_values:
            in_edges = graph.in_edges(boundary)
            out_edges = graph.out_edges(boundary)
            assert len(in_edges) + len(out_edges) == 1, "Boundary node with multiple edges"
            if len(in_edges) == 1:
                self._boundary_in_nodes.append(int(boundary))
            else:
                self._boundary_out_nodes.append(int(boundary))
        self._boundary_in_nodes = tuple(self._boundary_in_nodes)
        self._boundary_out_nodes = tuple(self._boundary_out_nodes)

        # Create mesh segments
        # TODO: Extract graph coloring coloring information to reduce the number of unique cell markers,
        # which results in a reduction in the number of submeshes.
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

                    internal_line_coords = start * (1 - line_weights) + end * line_weights
                    mesh_nodes = np.vstack((mesh_nodes, internal_line_coords))
                    cells.append([segment[0], start_coord_pos])
                    segment_connectivity = (
                        np.repeat(np.arange(internal_line_coords.shape[0]), 2)[1:-1].reshape(
                            internal_line_coords.shape[0] - 1, 2
                        )
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
            mesh_nodes = np.empty((0, self._geom_dim), dtype=np.float64)
            cell_markers = np.empty((0,), dtype=np.int32)

        partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
        graph_mesh = mesh.create_mesh(
            MPI.COMM_WORLD,
            x=mesh_nodes,
            cells=cells,
            e=ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(self._geom_dim,))),
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
            _graph.adjacencylist(local_entities),
            local_values,
        )

        if MPI.COMM_WORLD.rank == 0:
            lv = graph_nodes.astype(np.int64).reshape((-1, 1))
            lvv = np.arange(len(graph_nodes), dtype=np.int32)
        else:
            lv = np.empty((0, 1), dtype=np.int64)
            lvv = np.empty((0,), dtype=np.int32)
        self.mesh.topology.create_connectivity(0, 1)
        local_vertices, local_vertex_values = _io.distribute_entity_data(self.mesh, 0, lv, lvv)
        self._facet_markers = mesh.meshtags_from_entities(
            self.mesh,
            0,
            _graph.adjacencylist(local_vertices),
            local_vertex_values,
        )

        self.subdomains.name = "subdomains"
        self.boundaries.name = "bifurcations"
        if self.cfg.export:
            with _io.XDMFFile(self.mesh.comm, self.cfg.outdir + "/mesh/mesh.xdmf", "w") as file:
                file.write_mesh(self.mesh)
                file.write_meshtags(self.subdomains, self.mesh.geometry)
                file.write_meshtags(self.boundaries, self.mesh.geometry)

    @property
    def in_nodes(self) -> tuple[int, ...]:
        return self._boundary_in_nodes

    @property
    def out_nodes(self) -> tuple[int, ...]:
        return self._boundary_out_nodes

    @timeit
    def _build_network_submeshes(self):
        """Create submeshes for each edge in the network."""
        assert self._msh is not None
        assert self._subdomains is not None
        assert self._facet_markers is not None
        self._edge_meshes = []
        self._edge_entity_maps = []
        self._submesh_facet_markers = []
        for i in range(self._num_segments):
            edge_subdomain = self.subdomains.indices[self.subdomains.values == i]
            edge_mesh, edge_map, vertex_map = mesh.create_submesh(
                self.mesh, self.mesh.topology.dim, edge_subdomain
            )[0:3]
            self._edge_meshes.append(edge_mesh)
            self._edge_entity_maps.append(edge_map)
            self._submesh_facet_markers.append(
                scifem.transfer_meshtags_to_submesh(
                    self._facet_markers, edge_mesh, vertex_map, edge_map
                )[0]
            )

    @property
    def submesh_facet_markers(self) -> list[mesh.MeshTags]:
        if self._submesh_facet_markers is None:
            raise RuntimeError("Mesh has no submesh facet markers")
        return self._submesh_facet_markers

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

    @property
    def submeshes(self):
        if len(self._edge_meshes) == 0:
            raise RuntimeError(
                "Submeshes have not been built yet. Call build_network_submeshes() first."
            )
        return self._edge_meshes

    @property
    def entity_maps(self):
        if len(self._edge_entity_maps) == 0:
            raise RuntimeError(
                "Entity maps have not been built yet. Call build_network_submeshes() first."
            )
        return self._edge_entity_maps

    @property
    def tangent(self):
        return self._tangent

    def export_tangent(self):
        if self.cfg.export:
            with _io.XDMFFile(self.comm, self.cfg.outdir + "/mesh/tangent.xdmf", "w") as file:
                file.write_mesh(self.mesh)
                file.write_function(self.tangent)
        else:
            print("Export of tangent skipped as cfg.export is set to False.")

    @property
    def bifurcation_values(self) -> npt.NDArray[np.int32]:
        return self._bifurcation_values

    @property
    def boundary_values(self) -> npt.NDArray[np.int32]:
        return self._boundary_values

    def in_edges(self, bifurcation: int) -> list[int]:
        return self._bifurcation_in_edges[bifurcation]

    def out_edges(self, bifurcation: int) -> list[int]:
        return self._bifurcation_out_edges[bifurcation]


def compute_tangent(domain: mesh.Mesh) -> fem.Function:
    """Compute tangent vector for all cells.

    Tangent is oriented according to positive y-axis.
    If perpendicular to y-axis, align with x-axis.

    Note:
        Assuming that the mesh is affine.
    """
    cell_map = domain.topology.index_map(domain.topology.dim)
    geom_indices = mesh.entities_to_geometry(
        domain,
        domain.topology.dim,
        np.arange(cell_map.size_local + cell_map.num_ghosts, dtype=np.int32),
    )
    geom_coordinates = domain.geometry.x[geom_indices]
    tangent = geom_coordinates[:, 0, :] - geom_coordinates[:, 1, :]
    global_orientation = np.sign(np.dot(tangent, [0, 1, 0]))
    is_x_aligned = np.isclose(global_orientation, 0)
    global_orientation[is_x_aligned] = np.sign(np.dot(tangent[is_x_aligned], [1, 0, 0]))
    tangent *= global_orientation[:, None]
    assert np.all(np.linalg.norm(tangent, axis=1) > 0), "Zero-length tangent vector detected"
    gdim = domain.geometry.dim
    DG0 = fem.functionspace(domain, ("DG", 0, (gdim,)))
    global_tangent = fem.Function(DG0)
    global_tangent.x.array[:] = tangent.flatten()
    return global_tangent
