# Copyright (C) Simula Research Laboratory and Jørgen S. Dokken
# SPDX-License-Identifier:    MIT
"""
Interface for converting a :py:class:`directional networkx graph<networkx.DiGraph>`
into a :py:class:`dolfinx.mesh.Mesh`.

This idea stems from the Graphnics project (https://doi.org/10.48550/arXiv.2212.02916),
https://github.com/IngeborgGjerde/fenics-networks by Ingeborg Gjerde.
"""

from typing import Callable, Iterable

from mpi4py import MPI

import networkx as nx
import numpy as np
import numpy.typing as npt

import basix.ufl
import ufl
from dolfinx import common, fem, mesh
from dolfinx import graph as _graph
from dolfinx import io as _io

__all__ = ["NetworkMesh"]


@common.timed("nxfx:color_graph")
def color_graph(
    graph: nx.DiGraph,
    strategy: str | Callable[[nx.Graph, dict[int, int]], Iterable[int]] | None,
) -> dict[tuple[int, int], int]:
    """
    Color the edges of a graph.
    """
    if strategy is not None:
        undirected_edge_graph = nx.line_graph(graph.to_undirected())
        edge_coloring = nx.coloring.greedy_color(undirected_edge_graph, strategy=strategy)
    else:
        edge_coloring = {edge: i for i, edge in enumerate(graph.edges)}
    return edge_coloring


class NetworkMesh:
    """A representation of a :py:class:`Directional Networkx<networkx.DiGraph>` graph in
    :py:mod:`DOLFINx<dolfinx>`.

    Stores the resulting :py:class:`mesh<dolfinx.mesh.Mesh>`,
    :py:class:`subdomains<dolfinx.mesh.MeshTags>`,
    and :py:class:`facet markers<dolfinx.mesh.MeshTags>` for bifurcations and boundary nodes.
    Has a :py:class:`submesh<dolfinx.mesh.Mesh>` for each edge in the
    :py:class:`Networkx graph<networkx.DiGraph>`.

    Args:
        graph: The directional networkx graph to convert.
        N: Number of elements per segment.
        color_strategy: Strategy to use for coloring the graph edges.
            If set to `None`, no-graphcoloring is used (not recommended).
        comm: The MPI communicator to distribute the mesh on.
        graph_rank: The MPI rank of the process that holds the graph.
    """

    # Graph properties
    _geom_dim: int
    _num_edge_colors: int
    _bifurcation_in_color: dict[int, list[int]]
    _bifurcation_out_color: dict[int, list[int]]

    # Mesh properties
    _msh: mesh.Mesh | None
    _subdomains: mesh.MeshTags | None
    _facet_markers: mesh.MeshTags | None
    _submesh_facet_markers: list[mesh.MeshTags]
    _edge_meshes: list[mesh.Mesh]
    _edge_entity_maps: list[mesh.EntityMap]
    _orientation: fem.Function
    _bifurcation_values: npt.NDArray[np.int32]
    _boundary_values: npt.NDArray[np.int32]
    _lm_mesh: mesh.Mesh | None
    _lm_map: mesh.EntityMap | None

    def __init__(
        self,
        graph: nx.DiGraph,
        N: int,
        color_strategy: str
        | Callable[[nx.Graph, dict[int, int]], Iterable[int]]
        | None = "largest_first",
        comm: MPI.Comm = MPI.COMM_WORLD,
        graph_rank: int = 0,
    ):
        self._build_mesh(
            graph, N=N, color_strategy=color_strategy, comm=comm, graph_rank=graph_rank
        )
        self._build_network_submeshes()
        self._create_lm_submesh()

    @property
    def lm_mesh(self) -> mesh.Mesh:
        """Lagrange multiplier mesh, a point-cloud mesh including each bifurcation."""
        if self._lm_mesh is None:
            raise RuntimeError("Lagrange multiplier submesh has not been created.")
        return self._lm_mesh

    @property
    def lm_map(self) -> mesh.EntityMap:
        """Entity map for the :py:meth:`Lagrange multiplier mesh<NetworkMesh.lm_mesh>`"""
        if self._lm_map is None:
            raise RuntimeError("Lagrange multiplier entity map has not been created.")
        return self._lm_map

    @property
    def comm(self) -> MPI.Comm:
        """MPI-communicator of the network mesh"""
        return self.mesh.comm

    @common.timed("nxfx:create_lm_submesh")
    def _create_lm_submesh(self):
        """Create a submesh for the Lagrange multipliers at the bifurcations.

        Note:
            This is an internal class function, that is not supposed to be called by
            users.
        """
        assert self._msh is not None
        assert self._facet_markers is not None
        bifurcation_indices = self._facet_markers.indices[
            np.isin(self._facet_markers.values, self.bifurcation_values)
        ]
        self._lm_mesh, self._lm_map = mesh.create_submesh(
            self.mesh,
            self.mesh.topology.dim - 1,
            bifurcation_indices,
        )[:2]
        # Workaround until: https://github.com/FEniCS/dolfinx/pull/3974 is merged
        self._lm_mesh.topology.create_entity_permutations()

    @common.timed("nxfx:build_mesh")
    def _build_mesh(
        self,
        graph: nx.DiGraph | None,
        N,
        color_strategy: str | Callable[[nx.Graph, dict[int, int]], Iterable[int]] | None,
        comm: MPI.Comm,
        graph_rank: int,
    ):
        """Convert the networkx graph into a {py:class}`dolfinx.mesh.Mesh`.

        Each segment in the networkx graph gets a unique subdomain marker.
        Each bifurcation and boundary node is marked on the facets with a unique integer.

        Note:
            This function attaches data to `self.mesh`, `self.subdomains` and
            `self.boundaries`.

        Args:
            graph: The networkx graph to convert.
            N: Number of elements per segment.
            color_strategy: Strategy to use for coloring the graph edges.
                If set to `None`, no-graphcoloring is used (not recommended).
            comm: The MPI communicator to distribute the mesh on.
            graph_rank: The MPI rank of the process that holds the graph.
        """
        # Extract geometric dimension from graph
        geom_dim = None
        num_edge_colors = None
        max_connections = None
        bifurcation_values = None
        boundary_values = None
        number_of_nodes = None
        bifurcation_in_color: dict[int, list[int]] | None = None
        bifurcation_out_color: dict[int, list[int]] | None = None
        if comm.rank == graph_rank:
            assert isinstance(graph, nx.DiGraph), f"Directional graph not present of {graph_rank}"
            geom_dim = len(graph.nodes[1]["pos"])
            edge_coloring = color_graph(graph, color_strategy)

            num_edge_colors = len(set(edge_coloring.values()))
            cells_array = np.asarray([[u, v] for u, v in graph.edges()])
            number_of_nodes = graph.number_of_nodes()
            nodes_with_degree = np.full(number_of_nodes, -1, dtype=np.int32)
            for node, degree in graph.degree():
                nodes_with_degree[node] = degree
            bifurcation_values = np.flatnonzero(nodes_with_degree > 1)
            boundary_values = np.flatnonzero(nodes_with_degree == 1)
            max_connections = np.max(nodes_with_degree)

            bifurcation_in_color = {}
            bifurcation_out_color = {}
            for bifurcation in bifurcation_values:
                in_edges = graph.in_edges(bifurcation)
                bifurcation_in_color[int(bifurcation)] = []
                for edge in in_edges:
                    bifurcation_in_color[int(bifurcation)].append(edge_coloring[edge])
                out_edges = graph.out_edges(bifurcation)
                bifurcation_out_color[int(bifurcation)] = []
                for edge in out_edges:
                    bifurcation_out_color[int(bifurcation)].append(edge_coloring[edge])

            # Map boundary_values to inlet and outlet data from graph
            boundary_in_nodes = []
            boundary_out_nodes = []
            for boundary in boundary_values:
                in_edges = graph.in_edges(boundary)
                out_edges = graph.out_edges(boundary)
                assert len(in_edges) + len(out_edges) == 1, "Boundary node with multiple edges"
                if len(in_edges) == 1:
                    boundary_in_nodes.append(int(boundary))
                else:
                    boundary_out_nodes.append(int(boundary))

        comm.bcast(num_edge_colors, root=graph_rank)
        num_edge_colors, number_of_nodes, max_connections, geom_dim = comm.bcast(
            (num_edge_colors, number_of_nodes, max_connections, geom_dim), root=graph_rank
        )
        comm.barrier()
        bifurcation_values, boundary_values = comm.bcast(
            (bifurcation_values, boundary_values), root=graph_rank
        )
        comm.barrier()
        bifurcation_in_color, bifurcation_out_color = comm.bcast(
            (bifurcation_in_color, bifurcation_out_color), root=graph_rank
        )
        comm.barrier()

        self._geom_dim = geom_dim
        self._num_edge_colors = num_edge_colors
        self._bifurcation_values = bifurcation_values
        self._boundary_values = boundary_values

        # Create lookup of in and out colors for each bifurcation
        self._bifurcation_in_color = bifurcation_in_color
        self._bifurcation_out_color = bifurcation_out_color

        # Generate mesh
        if comm.rank == graph_rank:
            assert isinstance(graph, nx.DiGraph), (
                f"No directional graph present on rank {comm.rank}"
            )
            vertex_coords = np.asarray([graph.nodes[v]["pos"] for v in graph.nodes()])
            line_weights = np.linspace(0, 1, N, endpoint=False)[1:][:, None]
            mesh_nodes = vertex_coords.copy()
            cells = []
            cell_markers = []
            if len(line_weights) == 0:
                for segment in cells_array:
                    cells.append(np.array([segment[0], segment[1]], dtype=np.int64))
                    cell_markers.append(edge_coloring[(segment[0], segment[1])])
                    start = vertex_coords[segment[0]]
                    end = vertex_coords[segment[1]]
                    in_order = segment[0] < segment[1]
            else:
                for segment in cells_array:
                    start_coord_pos = mesh_nodes.shape[0]
                    start = vertex_coords[segment[0]]
                    end = vertex_coords[segment[1]]
                    internal_line_coords = start * (1 - line_weights) + end * line_weights

                    in_order = segment[0] < segment[1]

                    mesh_nodes = np.vstack((mesh_nodes, internal_line_coords))
                    cells.append(np.array([segment[0], start_coord_pos], dtype=np.int64))
                    segment_connectivity = (
                        np.repeat(np.arange(internal_line_coords.shape[0]), 2)[1:-1].reshape(
                            internal_line_coords.shape[0] - 1, 2
                        )
                        + start_coord_pos
                    )
                    cells.append(segment_connectivity)
                    cells.append(
                        np.array(
                            [
                                start_coord_pos + internal_line_coords.shape[0] - 1,
                                segment[1],
                            ],
                            dtype=np.int64,
                        )
                    )
                    cell_markers.extend(
                        np.full(
                            internal_line_coords.shape[0] + 1,
                            edge_coloring[(segment[0], segment[1])],
                            dtype=np.int32,
                        )
                    )

            cells_ = np.vstack(cells).astype(np.int64)
            cell_markers_ = np.array(cell_markers, dtype=np.int32)

            orientations = np.full_like(cell_markers_, 1.0, dtype=np.float64)
            orientations[cells_[:, 0] > cells_[:, 1]] = -1.0

            assert cell_markers_.shape == orientations.shape
        else:
            cells_ = np.empty((0, 2), dtype=np.int64)
            mesh_nodes = np.empty((0, self._geom_dim), dtype=np.float64)
            cell_markers_ = np.empty((0,), dtype=np.int32)
            orientations = np.empty(0, dtype=np.float64)

        partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
        graph_mesh = mesh.create_mesh(
            comm,
            x=mesh_nodes,
            cells=cells_,
            e=ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(self._geom_dim,))),
            partitioner=partitioner,
            max_facet_to_cell_links=np.max(max_connections),
        )
        self._msh = graph_mesh

        tdim = self.mesh.topology.dim

        # Compute subdomain tags.
        local_entities, local_values = _io.distribute_entity_data(
            self.mesh, tdim, cells_, cell_markers_
        )

        self._subdomains = mesh.meshtags_from_entities(
            self.mesh,
            self.mesh.topology.dim,
            _graph.adjacencylist(local_entities),
            local_values,
        )

        # Distribute orientations as meshtag.
        local_cells, local_orientations = _io.distribute_entity_data(
            self.mesh, tdim, cells_, orientations
        )

        meshtag_orientation = mesh.meshtags_from_entities(
            self.mesh, tdim, _graph.adjacencylist(local_cells), local_orientations
        )

        # Assemble orientations function
        orientation_space = fem.functionspace(graph_mesh, ("DG", 0))
        self._orientation = fem.Function(orientation_space)
        self._orientation.x.array[meshtag_orientation.indices] = meshtag_orientation.values

        # Correct orientations for possible reorder
        e_idx = np.arange(self.mesh.topology.index_map(tdim).size_local, dtype=np.int32)
        e_geo = mesh.entities_to_geometry(self.mesh, tdim, e_idx)

        global_input = self.mesh.geometry.input_global_indices
        in_order = global_input[e_geo[:, 0]] < global_input[e_geo[:, 1]]

        # Four cases that might arise (per edge), with naming i := input_in_order, n := in_order:
        # 1) i & n:
        #   orientation = 1 (input=1, no action needed)
        # 2) i & !n:
        #   orientation = -1 (input=1, need sign flip)
        # 3) !i & n:
        #   orientation = -1 (input=-1, no action needed)
        # 4) !i & !n
        #   orientation = 1 (input=-1, need sign flip)
        #
        # so we need to sign flip when
        #   (i & !n) | (!i & !n) = !n
        self.orientation.x.array[: e_idx.size][~in_order] *= -1

        self.orientation.x.scatter_forward()

        self._in_marker = 3 * number_of_nodes
        self._out_marker = 5 * number_of_nodes
        if comm.rank == graph_rank:
            lv = np.arange(number_of_nodes, dtype=np.int64).reshape(-1, 1)
            lvv = np.arange(number_of_nodes, dtype=np.int32)
            lvv[boundary_in_nodes] = self._in_marker
            lvv[boundary_out_nodes] = self._out_marker
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

    @common.timed("nxfx:build_network_submeshes")
    def _build_network_submeshes(self):
        """Create submeshes for each edge in the network."""
        assert self._msh is not None
        assert self._subdomains is not None
        assert self._facet_markers is not None
        self._edge_meshes = []
        self._edge_entity_maps = []
        self._submesh_facet_markers = []
        parent_vertex_map = self.mesh.topology.index_map(0)
        num_vertices_parent = parent_vertex_map.size_local + parent_vertex_map.num_ghosts
        parent_vertex_marker = np.full(num_vertices_parent, -1, dtype=np.int32)
        parent_vertex_marker[self._facet_markers.indices] = self._facet_markers.values
        for i in range(self._num_edge_colors):
            # Create submesh of color i
            edge_subdomain = self.subdomains.indices[self.subdomains.values == i]
            edge_mesh, edge_map, vertex_map = mesh.create_submesh(
                self.mesh, self.mesh.topology.dim, edge_subdomain
            )[0:3]
            self._edge_meshes.append(edge_mesh)
            self._edge_entity_maps.append(edge_map)

            # Map all submesh vertices to parent
            num_submesh_vertices = (
                edge_mesh.topology.index_map(0).size_local
                + edge_mesh.topology.index_map(0).num_ghosts
            )
            parent_vertices = vertex_map.sub_topology_to_topology(
                np.arange(num_submesh_vertices, dtype=np.int32), inverse=False
            )
            sub_topology_values = parent_vertex_marker[parent_vertices]
            marked_vertices = np.flatnonzero(sub_topology_values >= 0)
            marked_values = sub_topology_values[marked_vertices].copy()
            self._submesh_facet_markers.append(
                mesh.meshtags(edge_mesh, 0, marked_vertices, marked_values)
            )

    def in_edges(self, bifurcation: int) -> list[int]:
        return self._bifurcation_in_color[bifurcation]

    def out_edges(self, bifurcation: int) -> list[int]:
        return self._bifurcation_out_color[bifurcation]

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
    def orientation(self):
        """Return DG-0 field containing the tangent vector of the graph."""
        return self._orientation

    def export_orientation(self):
        if self.cfg.export:
            with _io.XDMFFile(self.comm, self.cfg.outdir / "mesh/orientation.xdmf", "w") as file:
                file.write_mesh(self.mesh)
                file.write_function(self.orientation)
        else:
            print("Export of tangent skipped as cfg.export is set to False.")

    @property
    def bifurcation_values(self) -> npt.NDArray[np.int32]:
        return self._bifurcation_values

    @property
    def boundary_values(self) -> npt.NDArray[np.int32]:
        return self._boundary_values

    @property
    def in_marker(self) -> int:
        return self._in_marker

    @property
    def out_marker(self) -> int:
        return self._out_marker
