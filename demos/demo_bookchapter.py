# This file is based on the graphnics project (https://arxiv.org/abs/2212.02916), https://github.com/IngeborgGjerde/fenics-networks - forked on August 2022
# Copyright (C) 2022-2023 by Ingeborg Gjerde
# You can freely redistribute it and/or modify it under the terms of the GNU General Public License, version 3.0, provided that the above copyright notice
# is kept intact and that the source code is made available under an open-source license.

import networkx as nx
import matplotlib.pyplot as plt
import gmsh
from collections import defaultdict
from dolfinx import io, mesh, fem
from mpi4py import MPI
import numpy as np
from enum import Enum
import ufl
from petsc4py import PETSc
from dolfinx.common import Timer
import json
from basix.ufl import element
import basix
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block


def create_tree(n: int, H: float, W: float):
    """
    Create a tree graph.

    Args:
        n: number of generations
        H: height
        W: width

    Returns:
        A tree graph and the nodal coordinates
    """

    # FIXME: add parameter r (branching factor of the tree i.e. each
    # node has r children)
    r = 2
    G = nx.DiGraph()

    nb_nodes_gen = []
    for i in range(n):
        nb_nodes_gen.append(pow(r, i))

    nb_nodes = 1 + sum(nb_nodes_gen)
    nb_nodes_last = pow(r, n - 1)

    G.add_nodes_from(range(nb_nodes))

    x_offset = W / (2 * (nb_nodes_last - 1))
    y_offset = H / n

    # Add two first nodes
    idx = 0
    pos = {}
    pos[idx] = np.array([0, 0])
    pos[idx + 1] = np.array([0, y_offset])
    idx = idx + 2

    # Add nodes for rest of the tree
    for gen in range(1, n):
        factor = pow(2, n - gen)
        x = x_offset * (factor / 2)
        y = y_offset * (gen + 1)
        x_coord = []
        nb_nodes_ = int(nb_nodes_gen[gen] / 2)
        for i in range(nb_nodes_):
            x_coord.append(x)
            x_coord.append(-x)
            x = x + x_offset * factor
        # Add nodes to G, from sorted x_coord array
        x_coord.sort()
        for x in x_coord:
            pos[idx] = np.array([x, y])
            idx = idx + 1

    # TODO Allocate memory before
    G.add_edge(0, 1)
    current_node = 1
    node_to_add = 2
    arity = 2
    for level in range(n - 1):
        nodes_in_level = 2**level
        for i in range(nodes_in_level):
            for j in range(arity):
                G.add_edge(current_node, node_to_add)
                node_to_add += 1
            current_node += 1
    return G, pos


def colour_graph(G):
    """
    Colour the graph

    Args:
        G: The grpah

    Returns:
        A tuple whose first entry are the unique colours in graph
        and whose second entry is a dictionary with keys representing
        edges and values representing corresponding colouring.
    """
    L = nx.line_graph(G.to_undirected())
    colouring = nx.coloring.greedy_color(L, strategy="largest_first")
    colours = np.unique(list(colouring.values()))
    return colours, colouring


def draw_graph(G, pos, G_edge_colouring):
    """
    Draw a graph.

    Args:
        G: The graph
        pos: The nodal coordinates
        G_edge_colouring: The edge colouring
    """
    nx.draw_networkx_nodes(G, pos=pos)
    # FIXME: Generalise for graphs with more colours
    colours = ["red", "blue", "green", "yellow"]
    nx.draw_networkx_edges(
        G, pos=pos, edge_color=[colours[G_edge_colouring[e]] for e in G.edges()]
    )
    nx.draw_networkx_labels(G, pos)
    plt.show()


def create_mesh_gmsh(comm, G, pos, mesh_size=1000):
    """
    Create a mesh of a network using gmsh.

    Args:
        comm: The MPI communicator
        G: The graph representing the network
        pos: The nodal coordinates
        mesh_size: The maximum cell diameter

    Returns:
        The mesh, cell tags, and facet tags
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    factory = gmsh.model.geo
    # FIXME Check ith entry in pos.items() corresponds to node i
    points = [
        factory.addPoint(x[0], x[1], 0.0, meshSize=mesh_size)
        for (node, x) in pos.items()
    ]
    lines = [factory.addLine(points[e[0]], points[e[1]]) for e in G.edges()]
    factory.synchronize()

    for i, point in enumerate(points):
        gmsh.model.add_physical_group(0, [point], i)
    for i, line in enumerate(lines):
        gmsh.model.addPhysicalGroup(1, [line], i)
    gmsh.model.mesh.generate(1)
    # gmsh.fltk.run()
    return io.gmshio.model_to_mesh(gmsh.model, comm=comm, rank=0, gdim=3)


def create_mesh(comm, G, pos):
    """
    Create a mesh of a network.

    Args:
        comm: The MPI communicator
        G: The graph representing the network
        pos: The nodal coordinates

    Returns:
        The mesh, cell tags, and facet tags
    """
    # FIXME Change how pos is created so this isn't needed
    d = 3
    x = np.zeros(d * len(pos))
    for node, coord in pos.items():
        x[d * node:d * node + 2] = coord[:]
    x = np.reshape(x, (len(pos), d))

    cells = np.array(list(G.edges))
    domain = ufl.Mesh(
        element(basix.ElementFamily.P, basix.CellType.interval, 1, shape=(3,))
    )
    msh = mesh.create_mesh(comm, cells, x, domain)
    msh.topology.create_connectivity(0, 1)

    cell_imap = msh.topology.index_map(1)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts
    entities = np.arange(num_cells, dtype=np.int32)
    values = np.array(msh.topology.original_cell_index, dtype=np.int32)
    ct = mesh.meshtags(msh, 1, entities, values)

    vertex_imap = msh.topology.index_map(0)
    num_vertices = vertex_imap.size_local + vertex_imap.num_ghosts
    vertices = np.arange(num_vertices, dtype=np.int32)
    vertex_values = np.array(msh.geometry.input_global_indices, dtype=np.int32)
    ft = mesh.meshtags(msh, 0, vertices, vertex_values)

    return msh, ct, ft


def create_colour_to_edges(G, G_edge_colouring):
    """
    Create a dictionary mapping a colour to graph edges.

    Args:
        G: The graph
        G_edge_colouring: The edge colouring of G

    Returns:
        A dictionary mapping a colour to graph edges
    """
    colour_to_edges = defaultdict(list)
    for i, e in enumerate(G.edges()):
        colour_to_edges[G_edge_colouring[e]].append(i)
    return colour_to_edges


def create_submeshes(colours, msh, ct, colour_to_edges):
    """
    Create a sub-mesh of edges of each colour. The submesh at
    index is has edges of colour i.

    Args:
        colours: The unique colours in the graph
        msh: The mesh
        ct: Cell tags
        colour_to_edges: A map from colours to edges of that colour

    Returns:
        A list of submeshes (including entity, geometry, and vertex maps) for
        each colour
    """
    submeshes = []
    for c in colours:
        edges = colour_to_edges[c]
        entities = ct.indices[np.isin(ct.values, edges)]
        sm = mesh.create_submesh(msh, msh.topology.dim, entities)
        sm[0].topology.create_connectivity(0, 1)
        submeshes.append(sm)
    return submeshes


class NodeType(Enum):
    # Boundary
    BOUND = 1
    # Bifurcation
    BI = 2
    # Inlet boundary
    BOUND_IN = 3
    # Outlet boundary
    BOUND_OUT = 4
    # Inlet bifurcation
    BI_IN = 5
    # Outlet bifurcation
    BI_OUT = 6


def compute_node_types(G):
    """
    Compute the type of each node in a graph

    Args:
        G: The graph

    Returns:
        A list whose ith entry is the node type of node i.
    """
    node_types = []
    for node in G.nodes:
        num_conn_edges = len(G.in_edges(node)) + len(G.out_edges(node))
        assert num_conn_edges > 0
        if num_conn_edges == 1:
            node_types.append(NodeType.BOUND)
        else:
            node_types.append(NodeType.BI)
    return node_types


def create_msh_to_sm_vertex_maps(submeshes):
    """
    Create maps relating mesh vertices to submesh vertices

    Args:
        submeshes: A list with each entry of the form
            [submesh, entitiy_map, vertex_map, geom_map]

    Returns:
        Maps relating msh vertices to submesh vertices for each
        submesh in submeshes
    """
    msh_to_sm_vertex_maps = []
    vertex_imap = msh.topology.index_map(0)
    num_vertices = vertex_imap.size_local + vertex_imap.num_ghosts
    for sm in submeshes:
        sm_to_msh_vertex_map = sm[2]
        msh_to_sm_vertex_map = np.full(num_vertices, -1)
        msh_to_sm_vertex_map[sm_to_msh_vertex_map] = np.arange(
            len(sm_to_msh_vertex_map)
        )
        msh_to_sm_vertex_maps.append(msh_to_sm_vertex_map)
    return msh_to_sm_vertex_maps


def create_node_to_vertex_map(ft):
    """
    Create a map from graph nodes to mesh vertices

    Args:
        ft: The facet tags for the mesh

    Returns:
        An array whose ith entry is the vertex corresponding to graph node i
    """
    node_to_vertex = np.full(len(ft.indices), -1)
    for node, vertex in zip(ft.values, ft.indices):
        node_to_vertex[node] = vertex
    return node_to_vertex


def compute_submesh_facet_tags(
    submeshes, colours, msh_to_sm_vertex_maps, colour_to_edges, G, node_to_vertex
):
    """
    Compute facet tags for a submesh

    Args:
        submeshes: A list of submeshes
        colours: The graph colours
        msh_to_sm_vertex_maps: Maps relating mesh vertices to submesh vertices
        colour_to_edges: A map from graph colours to edges of that colour
        G: The graph
        node_to_vertex: A map from graph nodes to mesh vertices

    Returns:
        The facet tags for each submesh
    """
    sm_facet_tags = []
    for c in colours:
        sm = submeshes[c]
        msh_to_sm_vertex_map = msh_to_sm_vertex_maps[c]
        indices = []
        tags = []
        edges = list(G.edges)
        for e in colour_to_edges[c]:
            nodes = edges[e]
            for i, node in enumerate(nodes):
                vertex = node_to_vertex[node]
                vertex_sm = msh_to_sm_vertex_map[vertex]
                assert vertex_sm >= 0

                node_type = node_types[node]
                if node_type == NodeType.BOUND and i == 0:
                    tag = NodeType.BOUND_IN
                elif node_type == NodeType.BOUND and i == 1:
                    tag = NodeType.BOUND_OUT
                elif node_type == NodeType.BI and i == 0:
                    tag = NodeType.BI_IN
                elif node_type == NodeType.BI and i == 1:
                    tag = NodeType.BI_OUT

                indices.append(vertex_sm)
                tags.append(tag.value)
        indices = np.array(indices, dtype=np.int32)
        tags = np.array(tags, dtype=np.int32)
        perm = np.argsort(indices)
        sm_facet_tags.append(mesh.meshtags(sm[0], 0, indices[perm], tags[perm]))
    return sm_facet_tags


def create_cell_to_edge_map(ct, num_cells):
    """
    Create a map from mesh cells to graph edges.

    Args:
        ct: Cell tags
        num_cells: Number of cells in mesh

    Returns:
        An array whose ith entry is the graph edge corresponding to cell i
    """
    cell_to_edge = np.full(num_cells, -1)
    for cell, edge in zip(ct.indices, ct.values):
        cell_to_edge[cell] = edge
    return cell_to_edge


def compute_global_tangent(G, msh, num_cells, cell_to_edge):
    """
    Compute the tangent vector to network edges.

    Args:
        G: The graph
        msh: The mesh
        num_cells: Number of cells in the mesh
        cell_to_edge: Map relating mesh cells to graph edges

    Returns:
        A function representing the tangent vector to each edge in the network.
    """
    edge_to_tangent = []
    for e in G.edges():
        t = pos[e[1]] - pos[e[0]]
        t *= 1 / np.linalg.norm(t)
        edge_to_tangent.append(t)
    V_gt = fem.functionspace(msh, ("Discontinuous Lagrange", 0, (msh.geometry.dim,)))
    gt = fem.Function(V_gt)
    block_size = V_gt.dofmap.index_map_bs
    for c in range(num_cells):
        edge = cell_to_edge[c]
        dof = V_gt.dofmap.cell_dofs(c)
        t = edge_to_tangent[edge]
        gt.x.array[dof * block_size] = t[0]
        gt.x.array[dof * block_size + 1] = t[1]
    return gt


def recover_solution(x, colours, V, Q):
    """
    Recover the fluxes and pressure from the global solution vector.

    Args:
        x: Global solution vector
        colours: Unique colours in the graph
        V: Flux function spaces
        Q: Pressure function space

    Returns:
        The fluxes and the pressure
    """
    fluxes = []
    start = 0
    for c in colours:
        q = fem.Function(V[c])
        offset = V[c].dofmap.index_map.size_local * V[c].dofmap.index_map_bs
        q.x.array[:offset] = x.array_r[start:start + offset]
        q.x.scatter_forward()
        start += offset
        fluxes.append(q)

    p = fem.Function(Q)
    offset = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
    p.x.array[:offset] = x.array_r[start:start + offset]
    p.x.scatter_forward()
    return fluxes, p


def dds(f):
    """
    Compute the derivative df/ds, where s is the arc length along a graph edge.

    Args:
        f: The function whose derivative to compute

    Return:
        The derivative df/ds
    """

    return ufl.dot(ufl.grad(f), gt)


def p_bc_expr(x):
    return 1 - x[1]


def create_lm_bc(node_types, ft):
    """
    Create Dirichlet boundary conditions for lagrange multipliers

    Args:
        node_types: The type of each node in the graph
        ft: the facet tags

    Returns:
        A Dirichlet boundary condition
    """
    # FIXME Do this efficiently
    unconstrained_nodes = []
    for node, node_type in enumerate(node_types):
        if node_type == NodeType.BI:
            unconstrained_nodes.append(node)
    # NOTE: Can't just invert isin because ft only contains vertices at nodes
    unconstrained_vertices = ft.indices[np.isin(ft.values, unconstrained_nodes)]
    vertex_imap = msh.topology.index_map(0)
    num_vertices = vertex_imap.size_local + vertex_imap.num_ghosts
    vertices = np.arange(num_vertices)
    constrained_vertices = vertices[
        np.isin(vertices, unconstrained_vertices, invert=True)
    ]
    constrained_dofs = fem.locate_dofs_topological(W, 0, constrained_vertices)
    return fem.dirichletbc(0.0, constrained_dofs, W)


def create_timer(name):
    print(name)
    return Timer(name)


data_dict = {}

t = create_timer("Create tree")
n = 6  # Number of generations
G, pos = create_tree(n, n, n)
data_dict["create_tree"] = t.stop()
print("Done.")

t = create_timer("Colour graph")
colours, G_edge_colouring = colour_graph(G)
colour_to_edges = create_colour_to_edges(G, G_edge_colouring)
data_dict["colour_graph"] = t.stop()
print("Done.")

# draw_graph(G, pos, G_edge_colouring)
# exit()

t = create_timer("Create mesh")
comm = MPI.COMM_WORLD
# NOTE NEED TO DISABLE GPS reordering in FEniCSx for performance
msh, ct, ft = create_mesh(comm, G, pos)
data_dict["create_mesh"] = t.stop()
print("Done.")

imap = msh.topology.index_map(1)
num_cells = imap.size_local + imap.num_ghosts
print(f"Number of cells = {num_cells}")
data_dict["num_cells"] = num_cells

# with io.XDMFFile(comm, "mesh.xdmf", "w") as file:
#     file.write_mesh(msh)
#     # file.write_meshtags(ct)
#     file.write_meshtags(ft)

t = create_timer("Create submeshes")
submeshes = create_submeshes(colours, msh, ct, colour_to_edges)
data_dict["create_submeshes"] = t.stop()
print("Done.")

t = create_timer("Compute node types")
node_types = compute_node_types(G)
data_dict["compute_node_types"] = t.stop()
print("Done.")

t = create_timer("Create vertex maps")
msh_to_sm_vertex_maps = create_msh_to_sm_vertex_maps(submeshes)
data_dict["create_vertex_maps"] = t.stop()
print("Done.")

t = create_timer("Create node to vertex map")
node_to_vertex = create_node_to_vertex_map(ft)
data_dict["create_node_to_ver_map"] = t.stop()
print("Done.")

t = create_timer("Create submesh facet tags")
fts_sm = compute_submesh_facet_tags(
    submeshes, colours, msh_to_sm_vertex_maps, colour_to_edges, G, node_to_vertex
)
data_dict["create_sm_facet_tags"] = t.stop()
print("Done.")

# for c, sm in enumerate(submeshes):
#     with io.XDMFFile(comm, f"sm_{c}.xdmf", "w") as file:
#         file.write_mesh(sm[0])
#         file.write_meshtags(fts_sm[c])

t = create_timer("Create function space")
k = 1
# Flux function space
V = [fem.functionspace(sm[0], ("Lagrange", k)) for sm in submeshes]
# Pressure function space
Q = fem.functionspace(msh, ("Discontinuous Lagrange", k - 1))
# Lagrange multiplier function space
W = fem.functionspace(msh, ("Lagrange", k))
data_dict["create_func_spaces"] = t.stop()
print("Done.")

t = create_timer("Create cell to edge map")
cell_to_edge = create_cell_to_edge_map(ct, num_cells)
data_dict["create_cell_to_edge"] = t.stop()
print("Done.")

t = create_timer("Compute global tangent")
gt = compute_global_tangent(G, msh, num_cells, cell_to_edge)
data_dict["create_global_tangent"] = t.stop()
print("Done.")

t = create_timer("Compute forms")
# TODO Need to add other blocks
num_colours = len(colours)
num_blocks = num_colours + 2
a = [[None] * num_blocks for i in range(num_blocks)]
L = [None] * num_blocks

p = ufl.TrialFunction(Q)
phi = ufl.TestFunction(Q)

lmbda = ufl.TrialFunction(W)
mu = ufl.TestFunction(W)

cffi_options = ["-Ofast", "-march=native"]
jit_options = {"cffi_extra_compile_args": cffi_options}

for c in colours:
    sm = submeshes[c]
    dx = ufl.Measure("dx", domain=sm[0])

    q = ufl.TrialFunction(V[c])
    v = ufl.TestFunction(V[c])

    a[c][c] = fem.form(q * v * dx, jit_options=jit_options)

    entity_maps = {msh: sm[1]}
    a[num_colours][c] = fem.form(
        phi * dds(q) * dx, entity_maps=entity_maps, jit_options=jit_options
    )
    a[c][num_colours] = fem.form(
        -p * dds(v) * dx, entity_maps=entity_maps, jit_options=jit_options
    )

    p_bc = fem.Function(V[c])
    p_bc.interpolate(p_bc_expr)

    ds = ufl.Measure("ds", domain=sm[0], subdomain_data=fts_sm[c])
    L[c] = fem.form(
        p_bc * v * ds(NodeType.BOUND_IN.value) - p_bc * v * ds(NodeType.BOUND_OUT.value),
        jit_options=jit_options,
    )

    a[num_colours + 1][c] = fem.form(
        mu * q * ds(NodeType.BI_IN.value) - mu * q * ds(NodeType.BI_OUT.value),
        entity_maps=entity_maps,
        jit_options=jit_options,
    )
    a[c][num_colours + 1] = fem.form(
        lmbda * v * ds(NodeType.BI_IN.value) - lmbda * v * ds(NodeType.BI_OUT.value),
        entity_maps=entity_maps,
        jit_options=jit_options,
    )

dx = ufl.Measure("dx", domain=msh)
a[num_colours + 1][num_colours + 1] = fem.form(
    1e-16 * lmbda * mu * dx, jit_options=jit_options
)
L[num_colours] = fem.form(1e-16 * phi * dx, jit_options=jit_options)
L[num_colours + 1] = fem.form(1e-16 * mu * dx, jit_options=jit_options)
data_dict["create_forms"] = t.stop()
print("Done.")

# Create bc for Lagrange multipliers
t = create_timer("Create LMBC")
lm_bc = create_lm_bc(node_types, ft)
data_dict["create_lmbc"] = t.stop()
print("Done.")

t = create_timer("Assemble matrix")
A = assemble_matrix_block(a, bcs=[lm_bc])
A.assemble()
data_dict["assemble_mat"] = t.stop()
print("Done.")

t = create_timer("Assemble vector")
b = assemble_vector_block(L, a, bcs=[lm_bc])
b.assemble()
data_dict["assemble_vec"] = t.stop()
print("Done.")

t = create_timer("Solve")
ksp = PETSc.KSP().create(comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")
x = A.createVecLeft()
ksp.solve(b, x)
data_dict["solve"] = t.stop()
print("Done.")

t = create_timer("Recover solution")
fluxes, p = recover_solution(x, colours, V, Q)
data_dict["recover_sol"] = t.stop()
print("Done.")

# Write to file
with io.VTXWriter(msh.comm, "p.bp", p, "BP4") as f:
    f.write(0.0)

for c, q in enumerate(fluxes):
    with io.VTXWriter(msh.comm, f"q_{c}.bp", q, "BP4") as f:
        f.write(0.0)

file_name = "results.json"
try:
    with open(file_name, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    data = {}

data[n] = data_dict

with open(file_name, "w") as f:
    json.dump(data, f)
