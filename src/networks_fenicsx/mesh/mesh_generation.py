from . import mesh
import networkx as nx
from networks_fenicsx import config
import numpy as np

"""
This file is based on the graphnics project (https://arxiv.org/abs/2212.02916), https://github.com/IngeborgGjerde/fenics-networks - forked on August 2022
Copyright (C) 2022-2023 by Ingeborg Gjerde

You can freely redistribute it and/or modify it under the terms of the GNU General Public License, version 3.0, provided that the above copyright notice is kept intact and that the source code is made available under an open-source license.

Modified by Cécile Daversin-Catty - 2023
"""


def make_line_graph(n, cfg: config.Config, dim=3):
    """
    Make a graph along the unit x-axis with n nodes
    """

    G = mesh.NetworkGraph(cfg)

    dx = 1 / (n - 1)
    G.add_nodes_from(range(0, n))
    for i in range(0, n):
        if dim == 2:
            G.nodes[i]["pos"] = [i * dx, 0]
        else:
            G.nodes[i]["pos"] = [i * dx, 0, 0]

    for i in range(0, n - 1):
        G.add_edge(i, i + 1)

    G.build_mesh()
    G.build_network_submeshes()
    G.compute_tangent()

    return G


def make_Y_bifurcation(cfg: config.Config, dim=3):
    """
    Make a 3 branches network with one bifurcation
    """

    G = mesh.NetworkGraph(cfg)

    G.add_nodes_from([0, 1, 2, 3])
    if dim == 2:
        G.nodes[0]["pos"] = [0, 0]
        G.nodes[1]["pos"] = [0, 0.5]
        G.nodes[2]["pos"] = [-0.5, 1]
        G.nodes[3]["pos"] = [0.5, 1]
    else:
        G.nodes[0]["pos"] = [0, 0, 0]
        G.nodes[1]["pos"] = [0, 0.5, 0]
        G.nodes[2]["pos"] = [-0.5, 1, 0]
        G.nodes[3]["pos"] = [0.5, 1, 0]

    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(1, 3)

    G.build_mesh()
    G.build_network_submeshes()
    G.compute_tangent()

    return G


def make_double_Y_bifurcation(cfg: config.Config, dim=3):
    G = mesh.NetworkGraph(cfg)

    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7])
    if dim == 2:
        G.nodes[0]["pos"] = [0, 0]
        G.nodes[1]["pos"] = [0, 0.5]
        G.nodes[2]["pos"] = [-0.5, 1]
        G.nodes[3]["pos"] = [0.5, 1]
        G.nodes[4]["pos"] = [-0.75, 1.5]
        G.nodes[5]["pos"] = [-0.25, 1.5]
        G.nodes[6]["pos"] = [0.25, 1.5]
        G.nodes[7]["pos"] = [0.75, 1.5]
    else:
        G.nodes[0]["pos"] = [0, 0, 0]
        G.nodes[1]["pos"] = [0, 0.5, 0]
        G.nodes[2]["pos"] = [-0.5, 1, 0]
        G.nodes[3]["pos"] = [0.5, 1, 0]
        G.nodes[4]["pos"] = [-0.75, 1.5, 0]
        G.nodes[5]["pos"] = [-0.25, 1.5, 0]
        G.nodes[6]["pos"] = [0.25, 1.5, 0]
        G.nodes[7]["pos"] = [0.75, 1.5, 0]

    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(1, 3)

    G.add_edge(2, 4)
    G.add_edge(2, 5)
    G.add_edge(3, 6)
    G.add_edge(3, 7)

    G.build_mesh()
    G.build_network_submeshes()
    G.compute_tangent()

    return G


def tree_edges(n, r):
    # helper function for trees
    # yields edges in rooted tree at 0 with n nodes and branching ratio r
    if n == 0:
        return
    # Root branch
    source = 0
    target = 1
    yield source, target
    # Other branches
    nodes = iter(range(1, n))
    parents = [next(nodes)]  # stack of max length r
    while parents:
        source = parents.pop(0)
        for i in range(r):
            try:
                target = next(nodes)
                parents.append(target)
                yield source, target
            except StopIteration:
                break


def make_tree(n: int, H: float, W: float, cfg: config.Config, dim=3):
    """
    n : number of generations
    H : height
    W : width
    """

    # FIXME : add parameter r : branching factor of the tree (each node has r children)
    r = 2
    G = mesh.NetworkGraph(cfg)

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
    if dim == 2:
        G.nodes[idx]["pos"] = [0, 0]
        G.nodes[idx + 1]["pos"] = [0, y_offset]
    else:
        G.nodes[idx]["pos"] = [0, 0, 0]
        G.nodes[idx + 1]["pos"] = [0, y_offset, 0]
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
            if dim == 2:
                G.nodes[idx]["pos"] = [x, y]
            else:
                G.nodes[idx]["pos"] = [x, y, 0]
            idx = idx + 1

    edges = tree_edges(nb_nodes, r)
    for e0, e1 in list(edges):
        G.add_edge(e0, e1)

    G.build_mesh()
    G.build_network_submeshes()
    G.compute_tangent()

    return G


def make_honeycomb(n, m, cfg: config.Config, dim=3):
    # Make hexagonal mesh
    G_ = nx.hexagonal_lattice_graph(n, m)
    G_ = nx.convert_node_labels_to_integers(G_)

    G = mesh.NetworkGraph(config=cfg, graph=G_)

    # Hexagonal mesh contains edges directed in both directions
    # Remove double defined edges (keep only one direction)
    edges_to_remove = []
    for e1, e2 in G.edges():
        if e2 < e1:
            edges_to_remove.append((e1, e2))
    G.remove_edges_from(edges_to_remove)

    # Add inlet (bottom left)
    G.add_node(len(G.nodes))
    G.nodes[len(G.nodes) - 1]["pos"] = [0, -1]
    G.add_edge(len(G.nodes) - 1, 0)

    inlet_node = len(G.nodes) - 1
    outlet_node = len(G.nodes)

    # Add outlet (top right)
    pos = nx.get_node_attributes(G, "pos")
    all_coords = np.asarray(list(pos.values()))
    all_node_dist_from_origin = np.linalg.norm(all_coords, axis=1)
    furthest_node_ix = np.argmax(all_node_dist_from_origin, axis=0)
    coord_furthest_node = all_coords[furthest_node_ix, :]

    # Add new node a bit above the furthest one
    G.add_node(outlet_node)
    G.nodes[len(G.nodes) - 1]["pos"] = coord_furthest_node + np.asarray([0.7, 1])
    G.add_edge(len(G.nodes) - 1, furthest_node_ix)

    # The inlet edge might be oriented outwards, we want it inwards
    if (0, inlet_node) in G.edges():
        G.remove_edge(0, inlet_node)
        G.add_edge(inlet_node, 0)

    # The outlet edge might be oriented inwards, we want it outwards
    if (outlet_node, inlet_node - 1) in G.edges():
        G.remove_edge(outlet_node, inlet_node - 1)
        G.add_edge(inlet_node - 1, outlet_node)

    # 2D to 3D points if needed
    if dim == 3:
        for idx in range(len(G.nodes())):
            G.nodes[idx]["pos"] = [G.nodes[idx]["pos"][0], G.nodes[idx]["pos"][1], 0]

    G.build_mesh()
    G.build_network_submeshes()
    G.compute_tangent()

    return G


if __name__ == "__main__":
    make_Y_bifurcation(cfg=config.Config())
