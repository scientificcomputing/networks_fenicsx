"""Utilities for generating :py:class:`Networkx Directional graphs<networkx.DiGraph>`"""

__all__ = ["make_tree"]

import networkx as nx

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


def make_tree(n: int, H: float, W: float, dim=3):
    """
    n : number of generations
    H : height
    W : width
    """

    # FIXME : add parameter r : branching factor of the tree (each node has r children)
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
    return G
