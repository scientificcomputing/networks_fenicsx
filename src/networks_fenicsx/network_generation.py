# Copyright (C) Alexandra Vallet, Simula Research Laboratory, Cécile Daversin-Catty
# and Jørgen S. Dokken
# SPDX-License-Identifier:    MIT

"""Utilities for generating :py:class:`Networkx Directional graphs<networkx.DiGraph>`"""

__all__ = ["make_tree", "make_arterial_tree"]

import networkx as nx
import numpy as np
import numpy.typing as npt
from typing import Callable


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


def _default_normal(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Surface plane normal orientation for the xy-plane"""
    output = np.zeros_like(x)
    output[2] = 1
    return output


def _project_onto_plane(x, n):
    """Project `x` onto plane with normal `n`"""
    d = np.dot(x, n) / np.linalg.norm(n)
    p = d * n / np.linalg.norm(n)
    return x - p


def _rotate_in_plane(
    x: npt.NDArray[np.floating], axis: npt.NDArray[np.floating], angle: float
) -> npt.NDArray[np.floating]:
    """Use Rodrigues formula to rotate a vector in space given an axis and
    angle of rotation. Ref: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    theta = np.radians(angle)
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return np.dot(R, x)


def _compute_vessel_endpoint(previousvessel, surfacenormal, angle, length):
    """From a previous vessel defined in 3D,
    the brain surface at the end of the previous vessel, angle and length :
    compute the coordinate of the end node of the current vessel"""
    # project the previous vessel in the current plane
    # and get the direction vector of the previous vessel
    pm1 = previousvessel[0]
    p0 = previousvessel[1]
    vector_previous = p0 - pm1
    previousdir = _project_onto_plane(vector_previous, surfacenormal)
    # compute the direction vector of the new vessel with the angle
    newdir = _rotate_in_plane(previousdir, surfacenormal, angle)
    # compute the location of the end of the new vessel
    return _translate(p0, newdir, length)


def _translate(
    p0: npt.NDArray[np.floating], direction: npt.NDArray[np.floating], length: float
) -> npt.NDArray[np.floating]:
    """Compute `p1 = p0 + length * direction/|direction|`"""
    # compute the location of the end of the new vessel
    assert len(p0) == len(direction)
    return p0 + length * direction / np.linalg.norm(direction, axis=-1)


def make_arterial_tree(
    N: int,
    p0: np.ndarray[tuple[float, float, float], np.dtype[np.floating]] = np.zeros(
        3, dtype=np.float64
    ),
    direction: np.ndarray[tuple[float, float, float], np.dtype[np.floating]] = np.array(
        [0, 1, 0], dtype=np.float64
    ),
    D0: float = 2.0,
    lmbda: float = 8.0,
    gamma: float = 0.8,
    normal: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]] = _default_normal,
    random: bool = False,
) -> nx.DiGraph:
    """Create an arterial tree.

    Murray's law
    We consider that the sum of power 3 of daughter vessels radii equal
    to the power three of the parent vessel radius :math:`D0^3=D1^3 + D2^3`
    We consider that the ratio between two daughter vessels radii is gam
    :math:`D2/D1=gam`.
    Then we have
    
    .. math::
        D2=D0(\\gammma^3+1)^(-1/3)\\\\
        D1=\\gamma D2

    We consider that the length `L` of a vessel segment in a given network is
    related to its diameter `d` via :math:`L = \\lambda d`,
    where the positive constant :math:`\\lambda` is network-specific.

    The angle of bifurcation can be expressed based on blood volume conservation
    and minimum energy hypothesis
    FUNG, Y. (1997). Biomechanics: Circulation. 2nd edition. Heidelberg: Springer.
    HUMPHREY, J. and DELANGE, S. (2004). An Introduction to Biomechanics. Heidelberg: Springer.
    VOGEL, S. (1992). Vital Circuits. Oxford: Oxford University Press.
    See here for derivation : https://www.montogue.com/blog/murrays-law-and-arterial-bifurcations/

    Note:
        Code based on the repository: https://gitlab.com/ValletAlexandra/NetworkGen
        by Alexandra Vallet under MIT License.
    
    Args:
        N: Number of generations of vessels
        p0: Origin location
        direction: Initial direction
        D0: First vessel diameter
        lmbda: Network specific constant relating vessel segment length to diameter `d`.
        gamma: Ratio between two daughter vessels.
        normal: Function that computes the normal to the plane at any point (x,y,z)
        random: If True set randomly which vessel goes right or left. If False,
            the biggest vessel is always the second one
    """
    if gamma > 1:
        raise ValueError("Please choose a gamma lower or equal to 1")

    # Create directed graph
    G = nx.DiGraph()

    G.add_edge(0, 1)

    # Set initial vessel
    nx.set_node_attributes(G, p0, "pos")
    nx.set_edge_attributes(G, D0 / 2, "radius")
    L = D0 * lmbda
    G.nodes[1]["pos"] = _translate(p0, direction, L)

    inode = 1

    # List of vessels from the previous generation
    previous_edges = [(0, 1)]
    previous_vessel = np.empty((2, 3), dtype=p0.dtype)
    for _ in range(1, N):
        current_edges = []
        for e in previous_edges:
            # Parent vessel properties
            previous_vessel[0, :] = G.nodes[e[0]]["pos"]
            previous_vessel[1, :] = G.nodes[e[1]]["pos"]
            D0 = G.edges[e]["radius"] * 2

            # Daughter diameters
            D2 = D0 * (gamma**3 + 1) ** (-1 / 3)
            D1 = gamma * D2
            # Daughter lenghts
            L2 = lmbda * D2
            L1 = lmbda * D1
            # Bifurcation angles
            # angle for the smallest vessel
            cos1 = (D0**4 + D1**4 - (D0**3 - D1**3) ** (4 / 3)) / (2 * D0**2 * D1**2)
            angle1 = np.degrees(np.arccos(cos1))
            # angle for the biggest vessel
            cos2 = (D0**4 + D2**4 - (D0**3 - D2**3) ** (4 / 3)) / (2 * D0**2 * D2**2)
            angle2 = np.degrees(np.arccos(cos2))
            sign1 = 1 if not random else np.rand.choice([-1, 1])
            sign2 = -sign1

            # Add first daughter vessel
            inode += 1
            new_edge = (e[1], inode)
            G.add_edge(*new_edge)

            # Compute endpoint of vessel
            G.nodes[inode]["pos"] = _compute_vessel_endpoint(
                previous_vessel, normal(previous_vessel[1]), sign1 * angle1, L1
            )

            # Set radius
            G.edges[new_edge]["radius"] = D1 / 2

            # Add to the pool of vessels for this generation
            current_edges.append(new_edge)

            inode += 1
            new_edge = (e[1], inode)
            G.add_edge(*new_edge)

            # Set the location according to length and angle
            G.nodes[inode]["pos"] = _compute_vessel_endpoint(
                previous_vessel, normal(previous_vessel[1]), sign2 * angle2, L2
            )

            # Set radius
            G.edges[new_edge]["radius"] = D2 / 2

            # Add to the pool of vessels for this generation
            current_edges.append(new_edge)
        previous_edges = current_edges
    return G
