import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import dolfinx
import networks_fenicsx


def get_warren_bridge(
    length: float = 20.0, radius: float = 40.0, offset: float = 10.0, spacing: float = 5.0
):
    """
    length:
        length of the bridge.
    radius:
        radius of arc
    offset:
        z-axis offset of the arcs center to the base plane.
    spacing:
        distance in between vertical members
    """

    base_start = np.array([spacing / 2, 0, 0], dtype=np.float64)
    base_end = np.array([np.sqrt(radius**2 - offset**2), 0, 0], dtype=np.float64)
    base = np.linspace(base_start, base_end, 10)
    base = np.append(np.flip(-base, 0), base, axis=0)

    # TODO: arc
    top = base[:-1] + np.array([spacing / 2, 2, 0])
    # top = base[:-1] + np.sqrt(radius**2 - )

    arc_center = np.array([0, -offset, 0], dtype=np.float64)

    plt.scatter(arc_center[0], arc_center[1], marker="x")
    plt.scatter(base[:, 0], base[:, 1])
    plt.scatter(top[:, 0], top[:, 1])
    # plt.show()

    G = nx.DiGraph()
    for i in range(base.shape[0]):
        G.add_node(i, pos=base[i])
    for i in range(top.shape[0]):
        G.add_node(base.shape[0] + i, pos=top[i])

    for i in range(base.shape[0] - 1):
        G.add_edge(i, i + 1)

    for i in range(top.shape[0] - 1):
        G.add_edge(base.shape[0] + i, base.shape[0] + i + 1)

    for i in range(base.shape[0] - 1):
        G.add_edge(i, base.shape[0] + i)
        G.add_edge(base.shape[0] + i, i + 1)
    # nx.draw(G)
    # plt.show()
    return G


G = get_warren_bridge()

mesh = networks_fenicsx.mesh.NetworkMesh(G, 1)

with dolfinx.io.XDMFFile(mesh.comm, "truss.xdmf", "w") as file:
    file.write_mesh(mesh.mesh)
