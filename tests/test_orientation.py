import networkx as nx
import numpy as np
import pytest

import dolfinx
import ufl
from networks_fenicsx.mesh import NetworkMesh


def linear_graph(n: int, dim: int = 2, ordered=lambda _: True) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for i in range(n - 1):
        if ordered(i):
            G.add_edge(i, i + 1)
        else:
            G.add_edge(i + 1, i)

    for i in range(n):
        pos = np.zeros(dim)
        pos[0] = i / (n - 1)
        G.nodes[i]["pos"] = pos

    return G


@pytest.mark.parametrize("n", [30])
@pytest.mark.parametrize("order", ["in", "reverse", "alternating"])
@pytest.mark.parametrize("N", [1, 4, 8])
def test_orientation(n: int, order: str, N: int) -> None:
    if order == "in":
        ordered = lambda _: True
    elif order == "reverse":
        ordered = lambda _: False
    elif order == "alternating":
        ordered = lambda k: k % 2
    else:
        raise RuntimeError()

    G = linear_graph(n, ordered=ordered)
    # import matplotlib.pyplot as plt
    # nx.draw(G)
    # plt.show()

    network_mesh = NetworkMesh(G, N=N)

    J = ufl.Jacobian(network_mesh.mesh)
    t = J[:, 0]
    t /= ufl.sqrt(ufl.inner(t, t))
    f = dolfinx.fem.form(ufl.inner(ufl.as_vector((1, 0)), t) * network_mesh.orientation * ufl.dx)

    val = network_mesh.comm.allreduce(dolfinx.fem.assemble_scalar(f))

    if order == "in":
        assert np.isclose(val, 1.0)
    elif order == "reverse":
        assert np.isclose(val, -1.0)
    else:
        edge_count = n - 1
        assert np.isclose(val, edge_count % 2 * -1 / edge_count)
