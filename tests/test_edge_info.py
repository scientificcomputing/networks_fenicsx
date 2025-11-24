import networkx as nx
import numpy as np
import pytest

from networks_fenicsx import NetworkMesh


@pytest.mark.parametrize("N", [10, 50])
def test_edge_info(N: int):
    # Create manual graph where we have five bifurcations.
    # One inlet (0) -> (1) -> (7).
    G = nx.DiGraph()
    G.add_node(0, pos=np.zeros(3))
    G.add_node(1, pos=np.array([0.0, 0.0, 1.0]))
    G.add_node(2, pos=np.array([0.2, 0.2, 2.0]))
    G.add_node(3, pos=np.array([-0.2, 0.3, 2.0]))
    G.add_node(4, pos=np.array([0.0, 0.1, 2.1]))
    G.add_node(5, pos=np.array([0.1, -0.1, 3.0]))
    G.add_node(6, pos=np.array([-0.3, 0.4, 4.0]))
    G.add_node(7, pos=1.1 * G.nodes[1]["pos"])
    G.add_edge(0, 1)
    G.add_edge(1, 7)
    # Split into three bifurcations (7)->(2), (7)->(3), (7)->(4)
    G.add_edge(7, 2)
    # First branch goes right to gathering
    G.add_edge(2, 5)
    # Second branch goes through an extra point(3), before meeting at point (4)
    G.add_edge(7, 3)
    G.add_edge(3, 4)
    G.add_edge(4, 5)
    # Last branch goes directly to gathering
    G.add_edge(7, 4)
    G.add_edge(5, 6)

    network_mesh = NetworkMesh(G, N=N)
    assert len(network_mesh.bifurcation_values) == 6
    # Bifurcation values are sorted in increasing order
    np.testing.assert_allclose([1, 2, 3, 4, 5, 7], network_mesh.bifurcation_values)
    assert len(network_mesh.in_edges(0)) == 1
    assert len(network_mesh.out_edges(0)) == 1

    assert len(network_mesh.in_edges(1)) == 1
    assert len(network_mesh.out_edges(1)) == 1

    assert len(network_mesh.in_edges(2)) == 1
    assert len(network_mesh.out_edges(2)) == 1

    assert len(network_mesh.in_edges(3)) == 2
    assert len(network_mesh.out_edges(3)) == 1

    assert len(network_mesh.in_edges(4)) == 2
    assert len(network_mesh.out_edges(4)) == 1

    assert len(network_mesh.in_edges(5)) == 1
    assert len(network_mesh.out_edges(5)) == 3
