import pytest

from networks_fenicsx import NetworkMesh, network_generation


@pytest.mark.parametrize("gdim", [2, 3])
@pytest.mark.parametrize("N", [1, 4, 10])
@pytest.mark.parametrize("n", [2, 5, 7])
@pytest.mark.parametrize("H", [1, 2])
def test_make_tree(n: int, H: int, gdim: int, N: int):
    G = network_generation.make_tree(n=n, H=H, W=1, dim=gdim)
    network_mesh = NetworkMesh(G, N=N)

    domain = network_mesh.mesh

    tdim = domain.topology.dim
    assert tdim == 1
    assert domain.geometry.dim == gdim
    num_cells_global = domain.topology.index_map(tdim).size_global

    num_segments = sum(2**i for i in range(n))
    assert num_cells_global == N * num_segments
    num_vertices_global = domain.topology.index_map(0).size_global
    assert num_vertices_global == N + 1 + (num_segments - 1) * N
