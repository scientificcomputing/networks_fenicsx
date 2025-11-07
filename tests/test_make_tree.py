import pytest

from networks_fenicsx import Config, NetworkMesh, network_generation


@pytest.mark.parametrize("gdim", [2, 3])
@pytest.mark.parametrize("lcar", [1.0, 0.5, 0.1])
@pytest.mark.parametrize("n", [2, 5, 7])
@pytest.mark.parametrize("H", [1, 2])
def test_make_tree(n: int, H: int, gdim: int, lcar: float):
    G = network_generation.make_tree(n=n, H=H, W=1, dim=gdim)
    config = Config()
    config.lcar = lcar
    network_mesh = NetworkMesh(G, config)

    domain = network_mesh.mesh

    tdim = domain.topology.dim
    assert tdim == 1
    assert domain.geometry.dim == gdim
    num_cells_global = domain.topology.index_map(tdim).size_global

    num_elements_per_segment = max((H / n), 1) / lcar
    num_segments = sum(2**i for i in range(n))
    assert num_cells_global == num_elements_per_segment * num_segments
    num_vertices_global = domain.topology.index_map(0).size_global
    assert (
        num_vertices_global
        == num_elements_per_segment + 1 + (num_segments - 1) * num_elements_per_segment
    )
