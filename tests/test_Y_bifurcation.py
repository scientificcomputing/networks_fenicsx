import pytest

from networks_fenicsx import Config, NetworkMesh, network_generation


@pytest.mark.parametrize("gdim", [2, 3])
@pytest.mark.parametrize("lcar", [1, 0.5, 0.1])
def test_Y_bifurcation(gdim, lcar):
    # Create Y bifurcation graph
    G = network_generation.make_tree(n=2, H=1, W=1, dim=gdim)
    config = Config()
    config.lcar = lcar
    config.geometry_dim = gdim
    network_mesh = NetworkMesh(G, config)

    domain = network_mesh.mesh

    tdim = domain.topology.dim
    assert tdim == 1
    assert domain.geometry.dim == gdim
    num_cells_global = domain.topology.index_map(tdim).size_global
    assert num_cells_global == 3 * 1 / config.lcar
    num_vertices_global = domain.topology.index_map(0).size_global
    assert num_vertices_global == 3 * 1 / config.lcar + 1
