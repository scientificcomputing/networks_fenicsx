# MPI compatible implementation of graph networks in FEniCSx

![formatting](https://github.com/cdaversin/networks_fenicsx/actions/workflows/check_formatting.yml/badge.svg)
![pytest](https://github.com/cdaversin/networks_fenicsx/actions/workflows/test_package.yml/badge.svg)

This repository contains a re-implementation of [GraphNics](https://github.com/IngeborgGjerde/graphnics/)
and [FEniCS-Networks](https://github.com/IngeborgGjerde/fenics-networks)
by I. Gjerde (DOI: [10.48550/arXiv.2212.02916](https://doi.org10.48550/arXiv.2212.02916)).

An initial implementation compatible with [DOLFINx](https://github.com/FEniCS/dolfinx/)
(I. Baratta et al, DOI: [10.5281/zenodo.10447665](https://doi.org/10.5281/zenodo.10447665)) with performance benchmarks presented by C. Daversin-Catty et al. in
[Finite Element Software and Performance for Network Models with Multipliers](https://doi.org/10.1007/978-3-031-58519-7_4).

However, this implementation was not **MPI compatible**.
This repository contains an **MPI compatible** implementation of graph support in DOLFINx (v0.10.0).

Please cite usage of this repository by using the [CITATION-file](./CITATION.cff).

## Installation

The easiest way to install `networks-fenicsx` on your system (given that you have DOLFINx installed) is by calling

```bash
python3 -m pip install networks-fenicsx
```

or through the git repo

```bash
python3 -m pip install git+https://github.com/scientificcomputing/networks_fenicsx.git
```

### Spack

If you want to use Spack (recommended on HPC systems), you should first check if Spack is installed on your system.
If not you can clone and activate it with:

```bash
git clone https://github.com/spack/spack.git
. spack/share/spack/setup-env.sh
```

Furthermore, you should add the [FEniCS spack repo](https://github.com/FEniCS/spack-fenics/) and [Scientific Computing spack repo](https://github.com/scientificcomputing/spack_repos) to overload the packages that exist in the main [spack-packages](https://github.com/spack/spack-packages/) registry.

```bash
spack repo add https://github.com/FEniCS/spack-fenics.git
spack repo add https://github.com/scientificcomputing/spack_repos.git
```

To install `py-networks-fenicsx` with ADIOS2 IO for DOLFINx and PETSc compiled with mumps, one can call:

```bash
spack add py-networks-fenicsx ^py-fenics-dolfinx+petsc4py ^petsc+mumps ^fenics-dolfinx+adios2
spack concretize
spack install -j 4
```
