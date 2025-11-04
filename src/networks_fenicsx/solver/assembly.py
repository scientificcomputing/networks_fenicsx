"""
This file is based on the graphnics project (https://arxiv.org/abs/2212.02916), https://github.com/IngeborgGjerde/fenics-networks - forked on August 2022
Copyright (C) 2022-2023 by Ingeborg Gjerde

You can freely redistribute it and/or modify it under the terms of the GNU General Public License, version 3.0, provided that the above copyright notice is kept intact and that the source code is made available under an open-source license.

Modified by Cécile Daversin-Catty - 2023
Modified by Joseph P. Dean - 2023
Modified by Jørgen S. Dokken - 2025
"""

from operator import add
from dolfinx import fem, mesh as _mesh
import ufl


# from mpi4py import MPI
import basix
from petsc4py import PETSc
import logging
import numpy as np
import numpy.typing as npt
from networks_fenicsx.mesh import mesh
from networks_fenicsx.utils.timers import timeit
from networks_fenicsx import config

__all__ = ["Assembler"]


def flux_term(
    q: ufl.core.expr.Expr,
    facet_marker: _mesh.MeshTags | list[tuple[int, npt.NDArray[np.int32]]],
    tag: int,
) -> ufl.Form:
    ds = ufl.Measure(
        "ds",
        domain=q.ufl_function_space().mesh,
        subdomain_data=facet_marker,
        subdomain_id=tag,
    )
    return q * ds


@timeit
def compute_integration_data(
    network_mesh: mesh.NetworkMesh,
) -> tuple[dict[int, npt.NDArray[np.int32]], dict[int, npt.NDArray[np.int32]]]:
    """Given a network mesh, compute integration entities for the "parent" network mesh
    for each bifuraction in the mesh.

    Args:
        network_mesh: The network mesh

    Returns:
        A tuple `(in_entities, out_entities) mapping integration entities on each edge of the network
        (marked by color) to its integration entities on the parent mesh.

    """

    # Pack all bifurcation in and out fluxes per colored edge in graph
    influx_color_to_bifurcations = {
        int(color): [] for color in range(network_mesh._num_edge_colors)
    }
    outflux_color_to_bifurcations = {
        int(color): [] for color in range(network_mesh._num_edge_colors)
    }
    for bifurcation in network_mesh.bifurcation_values:
        for color in network_mesh.in_edges(bifurcation):
            influx_color_to_bifurcations[color].append(int(bifurcation))
        for color in network_mesh.out_edges(bifurcation):
            outflux_color_to_bifurcations[color].append(int(bifurcation))
    # Accumulate integration data for all in-edges on the same submesh.
    in_flux_entities: dict[int, npt.NDArray[np.int32]] = {}
    out_flux_entities: dict[int, npt.NDArray[np.int32]] = {}

    for color in range(network_mesh._num_edge_colors):
        sm = network_mesh.submeshes[color]
        smfm = network_mesh.submesh_facet_markers[color]
        sm.topology.create_connectivity(sm.topology.dim - 1, sm.topology.dim)

        # Compute influx entities
        submesh_influx_entities = fem.compute_integration_domains(
            fem.IntegralType.exterior_facet,
            sm.topology,
            smfm.indices[np.isin(smfm.values, influx_color_to_bifurcations[color])],
        ).reshape(-1, 2)
        parent_to_sub_influx = network_mesh.entity_maps[color].sub_topology_to_topology(
            submesh_influx_entities[:, 0].copy(), inverse=False
        )
        submesh_influx_entities[:, 0] = parent_to_sub_influx
        in_flux_entities[color] = submesh_influx_entities.flatten()

        # Compute influx entities
        submesh_outflux_entities = fem.compute_integration_domains(
            fem.IntegralType.exterior_facet,
            sm.topology,
            smfm.indices[np.isin(smfm.values, outflux_color_to_bifurcations[color])],
        ).reshape(-1, 2)
        parent_to_sub_outflux = network_mesh.entity_maps[color].sub_topology_to_topology(
            submesh_outflux_entities[:, 0].copy(), inverse=False
        )
        submesh_outflux_entities[:, 0] = parent_to_sub_outflux
        out_flux_entities[color] = submesh_outflux_entities.flatten()
    return in_flux_entities, out_flux_entities


class Assembler:
    _network_mesh: mesh.NetworkMesh
    _flux_spaces: list[fem.FunctionSpace]
    _pressure_space: fem.FunctionSpace
    _lm_space: fem.FunctionSpace
    _cfg = config.Config
    _in_idx: int  # Starting point for each influx interior bifurcation integral
    _out_idx: int  # Starting point for each outflux interior bifurcation integral
    _in_keys: tuple[int]  # Set of unique markers for all influx conditions
    _out_keys: tuple[int]  # Set of unique markers for all outflux conditions

    def __init__(self, config: config.Config, mesh: mesh.NetworkMesh):
        self._network_mesh = mesh
        self._cfg = config
        self._A = None
        self._b = None
        submeshes = self._network_mesh.submeshes

        # Flux spaces on each segment, ordered by the edge list
        # Using equispaced elements to match with legacy FEniCS
        flux_degree = self.cfg.flux_degree
        flux_element = basix.ufl.element(
            family="Lagrange",
            cell="interval",
            degree=flux_degree,
            lagrange_variant=basix.LagrangeVariant.equispaced,
        )
        Pqs = [fem.functionspace(submsh, flux_element) for submsh in submeshes]

        pressure_degree = self.cfg.pressure_degree
        if pressure_degree == 0:
            discontinuous = True
        else:
            discontinuous = False
        pressure_element = basix.ufl.element(
            family="Lagrange",
            cell="interval",
            degree=pressure_degree,
            lagrange_variant=basix.LagrangeVariant.equispaced,
            discontinuous=discontinuous,
        )
        Pp = fem.functionspace(self._network_mesh.mesh, pressure_element)

        self._flux_spaces = Pqs
        self._pressure_space = Pp
        self._lm_space = fem.functionspace(self._network_mesh.lm_mesh, ("DG", 0))

        # Initialize forms
        num_qs = self._network_mesh._num_edge_colors
        num_blocks = num_qs + 2
        self.a = [[None] * num_blocks for _ in range(num_blocks)]
        self.L = [None] * num_blocks

        # Compute integration data for network mesh
        self._integration_data = []
        self._in_idx = max(mesh.in_marker, mesh.out_marker) + 1
        in_flux_entities, out_flux_entities = compute_integration_data(self._network_mesh)
        self._in_keys = tuple(in_flux_entities.keys())
        self._out_keys = tuple(out_flux_entities.keys())
        for color in self._in_keys:
            self._integration_data.append((self._in_idx + color, in_flux_entities[color]))
        self._out_idx = self._in_idx + len(out_flux_entities)
        for color in self._out_keys:
            self._integration_data.append((self._out_idx + color, out_flux_entities[color]))

    @property
    def cfg(self):
        return self._cfg

    def dds(self, f):
        """
        function for derivative df/ds along graph
        """
        return ufl.dot(ufl.grad(f), self._network_mesh.tangent)

    @timeit
    def compute_forms(
        self,
        f=None,
        p_bc_ex=None,
        jit_options: dict | None = None,
        form_compiler_options: dict | None = None,
    ):
        """
        Compute forms for hydraulic network model
            R q + d/ds p = 0
            d/ds q = f
        on graph G, with bifurcation condition q_in = q_out
        and jump vectors the bifurcation conditions

        Args:
           f (dolfinx.fem.function): source term
           p_bc (class): neumann bc for pressure
        """

        if f is None:
            f = fem.Constant(self._network_mesh.mesh, 0.0)
        # Fluxes on each branch
        qs = []
        vs = []
        for Pq in self._flux_spaces:
            qs.append(ufl.TrialFunction(Pq))
            vs.append(ufl.TestFunction(Pq))

        # Lagrange multipliers
        lmbda = ufl.TrialFunction(self.lm_space)
        mu = ufl.TestFunction(self.lm_space)

        # Pressure
        p = ufl.TrialFunction(self._pressure_space)
        phi = ufl.TestFunction(self._pressure_space)
        # Assemble variational formulation
        network_mesh = self._network_mesh

        # Assemble edge contributions to a and L
        num_qs = len(self._network_mesh.submeshes)
        P1_e = fem.functionspace(network_mesh.mesh, ("Lagrange", 1))
        p_bc = fem.Function(P1_e)
        p_bc.interpolate(p_bc_ex.eval)
        for i, (submesh, entity_map, facet_marker) in enumerate(
            zip(
                network_mesh.submeshes,
                network_mesh.entity_maps,
                network_mesh.submesh_facet_markers,
            )
        ):
            dx_edge = ufl.Measure("dx", domain=submesh)
            ds_edge = ufl.Measure("ds", domain=submesh, subdomain_data=facet_marker)

            self.a[i][i] = fem.form(
                qs[i] * vs[i] * dx_edge,
                jit_options=jit_options,
                form_compiler_options=form_compiler_options,
            )
            self.a[num_qs][i] = fem.form(
                phi * self.dds(qs[i]) * dx_edge,
                entity_maps=[entity_map],
                jit_options=jit_options,
                form_compiler_options=form_compiler_options,
            )
            self.a[i][num_qs] = fem.form(
                -p * self.dds(vs[i]) * dx_edge,
                entity_maps=[entity_map],
                jit_options=jit_options,
                form_compiler_options=form_compiler_options,
            )

            # Add all boundary contributions
            self.L[i] = fem.form(
                p_bc * vs[i] * ds_edge(network_mesh.in_marker)
                - p_bc * vs[i] * ds_edge(network_mesh.out_marker),
                jit_options=jit_options,
                form_compiler_options=form_compiler_options,
                entity_maps=[entity_map],
            )
        # Multiplier mesh and flux share common parent mesh.
        # We create unique integration entities for each in and out branch
        # with a common ufl Measure
        # Map (bifurcation, edge) to local integration measure index

        for edge in range(self._network_mesh._num_edge_colors):
            self.a[edge][-1] = ufl.ZeroBaseForm((lmbda, vs[edge]))
            self.a[-1][edge] = ufl.ZeroBaseForm((mu, qs[edge]))

        ds = ufl.Measure(
            "ds", domain=self._network_mesh.mesh, subdomain_data=self._integration_data
        )
        for color in self._in_keys:
            self.a[-1][color] += mu * qs[color] * ds(self._in_idx + color)
            self.a[color][-1] += lmbda * vs[color] * ds(self._in_idx + color)

        for color in self._out_keys:
            self.a[-1][color] -= mu * qs[color] * ds(self._out_idx + color)
            self.a[color][-1] -= lmbda * vs[color] * ds(self._out_idx + color)

        self.L[-1] = fem.form(ufl.ZeroBaseForm((mu,)))
        entity_maps = [self._network_mesh.lm_map, *self._network_mesh.entity_maps]
        for i in range(self._network_mesh._num_edge_colors):
            self.a[i][-1] = fem.form(
                self.a[i][-1],
                entity_maps=entity_maps,
                jit_options=jit_options,
                form_compiler_options=form_compiler_options,
            )
            self.a[-1][i] = fem.form(
                self.a[-1][i],
                entity_maps=entity_maps,
                jit_options=jit_options,
                form_compiler_options=form_compiler_options,
            )
   
        # Add zero to uninitialized diagonal blocks (needed by petsc)
        self.a[num_qs][num_qs] = fem.form(
            ufl.ZeroBaseForm((p, phi)),
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
        )
        self.L[num_qs] = fem.form(
            ufl.ZeroBaseForm((phi,)),
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
        )

    @property
    def lm_space(self):
        return self._lm_space

    @property
    def pressure_space(self):
        return self._pressure_space

    @property
    def flux_spaces(self):
        return self._flux_spaces

    @property
    def function_spaces(self):
        return [*self._flux_spaces, self._pressure_space, self._lm_space]

    @timeit
    def assemble(
        self, A: PETSc.Mat | None = None, b: PETSc.Mat | None = None
    ) -> tuple[PETSc.Mat, PETSc.Vec]:
        """Assemble system matrix and rhs vector

        Args:
            A: PETSc matrix to assemble `self.a` into. This matrix is just a subset of the system
               if using the old Lagrange-multiplier approach
            b: PETSc vector to assemble `self.L` into. This vector is just a subset of the system
               if using the old Lagrange-multiplier approach
        """
        if A is None:
            A = fem.petsc.create_matrix(self.a)
        A = fem.petsc.assemble_matrix(A, self.a)
        A.assemble()
        if b is None:
            b = fem.petsc.create_vector(fem.extract_function_spaces(self.L))
        b = fem.petsc.assemble_vector(self.L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return (A, b)
      

    def bilinear_forms(self):
        if self.a is None:
            logging.error("Bilinear forms haven't been computed. Need to call compute_forms()")
        else:
            return self.a

    def bilinear_form(self, i: int, j: int):
        a = self.bilinear_forms()
        if i > len(a) or j > len(a[i]):
            logging.error("Bilinear form a[" + str(i) + "][" + str(j) + "] out of range")
        return a[i][j]

    def linear_forms(self):
        if self.L is None:
            logging.error("Linear forms haven't been computed. Need to call compute_forms()")
        else:
            return self.L

    def linear_form(self, i: int):
        L = self.linear_forms()
        if i > len(L):
            logging.error("Linear form L[" + str(i) + "] out of range")
        return L[i]
