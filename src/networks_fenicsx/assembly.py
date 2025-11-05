# Copyright (C) Simula Research Laboratory, Cécile Daversin-Catty, Joe P. Dean and Jørgen S. Dokken
# SPDX-License-Identifier:    MIT
"""Assembly routine for Hydraulic network."""

from petsc4py import PETSc
from typing import Protocol
from dolfinx import fem, mesh as _mesh
import dolfinx.la.petsc as _petsc_la
import ufl

import typing
import basix
import logging
import numpy as np
import numpy.typing as npt
from .mesh import NetworkMesh
from .timers import timeit
from networks_fenicsx import config

__all__ = ["HydraulicNetworkAssembler", "PressureFunction"]


class PressureFunction(Protocol):
    def eval(x: npt.NDArray[np.floating]) -> npt.NDArray[np.inexact]: ...


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
    network_mesh: NetworkMesh,
) -> tuple[dict[int, npt.NDArray[np.int32]], dict[int, npt.NDArray[np.int32]]]:
    """Given a network mesh, compute integration entities for the "parent" network mesh
    for each bifuraction in the mesh.

    Args:
        network_mesh: The network mesh

    Returns:
        A tuple `(in_entities, out_entities) mapping integration entities
        on each edge of the network (marked by color) to its integration
        entities on the parent mesh.
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


class HydraulicNetworkAssembler:
    """
    Assembler for the variational formulation of an hydraulic network.

    .. math::
        R q + \\frac{\\mathrm{d}}{\\mathrm{d}s} p = 0 \\\\
        \\frac{\\mathrm{d}}{\\mathrm{d}s} q  = f

    Args:
        config: The configuration file, selecting the degree of flux and
            pressure spaces.
        mesh: The network mesh
    """

    _network_mesh: NetworkMesh
    _flux_spaces: list[fem.FunctionSpace]
    _pressure_space: fem.FunctionSpace
    _lm_space: fem.FunctionSpace
    _cfg = config.Config
    _in_idx: int  # Starting point for each influx interior bifurcation integral
    _out_idx: int  # Starting point for each outflux interior bifurcation integral
    _in_keys: tuple[int]  # Set of unique markers for all influx conditions
    _out_keys: tuple[int]  # Set of unique markers for all outflux conditions
    _a: list[list[fem.Form | None]]  # Bilinear forms
    _L: list[fem.Form | None]  # Linear forms

    def __init__(self, config: config.Config, mesh: NetworkMesh):
        self._network_mesh = mesh
        self._cfg = config
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
        self._a = [[None] * num_blocks for _ in range(num_blocks)]
        self._L = [None] * num_blocks

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
        p_bc_ex: PressureFunction,
        f: ufl.core.expr.Expr | None = None,
        jit_options: dict | None = None,
        form_compiler_options: dict | None = None,
    ):
        """
        Compute forms for hydraulic network model

        .. math::
            R q + \\frac{\\mathrm{d}}{\\mathrm{d}s} p = 0\\\\
            \\frac{\\mathrm{d}}{\\mathrm{d}s} q  = f


        on graph G, with bifurcation condition q_in = q_out
        and jump vectors the bifurcation conditions

        Args:
           f: source term
           p_bc: neumann bc for pressure
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

            self._a[i][i] = fem.form(
                qs[i] * vs[i] * dx_edge,
                jit_options=jit_options,
                form_compiler_options=form_compiler_options,
            )
            self._a[num_qs][i] = fem.form(
                phi * self.dds(qs[i]) * dx_edge,
                entity_maps=[entity_map],
                jit_options=jit_options,
                form_compiler_options=form_compiler_options,
            )
            self._a[i][num_qs] = fem.form(
                -p * self.dds(vs[i]) * dx_edge,
                entity_maps=[entity_map],
                jit_options=jit_options,
                form_compiler_options=form_compiler_options,
            )

            # Add all boundary contributions
            self._L[i] = fem.form(
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
            self._a[edge][-1] = ufl.ZeroBaseForm((lmbda, vs[edge]))
            self._a[-1][edge] = ufl.ZeroBaseForm((mu, qs[edge]))

        ds = ufl.Measure(
            "ds", domain=self._network_mesh.mesh, subdomain_data=self._integration_data
        )
        for color in self._in_keys:
            self._a[-1][color] += mu * qs[color] * ds(self._in_idx + color)
            self._a[color][-1] += lmbda * vs[color] * ds(self._in_idx + color)

        for color in self._out_keys:
            self._a[-1][color] -= mu * qs[color] * ds(self._out_idx + color)
            self._a[color][-1] -= lmbda * vs[color] * ds(self._out_idx + color)

        self._L[-1] = fem.form(ufl.ZeroBaseForm((mu,)))
        entity_maps = [self._network_mesh.lm_map, *self._network_mesh.entity_maps]
        for i in range(self._network_mesh._num_edge_colors):
            self._a[i][-1] = fem.form(
                self._a[i][-1],
                entity_maps=entity_maps,
                jit_options=jit_options,
                form_compiler_options=form_compiler_options,
            )
            self._a[-1][i] = fem.form(
                self._a[-1][i],
                entity_maps=entity_maps,
                jit_options=jit_options,
                form_compiler_options=form_compiler_options,
            )

        # Add zero to uninitialized diagonal blocks (needed by petsc)
        self._a[num_qs][num_qs] = fem.form(
            ufl.ZeroBaseForm((p, phi)),
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
        )
        self._L[num_qs] = fem.form(
            ufl.ZeroBaseForm((phi,)),
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
        )

    @property
    def lm_space(self) -> fem.FunctionSpace:
        return self._lm_space

    @property
    def pressure_space(self) -> fem.FunctionSpace:
        return self._pressure_space

    @property
    def flux_spaces(self) -> list[fem.FunctionSpace]:
        return self._flux_spaces

    @property
    def function_spaces(self) -> list[fem.FunctionSpace]:
        return [*self._flux_spaces, self._pressure_space, self._lm_space]

    @property
    def network(self) -> NetworkMesh:
        return self._network_mesh

    @timeit
    def assemble(
        self,
        A: PETSc.Mat | None = None,
        b: PETSc.Mat | None = None,
        assemble_lhs: bool = True,
        assemble_rhs: bool = True,
        kind: str | typing.Sequence[typing.Sequence[str]] | None = None,
    ) -> tuple[PETSc.Mat, PETSc.Vec]:
        """Assemble system matrix and rhs vector.

        Note:
            If neither `A` or `b` is provided, they are created inside this class.

        Args:
            A: :py:class:`PETSc matrix<petsc4py.PETSc.Mat>` to assemble
                :py:attr:`HydraulicNetworkAssembler.bilinear_forms` into.
            b: :py:class:`PETSc vector<petsc4py.PETSc.Vec>` to assemble
                :py:attr:`HydraulicNetworkAssembler.linear_forms` into.
            assemble_lhs: Whether to assemble the system matrix.
            assemble_rhs: Whether to assemble the rhs vector.
            kind: If no matrix or vector is provided, "kind" is used to determine what
                kind of matrix/vector to create.
        """
        if assemble_lhs:
            if A is None:
                A = fem.petsc.create_matrix(self._a, kind=kind)
            A = fem.petsc.assemble_matrix(A, self._a, bcs=[])
            A.assemble()
            kind = "nest" if A.getType() == PETSc.Mat.Type.NEST else kind  # type: ignore[attr-defined]
        if assemble_rhs:
            if b is None:
                b = fem.petsc.create_vector(fem.extract_function_spaces(self._L), kind=kind)
            b = fem.petsc.assemble_vector(b, self._L)
            _petsc_la._ghost_update(
                b, insert_mode=PETSc.InsertMode.ADD_VALUES, scatter_mode=PETSc.ScatterMode.REVERSE
            )
        return (A, b)

    @property
    def bilinear_forms(self) -> list[list[fem.Form]]:
        """Nested list of the compiled, bilinear forms."""
        if self._a is None:
            logging.error("Bilinear forms haven't been computed. Need to call compute_forms()")
        else:
            return self._a

    def bilinear_form(self, i: int, j: int) -> fem.Form:
        """Extract the i,j bilinear form."""
        a = self.bilinear_forms
        if i > len(a) or j > len(a[i]):
            logging.error("Bilinear form a[" + str(i) + "][" + str(j) + "] out of range")
        return a[i][j]

    @property
    def linear_forms(self) -> list[fem.Form]:
        """Extract the linear form."""
        if self._L is None:
            logging.error("Linear forms haven't been computed. Need to call compute_forms()")
        else:
            return self._L

    def linear_form(self, i: int) -> fem.Form:
        """Return the i-th block of the linear form"""
        L = self.linear_forms
        if i > len(L):
            logging.error("Linear form L[" + str(i) + "] out of range")
        return L[i]
