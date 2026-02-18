# Copyright (C) Simula Research Laboratory, Cécile Daversin-Catty, Joe P. Dean and Jørgen S. Dokken
# SPDX-License-Identifier:    MIT
"""Assembly routine for Hydraulic network."""

import logging
import typing
from typing import Protocol

from petsc4py import PETSc

import numpy as np
import numpy.typing as npt

import basix
import dolfinx.la.petsc as _petsc_la
import ufl
from dolfinx import common, fem

from .mesh import NetworkMesh

__all__ = ["HydraulicNetworkAssembler", "PressureFunction"]


class PressureFunction(Protocol):
    def eval(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.inexact]: ...


@common.timed("nxfx:compute_integration_data")
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
    influx_color_to_bifurcations: dict[int, npt.NDArray[np.int32]] = {
        int(color): np.empty(0, dtype=np.int32) for color in range(network_mesh.num_edge_colors)
    }
    outflux_color_to_bifurcations: dict[int, npt.NDArray[np.int32]] = {
        int(color): np.empty(0, dtype=np.int32) for color in range(network_mesh.num_edge_colors)
    }
    for i, bifurcation in enumerate(network_mesh.bifurcation_values):
        for color in network_mesh.in_edges(i):
            influx_color_to_bifurcations[color] = np.append(
                influx_color_to_bifurcations[color], bifurcation
            )
        for color in network_mesh.out_edges(i):
            outflux_color_to_bifurcations[color] = np.append(
                outflux_color_to_bifurcations[color], bifurcation
            )
    # Accumulate integration data for all in-edges on the same submesh.
    in_flux_entities: dict[int, npt.NDArray[np.int32]] = {}
    out_flux_entities: dict[int, npt.NDArray[np.int32]] = {}

    for color in range(network_mesh.num_edge_colors):
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
        mesh: The network mesh
        flux_degree: The polynomial degree for the flux functions
        pressure_degree: The polynomial degree for the pressure functions
    """

    _network_mesh: NetworkMesh
    _flux_spaces: list[fem.FunctionSpace]
    _pressure_space: fem.FunctionSpace
    _lm_space: fem.FunctionSpace
    _in_idx: int  # Starting point for each influx interior bifurcation integral
    _out_idx: int  # Starting point for each outflux interior bifurcation integral
    _in_keys: tuple[int, ...]  # Set of unique markers for all influx conditions
    _out_keys: tuple[int, ...]  # Set of unique markers for all outflux conditions
    _a: list[list[fem.Form]]  # Bilinear forms
    _L: list[fem.Form]  # Linear forms

    @common.timed("nxfx:HydraulicNetworkAssembler:__init__")
    def __init__(self, mesh: NetworkMesh, flux_degree: int = 1, pressure_degree: int = 0):
        self._network_mesh = mesh
        submeshes = self._network_mesh.submeshes

        # Flux spaces on each segment, ordered by the edge list
        # Using equispaced elements to match with legacy FEniCS
        flux_element = basix.ufl.element(
            family="Lagrange",
            cell="interval",
            degree=flux_degree,
            lagrange_variant=basix.LagrangeVariant.equispaced,
        )
        Pqs = [fem.functionspace(submsh, flux_element) for submsh in submeshes]

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

    @common.timed("nxfx:HydraulicNetworkAssembler:compute_forms")
    def compute_forms(
        self,
        p_bc_ex: typing.Callable[[npt.NDArray[np.floating]], npt.NDArray[np.inexact]]
        | fem.Expression
        | ufl.core.expr.Expr,
        f: ufl.core.expr.Expr | None = None,
        R: ufl.core.expr.Expr | None = None,
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
        num_flux_spaces = self._network_mesh.num_edge_colors

        test_functions = [ufl.TestFunction(fs) for fs in self.function_spaces]
        trial_functions = [ufl.TrialFunction(fs) for fs in self.function_spaces]
        a: list[list[ufl.Form | ufl.ZeroBaseForm | None]] = [
            [ufl.ZeroBaseForm((ui, vj)) for vj in test_functions] for ui in trial_functions
        ]
        L: list[ufl.Form | ufl.ZeroBaseForm | None] = [
            ufl.ZeroBaseForm((vj,)) for vj in test_functions
        ]

        if f is None:
            f = fem.Constant(self._network_mesh.mesh, 0.0)

        if R is None:
            R = fem.Constant(self._network_mesh.mesh, 1.0)

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
        P1_e = fem.functionspace(network_mesh.mesh, ("Lagrange", 1))
        p_bc = fem.Function(P1_e)
        if isinstance(p_bc_ex, ufl.core.expr.Expr):
            try:
                expr = fem.Expression(p_bc_ex, P1_e.element.interpolation_points())  # type: ignore[operator]
            except TypeError:
                expr = fem.Expression(p_bc_ex, P1_e.element.interpolation_points)
            p_bc.interpolate(expr)
        else:
            p_bc.interpolate(p_bc_ex)

        dx_global = ufl.Measure("dx", domain=network_mesh.mesh)

        tangent = self._network_mesh.tangent
        for i, (submesh, entity_map, facet_marker) in enumerate(
            zip(
                network_mesh.submeshes,
                network_mesh.entity_maps,
                network_mesh.submesh_facet_markers,
            )
        ):
            dx_edge = ufl.Measure("dx", domain=submesh)
            ds_edge = ufl.Measure("ds", domain=submesh, subdomain_data=facet_marker)

            a[i][i] += R * qs[i] * vs[i] * dx_edge
            a[num_flux_spaces][i] += phi * ufl.dot(ufl.grad(qs[i]), tangent) * dx_edge
            a[i][num_flux_spaces] = -p * ufl.dot(ufl.grad(vs[i]), tangent) * dx_edge

            # Add all boundary contributions
            L[i] = p_bc * vs[i] * ds_edge(network_mesh.in_marker) - p_bc * vs[i] * ds_edge(
                network_mesh.out_marker
            )

        L[num_flux_spaces] += f * phi * dx_global

        # Multiplier mesh and flux share common parent mesh.
        # We create unique integration entities for each in and out branch
        # with a common ufl Measure
        # Map (bifurcation, edge) to local integration measure index
        ds = ufl.Measure(
            "ds", domain=self._network_mesh.mesh, subdomain_data=self._integration_data
        )
        for color in self._in_keys:
            a[-1][color] += mu * qs[color] * ds(self._in_idx + color)
            a[color][-1] += lmbda * vs[color] * ds(self._in_idx + color)

        for color in self._out_keys:
            a[-1][color] -= mu * qs[color] * ds(self._out_idx + color)
            a[color][-1] -= lmbda * vs[color] * ds(self._out_idx + color)

        entity_maps = [entity_map, self._network_mesh.lm_map, *self._network_mesh.entity_maps]

        # Replace remaining ZeroBaseForm with None
        # This is because `dolfinx.fem.forms._zero_form`
        # assumes single domain (no entity maps)
        for i, ai in enumerate(a):
            for j, aij in enumerate(ai):
                if isinstance(aij, ufl.ZeroBaseForm):
                    a[i][j] = None
        self._a = fem.form(
            a,  # type: ignore[arg-type]
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
            entity_maps=entity_maps,
        )
        self._L = fem.form(
            L,  # type: ignore[arg-type]
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
            entity_maps=entity_maps,
        )

    @property
    def lm_space(self) -> fem.FunctionSpace:
        """The function space of the bifurcation Lagrange multipliers"""
        return self._lm_space

    @property
    def pressure_space(self) -> fem.FunctionSpace:
        """The function space of the pressure function"""
        return self._pressure_space

    @property
    def flux_spaces(self) -> list[fem.FunctionSpace]:
        """List of function spaces for each flux function. The ith function corresponds
        to the edges of the :py:class:`networkx.DiGraph` that were colored with color `i`."""
        return self._flux_spaces

    @property
    def function_spaces(self) -> list[fem.FunctionSpace]:
        """List of all function-spaces in the order `[flux, pressure, lm]`,
        the same order as used by the :py:meth:`assemble`"""
        return [*self._flux_spaces, self._pressure_space, self._lm_space]

    @property
    def network(self) -> NetworkMesh:
        """Return the underlying network mesh."""
        return self._network_mesh

    @common.timed("nxfx:HydraulicNetworkAssembler:assemble")
    def assemble(
        self,
        A: PETSc.Mat | None = None,  # type: ignore[name-defined]
        b: PETSc.Mat | None = None,  # type: ignore[name-defined]
        assemble_lhs: bool = True,
        assemble_rhs: bool = True,
        kind: str | typing.Sequence[typing.Sequence[str]] | None = None,
    ) -> tuple[PETSc.Mat, PETSc.Vec]:  # type: ignore[name-defined]
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
                A = fem.petsc.create_matrix([[aij for aij in ai] for ai in self._a], kind=kind)
            A = fem.petsc.assemble_matrix(A, self._a, bcs=[])  # type: ignore
            A.assemble()
            kind = "nest" if A.getType() == PETSc.Mat.Type.NEST else kind  # type: ignore[attr-defined]
        if assemble_rhs:
            if b is None:
                assert isinstance(kind, str) or kind is None
                b = fem.petsc.create_vector(fem.extract_function_spaces(self._L), kind=kind)
            b = fem.petsc.assemble_vector(b, self._L)  # type: ignore
            _petsc_la._ghost_update(
                b,
                insert_mode=PETSc.InsertMode.ADD_VALUES,  # type: ignore[attr-defined]
                scatter_mode=PETSc.ScatterMode.REVERSE,  # type: ignore[attr-defined]
            )
        return (A, b)

    @property
    def bilinear_forms(self) -> typing.Sequence[typing.Sequence[fem.Form]]:
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
    def linear_forms(self) -> typing.Sequence[fem.Form]:
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
