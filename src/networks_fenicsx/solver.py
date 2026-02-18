# Copyright (C) Simula Research Laboratory and Jørgen S. Dokken, Cécile Daversin-Catty
# SPDX-License-Identifier:    MIT
"""Solver interface for graphs."""

import typing

from petsc4py import PETSc

import dolfinx.fem.petsc
import dolfinx.la.petsc
from networks_fenicsx import assembly

__all__ = ["Solver"]


class Solver:
    """PETSc solver interface for the Network problems.

    Args:
        assembler: The hydraulic network assembler.
        petsc_options_prefix: Prefix for PETSc options.
        petsc_options: Dictionary of PETSc options.
        kind: Kind of PETSc matrix and vectors to create.
    """

    _ksp: PETSc.KSP | None = None  # type: ignore[name-defined]
    _A: PETSc.Mat | None = None  # type: ignore[name-defined]
    _b: PETSc.Vec | None = None  # type: ignore[name-defined]
    _x: PETSc.Vec | None = None  # type: ignore[name-defined]
    _assembler: assembly.HydraulicNetworkAssembler

    def __init__(
        self,
        assembler: assembly.HydraulicNetworkAssembler,
        petsc_options_prefix: str = "NetworkSolver_",
        petsc_options: dict | None = None,
        kind: str | typing.Sequence[typing.Sequence[str]] | None = None,
    ):
        self._assembler = assembler

        self._ksp = PETSc.KSP().create(self._assembler.network.comm)  # type: ignore[attr-defined]

        self._A = dolfinx.fem.petsc.create_matrix(self._assembler.bilinear_forms, kind=kind)
        kind = "nest" if self._A.getType() == "nest" else kind  # type: ignore[attr-defined]
        assert isinstance(kind, str) or kind is None
        self._b = dolfinx.fem.petsc.create_vector(
            dolfinx.fem.extract_function_spaces(self._assembler.linear_forms), kind=kind
        )
        self._x = dolfinx.fem.petsc.create_vector(self._assembler.function_spaces, kind)

        self.ksp.setOperators(self.A)

        # Set PETSc options
        self.ksp.setOptionsPrefix(petsc_options_prefix)
        self._A.setOptionsPrefix(f"{petsc_options_prefix}A_")
        self._b.setOptionsPrefix(f"{petsc_options_prefix}b_")

        if petsc_options is None:
            petsc_options = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "ksp_monitor": None,
                "ksp_error_if_not_converged": True,
            }
        opts = PETSc.Options()  # type: ignore[attr-defined]
        opts.prefixPush(self.ksp.getOptionsPrefix())
        for key, value in petsc_options.items():
            opts[key] = value
        self.ksp.setFromOptions()
        self._A.setFromOptions()
        self._b.setFromOptions()
        opts.prefixPop()

    @property
    def assembler(self) -> assembly.HydraulicNetworkAssembler:
        """The hydraulic network assembler."""
        return self._assembler

    @property
    def A(self) -> PETSc.Mat:  # type: ignore[name-defined]
        """System matrix."""
        return self._A

    @property
    def b(self) -> PETSc.Vec:  # type: ignore[name-defined]
        """Right-hand side vector."""
        return self._b

    def assemble(self, lhs: bool = True, rhs: bool = True):
        """Assemble the system matrix and rhs vector.

        Args:
            lhs: Whether to assemble the system matrix.
            rhs: Whether to assemble the rhs vector.
        """
        if self._A is not None and lhs:
            self._A.zeroEntries()
        if self._b is not None and rhs:
            dolfinx.la.petsc._zero_vector(self._b)
        self.assembler.assemble(self._A, self._b, assemble_lhs=lhs, assemble_rhs=rhs)

    @property
    def ksp(self) -> PETSc.KSP:  # type: ignore[name-defined]
        return self._ksp

    @dolfinx.common.timed("nxfx:Solver:solve")
    def solve(
        self, functions: list[dolfinx.fem.Function] | None = None
    ) -> list[dolfinx.fem.Function]:
        """Solve the linear system of equations and assign them to a set of corresponding
        DOLFINx functions.

        Args:
            functions: List of DOLFINx functions to assign the solution to.
                If not provided they are created based on the assembler information.
        Returns:
            The functions.
        """
        if functions is None:
            functions = []
            for i, Vi in enumerate(self.assembler.flux_spaces):
                functions.append(dolfinx.fem.Function(Vi, name=f"flux_color_{i}"))
            functions.append(dolfinx.fem.Function(self.assembler.pressure_space, name="pressure"))
            functions.append(dolfinx.fem.Function(self.assembler.lm_space, name="global_flux"))

        self.ksp.solve(self.b, self._x)
        dolfinx.la.petsc._ghost_update(
            self._x,
            insert_mode=PETSc.InsertMode.INSERT,  # type: ignore[attr-defined]
            scatter_mode=PETSc.ScatterMode.FORWARD,  # type: ignore[attr-defined]
        )
        assert isinstance(self._x, PETSc.Vec)  # type: ignore[attr-defined]
        dolfinx.fem.petsc.assign(self._x, functions)
        return functions

    def __del__(self):
        if self._ksp is not None:
            self._ksp.destroy()
        if self._b is not None:
            self._b.destroy()
        if self._x is not None:
            self._x.destroy()
