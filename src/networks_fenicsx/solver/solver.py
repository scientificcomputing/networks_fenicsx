"""
This file is based on the graphnics project (https://arxiv.org/abs/2212.02916), https://github.com/IngeborgGjerde/fenics-networks - forked on August 2022
Copyright (C) 2023 by Cécile Daversin-Catty

You can freely redistribute it and/or modify it under the terms of the GNU General Public License, version 3.0, provided that the above copyright notice is kept intact and that the source code is made available under an open-source license.

"""

from mpi4py import MPI
from petsc4py import PETSc

from networks_fenicsx.mesh import mesh
from networks_fenicsx.solver import assembly
from networks_fenicsx.utils.timers import timeit
from networks_fenicsx import config


class Solver:
    def __init__(
        self,
        config: config.Config,
        network_mesh: mesh.NetworkMesh,
        assembler: assembly.Assembler,
    ):
        self.network_mesh = network_mesh
        self._ksp = PETSc.KSP().create(self.network_mesh.mesh.comm)

        self.assembler = assembler
        self.cfg = config

        if self.assembler is not None:
            self.A = assembler.assembled_matrix()
            self.b = assembler.assembled_rhs()

    @property
    def ksp(self):
        return self._ksp

    @timeit
    def solve(self) -> PETSc.Vec:
        # Configure solver
        self.ksp.setOperators(self.A)

        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")
        self.ksp.getPC().setFactorSolverType("mumps")
        self.ksp.setErrorIfNotConverged(True)

        # Solve
        x = self.A.createVecLeft()
        self.ksp.solve(self.b, x)

        return x


    def __del__(self):
        if self._ksp is not None:
            self._ksp.destroy()
        if self.b is not None:
            self.b.destroy()