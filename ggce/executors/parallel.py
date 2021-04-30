#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & John Sous"
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com"

import numpy as np
from scipy import linalg
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import time

from ggce.executors.base import BaseExecutor
from ggce.executors.serial import SerialSparseExecutor
from ggce.engine.physics import G0_k_omega

import petsc4py
from petsc4py import PETSc

BYTES_TO_MB = 1048576


class ParallelSparseExecutor(BaseExecutor):
    ''' A class to connect to PETSc powerful parallel sparse solver
        tools, to calculate G(k,w) in parallel.'''

    def _init_petsc(self, comm):

        ''' Quick helper function inserts the MPI communicator
            into the class scope and calculates the given process's rank.'''

        self.comm = comm
        self.rank = comm.getRank()

    def prime(self, comm):

        '''
        Prepare the executor for running by finding the
        system of equations and basis.

        Parameters
        ----------
        comm : MPI_COMM_WORLD
                Needed by PETSc to create the parallel matrices and vectors.
                Pass specifically PETSc's COMM version for optimal results.
        Returns
        -------

        '''

        self._init_petsc(comm)
        self._prime_parameters()
        self._prime_system()
        self._basis = self._system.get_basis(full_basis=True)
        # get the total size of the linear system -- needed by PETSc
        self.linsys_size = len(self._basis)
        # call structs to initialize the PETSc vectors and matrices
        self._setup_petsc_structs()

    def _setup_petsc_structs(self):

        ''' This function serves to initialize the various vectors
            and matrices (using PETSc data types) that are needed
            to solve the linear problem. They are setup using the
            sparse scheme, in parallel, so that each process owns
            only a small chunk of it.

        Parameters
        ----------
        Returns
        -------
            '''

        # initialize the parallel vector b from Ax = b
        self.v = PETSc.Vec().create(comm=self.comm)
        # need to set the total size of the vector
        self.v.setSizes(self.linsys_size)
        # this sets all the other PETSc options as defaults
        self.v.setFromOptions()
        # now we create the solution vector, x in Ax = b
        self.x = self.v.duplicate()
        ## now determine what is the local size PETSc picked
        self.nlocal = self.v.getLocalSize()
        ## Now we need to figure out what the given process owns
        self.rstart,self.rend = self.v.getOwnershipRange()

        ## now create the matrix for the linear problem
        self.X = PETSc.Mat().create(comm=self.comm)
        ## set the matrix dimensions
        ## input format is [(n,N),(m,M)] where capitals are total matrix
        ## dimensions and lowercase are local block dimensions
        ## see bottom of PETSc listserv entry
        ## https://lists.mcs.anl.gov/mailman/htdig/petsc-users/2015-March/024879.html
        ## for example
        self.X.setSizes( [(self.nlocal,self.linsys_size),\
                                    (self.nlocal,self.linsys_size)] )
        # this sets all the other PETSc options as defaults
        self.X.setFromOptions()
        ## this is needed for some reason before PETSc matrix can be used
        self.X.setUp()


    def _matr_from_eqs(self, k, w, eta):

        ''' This function iterates through the GGCE equations dicts
            to extract the row, column coordiante and value of the
            nonzero entries in the matrix. This is subsequently
            used to construct the parallel sparse system matrix.
            This is exactly the same as in the Serial class: however
            that method returns X, v whereas here we need row_ind/col_ind_dat.

        Parameters
        ----------
        k : float
            The momentum quantum number point of the calculation.
        w : float
            The frequency grid point of the calculation.
        eta : float, optional
            The artificial broadening parameter of the calculation (the default
            is None, which uses the value provided in parameter_dict at
            instantiation).

        Returns
        -------
        list, list, list
            The row and column coordinate lists, as well as a list of
            values of the matrix that are nonzero.
        '''

        row_ind = []
        col_ind = []
        dat = []

        total_bosons = np.sum(self._parameters.N)
        for n_bosons in range(total_bosons + 1):
            for eq in self._system.equations[n_bosons]:
                row_dict = dict()
                index_term_id = eq.index_term.identifier()
                ii_basis = self._basis[index_term_id]

                for term in eq.terms_list + [eq.index_term]:
                    jj = self._basis[term.identifier()]
                    try:
                        row_dict[jj] += term.coefficient(k, w, eta)
                    except KeyError:
                        row_dict[jj] = term.coefficient(k, w, eta)

                row_ind.extend([ii_basis for _ in range(len(row_dict))])
                col_ind.extend([key for key, _ in row_dict.items()])
                dat.extend([value for _, value in row_dict.items()])

        return row_ind, col_ind, dat

    def _assemble_matrix(self, k, w, eta):

        ''' The function uses the GGCE equation sparse format data
            to construct a sparse matrix in the PETSc scheme.

        Parameters
        ----------
        k : float
            The momentum quantum number point of the calculation.
        w : float
            The frequency grid point of the calculation.
        eta : float, optional
            The artificial broadening parameter of the calculation (the default
            is None, which uses the value provided in parameter_dict at
            instantiation).

        Returns
        -------
        The matrices self.X, self.v are constructed in-place,
        nothing is returned.
        '''

        row_ind, col_ind, dat = self._matr_from_eqs(k, w, eta)

        row_start = np.zeros(1, dtype='i4')
        col_pos = np.zeros(1,dtype='i4')
        val = np.zeros(1,dtype='complex128')
        loc_extent = range(self.rstart, self.rend)
        for i, row_coo in enumerate(row_ind):
            if row_coo in loc_extent:
                row_start, col_pos, val = row_coo, col_ind[i],dat[i]
                self.X.setValues(row_start, col_pos, val)

        ## assemble the matrix now that the values are filled in
        self.X.assemblyBegin(self.X.AssemblyType.FINAL)
        self.X.assemblyEnd(self.X.AssemblyType.FINAL)

        ## let us go ahead and assign values for the b vector
        self.v.setValues(self.linsys_size - 1,self._system.equations[0][0].bias(k,w))
        ## need to assemble before use
        self.v.assemblyBegin()
        self.v.assemblyEnd()

        # check memory usage
        ### to be implemented

    def solve(self, k, w, eta=None):

        """Solve the sparse-represented system using PETSc's KSP context.

        Parameters
        ----------
        k : float
            The momentum quantum number point of the calculation.
        w : float
            The frequency grid point of the calculation.
        eta : float, optional
            The artificial broadening parameter of the calculation (the default
            is None, which uses the value provided in parameter_dict at
            instantiation).

        Returns
        -------
        np.ndarray, dict
            The value of G and meta information, which in this case, is only
            specifically the time elapsed to solve for this (k, w) point
            using the PETSc KSP context.
        """


        ## function call to construct the sparse matrix into self.X
        self._assemble_matrix(k, w, eta)
        self._logger.debug(f"PETSc matrix assembled.")

        ## now construct the desired solver instance
        ksp = PETSc.KSP().create()
        ksp.setType('preonly') ##preonly for e.g. mumps and other external solvers
        ## define the linear system matrix and its preconditioner
        ksp.setOperators(self.X,self.X)

        ## set preconditioner options
        ## see PETSc manual for details
        pc = ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')
        # set tolerance and options
        ksp.setTolerances(rtol=1.e-15)
        ksp.setFromOptions()

        self._logger.debug(f"KSP and PC contexts initialized.")

        ## call the solve method
        t0 = time.time()
        res = ksp.solve(self.v,self.x)
        dt = time.time() - t0

        self._logger.debug("Sparse matrices solved", elapsed=dt)

        # assemble the solution vector
        self.x.assemblyBegin()
        self.x.assemblyEnd()

        self.comm.barrier()

        ## the last rank has the end of the solution vector, which contains G
        ## G is the last entry
        if self.rank == self.comm.getSize()-1: ## for future: change to rank 0
            G = self.x.getValue(self.linsys_size-1)
            return np.array(G), {'time': [dt]}

''' NEED TO FIGURE OUT HOW TO GATHER THE VECTOR INTO A SERIAL FORM USING PETSC MPI_COMM '''

        # G = res[self._basis['{G}(0.0)']]
        #
        # if -G.imag / np.pi < 0.0:
        #     self._logger.error(
        #         f"Negative A({k:.02f}, {w:.02f}): {(-G.imag / np.pi):.02f}"
        #     )

''' END OF : NEED TO FIGURE OUT HOW TO BCAST THE VECTOR INTO A SERIAL FORM '''
