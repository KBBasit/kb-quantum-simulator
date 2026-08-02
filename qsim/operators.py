'''
This module contains the implementation of operators used
in the simulation of quantum circuits. It provides classes
and functions to define and manipulate various quantum
gates, including single and multiple qubit operations.
'''

import numpy as np
from qsim.states import StateVector



class Operator:
    '''
    A class representing a quantum operator (gate).
    '''
    def __init__(self, matrix):
        matrix = np.asarray(matrix, dtype=np.complex128)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Operator must be a square matrix.")

        if not np.allclose(matrix.conj().T @ matrix, np.eye(matrix.shape[0]), dtype=np.complex128):
            raise ValueError("Operator must be unitary.")

        self.matrix = matrix


    @property
    def dimension(self):
        '''
        Return the dimension of the operator.
        '''
        return self.matrix.shape[0]


    def apply_operator_inplace(self, state_vector):
        '''
        Apply the operator to a given state vector in place.
        '''
        if not isinstance(state_vector, StateVector):
            raise ValueError("State vector must be a StateVector object.")

        if state_vector.dimension != self.dimension:
            raise ValueError("State vector size must match operator dimension.")

        # Perform matrix-vector multiplication
        result = self.matrix @ state_vector.amplitudes

        # Update the state vector in place
        state_vector.amplitudes[:] = result

    