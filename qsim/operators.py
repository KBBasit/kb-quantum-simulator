'''
This module contains the implementation of operators used
in the simulation of quantum circuits.
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

        self.matrix = matrix


    @property
    def dimension(self):
        '''
        Return the dimension of the operator.
        '''
        return self.matrix.shape[0]


    def apply_operator(self, state_vector):
        '''
        Apply the operator to a given state vector and return
        the resulting state vector.
        '''
        if not np.allclose(self.conj().T @ self.matrix, np.eye(self.dimension)):
            raise ValueError("Operator must be unitary.")

        if not isinstance(state_vector, StateVector):
            raise ValueError("State vector must be a StateVector object.")

        if state_vector.dimension != self.dimension:
            raise ValueError("State vector size must match operator dimension.")

        # Perform matrix-vector multiplication
        result = self.matrix @ state_vector.amplitudes

        return StateVector(result)


    def apply_operator_inplace(self, state_vector):
        '''
        Apply the operator to a given state vector in place.
        '''
        if not np.allclose(self.conj().T @ self.matrix, np.eye(self.dimension)):
            raise ValueError("Operator must be unitary.")

        if not isinstance(state_vector, StateVector):
            raise ValueError("State vector must be a StateVector object.")

        if state_vector.dimension != self.dimension:
            raise ValueError("State vector size must match operator dimension.")

        state_vector.amplitudes[:] = self.matrix @ state_vector.amplitudes

    
    def tensor_product(self, other):
        '''
        Compute the tensor product of this operator with another
        operator and return a new Operator instance.
        '''
        if not isinstance(other, Operator):
            raise ValueError("Other must be an Operator object.")

        tensor_matrix = np.kron(self.matrix, other.matrix)
        return Operator(tensor_matrix)