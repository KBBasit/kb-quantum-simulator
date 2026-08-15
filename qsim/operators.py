'''
This module contains the implementation of operators used
in the simulation of quantum circuits.
'''

import numpy as np
from qsim.states import StateVector



class Operator:
    '''
    A class for representing quantum operators.
    '''
    def __init__(self, matrix):
        matrix = np.asarray(matrix, dtype=np.complex128)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Operator must be a square matrix.")

        if matrix.shape[0] & (matrix.shape[0] - 1) != 0:
            raise ValueError("Operator dimension must be a power of 2.")

        self.matrix = matrix


    @property
    def dimension(self):
        '''
        Return the dimension of the operator.
        '''
        return self.matrix.shape[0]


    @property
    def is_unitary(self):
        '''
        Check if the operator is unitary.
        '''
        identity = np.eye(self.dimension, dtype=np.complex128)

        return np.allclose(self.matrix.conj().T @ self.matrix, identity)


    @property
    def operator_qubits(self):
        '''
        Return the number of qubits the operator acts on.
        '''
        return int(np.log2(self.matrix.shape[0]))


    def _validate_gate_application(self, state_vector):
        if not self.is_unitary:
            raise ValueError("Operator must be unitary.")
        if not isinstance(state_vector, StateVector):
            raise TypeError("State vector must be a StateVector object.")
        if state_vector.dimension != self.dimension:
            raise ValueError("State vector size must match operator dimension.")


    def apply_gate(self, state_vector):
        '''
        Apply the operator to a given state vector and
        return the resulting state vector.
        '''
        self._validate_gate_application(state_vector)

        # Perform matrix-vector multiplication
        result = self.matrix @ state_vector.amplitudes

        return StateVector(result)


    def apply_gate_inplace(self, state_vector):
        '''
        Apply the operator to a given state vector in
        place.
        '''
        self._validate_gate_application(state_vector)

        state_vector.amplitudes[:] = self.matrix @ state_vector.amplitudes

    
    def tensor_product(self, other):
        '''
        Compute the tensor product of this operator with
        another operator and return a new Operator
        instance.
        '''
        if not isinstance(other, Operator):
            raise TypeError("Other must be an Operator object.")

        tensor_matrix = np.kron(self.matrix, other.matrix)

        return Operator(tensor_matrix)
    

    def apply_single_qubit_gate_on_register(self, state_vector, target_qubit):
        '''
        Apply a single-qubit gate to one qubit in a
        multi-qubit state.
        '''
        if not isinstance(state_vector, StateVector):
            raise TypeError("State vector must be a StateVector object.")

        if self.dimension != 2:
            raise ValueError("This method only supports single-qubit operators.")

        if not isinstance(target_qubit, int):
            raise TypeError("Target qubit must be an integer.")

        if target_qubit < 0 or target_qubit >= state_vector.num_qubits:
            raise ValueError("Target qubit index must be non-negative and valid for the state vector.")

        if not self.is_unitary:
            raise ValueError("Operator must be unitary.")

        # Construct the full operator for the multi-qubit system
        full_operator = np.eye(1, dtype=np.complex128)
        for i in range(state_vector.num_qubits):
            if i == target_qubit:
                next_operator = self.matrix
            else:
                next_operator = np.eye(2, dtype=np.complex128)

            full_operator = np.kron(full_operator, next_operator)

        # Apply the full operator to the state vector
        result = full_operator @ state_vector.amplitudes

        return StateVector(result)