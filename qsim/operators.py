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


    def apply_gate(self, state_vector):
        '''
        Apply the operator to a given state vector and return
        the resulting state vector.
        '''
        if not self.is_unitary:
            raise ValueError("Operator must be unitary.")

        if not isinstance(state_vector, StateVector):
            raise TypeError("State vector must be a StateVector object.")

        if state_vector.dimension != self.dimension:
            raise ValueError("State vector size must match operator dimension.")

        # Perform matrix-vector multiplication
        result = self.matrix @ state_vector.amplitudes

        return StateVector(result)


    def apply_gate_inplace(self, state_vector):
        '''
        Apply the operator to a given state vector in place.
        '''
        if not self.is_unitary:
            raise ValueError("Operator must be unitary.")

        if not isinstance(state_vector, StateVector):
            raise TypeError("State vector must be a StateVector object.")

        if state_vector.dimension != self.dimension:
            raise ValueError("State vector size must match operator dimension.")

        state_vector.amplitudes[:] = self.matrix @ state_vector.amplitudes

    
    def tensor_product(self, other):
        '''
        Compute the tensor product of this operator with another
        operator and return a new Operator instance.
        '''
        if not isinstance(other, Operator):
            raise TypeError("Other must be an Operator object.")

        tensor_matrix = np.kron(self.matrix, other.matrix)

        return Operator(tensor_matrix)
    

    def apply_gate_on_register(self, state_vector, qubit_indices):
        '''
        Apply the operator to specific qubits in a multi-qubit
        state vector.
        '''
        if self.operator_qubits != len(qubit_indices) or len(qubit_indices) != state_vector.num_qubits:
            raise ValueError("Number of qubit indices must match operator qubits.")

        if not self.is_unitary:
            raise ValueError("Operator must be unitary.")

        if not isinstance(state_vector, StateVector):
            raise TypeError("State vector must be a StateVector object.")

        num_qubits = state_vector.num_qubits
        if any(qi < 0 or qi >= num_qubits for qi in qubit_indices):
            raise ValueError("Qubit indices must be valid for the state vector.")

        # Create a full operator for the entire register
        full_operator = np.eye(1, dtype=np.complex128)
        for i in range(num_qubits):
            if i in qubit_indices:
                full_operator = np.kron(full_operator, self.matrix)
            else:
                full_operator = np.kron(full_operator, np.eye(2, dtype=np.complex128))

        # Apply the full operator to the state vector
        result = full_operator @ state_vector.amplitudes

        return StateVector(result)