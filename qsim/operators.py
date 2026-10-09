"""
This module contains the implementation of operators used
in the simulation of quantum circuits.
"""

from .states import StateVector
import numpy as np



class Operator:
    """
    A class for representing quantum operators.

    Attributes
    ----------
    matrix : numpy.ndarray
        The matrix representation of the operator.
    dimension : int
        The dimension of the operator.
    is_unitary : bool
        Whether the operator is unitary.
    operator_qubits : int
        The number of qubits the operator acts on.
    """
    def __init__(self, matrix):
        """
        Initialise an Operator.

        Parameters
        ----------
        matrix : (array-like)
            The matrix representation of the operator.
        """
        matrix = np.asarray(matrix, dtype=np.complex128)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Operator must be a square matrix.")

        if matrix.shape[0] & (matrix.shape[0] - 1) != 0:
            raise ValueError("Operator dimension must be a power of 2.")

        self.matrix = matrix


    @property
    def dimension(self):
        """
        Return the dimension of the operator.
        """
        return self.matrix.shape[0]


    @property
    def is_unitary(self):
        """
        Check if the operator is unitary.
        """
        identity = np.eye(self.dimension, dtype=np.complex128)

        return np.allclose(self.matrix.conj().T @ self.matrix, identity)


    @property
    def operator_qubits(self):
        """
        Return the number of qubits the operator acts on.
        """
        return int(np.log2(self.matrix.shape[0]))


    def _validate_gate_application(self, state_vector, target_qubits):
        if not isinstance(state_vector, StateVector):
            raise TypeError("State vector must be a StateVector object.")
        if not self.is_unitary:
            raise ValueError("Operator must be unitary.")
        if not isinstance(target_qubits, tuple):
            raise TypeError("Target qubits must be a tuple.")
        if not all(isinstance(q, int) for q in target_qubits):
            raise TypeError("Target qubits must be integers.")
        if len(target_qubits) != self.operator_qubits:
            raise ValueError("Number of target qubits must match the operator's qubit count.")
        if any(q < 0 or q >= state_vector.num_qubits for q in target_qubits):
            raise ValueError("Target qubits must be within the range of the state vector's qubits.") 
        if len(set(target_qubits)) != len(target_qubits):
            raise ValueError("Target qubits must be unique.")


    def apply_gate(self, state_vector, target_qubits = (0,), inplace=False):
        """
        Apply the operator to the given state vector on the
        specified target qubits.

        Parameters
        ----------
        state_vector : StateVector
            The quantum state vector to which the operator is applied.
        target_qubits : tuple, optional
            A tuple of qubit indices on which the operator acts. Default is (0,).
        inplace : bool, optional
            If True, modify the state vector in place. If False, return a new state vector. Default is False.

        Returns
        -------
        StateVector
            The resulting state vector after applying the operator.
        """
        self._validate_gate_application(state_vector, target_qubits)

        tensor = state_vector.amplitudes.reshape([2] * state_vector.num_qubits)
        tensor = np.moveaxis(tensor, target_qubits, range(len(target_qubits)))

        matrix = tensor.reshape(self.dimension, -1)

        result = self.matrix @ matrix
        result = result.reshape([2] * state_vector.num_qubits)
        result = np.moveaxis(result, range(len(target_qubits)), target_qubits)
        result = result.reshape(2 ** state_vector.num_qubits)

        if inplace:
            state_vector.amplitudes[:] = result
            return state_vector

        return StateVector(result)