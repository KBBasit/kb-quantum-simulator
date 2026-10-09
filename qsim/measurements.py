"""
This module contains functions for performing measurements
on quantum states.
"""

from .states import StateVector
import numpy as np



def measurement_probabilities(state_vector):
    """
    Calculate the measurement probabilities for each basis
    state.

    Parameters
    ----------
        state_vector : StateVector
            The quantum state vector.

    Returns
    -------
        numpy.ndarray
            The measurement probabilities for each basis state.
    """
    if not isinstance(state_vector, StateVector):
        raise TypeError("Input must be a StateVector object.")

    return np.abs(state_vector.amplitudes) ** 2


def _validate_state_type_and_qubit_inputs(state_vector, qubits):
    if not isinstance(state_vector, StateVector):
        raise TypeError("Input must be a StateVector object.")
    if not isinstance(qubits, (tuple, type(None))):
        raise TypeError("Qubits must be a tuple of indices or None.")
    if qubits is not None:
        if not all(isinstance(q, int) for q in qubits):
            raise TypeError("All qubit indices must be integers.")
        if any(q < 0 or q >= state_vector.num_qubits for q in qubits):
            raise ValueError("Qubit indices must be within the range of the state vector's qubits.")
        if len(set(qubits)) != len(qubits):
            raise ValueError("Qubit indices must be unique.")


def measure(state_vector, qubits=None):
    """
    Perform a measurement on the specified qubits of the
    state vector and collapse the state accordingly.

    Parameters
    ----------
        state_vector : StateVector
            The quantum state vector.
        qubits : tuple, optional
            A tuple of qubit indices to measure. If None, all qubits are measured. Default is None.

    Returns
    -------
        str
            The bitstring of the measured basis state.
    """
    _validate_state_type_and_qubit_inputs(state_vector, qubits)

    if qubits is not None:
        tensor = state_vector.amplitudes.reshape([2] * state_vector.num_qubits)
        tensor = np.moveaxis(tensor, qubits, range(len(qubits)))

        matrix = tensor.reshape(2 ** len(qubits), -1)

        probabilities = np.sum(np.abs(matrix) ** 2, axis=1)
        outcome = np.random.choice(len(probabilities), p=probabilities)

        collapsed_matrix = np.zeros_like(matrix)
        collapsed_matrix[outcome] = matrix[outcome]

        collapsed_tensor = collapsed_matrix.reshape([2] * state_vector.num_qubits)
        collapsed_tensor = np.moveaxis(collapsed_tensor, range(len(qubits)), qubits)
        collapsed_state_vector = collapsed_tensor.reshape(2 ** state_vector.num_qubits)
        collapsed_state_vector /= np.sqrt(probabilities[outcome])
        state_vector.amplitudes = collapsed_state_vector

        bitstring = format(outcome, f'0{len(qubits)}b')
    else:
        probabilities = measurement_probabilities(state_vector)
        outcome = np.random.choice(len(probabilities), p=probabilities)

        state_vector.amplitudes = np.zeros_like(state_vector.amplitudes)
        state_vector.amplitudes[outcome] = 1.0

        bitstring = format(outcome, f'0{state_vector.num_qubits}b')

    return bitstring


def sample(state_vector, shots, qubits=None):
    """
    Sample measurement outcomes from the state vector's
    probability distribution.

    Parameters
    ----------
        state_vector : StateVector
            The quantum state vector.
        qubits : tuple, optional
            A tuple of qubit indices to measure. If None, all qubits are measured. Default is None.
        shots : int
            The number of measurements to perform.

    Returns
    -------
        list
            A list of bitstrings corresponding to the measured basis states.
    """
    _validate_state_type_and_qubit_inputs(state_vector, qubits)
    if not isinstance(shots, int):
        raise TypeError("Shots must be an integer.")
    if shots <= 0:
        raise ValueError("Shots must be a positive integer.")

    if qubits is not None:
        tensor = state_vector.amplitudes.reshape([2] * state_vector.num_qubits)
        tensor = np.moveaxis(tensor, qubits, range(len(qubits)))
        matrix = tensor.reshape(2 ** len(qubits), -1)

        probabilities = np.sum(np.abs(matrix) ** 2, axis=1)
        outcomes = np.random.choice(len(probabilities), size=shots, p=probabilities)
        bitstrings = [format(outcome, f'0{len(qubits)}b') for outcome in outcomes]
    else:
        probabilities = measurement_probabilities(state_vector)
        outcomes = np.random.choice(len(probabilities), size=shots, p=probabilities)
        bitstrings = [format(outcome, f'0{state_vector.num_qubits}b') for outcome in outcomes]

    return bitstrings