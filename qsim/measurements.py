import numpy as np
from qsim.states import StateVector



def measurement_probabilities(state_vector):
    """
    Calculate the measurement probabilities for each basis 
    state.

    Parameters:
        state_vector (StateVector): The quantum state vector.

    Returns:
        numpy.ndarray: An array of measurement probabilities for each basis state.
    
    Raises:
        TypeError: If state_vector is not an instance of StateVector.
    """
    if not isinstance(state_vector, StateVector):
        raise TypeError("Input must be a StateVector object.")

    return np.abs(state_vector.amplitudes) ** 2


def measure(state_vector):
    """
    Perform a measurement on the quantum state vector and
    return the corresponding bitstring of the measured
    basis state. Collapse the state vector to the measured 
    state.

    Parameters:
        state_vector (StateVector): The quantum state vector.

    Returns:
        str: The bitstring of the measured basis state.

    Raises:
        TypeError: If state_vector is not an instance of StateVector.
    """
    if not isinstance(state_vector, StateVector):
        raise TypeError("Input must be a StateVector object.")

    probabilities = measurement_probabilities(state_vector)
    outcome = np.random.choice(len(probabilities), p=probabilities)

    state_vector.amplitudes = np.zeros_like(state_vector.amplitudes)
    state_vector.amplitudes[outcome] = 1.0

    bitstring = format(outcome, f'0{state_vector.num_qubits}b')

    return bitstring


def sample(state_vector, shots):
    """
    Sample measurement outcomes from the state vecotr's
    probability distribution.

    Parameters:
        state_vector (StateVector): The quantum state vector.
        shots (int): The number of measurements to perform.

    Returns:
        list: A list of bitstrings corresponding to the measured basis states.

    Raises:
        TypeError: If state_vector is not an instance of StateVector.
        ValueError: If shots is not a positive integer.
    """
    if not isinstance(state_vector, StateVector):
        raise TypeError("Input must be a StateVector object.")
    if not isinstance(shots, int):
        raise TypeError("Shots must be an integer.")
    if shots <= 0:
        raise ValueError("Shots must be a positive integer.")

    probabilities = measurement_probabilities(state_vector)
    outcomes = np.random.choice(len(probabilities), size=shots, p=probabilities)

    bitstrings = [format(outcome, f'0{state_vector.num_qubits}b') for outcome in outcomes]

    return bitstrings