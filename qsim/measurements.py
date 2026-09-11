import numpy as np
from qsim.states import StateVector



def measurement_probabilities(state_vector):
    """
    Calculate the measurement probabilities for each basis state.

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
    Perform a measurement on the quantum state vector.

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
    bitstring = format(outcome, f'0{state_vector.num_qubits}b')

    return bitstring