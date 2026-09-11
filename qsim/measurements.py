import numpy as np
from qsim.states import StateVector



def measurement_probabilities(state_vector):
    '''
    Calculate the measurement probabilities for each basis state.

    Parameters:
        state_vector (StateVector): The quantum state vector.

    Returns:
        numpy.ndarray: An array of measurement probabilities for each basis state.
    '''
    if not isinstance(state_vector, StateVector):
        raise TypeError("Input must be a StateVector object.")

    return np.abs(state_vector.amplitudes) ** 2