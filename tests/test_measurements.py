from qsim import states
from qsim import measurements
import numpy as np
import pytest



def test_invalid_input_for_probability_distribution():
    with pytest.raises(TypeError, match="Input must be a StateVector object."):
        measurements.measurement_probabilities("This is not a state vector object")


def test_probability_distribution():
    state_vector = states.StateVector(np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]))
    prob_dist = measurements.measurement_probabilities(state_vector)
    expected_prob_dist = np.array([0.5, 0.0, 0.0, 0.5])
    assert np.allclose(prob_dist, expected_prob_dist)