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


def test_invalid_input_for_measurement():
    with pytest.raises(TypeError, match="Input must be a StateVector object."):
        measurements.measure("This is not a state vector object")


def test_measurement_and_collapse():
    state_vector = states.StateVector(1/np.sqrt(2) * np.array([1, 0, 0, 1]))
    bitstring = measurements.measure(state_vector)

    assert bitstring in ['00', '11']
    if bitstring == '00':
        assert np.allclose(state_vector.amplitudes, [1.0, 0.0, 0.0, 0.0])
    else:
        assert np.allclose(state_vector.amplitudes, [0.0, 0.0, 0.0, 1.0])


def test_invalid_input_for_sampling():
    state = states.StateVector(1/np.sqrt(2) * np.array([1, 0, 0, 1]))

    with pytest.raises(TypeError, match="Input must be a StateVector object."):
        measurements.sample("This is not a state vector object", shots=10)
    with pytest.raises(TypeError, match="Shots must be an integer."):
        measurements.sample(state, shots="ten")
    with pytest.raises(ValueError, match="Shots must be a positive integer."):
        measurements.sample(state, shots=-5)


def test_sampling():
    state_vector = states.StateVector(1/np.sqrt(2) * np.array([1, 0, 0, 1]))
    shots = 1000
    samples = measurements.sample(state_vector, shots)

    assert len(samples) == shots
    assert all(bitstring in ['00', '11'] for bitstring in samples)