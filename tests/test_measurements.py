from qsim.states import StateVector
from qsim.measurements import measurement_probabilities, measure, sample
import numpy as np
import pytest



def test_invalid_input_for_probability_distribution():
    with pytest.raises(TypeError, match="Input must be a StateVector object."):
        measurement_probabilities("This is not a state vector object")


def test_probability_distribution():
    state_vector = StateVector(np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]))
    prob_dist = measurement_probabilities(state_vector)
    expected_prob_dist = np.array([0.5, 0.0, 0.0, 0.5])
    assert np.allclose(prob_dist, expected_prob_dist)


def test_invalid_inputs_for_measurements():
    with pytest.raises(TypeError, match="Input must be a StateVector object."):
        measure("This is not a state vector object")
    with pytest.raises(TypeError, match="Qubits must be a tuple of indices or None."):
        measure(StateVector(np.array([1, 0])), qubits="not a tuple")
    with pytest.raises(TypeError, match="All qubit indices must be integers."):
        measure(StateVector(np.array([1, 0])), qubits=(0, "not an int"))
    with pytest.raises(ValueError, match="Qubit indices must be within the range of the state vector's qubits."):
        measure(StateVector(np.array([1, 0])), qubits=(0, 2))
    with pytest.raises(ValueError, match="Qubit indices must be unique."):
        measure(StateVector(1/np.sqrt(2) * np.array([1, 0, 0, 1])), qubits=(0, 0))


def test_measurement_and_collapse():
    state_vector_1 = StateVector(1/np.sqrt(2) * np.array([1, 0, 0, 1]))
    bitstring_1 = measure(state_vector_1)
    assert bitstring_1 in ['00', '11']
    if bitstring_1 == '00':
        assert np.allclose(state_vector_1.amplitudes, [1.0, 0.0, 0.0, 0.0])
    else:
        assert np.allclose(state_vector_1.amplitudes, [0.0, 0.0, 0.0, 1.0])

    state_vector_2 = StateVector(1/np.sqrt(2) * np.array([1, 0, 0, 0, 0, 0, 0, 1]))
    bitstring_2 = measure(state_vector_2, qubits=(0, 2))
    assert bitstring_2 in ['00', '11']
    if bitstring_2 == '00':
        assert np.allclose(state_vector_2.amplitudes, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        assert np.allclose(state_vector_2.amplitudes, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    state_vector_3 = StateVector(1/2 * np.array([1, 1, 0, 0, 0, 0, 1, 1]))
    bitstring_3 = measure(state_vector_3, qubits=(0, 1))
    assert bitstring_3 in ['00', '11']
    if bitstring_3 == '00':
        assert np.allclose(state_vector_3.amplitudes, [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] / np.sqrt(2))
    else:
        assert np.allclose(state_vector_3.amplitudes, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0] / np.sqrt(2))


def test_invalid_input_for_sampling():
    state = StateVector(1/np.sqrt(2) * np.array([1, 0, 0, 1]))

    with pytest.raises(TypeError, match="Shots must be an integer."):
        sample(state, shots="ten")
    with pytest.raises(ValueError, match="Shots must be a positive integer."):
        sample(state, shots=-5)
    with pytest.raises(ValueError, match="Shots must be a positive integer."):
        sample(state, shots=0)


def test_sampling():
    shots = 1000
    state_vector_1 = StateVector(1/np.sqrt(2) * np.array([1, 0, 0, 1]))
    state_vector_2 = StateVector(1/2 * np.array([1, 0, 1, 0, 0, 1, 0, 1]))
    state_vector_3 = StateVector(1/np.sqrt(2) * np.array([1, 0, 0, 0, 0, 0, 0, 1]))

    samples_1 = sample(state_vector_1, shots=shots)
    samples_2 = sample(state_vector_2, qubits=(0, 2), shots=shots)
    samples_3 = sample(state_vector_3, qubits=(0, 1), shots=shots)

    assert len(samples_1) == shots
    assert all(bitstring in ['00', '11'] for bitstring in samples_1)

    assert len(samples_2) == shots
    assert all(bitstring in ['00', '11'] for bitstring in samples_2)

    assert len(samples_3) == shots
    assert all(bitstring in ['00', '11'] for bitstring in samples_3)


def test_sampling_does_not_modify_state_vector():
    state_vector = StateVector(1/np.sqrt(2) * np.array([1, 0, 0, 1]))
    original_amplitudes = state_vector.amplitudes.copy()
    shots = 1000
    sample(state_vector, shots)

    assert np.allclose(state_vector.amplitudes, original_amplitudes)