from qsim import states
import numpy as np



def test_state_vector_initialization():
    amplitudes = [1/np.sqrt(2), 1/np.sqrt(2)]
    state_vector = states.StateVector(amplitudes)

    assert state_vector.num_qubits == 1
    assert state_vector.dimension == 2
    assert np.isclose(np.linalg.norm(state_vector.amplitudes), 1.0)
    assert np.allclose(state_vector.amplitudes, amplitudes)


def test_zero_state_properties():
    num_qubits = 3
    zero_state = states.StateVector.zero_state(num_qubits)

    assert zero_state.num_qubits == num_qubits
    assert zero_state.dimension == 2 ** num_qubits
    assert np.isclose(np.linalg.norm(zero_state.amplitudes), 1.0)
    assert np.isclose(zero_state.amplitudes[0], 1.0)
    assert np.allclose(zero_state.amplitudes[1:], 0.0)


def test_copy_state_vector():
    amplitudes = [1/np.sqrt(2), 1/np.sqrt(2)]
    state_vector = states.StateVector(amplitudes)
    copied_state_vector = state_vector.copy()

    assert copied_state_vector.num_qubits == state_vector.num_qubits
    assert copied_state_vector.dimension == state_vector.dimension
    assert np.allclose(copied_state_vector.amplitudes, state_vector.amplitudes)