from qsim import states
import numpy as np



def test_zero_state_properties():
    num_qubits = 3
    zero_state = states.StateVector.zero_state(num_qubits)

    assert zero_state.num_qubits == num_qubits
    assert zero_state.number_of_dimensions == 2 ** num_qubits
    assert np.isclose(np.linalg.norm(zero_state.amplitudes), 1.0)
    assert np.isclose(zero_state.amplitudes[0], 1.0)
    assert np.allclose(zero_state.amplitudes[1:], 0.0)