from qsim import states
from qsim import circuits
from qsim import simulator
import numpy as np
import pytest



def test_statevector_simulator_run_validations():
    sim = simulator.StatevectorSimulator()
    circuit = circuits.Circuit(num_qubits=2)

    with pytest.raises(TypeError, match="Circuit must be an instance of the Circuit class."):
        sim.run("not_a_circuit")

    with pytest.raises(TypeError, match="Initial state must be an instance of the StateVector class."):
        sim.run(circuit, initial_state="not_a_state_vector")

    initial_state = states.StateVector.zero_state(1)
    with pytest.raises(ValueError, match="Initial state must have the same number of qubits as the circuit."):
        sim.run(circuit, initial_state=initial_state)


def test_statevector_simulator_run():
    sim = simulator.StatevectorSimulator()

    circuit_1 = circuits.Circuit(num_qubits=2)
    circuit_1.hadamard(0)
    circuit_1.cnot(0, 1)
    final_state_1, _ = sim.run(circuit_1)
    expected_state_1 = 1/np.sqrt(2) * np.array([1, 0, 0, 1])
    assert np.allclose(final_state_1.amplitudes, expected_state_1)

    circuit_2 = circuits.Circuit(num_qubits=2)
    circuit_2.pauli_x(0)
    initial_state_2 = states.StateVector(np.array([0, 0, 1, 0]))
    final_state_2, _ = sim.run(circuit_2, initial_state=initial_state_2)
    expected_state_2 = np.array([1, 0, 0, 0])
    assert np.allclose(final_state_2.amplitudes, expected_state_2)

    circuit_3 = circuits.Circuit(num_qubits=3)
    final_state_3, _ = sim.run(circuit_3)
    expected_state_3 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    assert np.allclose(final_state_3.amplitudes, expected_state_3)

    circuit_4 = circuits.Circuit(num_qubits=1)
    circuit_4.pauli_y(0)
    initial_state_4 = states.StateVector([1, 0])
    initial_state_4_amplitudes = initial_state_4.amplitudes.copy()
    sim.run(circuit_4, initial_state=initial_state_4)
    assert np.allclose(initial_state_4.amplitudes, initial_state_4_amplitudes)