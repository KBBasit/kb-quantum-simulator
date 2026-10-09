from qsim.gates import pauli_x
from qsim.states import StateVector
from qsim.operators import Operator
from qsim.circuits import Circuit
from qsim.simulator import StatevectorSimulator
import numpy as np
import pytest



def test_statevector_simulator_run_validations():
    sim = StatevectorSimulator()
    circuit = Circuit(num_qubits=2)

    with pytest.raises(TypeError, match="Circuit must be an instance of the Circuit class."):
        sim.run("not_a_circuit")

    with pytest.raises(TypeError, match="Initial state must be an instance of the StateVector class."):
        sim.run(circuit, initial_state="not_a_state_vector")

    initial_state = StateVector.zero_state(1)
    with pytest.raises(ValueError, match="Initial state must have the same number of qubits as the circuit."):
        sim.run(circuit, initial_state=initial_state)


def test_statevector_simulator_run():
    sim = StatevectorSimulator()

    circuit_1 = Circuit(num_qubits=2)
    circuit_1.hadamard(0)
    circuit_1.cnot(0, 1)
    final_state_1, _ = sim.run(circuit_1)
    expected_state_1 = 1/np.sqrt(2) * np.array([1, 0, 0, 1])
    assert np.allclose(final_state_1.amplitudes, expected_state_1)

    circuit_2 = Circuit(num_qubits=2)
    circuit_2.pauli_x(0)
    initial_state_2 = StateVector(np.array([0, 0, 1, 0]))
    final_state_2, _ = sim.run(circuit_2, initial_state=initial_state_2)
    expected_state_2 = np.array([1, 0, 0, 0])
    assert np.allclose(final_state_2.amplitudes, expected_state_2)

    circuit_3 = Circuit(num_qubits=3)
    final_state_3, _ = sim.run(circuit_3)
    expected_state_3 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    assert np.allclose(final_state_3.amplitudes, expected_state_3)

    circuit_4 = Circuit(num_qubits=1)
    circuit_4.pauli_y(0)
    initial_state_4 = StateVector([1, 0])
    initial_state_4_amplitudes = initial_state_4.amplitudes.copy()
    sim.run(circuit_4, initial_state=initial_state_4)
    assert np.allclose(initial_state_4.amplitudes, initial_state_4_amplitudes)

    circuit_5 = Circuit(num_qubits=2)
    circuit_5.pauli_x(0)
    circuit_5.add_measurement(key="alice", qubits=(1,))
    _, measurement_results_5 = sim.run(circuit_5)
    assert "alice" in measurement_results_5
    assert measurement_results_5["alice"] in ["0", "1"]

    circuit_6 = Circuit(num_qubits=1)
    circuit_6.hadamard(0)
    circuit_6.add_measurement(key="alice", qubits=(0,))
    final_state_6, measurement_results_6 = sim.run(circuit_6)
    if measurement_results_6["alice"] == "0":
        assert np.allclose(final_state_6.amplitudes, np.array([1, 0]))
    else:
        assert np.allclose(final_state_6.amplitudes, np.array([0, 1]))

    circuit_7 = Circuit(num_qubits=1)
    circuit_7.pauli_x(0)
    circuit_7.add_measurement(key="alice", qubits=(0,))
    circuit_7.pauli_x(0)
    final_state_7, measurement_results_7 = sim.run(circuit_7)
    if measurement_results_7["alice"] == "0":
        assert np.allclose(final_state_7.amplitudes, np.array([0, 1]))
    else:
        assert np.allclose(final_state_7.amplitudes, np.array([1, 0]))

    circuit_8 = Circuit(num_qubits=2)
    circuit_8.hadamard(0)
    circuit_8.add_measurement(key="alice", qubits=(0,))
    circuit_8.c_apply(operator=Operator(pauli_x()), measurement_key="alice", measurement_condition="1", target_qubits=(1,))
    final_state_8, measurement_results_8 = sim.run(circuit_8)
    if measurement_results_8["alice"] == "1":
        assert np.allclose(final_state_8.amplitudes, np.array([0, 0, 0, 1]))
    else:
        assert np.allclose(final_state_8.amplitudes, np.array([1, 0, 0, 0]))


def test_qft():
    sim = StatevectorSimulator()

    circuit_1 = Circuit(num_qubits=2)
    circuit_1.quantum_fourier_transform(target_qubits=(0,1))
    final_state_1, _ = sim.run(circuit_1)
    assert np.allclose(final_state_1.amplitudes, 1/2 * np.array([1,1,1,1]))

    circuit_2 = Circuit(num_qubits=4)
    circuit_2.quantum_fourier_transform(target_qubits=(0,3,2))
    final_state_2, _ = sim.run(circuit_2, initial_state=StateVector([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    expected_state_2 = 1/np.sqrt(8) * np.array([1,-1,1j,-1j,0,0,0,0,1,-1,1j,-1j,0,0,0,0])
    assert np.allclose(final_state_2.amplitudes, expected_state_2)


def test_inverse_qft():
    sim = StatevectorSimulator()

    circuit_1 = Circuit(num_qubits=2)
    circuit_1.inverse_quantum_fourier_transform(target_qubits=(0,1))
    final_state_1, _ = sim.run(circuit_1, initial_state=StateVector([1/2] * 4))
    assert np.allclose(final_state_1.amplitudes, np.array([1,0,0,0]))

    circuit_2 = Circuit(num_qubits=4)
    circuit_2.quantum_fourier_transform(target_qubits=(0,3,2))
    circuit_2.inverse_quantum_fourier_transform(target_qubits=(0,3,2))
    final_state_2, _ = sim.run(circuit_2, initial_state=StateVector([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    assert np.allclose(final_state_2.amplitudes, np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))


def test_unknown_operation_type():
    circuit = Circuit(num_qubits=1)
    circuit.operations.append(("unknown", None, None))

    sim = StatevectorSimulator()

    with pytest.raises(ValueError, match="Unknown operation type: unknown"):
        sim.run(circuit)