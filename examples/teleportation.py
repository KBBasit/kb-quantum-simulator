"""
Quantum Teleportation
"""

import numpy as np
from kb_quantum_simulator.gates import pauli_x, pauli_z
from kb_quantum_simulator.states import StateVector
from kb_quantum_simulator.operators import Operator
from kb_quantum_simulator.circuits import Circuit
from kb_quantum_simulator.simulator import StatevectorSimulator



def random_single_qubit_state():
    """
    Generates a random normalised numpy array.

    Returns
    -------
    StateVector
        A random single-qubit state vector.
    """
    state = np.random.rand(2) + 1j * np.random.rand(2)
    state /= np.linalg.norm(state)
    state = StateVector(state)

    return state


def teleportation(state_vector):
    """
    Simulates the quantum teleportation protocol

    Parameters
    ----------
    state_vector : StateVector
        The state vector to be teleported.

    Returns
    -------
    tuple[StateVector, dict]
        The final state vector and measurement results.
    """
    # Initialisation
    circuit = Circuit(num_qubits=3)
    initial_state_arr = np.kron(state_vector.amplitudes, np.array([1, 0, 0, 0]))
    initial_state = StateVector(initial_state_arr)

    # Form the entangled pair (Bell state) between qubits 1 and 2
    circuit.hadamard(1)
    circuit.cnot(1, 2)

    # Prepare for Bell basis measurement on qubits 0 and 1
    circuit.cnot(0, 1)
    circuit.hadamard(0)

    circuit.add_measurement(key="alice", qubits=(0, 1))

    # Apply operations to qubit 2 based on Alice's measurement
    circuit.c_apply(operator=Operator(pauli_z()), measurement_key="alice", measurement_condition="10", target_qubits=(2,))

    circuit.c_apply(operator=Operator(pauli_x()), measurement_key="alice", measurement_condition="01", target_qubits=(2,))

    circuit.c_apply(operator=Operator(pauli_x()), measurement_key="alice", measurement_condition="11", target_qubits=(2,))
    circuit.c_apply(operator=Operator(pauli_z()), measurement_key="alice", measurement_condition="11", target_qubits=(2,))

    sim = StatevectorSimulator()
    final_state, measurement_result = sim.run(circuit, initial_state=initial_state)

    return final_state, measurement_result



if __name__ == "__main__":
    state_to_teleport = random_single_qubit_state()

    final_state, measurement_result = teleportation(state_to_teleport)

    measurement_result_str = measurement_result["alice"]
    alice_qubit_0 = int(measurement_result_str[0])
    alice_qubit_1 = int(measurement_result_str[1])

    bob_qubit = final_state.amplitudes.reshape(2, 2, 2)[alice_qubit_0, alice_qubit_1, :]

    print("Initial state to teleport:", state_to_teleport.amplitudes)
    print("Final state of Bob's qubit:", bob_qubit)

    if np.allclose(state_to_teleport.amplitudes, bob_qubit):
        print("Teleportation successful!")
    else:
        print("Teleportation failed.")