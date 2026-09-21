'''
Quantum Phase Estimation
'''

from qsim.circuits import Circuit
from qsim.simulator import StatevectorSimulator
import numpy as np



def phase_estimation(control_qubits, theta):
    """
    Constructs a quantum circuit for phase estimation.

    Parameters:
        control_qubits (int): The number of control qubits.
        theta (float): The phase to be estimated.
    Returns:
        phase (float): The estimated phase.
    """
    sim = StatevectorSimulator()

    circuit = Circuit(num_qubits=control_qubits + 1)

    target_qubit = circuit.num_qubits - 1

    circuit.pauli_x(target_qubit=target_qubit)

    for control_qubit in range(control_qubits):
        circuit.hadamard(target_qubit=control_qubit)

    for control_qubit in range(control_qubits):
        circuit.controlled_phase(control_qubit=control_qubit, target_qubit=target_qubit, theta=theta * (2 ** (control_qubits - control_qubit - 1)))
    
    circuit.inverse_quantum_fourier_transform(target_qubits=tuple(range(control_qubits)))

    circuit.add_measurement(key = "Measurement", qubits = tuple(range(control_qubits)))

    _, measurement_results = sim.run(circuit)

    bitstring = measurement_results["Measurement"]
    m = int(bitstring, 2)
    phase = 2 * np.pi * m / (2 ** control_qubits)

    return phase



if __name__ == "__main__":
    control_qubits = 5
    theta_values = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 3 * np.pi / 16]

    estimated_phases = [phase_estimation(control_qubits, theta) for theta in theta_values]

    print(f"Estimated phases: {estimated_phases}")
    print(f"Actual phases: {theta_values}")

    if np.allclose(estimated_phases, theta_values):
        print("Phase estimation successful!")
    else:
        print("Phase estimation failed.")