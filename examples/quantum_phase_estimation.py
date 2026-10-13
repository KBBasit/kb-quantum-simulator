"""
Quantum Phase Estimation
"""

import numpy as np
from kb_quantum_simulator.circuits import Circuit
from kb_quantum_simulator.simulator import StatevectorSimulator



def phase_estimation(control_qubits, theta):
    """
    Constructs a quantum circuit for phase estimation.

    Parameters
    ----------
    control_qubits : int
        The number of control qubits.
    theta : float
        The phase to be estimated.

    Returns
    -------
    float
        The estimated phase.
    """
    sim = StatevectorSimulator()

    # Initialisation
    circuit = Circuit(num_qubits=control_qubits + 1)
    target_qubit = circuit.num_qubits - 1

    # Prepare target qubit in the state |1>
    circuit.pauli_x(target_qubit=target_qubit)

    # Create uniform superposition over the control qubits
    for control_qubit in range(control_qubits):
        circuit.hadamard(target_qubit=control_qubit)

    # Apply controlled phase gates
    for control_qubit in range(control_qubits):
        circuit.controlled_phase(control_qubit=control_qubit, target_qubit=target_qubit, theta=theta * (2 ** (control_qubits - control_qubit - 1)))

    # Apply the inverse Quantum Fourier Transform
    circuit.inverse_quantum_fourier_transform(target_qubits=tuple(range(control_qubits)))

    # Measure the control qubits
    circuit.add_measurement(key = "Measurement", qubits = tuple(range(control_qubits)))

    _, measurement_results = sim.run(circuit)

    bitstring = measurement_results["Measurement"]
    m = int(bitstring, 2)
    phase = float(2 * np.pi * m / (2 ** control_qubits))

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