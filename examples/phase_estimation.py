'''
Quantum Phase Estimation
'''

from qsim.gates import phase_gate
from qsim.circuits import Circuit



def phase_estimation(control_qubits, theta):
    """
    Constructs a quantum circuit for phase estimation.

    Parameters:
        control_qubits (int): The number of control qubits.
        theta (float): The phase to be estimated.
    Returns:
        phase (float): The estimated phase.
    """
    circuit = Circuit(num_qubits=control_qubits + 1)

    for control_qubit in range(control_qubits):
        circuit.hadamard(control_qubit)

    for control_qubit in range(control_qubits):
        circuit.controlled_phase(control_qubit, circuit.num_qubits - 1, theta * (2 ** control_qubit))