from qsim import gates
from qsim import circuits
import pytest



def test_circuit_initialisation():
    circuit = circuits.Circuit(num_qubits=3)

    assert circuit.num_qubits == 3
    assert len(circuit) == 0

    with pytest.raises(ValueError, match="Number of qubits must be a positive integer."):
        circuits.Circuit(num_qubits=0)

    with pytest.raises(ValueError, match="Number of qubits must be a positive integer."):
        circuits.Circuit(num_qubits=-1)


def test_add_gate_validations():
    circuit = circuits.Circuit(num_qubits=3)
    operator = circuits.Operator(gates.cnot())

    with pytest.raises(TypeError, match="Operator must be an instance of the Operator class."):
        circuit.add_gate("not_an_operator", target_qubits=(0,))

    with pytest.raises(TypeError, match="Target qubits must be a tuple."):
        circuit.add_gate(operator, target_qubits=[0])

    with pytest.raises(TypeError, match="Target qubits must be integers."):
        circuit.add_gate(operator, target_qubits=(0, '1'))

    with pytest.raises(ValueError, match="Target qubits must be unique."):
        circuit.add_gate(operator, target_qubits=(0, 0))

    with pytest.raises(ValueError, match="Number of target qubits must match the operator's qubit count."):
        circuit.add_gate(operator, target_qubits=(0, 1, 2))

    with pytest.raises(ValueError, match="Target qubits must be within the range of the circuit's qubits."):
        circuit.add_gate(operator, target_qubits=(0, 3))