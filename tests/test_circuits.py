import numpy as np
import pytest
from kb_quantum_simulator import gates
from kb_quantum_simulator.operators import Operator
from kb_quantum_simulator.circuits import Circuit



def test_circuit_initialisation():
    circuit = Circuit(num_qubits=3)

    assert circuit.num_qubits == 3
    assert len(circuit) == 0

    with pytest.raises(ValueError, match="Number of qubits must be a positive integer."):
        Circuit(num_qubits=0)

    with pytest.raises(ValueError, match="Number of qubits must be a positive integer."):
        Circuit(num_qubits=-1)


def test_add_gate_validations():
    circuit = Circuit(num_qubits=3)
    operator = Operator(gates.cnot())

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


def test_add_measurement_validations():
    circuit = Circuit(num_qubits=3)

    with pytest.raises(TypeError, match="Target qubits must be a tuple or None."):
        circuit.add_measurement(key="alice", qubits=[0])

    with pytest.raises(TypeError, match="All target qubit indices must be integers."):
        circuit.add_measurement(key="alice", qubits=(0, '1'))

    with pytest.raises(ValueError, match="Target qubit indices must be unique."):
        circuit.add_measurement(key="alice", qubits=(0, 0))

    with pytest.raises(ValueError, match="Target qubit indices must be within the range of the circuit's qubits."):
        circuit.add_measurement(key="alice", qubits=(0, 3))

    with pytest.raises(TypeError, match="Measurement key must be a string."):
        circuit.add_measurement(key=123, qubits=(0,))

    circuit.add_measurement(key="alice", qubits=(0,))
    with pytest.raises(ValueError, match="Measurement key 'alice' already exists in the circuit."):
        circuit.add_measurement(key="alice", qubits=(1,))


def test_c_apply_validations():
    circuit = Circuit(num_qubits=3)
    operator = Operator(gates.cnot())
    circuit.add_measurement(key="alice", qubits=(0,1))
    circuit.add_measurement(key="Eve", qubits=None)

    with pytest.raises(TypeError, match="Operator must be an instance of the Operator class."):
        circuit.c_apply("not_an_operator", measurement_key="alice", measurement_condition="00", target_qubits=(1,))

    with pytest.raises(TypeError, match="Target qubits must be a tuple of integers."):
        circuit.c_apply(operator, measurement_key="alice", measurement_condition="00", target_qubits=[0,1])

    with pytest.raises(TypeError, match="All target qubit indices must be integers."):
        circuit.c_apply(operator, measurement_key="alice", measurement_condition="01", target_qubits=(0,'1'))

    with pytest.raises(ValueError, match="Number of target qubits must match the operator's qubit count."):
        circuit.c_apply(operator, measurement_key="alice", measurement_condition="00", target_qubits=(0,))

    with pytest.raises(ValueError, match="Target qubit indices must be unique."):
        circuit.c_apply(operator, measurement_key="alice", measurement_condition="00", target_qubits=(0,0))

    with pytest.raises(ValueError, match="Target qubit indices must be within the range of the circuit's qubits."):
        circuit.c_apply(operator, measurement_key="alice", measurement_condition="00", target_qubits=(0,3))

    with pytest.raises(TypeError, match="Measurement condition must be a string."):
        circuit.c_apply(operator, measurement_key="alice", measurement_condition=00, target_qubits=(0,1))

    with pytest.raises(ValueError, match="Measurement condition must be a valid bitstring."):
        circuit.c_apply(operator, measurement_key="alice", measurement_condition="20", target_qubits=(0,1))

    with pytest.raises(ValueError, match="Length of measurement condition must match the number of measured qubits."):
        circuit.c_apply(operator, measurement_key="alice", measurement_condition="0", target_qubits=(0,1))

    with pytest.raises(ValueError, match="Measurement key 'bob' does not exist in the circuit."):
        circuit.c_apply(operator, measurement_key="bob", measurement_condition="00", target_qubits=(0,1))

    with pytest.raises(TypeError, match="Measurement key must be a string."):
        circuit.c_apply(operator, measurement_key=123, measurement_condition="00", target_qubits=(0,1))

    with pytest.raises(ValueError, match="Length of measurement condition must match the number of measured qubits."):
        circuit.c_apply(operator, measurement_key="Eve", measurement_condition="0", target_qubits=(0,1))


def test_c_apply():
    circuit = Circuit(num_qubits=3)
    circuit.hadamard(target_qubit=0)
    circuit.add_measurement(key="alice", qubits=(0,))
    circuit.add_measurement(key="eve", qubits=None)
    circuit.c_apply(operator=Operator(gates.cnot()), measurement_key="alice", measurement_condition="1", target_qubits=(1,2))
    circuit.c_apply(operator=Operator(gates.cnot()), measurement_key="eve", measurement_condition="000", target_qubits=(1,2))

    operation_type_1, operator_1, measurement_key_1, measurement_condition_1, target_qubits_1 = circuit.operations[3]
    assert operation_type_1 == "c_apply"
    assert np.allclose(operator_1.matrix, gates.cnot())
    assert measurement_key_1 == "alice"
    assert measurement_condition_1 == "1"
    assert target_qubits_1 == (1,2)

    operation_type_2, operator_2, measurement_key_2, measurement_condition_2, target_qubits_2 = circuit.operations[4]
    assert operation_type_2 == "c_apply"
    assert np.allclose(operator_2.matrix, gates.cnot())
    assert measurement_key_2 == "eve"
    assert measurement_condition_2 == "000"
    assert target_qubits_2 == (1,2)


def test_add_measurement():
    circuit = Circuit(num_qubits=2)
    circuit.add_measurement(key="alice", qubits=(0,))

    operation_type, qubits, key = circuit.operations[0]

    assert operation_type == "measure"
    assert qubits == (0,)
    assert key == "alice"


def test_pauli_x_gate_addition():
    circuit = Circuit(num_qubits=2)
    circuit.pauli_x(target_qubit=0)

    operation_type, operator, target_qubits = circuit.operations[0]

    assert operation_type == "gate"
    assert np.allclose(operator.matrix, gates.pauli_x())
    assert target_qubits == (0,)


def test_pauli_y_gate_addition():
    circuit = Circuit(num_qubits=2)
    circuit.pauli_y(target_qubit=1)

    operation_type, operator, target_qubits = circuit.operations[0]

    assert operation_type == "gate"
    assert np.allclose(operator.matrix, gates.pauli_y())
    assert target_qubits == (1,)


def test_pauli_z_gate_addition():
    circuit = Circuit(num_qubits=2)
    circuit.pauli_z(target_qubit=0)

    operation_type, operator, target_qubits = circuit.operations[0]
    
    assert operation_type == "gate"
    assert np.allclose(operator.matrix, gates.pauli_z())
    assert target_qubits == (0,)


def test_hadamard_gate_addition():
    circuit = Circuit(num_qubits=2)
    circuit.hadamard(target_qubit=1)

    operation_type, operator, target_qubits = circuit.operations[0]

    assert operation_type == "gate"
    assert np.allclose(operator.matrix, gates.hadamard())
    assert target_qubits == (1,)


def test_s_gate_addition():
    circuit = Circuit(num_qubits=2)
    circuit.s_gate(target_qubit=0)

    operation_type, operator, target_qubits = circuit.operations[0]

    assert operation_type == "gate"
    assert np.allclose(operator.matrix, gates.s_gate())
    assert target_qubits == (0,)


def test_t_gate_addition():
    circuit = Circuit(num_qubits=2)
    circuit.t_gate(target_qubit=1)

    operation_type, operator, target_qubits = circuit.operations[0]

    assert operation_type == "gate"
    assert np.allclose(operator.matrix, gates.t_gate())
    assert target_qubits == (1,)


def test_cnot_gate_addition():
    circuit = Circuit(num_qubits=2)
    circuit.cnot(control_qubit=0, target_qubit=1)

    operation_type, operator, target_qubits = circuit.operations[0]

    assert operation_type == "gate"
    assert np.allclose(operator.matrix, gates.cnot())
    assert target_qubits == (0, 1)


def test_controlled_z_gate_addition():
    circuit = Circuit(num_qubits=2)
    circuit.controlled_z(control_qubit=0, target_qubit=1)

    operation_type, operator, target_qubits = circuit.operations[0]

    assert operation_type == "gate"
    assert np.allclose(operator.matrix, gates.controlled_z())
    assert target_qubits == (0, 1)


def test_swap_gate_addition():
    circuit = Circuit(num_qubits=2)
    circuit.swap(qubit1=0, qubit2=1)

    operation_type, operator, target_qubits = circuit.operations[0]

    assert operation_type == "gate"
    assert np.allclose(operator.matrix, gates.swap())
    assert target_qubits == (0, 1)


def test_r_x_gate_addition():
    circuit = Circuit(num_qubits=2)
    theta = np.pi / 4
    circuit.r_x(target_qubit=0, theta=theta)

    operation_type, operator, target_qubits = circuit.operations[0]

    assert operation_type == "gate"
    assert np.allclose(operator.matrix, gates.r_x(theta))
    assert target_qubits == (0,)


def test_r_y_gate_addition():
    circuit = Circuit(num_qubits=2)
    theta = np.pi / 4
    circuit.r_y(target_qubit=1, theta=theta)

    operation_type, operator, target_qubits = circuit.operations[0]

    assert operation_type == "gate"
    assert np.allclose(operator.matrix, gates.r_y(theta))
    assert target_qubits == (1,)


def test_r_z_gate_addition():
    circuit = Circuit(num_qubits=2)
    theta = np.pi / 4
    circuit.r_z(target_qubit=0, theta=theta)

    operation_type, operator, target_qubits = circuit.operations[0]

    assert operation_type == "gate"
    assert np.allclose(operator.matrix, gates.r_z(theta))
    assert target_qubits == (0,)


def test_phase_gate_addition():
    circuit = Circuit(num_qubits=2)
    theta = np.pi / 4
    circuit.phase_gate(target_qubit=1, theta=theta)

    operation_type, operator, target_qubits = circuit.operations[0]

    assert operation_type == "gate"
    assert np.allclose(operator.matrix, gates.phase_gate(theta))
    assert target_qubits == (1,)


def test_controlled_phase_gate_addition():
    circuit = Circuit(num_qubits=2)
    theta = np.pi / 4
    circuit.controlled_phase(control_qubit=0, target_qubit=1, theta=theta)

    operation_type, operator, target_qubits = circuit.operations[0]

    assert operation_type == "gate"
    assert np.allclose(operator.matrix, gates.controlled_phase(theta))
    assert target_qubits == (0, 1)


def test_length_of_circuit():
    circuit = Circuit(num_qubits=2)
    assert len(circuit) == 0

    circuit.pauli_x(target_qubit=0)
    assert len(circuit) == 1

    circuit.cnot(control_qubit=0, target_qubit=1)
    assert len(circuit) == 2


def test_representation_of_circuit():
    circuit = Circuit(num_qubits=2)
    circuit.pauli_x(target_qubit=0)
    circuit.cnot(control_qubit=0, target_qubit=1)

    expected_repr = "Circuit(num_qubits=2, num_operations=2)"
    assert repr(circuit) == expected_repr


def test_qft_and_inverse_qft_validations():
    circuit = Circuit(num_qubits=3)

    with pytest.raises(TypeError, match="Target qubits must be a tuple."):
        circuit.quantum_fourier_transform(target_qubits=[0, 1, 2])
    with pytest.raises(TypeError, match="Target qubits must be a tuple."):
        circuit.inverse_quantum_fourier_transform(target_qubits=[0, 1, 2])

    with pytest.raises(TypeError, match="All qubit indices must be integers."):
        circuit.quantum_fourier_transform(target_qubits=(0, '1', 2))
    with pytest.raises(TypeError, match="All qubit indices must be integers."):
        circuit.inverse_quantum_fourier_transform(target_qubits=(0, '1', 2))

    with pytest.raises(ValueError, match="Qubit indices must be unique."):
        circuit.quantum_fourier_transform(target_qubits=(0, 1, 1))
    with pytest.raises(ValueError, match="Qubit indices must be unique."):
        circuit.inverse_quantum_fourier_transform(target_qubits=(0, 1, 1))

    with pytest.raises(ValueError, match="Qubit indices must be within the range of the circuit's qubits."):
        circuit.quantum_fourier_transform(target_qubits=(0, 1, 3))
    with pytest.raises(ValueError, match="Qubit indices must be within the range of the circuit's qubits."):
        circuit.inverse_quantum_fourier_transform(target_qubits=(0, 1, 3))


def test_qft_and_inverse_qft_addition():
    circuit = Circuit(num_qubits=3)
    circuit.quantum_fourier_transform(target_qubits=(0, 1, 2))
    circuit.inverse_quantum_fourier_transform(target_qubits=(0, 1, 2))

    assert circuit.operations[0] == ("qft", (0, 1, 2))
    assert circuit.operations[1] == ("inverse_qft", (0, 1, 2))