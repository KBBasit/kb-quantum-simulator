from qsim import states
from qsim import operators
import numpy as np
import pytest



def test_operator_initialization():
    matrix = [[1, 0], [0, 1]]
    operator = operators.Operator(matrix)

    assert operator.dimension == 2
    assert operator.is_unitary
    assert operator.operator_qubits == 1

    # Test invalid operator initialization (non-square matrix)
    with pytest.raises(ValueError, match="Operator must be a square matrix."):
        operators.Operator([[1, 0], [0, 1], [1, 0]])  # Non-square matrix

    # Test invalid operator initialization (dimension not a power of 2)
    with pytest.raises(ValueError, match="Operator dimension must be a power of 2."):
        operators.Operator([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Dimension not a power of 2


def test_operator_application_validations():
    cnot_matrix = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]]
    cnot_operator = operators.Operator(cnot_matrix)
    state_vector = states.StateVector([1, 0, 0, 0]) 

    # Test invalid state vector type
    with pytest.raises(TypeError, match="State vector must be a StateVector object."):
        cnot_operator.apply_gate([1, 0, 0, 0], target_qubits=(0, 1), inplace=False)

    # Test non-unitary operator
    non_unitary_matrix = [[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 1, 0]]
    non_unitary_operator = operators.Operator(non_unitary_matrix)
    with pytest.raises(ValueError, match="Operator must be unitary."):
        non_unitary_operator.apply_gate(state_vector, target_qubits=(0, 1), inplace=False)

    # Test invalid target qubits type 
    with pytest.raises(TypeError, match="Target qubits must be integers."):
        cnot_operator.apply_gate(state_vector, target_qubits=('0', '1'), inplace=False)

    # Test mismatched number of target qubits
    with pytest.raises(ValueError, match="Number of target qubits must match the operator's qubit count."):
        cnot_operator.apply_gate(state_vector, target_qubits=(0,), inplace=False)

    # Test out-of-bounds target qubit index
    with pytest.raises(ValueError, match="Target qubits must be within the range of the state vector's qubits."):
        cnot_operator.apply_gate(state_vector, target_qubits=(0, 2), inplace=False)

    # Test target qubits are unique
    with pytest.raises(ValueError, match="Target qubits must be unique."):
        cnot_operator.apply_gate(state_vector, target_qubits=(0, 0), inplace=False)


def test_operator_application():
    matrix = [[0, 1], [1, 0]]  # Pauli-X gate
    operator = operators.Operator(matrix)
    state_vector_1 = states.StateVector([1, 0])  # |0> state

    result_state_vector = operator.apply_gate(state_vector_1, target_qubits=(0,), inplace=False)

    assert np.allclose(result_state_vector.amplitudes, [0, 1])  # |1> state after applying X gate

