import numpy as np
import pytest
from qsim.gates import identity, cnot, pauli_x, pauli_y
from qsim.states import StateVector
from qsim.operators import Operator



def test_operator_initialisation():
    matrix = identity()
    operator = Operator(matrix)

    assert operator.dimension == 2
    assert operator.is_unitary
    assert operator.operator_qubits == 1

    with pytest.raises(ValueError, match="Operator must be a square matrix."):
        Operator([[1, 0], [0, 1], [1, 0]])

    with pytest.raises(ValueError, match="Operator dimension must be a power of 2."):
        Operator([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_operator_application_validations():
    cnot_matrix = cnot()
    cnot_operator = Operator(cnot_matrix)
    state_vector = StateVector([1, 0, 0, 0]) 

    with pytest.raises(TypeError, match="State vector must be a StateVector object."):
        cnot_operator.apply_gate([1, 0, 0, 0], target_qubits=(0, 1), inplace=False)

    non_unitary_matrix = [[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 1, 0]]
    non_unitary_operator = Operator(non_unitary_matrix)

    with pytest.raises(ValueError, match="Operator must be unitary."):
        non_unitary_operator.apply_gate(state_vector, target_qubits=(0, 1), inplace=False)

    with pytest.raises(TypeError, match="Target qubits must be a tuple."):
        cnot_operator.apply_gate(state_vector, target_qubits=[0, 1], inplace=False)

    with pytest.raises(TypeError, match="Target qubits must be integers."):
        cnot_operator.apply_gate(state_vector, target_qubits=('0', '1'), inplace=False)

    with pytest.raises(ValueError, match="Number of target qubits must match the operator's qubit count."):
        cnot_operator.apply_gate(state_vector, target_qubits=(0,), inplace=False)

    with pytest.raises(ValueError, match="Target qubits must be within the range of the state vector's qubits."):
        cnot_operator.apply_gate(state_vector, target_qubits=(0, 2), inplace=False)

    with pytest.raises(ValueError, match="Target qubits must be unique."):
        cnot_operator.apply_gate(state_vector, target_qubits=(0, 0), inplace=False)


def test_operator_application():
    matrix_1 = pauli_x()
    matrix_2 = pauli_y() 
    matrix_3 = cnot()
    operator_1 = Operator(matrix_1)
    operator_2 = Operator(matrix_2)
    operator_3 = Operator(matrix_3)

    state_vector_1 = StateVector([1, 0])
    state_vector_2 = StateVector([0, 1])
    state_vector_3 = StateVector(1/np.sqrt(2) * np.array([1, 0, 1, 0]))
    state_vector_4 = StateVector([1, 0, 0, 0, 0, 0, 0, 0])
    state_vector_5 = StateVector(1/np.sqrt(2) * np.array([1, 0, 0, 0, 1, 0, 0, 0]))

    result_state_vector_1 = operator_1.apply_gate(state_vector_1, target_qubits=(0,), inplace=False)
    assert result_state_vector_1 is not state_vector_1
    assert np.allclose(result_state_vector_1.amplitudes, [0, 1])

    result_state_vector_2 = operator_2.apply_gate(state_vector_2, target_qubits=(0,), inplace=True)
    assert np.allclose(result_state_vector_2.amplitudes, [-1j, 0])
    assert result_state_vector_2 is state_vector_2

    result_state_vector_3 = operator_3.apply_gate(state_vector_3, target_qubits=(0, 1), inplace=False)
    assert np.allclose(result_state_vector_3.amplitudes, 1/np.sqrt(2) * np.array([1, 0, 0, 1]))

    temp_state_vector_4 = state_vector_4.copy()
    for q in (0,2):
        temp_state_vector_4 = operator_1.apply_gate(temp_state_vector_4, target_qubits=(q,), inplace=False)
    assert np.allclose(temp_state_vector_4.amplitudes, [0, 0, 0, 0, 0, 1, 0, 0])

    result_state_vector_5 = operator_3.apply_gate(state_vector_5, target_qubits=(0, 2), inplace=False)
    assert np.allclose(result_state_vector_5.amplitudes, 1/np.sqrt(2) * np.array([1, 0, 0, 0, 0, 1, 0, 0]))