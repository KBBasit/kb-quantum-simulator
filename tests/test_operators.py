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


def test_operator_application():
    matrix = [[0, 1], [1, 0]]  # Pauli-X gate
    operator = operators.Operator(matrix)
    state_vector_1 = states.StateVector([1, 0])  # |0> state

    result_state_vector = operator.apply_gate(state_vector_1, target_qubits=(0,), inplace=False)

    assert np.allclose(result_state_vector.amplitudes, [0, 1])  # |1> state after applying X gate

