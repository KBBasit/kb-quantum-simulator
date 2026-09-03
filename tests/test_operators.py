from qsim import oprators
import pytest



def test_operator_initialization():
    matrix = [[1, 0], [0, 1]]
    operator = oprators.Operator(matrix)

    assert operator.dimension == 2
    assert operator.is_unitary
    assert operator.operator_qubits == 1

    # Test invalid operator initialization (non-square matrix)
    with pytest.raises(ValueError, match="Operator must be a square matrix."):
        oprators.Operator([[1, 0], [0, 1], [1, 0]])  # Non-square matrix

    # Test invalid operator initialization (dimension not a power of 2)
    with pytest.raises(ValueError, match="Operator dimension must be a power of 2."):
        oprators.Operator([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Dimension not a power of 2