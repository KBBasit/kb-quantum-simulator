from qsim import gates
import numpy as np



def test_validate_identity():
    x = gates.pauli_x()
    y = gates.pauli_y()
    z = gates.pauli_z()
    identity = gates.identity()

    assert np.allclose(identity @ x, x)
    assert np.allclose(identity @ y, y)
    assert np.allclose(identity @ z, z)


def is_matrix_squared_identity(matrix):
    identity = gates.identity()
    return np.allclose(matrix @ matrix, identity)


def is_unitary(matrix):
    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    return np.allclose(matrix.conj().T @ matrix, identity)


def test_pauli_x_properties():
    x = gates.pauli_x()

    assert is_matrix_squared_identity(x)
    assert is_unitary(x)


def test_pauli_y_properties():
    y = gates.pauli_y()

    assert is_matrix_squared_identity(y)
    assert is_unitary(y)


def test_pauli_z_properties():
    z = gates.pauli_z()

    assert is_matrix_squared_identity(z)
    assert is_unitary(z)


def test_hadamard_properties():
    h = gates.hadamard()

    assert is_matrix_squared_identity(h)
    assert is_unitary(h)


def test_s_gate_properties():
    s = gates.s_gate()

    assert is_unitary(s)
    assert np.allclose(s @ s, gates.pauli_z())


def test_t_gate_properties():
    t = gates.t_gate()

    assert is_unitary(t)
    assert np.allclose(t @ t, gates.s_gate())