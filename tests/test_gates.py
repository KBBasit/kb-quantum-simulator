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


def test_pauli_x_squared_is_identity():
    x = gates.pauli_x()
    identity = gates.identity()

    assert np.allclose(x @ x, identity)


def test_pauli_y_squared_is_identity():
    y = gates.pauli_y()
    identity = gates.identity()

    assert np.allclose(y @ y, identity)


def test_pauli_z_squared_is_identity():
    z = gates.pauli_z()
    identity = gates.identity()

    assert np.allclose(z @ z, identity)


def test_hadamard_squared_is_identity():
    h = gates.hadamard()
    identity = gates.identity()

    assert np.allclose(h @ h, identity)


def test_pauli_x_is_unitary():
    x = gates.pauli_x()
    identity = gates.identity()

    assert np.allclose(x @ np.transpose(x), identity)


def test_pauli_y_is_unitary():
    y = gates.pauli_y()
    identity = gates.identity()

    assert np.allclose(y @ np.transpose(y), identity)


def test_pauli_z_is_unitary():
    z = gates.pauli_z()
    identity = gates.identity()

    assert np.allclose(z @ np.transpose(z), identity)


def test_hadamard_is_unitary():
    h = gates.hadamard()
    identity = gates.identity()

    assert np.allclose(h @ np.transpose(h), identity)