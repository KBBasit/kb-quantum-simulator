from qsim import gates
import numpy as np



def _is_matrix_squared_identity(matrix):
    identity = gates.identity(dim=matrix.shape[0])
    return np.allclose(matrix @ matrix, identity)


def _is_unitary(matrix):
    identity = gates.identity(dim=matrix.shape[0])
    return np.allclose(matrix.conj().T @ matrix, identity)


def test_validate_identity():
    identity_2 = gates.identity()
    identity_4 = gates.identity(dim=4)

    assert identity_2.shape == (2, 2)
    assert identity_4.shape == (4, 4)

    assert np.allclose(identity_2 @ gates.pauli_x(), gates.pauli_x())
    assert np.allclose(identity_4 @ gates.cnot(), gates.cnot())


def test_pauli_x_properties():
    x = gates.pauli_x()

    assert _is_matrix_squared_identity(x)
    assert _is_unitary(x)


def test_pauli_y_properties():
    y = gates.pauli_y()

    assert _is_matrix_squared_identity(y)
    assert _is_unitary(y)


def test_pauli_z_properties():
    z = gates.pauli_z()

    assert _is_matrix_squared_identity(z)
    assert _is_unitary(z)


def test_hadamard_properties():
    h = gates.hadamard()

    assert _is_matrix_squared_identity(h)
    assert _is_unitary(h)


def test_s_gate_properties():
    s = gates.s_gate()

    assert _is_unitary(s)
    assert np.allclose(s @ s, gates.pauli_z())


def test_t_gate_properties():
    t = gates.t_gate()

    assert _is_unitary(t)
    assert np.allclose(t @ t, gates.s_gate())


def test_cnot_properties():
    cnot = gates.cnot()

    assert _is_matrix_squared_identity(cnot)
    assert _is_unitary(cnot)


def test_controlled_z_properties():
    cz = gates.controlled_z()

    assert _is_matrix_squared_identity(cz)
    assert _is_unitary(cz)


def test_swap_properties():
    swap = gates.swap()

    assert _is_matrix_squared_identity(swap)
    assert _is_unitary(swap)


def test_r_x_properties():
    theta = np.pi / 3
    rx = gates.r_x(theta)

    assert _is_unitary(rx)
    assert np.allclose(gates.r_x(0), gates.identity())
    assert np.allclose(gates.r_x(np.pi), -1j * gates.pauli_x())


def test_r_y_properties():
    theta = np.pi / 3
    ry = gates.r_y(theta)

    assert _is_unitary(ry)
    assert np.allclose(gates.r_y(0), gates.identity())
    assert np.allclose(gates.r_y(np.pi), -1j * gates.pauli_y())


def test_r_z_properties():
    theta = np.pi / 3
    rz = gates.r_z(theta)

    assert _is_unitary(rz)
    assert np.allclose(gates.r_z(0), gates.identity())
    assert np.allclose(gates.r_z(np.pi), -1j * gates.pauli_z())


def test_phase_gate_properties():
    theta = np.pi / 3
    phase = gates.phase_gate(theta)

    assert _is_unitary(phase)
    assert np.allclose(gates.phase_gate(0), gates.identity())
    assert np.allclose(gates.phase_gate(np.pi / 2), gates.s_gate())
    assert np.allclose(gates.phase_gate(np.pi / 4), gates.t_gate())


def test_cnot_action():
    cnot = gates.cnot()

    ket_00 = np.array([1, 0, 0, 0], dtype=np.complex128)
    ket_01 = np.array([0, 1, 0, 0], dtype=np.complex128)
    ket_10 = np.array([0, 0, 1, 0], dtype=np.complex128)
    ket_11 = np.array([0, 0, 0, 1], dtype=np.complex128)

    assert np.allclose(cnot @ ket_00, ket_00)
    assert np.allclose(cnot @ ket_01, ket_01)
    assert np.allclose(cnot @ ket_10, ket_11)
    assert np.allclose(cnot @ ket_11, ket_10)


def test_controlled_z_action():
    cz = gates.controlled_z()

    ket_00 = np.array([1, 0, 0, 0], dtype=np.complex128)
    ket_01 = np.array([0, 1, 0, 0], dtype=np.complex128)
    ket_10 = np.array([0, 0, 1, 0], dtype=np.complex128)
    ket_11 = np.array([0, 0, 0, 1], dtype=np.complex128)

    assert np.allclose(cz @ ket_00, ket_00)
    assert np.allclose(cz @ ket_01, ket_01)
    assert np.allclose(cz @ ket_10, ket_10)
    assert np.allclose(cz @ ket_11, -ket_11)


def test_swap_action():
    swap = gates.swap()

    ket_00 = np.array([1, 0, 0, 0], dtype=np.complex128)
    ket_01 = np.array([0, 1, 0, 0], dtype=np.complex128)
    ket_10 = np.array([0, 0, 1, 0], dtype=np.complex128)
    ket_11 = np.array([0, 0, 0, 1], dtype=np.complex128)

    assert np.allclose(swap @ ket_00, ket_00)
    assert np.allclose(swap @ ket_01, ket_10)
    assert np.allclose(swap @ ket_10, ket_01)
    assert np.allclose(swap @ ket_11, ket_11)