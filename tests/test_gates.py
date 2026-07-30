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


def test_pauli_x_sqaured_is_identity():
    x = gates.pauli_x()
    identity = gates.identity()

    assert np.allclose(x @ x, identity)