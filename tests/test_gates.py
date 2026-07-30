from qsim import gates
import numpy as np



def test_pauli_x_sqaured_is_identity():
    x = gates.pauli_x()
    identity = gates.identity()

    assert np.allclose(x @ x, identity)