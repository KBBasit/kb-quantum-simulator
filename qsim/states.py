"""
A quantum state is a vector in a Hilbert space. This module
provides the class for representing quantum states and
operations on them.
"""

import numpy as np



class StateVector:
    """
    A class for representing quantum state vectors.

    Attributes
    ----------
    amplitudes : numpy.ndarray
        The complex probability amplitudes of the quantum state.
    num_qubits : int
        The number of qubits for the quantum state.
    dimension : int
        The number of dimensions of the state vector.
    """
    def __init__(self, amplitudes):
        """
        Initialise a StateVector.

        Parameters
        ----------
        amplitudes : (array-like)
            Complex probability amplitudes of the quantum state.
        """
        amplitudes = np.asarray(amplitudes, dtype=np.complex128)

        if amplitudes.ndim != 1:
            raise ValueError("State vector must be a 1D array.")

        if amplitudes.size == 0 or amplitudes.size & (amplitudes.size - 1) != 0:
            raise ValueError("State vector size must be a power of 2.")

        if not np.isclose(np.linalg.norm(amplitudes), 1.0):
            raise ValueError("State vector must be normalised.")

        self.amplitudes = amplitudes


    @property
    def num_qubits(self):
        """
        Return the number of qubits represented by the state
        state vector.
        """
        return int(np.log2(self.amplitudes.size))


    @property
    def dimension(self):
        """
        Return the number of dimensions of the state
        vector.
        """
        return self.amplitudes.size


    @classmethod
    def zero_state(cls, num_qubits):
        """
        Create a state vector representing the |0> state
        for the given number of qubits.

        Parameters
        ----------
        num_qubits : int
            The number of qubits in the quantum state.

        Returns
        -------
        StateVector
            A StateVector object representing the |0> state.
        """
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1.")

        amplitudes = np.zeros(2 ** num_qubits, dtype=np.complex128)
        amplitudes[0] = 1.0

        return cls(amplitudes)


    def copy(self):
        """
        Make a copy of the state vector.

        Returns
        -------
        StateVector
            A copy of the current StateVector object.
        """
        return StateVector(self.amplitudes.copy())