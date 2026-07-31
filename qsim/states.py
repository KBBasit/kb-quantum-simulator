'''
A quantum state is a vector in a Hilbert space. This module
provides classes for representing quantum states and
operations on them.
'''

import numpy as np



class StateVector:
    '''
    A class representing a quantum state vector.
    '''
    def __init__(self, amplitudes):
        amplitudes = np.asarray(amplitudes, dtype=np.complex128)

        if amplitudes.ndim != 1:
            raise ValueError("State vector must be a 1D array.")

        if amplitudes.size == 0 or amplitudes.size & (amplitudes.size - 1) != 0:
            raise ValueError("State vector size must be a power of 2.")

        if not np.isclose(np.linalg.norm(amplitudes), 1.0):
            raise ValueError("State vector must be normalized.")

        self.amplitudes = amplitudes


    @property
    def num_qubits(self):
        '''
        Return the number of qubits represented by this state vector.
        '''
        return int(np.log2(self.amplitudes.size))


    @property
    def number_of_dimensions(self):
        '''
        Return the number of dimensions of the state vector.
        '''
        return self.amplitudes.size