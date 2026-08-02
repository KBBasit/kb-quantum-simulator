'''
This module contains the implementation of operators used
in the simulation of quantum circuits. It provides classes
and functions to define and manipulate various quantum
gates, including single and multiple qubit operations.
'''

import numpy as np



class Operator:
    '''
    A class representing a quantum operator (gate).
    '''
    def __init__(self, matrix):
        matrix = np.asarray(matrix, dtype=np.complex128)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Operator must be a square matrix.")

        self.matrix = matrix


    @property
    def dimension(self):
        '''
        Return the dimension of the operator.
        '''
        return self.matrix.shape[0]

    