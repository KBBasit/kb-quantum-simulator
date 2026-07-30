'''
Quantum gates are the building blocks of quantum circuits
that change the state of qubits.
This module is for the design of quantum gates using NumPy
arrays. Every quantum gate has a complex data type
specified for ease of later multiplication.
'''

import numpy as np



def pauli_x():
    ''' 
    Return the 2x2 Pauli X gate
    '''
    return np.array([
        [0,1],
        [1,0]], dtype=np.complex128)


def pauli_y():
    '''
    Return the 2x2 Pauli Y gate
    '''
    return np.array([
        [0,-1j],
        [1j,0]], dtype=np.complex128)


def pauli_z():
    '''
    Return the 2x2 Pauli Z gate
    '''
    return np.array([
        [1,0],
        [0,-1]], dtype=np.complex128)


def hadamard():
    '''
    Return the 2x2 Hadamard gate (45° roation in the X-Z
    plane, creating superposition from computational basis
    states)
    '''
    return 1 / np.sqrt(2) * np.array([
        [1,1],
        [1,-1]], dtype=np.complex128)


def s_gate():
    '''
    Return the 2x2 S gate (90° rotation about the Z axis)
    '''
    return np.array([
        [1,0],
        [0,1j]], dtype=np.complex128)


def t_gate():
    '''
    Return the 2x2 T gate (45° rotation about the Z axis)
    '''
    return np.array([
        [1,0],
        [0,np.exp(1j * np.pi / 4)]], dtype=np.complex128)


def identity():
    '''
    Return the 2x2 identity matrix
    '''
    return np.array([
        [1,0],
        [0,1]], dtype=np.complex128)