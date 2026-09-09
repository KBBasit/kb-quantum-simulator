'''
Quantum gates are the building blocks of quantum circuits
that change the state of qubits. This module provides
implementations of some common quantum gates using NumPy
arrays.
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
    Return the 2x2 Hadamard gate, maps computational basis
    states to equal superpositions
    '''
    return 1 / np.sqrt(2) * np.array([
        [1,1],
        [1,-1]], dtype=np.complex128)


def s_gate():
    '''
    Return the 2x2 S phase gate
    '''
    return np.array([
        [1,0],
        [0,1j]], dtype=np.complex128)


def t_gate():
    '''
    Return the 2x2 T phase gate
    '''
    return np.array([
        [1,0],
        [0,np.exp(1j * np.pi / 4)]], dtype=np.complex128)


def identity(dim=2):
    '''
    Return the identity gate of specified dimension.
    '''
    return np.eye(dim, dtype=np.complex128)


def cnot():
    '''
    Return the 4x4 CNOT gate, a two-qubit gate that flips
    the target qubit if the control qubit is in state |1>.
    '''
    return np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,0,1],
        [0,0,1,0]], dtype=np.complex128)


def controlled_z():
    '''
    Return the 4x4 Controlled-Z gate, a two-qubit gate that
    applies a -1 phase only when both qubits are 1.
    '''
    return np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,-1]], dtype=np.complex128)


def swap():
    '''
    Return the 4x4 SWAP gate, a two-qubit gate that swaps
    the states of the two qubits.
    '''
    return np.array([
        [1,0,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,0,1]], dtype=np.complex128)


def r_x(theta):
    '''
    Return the 2x2 single-qubit rotation gate representing
    a rotation by angle theta about the X-axis of the Bloch 
    sphere.
    '''
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)


def r_y(theta):
    '''
    Return the 2x2 single-qubit rotation gate representing
    a rotation by angle theta about the Y-axis of the Bloch 
    sphere.
    '''
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)


def r_z(theta):
    '''
    Return the 2x2 single-qubit rotation gate representing
    a rotation by angle theta about the Z-axis of the Bloch
    sphere.
    '''
    return np.array([
        [np.exp(-1j * theta/2), 0],
        [0, np.exp(1j * theta/2)]], dtype=np.complex128)


def phase_gate(theta):
    '''
    Return the 2x2 general phase gate that applies a phase
    shift of theta to the |1> state.
    '''
    return np.array([
        [1, 0],
        [0, np.exp(1j * theta)]], dtype=np.complex128)        