'''
Grover's Search Algorithm
'''

from qsim.operators import Operator
from qsim.circuits import Circuit
from qsim.simulator import StatevectorSimulator
import numpy as np



def make_f(marked_bit):
    """
    Creates a function that returns 1 for the marked bit and 0 for all other bits.
    Parameters:
        marked_bit (str): The input bit that the function returns 1 for
    Returns:
        function: A function that takes a binary string and returns 1 for the marked bit, else 0
    """
    def f(x):
        return 1 if x == marked_bit else 0

    return f


def make_phase_oracle(n, function):
    """
    Creates the phase oracle
    Parameters:
        n (int): The number of bits
        function (function): The function f to be evaluated.
    Returns:
        Operator: An operator object representing the phase oracle.
    """
    dimension = 2 ** n
    oracle = np.zeros((dimension, dimension), dtype=np.complex128)

    for x in range(2 ** n):
        x_binary = format(x, f'0{n}b')
        f_x = function(x_binary)

        oracle[x, x] = -1 ** f_x

    return Operator(oracle)


def grovers_search_algorithm(n, function):
    """
    Implements Grover's search algorithm.
    Parameters:
        n (int): The number of qubits
        function (function): The function f to be evaluated
    Returns:
        str: The marked bit found by Grover's algorithm
    """
    sim = StatevectorSimulator()


    # Initialisation
    circuit = Circuit(num_qubits=n)


    # Create uniform superposition
    for qubit in range(n):
        circuit.hadamard(qubit)


    # Apply Grover iteration (up to global phase)
    phase_oracle = make_phase_oracle(n, function)
    for _ in range(int(np.pi / 4 * np.sqrt(2 ** n))):
        circuit.add_gate(phase_oracle, tuple(range(n)))

        for qubit in range(n):
            circuit.hadamard(qubit)
        for qubit in range(n):
            circuit.pauli_x(qubit)
        # multi controlled Z gate implementation here

        for qubit in range(n):
            circuit.pauli_x(qubit)
        for qubit in range(n):
            circuit.hadamard(qubit)
        

        
