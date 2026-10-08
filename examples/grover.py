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


def make_oracle(n, function):
    """
    Creates the standard oracle
    Parameters:
        n (int): The number of input bits.
        function (function): The function f to be evaluated.
    Returns:
        Operator: An operator object representing the oracle.
    """
    dimension = 2 ** (n + 1)
    oracle = np.zeros((dimension, dimension), dtype=np.complex128)

    for x in range(2 ** n):
        x_binary = format(x, f'0{n}b')
        f_x = function(x_binary)

        for y in range(2):
            output_y = y ^ f_x

            input_index = 2 * x + y
            output_index = 2 * x + output_y

            oracle[output_index, input_index] = 1

    return Operator(oracle)


def grovers_search_algorithm(n):
    """
    
    """
    sim = StatevectorSimulator()

    # Initialisation
    circuit = Circuit(num_qubits=n)

    # Create uniform superposition
    for qubit in range(n):
        circuit.hadamard(qubit)
    