'''
Deutsch-Jozsa Algorithm
'''

from qsim.operators import Operator
from qsim.circuits import Circuit
from qsim.simulator import StatevectorSimulator
from qsim.measurements import measure
import numpy as np



def make_f(n, function_type):
    """
    Creates a function f: {0,1}^n -> {0,1} that is either constant or balanced.
    Parameters:
        n (int): The number of input bits.
        function_type (str): The type of function to create ('constant' or 'balanced').
    Returns:
        function: A function that takes a binary string of length n and returns 0 or 1.
    """
    if function_type == 'constant':
        return lambda x: 0
    elif function_type == 'balanced':
        def balanced_function(x):
            index = int(x, 2)
            return 0 if index < (2 ** n) / 2 else 1

        return balanced_function


def make_oracle(n, function):
    """
    Creates an oracle for the Deutsch-Jozsa algorithm.
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


def deutsch_jozsa_algorithm(n, function):
    '''
    Implements the Deutsch-Jozsa algorithm
    Parameters:
        n (int): The number of input bits.
        function (function): The function f to be evaluated.
    Returns:
        str: The result of the Deutsch-Jozsa algorithm ('constant' or 'balanced').
    '''
    circuit = Circuit(num_qubits=n + 1)
    oracle = make_oracle(n, function)
    ancilla = n

    circuit.pauli_x(ancilla)
    circuit.hadamard(ancilla)

    for qubit in range(n):
        circuit.hadamard(qubit)

    circuit.add_gate(oracle, tuple(range(n + 1)))

    for qubit in range(n):
        circuit.hadamard(qubit)

    sim = StatevectorSimulator()
    final_state = sim.run(circuit)

    measurement = measure(final_state)

    if measurement[:-1] == '0' * n:
        return 'constant'
    else:
        return 'balanced'



if __name__ == "__main__":
    n = 10

    constant_f = make_f(n, 'constant')
    balanced_f = make_f(n, 'balanced')

    constant_result = deutsch_jozsa_algorithm(n, constant_f)
    balanced_result = deutsch_jozsa_algorithm(n, balanced_f)

    print(f"Result for constant function: {constant_result}")
    print(f"Result for balanced function: {balanced_result}")