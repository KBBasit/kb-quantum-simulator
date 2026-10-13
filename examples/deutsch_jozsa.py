"""
Deutsch-Jozsa Algorithm
"""

import numpy as np
from kb_quantum_simulator.operators import Operator
from kb_quantum_simulator.circuits import Circuit
from kb_quantum_simulator.simulator import StatevectorSimulator



def make_f(n, function_type):
    """
    Creates a function f: {0,1}^n -> {0,1} that is either
    constant or balanced.

    Parameters
    ----------
    n : int 
        The number of input bits.
    function_type : str
        The type of function to create ('constant' or 'balanced').

    Returns
    -------
    callable
        A function that takes a binary string of length n and returns 0 or 1.
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

    Parameters
    ----------
    n : int
        The number of input bits.
    function : callable
        The function f to be evaluated.

    Returns
    -------
    Operator
        An operator object representing the oracle.
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
    """
    Implements the Deutsch-Jozsa algorithm

    Parameters
    ----------
    n : int
        The number of input bits.
    function : callable
        The function f to be evaluated.

    Returns
    -------
    str
        The result of the Deutsch-Jozsa algorithm ('constant' or 'balanced').
    """
    sim = StatevectorSimulator()

    # Initialisation
    circuit = Circuit(num_qubits=n + 1)
    oracle = make_oracle(n, function)
    ancilla = n

    # Prepare ancilla qubit in the state |->
    circuit.pauli_x(ancilla)
    circuit.hadamard(ancilla)

    # Create uniform superposition over the input qubits
    for qubit in range(n):
        circuit.hadamard(qubit)

    # Apply the oracle
    circuit.add_gate(oracle, tuple(range(n + 1)))

    # Apply Hadamard gates to input qubits
    for qubit in range(n):
        circuit.hadamard(qubit)

    # Measure the input qubits
    circuit.add_measurement(key="Measurement", qubits=tuple(range(n)))

    _, measurement = sim.run(circuit)

    measurement_result = measurement["Measurement"]

    if measurement_result == '0' * n:
        return 'constant'
    else:
        return 'balanced'



if __name__ == "__main__":
    all_passed = True
    for n in range(1, 7):
        constant_f = make_f(n, 'constant')
        balanced_f = make_f(n, 'balanced')

        constant_result = deutsch_jozsa_algorithm(n, constant_f)
        balanced_result = deutsch_jozsa_algorithm(n, balanced_f)

        print(f"Constant function result (number of input bits = {n}): {constant_result}")
        print(f"Balanced function result (number of input bits = {n}): {balanced_result}")

        if constant_result != 'constant' or balanced_result != 'balanced':
            all_passed = False
            break

    if all_passed:
        print("Deutsch-Jozsa algorithm successful!")
    else:
        print("Deutsch-Jozsa algorithm failed ")            