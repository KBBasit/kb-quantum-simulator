'''
Shor's Factoring Algorithm
'''

from qsim.operators import Operator
from qsim.circuits import Circuit
from qsim.simulator import StatevectorSimulator
import numpy as np



def make_modular_exponentiation_oracle(a, N, n, m):
    """
    Construct modular exponentiation oracle using binary exponentation
    Parameters:
        a (int): The base for modular exponentiation.
        N (int): The modulus for modular exponentiation.
        n (int): The number of qubits for the input register.
        m (int): The number of qubits for the output register.
    Returns:
        list: A list of operator objects for oracle operations
    """
    sequence = [a % N]
    for _ in range(n - 1):
        sequence.append((sequence[-1] ** 2) % N)

    oracle_operations = []
    for term in sequence:
        intermediate = np.zeros((2 ** (m+1), 2 ** (m+1)), dtype=np.complex128)
        for y in range(2 ** m):
            for x in range(2):
                input_index = x * (2 ** m) + y
                if x == 1 and y < N:
                    output_y = (y * term) % N
                else:
                    output_y = y
                output_index = x * (2 ** m) + output_y
                intermediate[output_index, input_index] = 1
        oracle_operations.append(Operator(intermediate))

    return oracle_operations


def shors_factoring_algorithm(N):
    """
    Implements Shor's factoring algorithm to find a non-trivial factor of a composite number N.
    Parameters:
        N (int): The composite number to be factored.
    Returns:
        int: A non-trivial factor of N
    """
    # Choose a random integer a such that 1 < a < N and gcd(a, N) = 1
    a = np.random.randint(2, N)
    gcd = np.gcd(a, N)
    if gcd != 1:
        return gcd

    # Initialisation
    n = (N ** 2 - 1).bit_length()
    m = (N - 1).bit_length()
    circuit = Circuit(num_qubits=n + m)

    # Create superposition
    for qubit in range(n):
        circuit.h(qubit)
