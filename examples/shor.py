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
        for x in range(2):
            for y in range(2 ** m):
                input_index = x * (2 ** m) + y
                if x == 1 and y < N:
                    output_y = (y * term) % N
                else:
                    output_y = y
                output_index = x * (2 ** m) + output_y
                intermediate[output_index, input_index] = 1
        oracle_operations.append(Operator(intermediate))

    return oracle_operations


def continued_fractions_algorithm(numerator, denominator):
    """
    Implements the continued fractions algorithm
    Parameters:
        numerator (int): The numerator of the fraction.
        denominator (int): The denominator of the fraction.
    Returns:
        tuple: A tuple containing the best rational approximation (p, q) such that p/q
    """
    fractions = []
    while denominator:
        fractions.append(numerator // denominator)
        numerator, denominator = denominator, numerator % denominator
    return fractions


def shors_factoring_algorithm(N):
    """
    Implements Shor's factoring algorithm to find a non-trivial factor of a composite number N.
    Parameters:
        N (int): The composite number to be factored.
    Returns:
        int: A non-trivial factor of N
    """
    sim = StatevectorSimulator()

    # Choose a random integer a such that 1 < a < N and gcd(a, N) = 1
    a = np.random.randint(2, N)
    gcd = np.gcd(a, N)
    if gcd != 1:
        return gcd

    # Initialisation
    n = (N ** 2 - 1).bit_length()
    m = (N - 1).bit_length()
    second_register_qubits = tuple(range(n, n + m))
    circuit = Circuit(num_qubits=n + m)
    circuit.pauli_x(n + m - 1)

    # Create superposition
    for qubit in range(n):
        circuit.hadamard(qubit)

    # Apply modular exponentiation oracle
    oracle_operations = make_modular_exponentiation_oracle(a, N, m, n)
    for i, operation in enumerate(oracle_operations):
        control_qubit = n - 1 - i
        circuit.add_gate(operation, target_qubits= (control_qubit,) + second_register_qubits)

    # Measure the second register
    circuit.add_measurement(key="y_0", qubits=second_register_qubits)

    # Apply inverse Quantum Fourier Transform to the first register
    circuit.inverse_quantum_fourier_transform(target_qubits=range(n))

    # Measure the first register
    circuit.add_measurement(key="c", qubits=tuple(range(n)))


    _, measurement_results = sim.run(circuit)
    c = int(measurement_results["c"], 2)

    return



if __name__ == "__main__":
    print(continued_fractions_algorithm(45, 16))
