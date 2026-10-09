"""
Shor's Factoring Algorithm
"""

from qsim.operators import Operator
from qsim.circuits import Circuit
from qsim.simulator import StatevectorSimulator
import numpy as np



def make_modular_exponentiation_oracle(a, N, n, m):
    """
    Construct modular exponentiation oracle using binary
    exponentation

    Parameters
    ----------
        a : int
            The base for modular exponentiation.
        N : int
            The modulus for modular exponentiation.
        n : int
            The number of qubits for the input register.
        m : int
            The number of qubits for the output register.

    Returns
    -------
        list
            A list of operator objects for oracle operations
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

    Parameters
    ----------
        numerator : int
            The numerator of the fraction.
        denominator : int
            The denominator of the fraction.

    Returns
    -------
        list
            A list of integers representing the continued fraction.
    """
    fractions = []
    while denominator:
        fractions.append(numerator // denominator)
        numerator, denominator = denominator, numerator % denominator

    return fractions


def find_convergents(fractions):
    """
    Finds the convergents of a continued fraction.

    Parameters
    ----------
        fractions : list
            A list of integers representing the continued fraction.

    Returns
    -------
        list
            A list of tuples representing the convergents (p, q).
    """
    convergents = []
    p0, p1 = 1, fractions[0]
    q0, q1 = 0, 1
    convergents.append((p1, q1))
    for i in range(1, len(fractions)):
        p2 = fractions[i] * p1 + p0
        q2 = fractions[i] * q1 + q0
        convergents.append((p2, q2))
        p0, p1 = p1, p2
        q0, q1 = q1, q2

    return convergents


def shors_factoring_algorithm(N, max_attempts=10):
    """
    Implements Shor's factoring algorithm to find 
    non-trivial factors of a composite number N.

    Parameters
    ----------
        N : int
            The composite number to be factored.
        max_attempts : int
            The maximum number of attempts to find the factors. Default is 10.

    Returns
    -------
        tuple
            A tuple containing two non-trivial factors of N, or None if no factors are found.
    """
    sim = StatevectorSimulator()

    for _ in range(max_attempts):
        # Choose a random integer a such that 1 < a < N and gcd(a, N) = 1
        a = np.random.randint(2, N)
        greatest_common_divisor = np.gcd(a, N)
        if greatest_common_divisor != 1:
            print(f"Factors of {N} found through classical GCD")
            return int(greatest_common_divisor), N // int(greatest_common_divisor)

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
        oracle_operations = make_modular_exponentiation_oracle(a, N, n, m)
        for i, operation in enumerate(oracle_operations):
            control_qubit = n - 1 - i
            circuit.add_gate(operation, target_qubits= (control_qubit,) + second_register_qubits)

        # Measure the second register
        circuit.add_measurement(key="y_0", qubits=second_register_qubits)

        # Apply inverse Quantum Fourier Transform to the first register
        circuit.inverse_quantum_fourier_transform(target_qubits=tuple(range(n)))

        # Measure the first register
        circuit.add_measurement(key="c", qubits=tuple(range(n)))

        _, measurement_results = sim.run(circuit)


        c = int(measurement_results["c"], 2)
        fractions = continued_fractions_algorithm(c, 2 ** n)
        convergents = find_convergents(fractions)

        for _, r_candidate in convergents:
            if r_candidate % 2 != 0:
                continue
            if pow(a, r_candidate, N) != 1:
                continue
            x = pow(a, r_candidate // 2, N)
            if x == 1 or x == N - 1:
                continue

            factor_1 = int(np.gcd(x - 1, N))
            factor_2 = int(np.gcd(x + 1, N))

            if 1 < factor_1 < N and 1 < factor_2 < N:
                return factor_1, factor_2

    return None



if __name__ == "__main__":
    N_values = [15, 21, 33, 35, 91]

    for N in N_values:
        print(f"Factoring {N}...")
        result = shors_factoring_algorithm(N, max_attempts=10)

        if result is None:
            print(f"Factorisation of {N} unsuccessful")
        else:
            factor_1, factor_2 = result
            print(f"Factorisation of {N} successful! Factors of {N} are: {factor_1} and {factor_2}")