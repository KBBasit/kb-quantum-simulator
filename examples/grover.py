'''
Grover's Search Algorithm
'''

from qsim.operators import Operator
from qsim.circuits import Circuit
from qsim.simulator import StatevectorSimulator
import numpy as np



def grovers_search_algorithm(n):
    """
    
    """
    sim = StatevectorSimulator()

    # Initialisation
    circuit = Circuit(num_qubits=n)

    # Create uniform superposition
    for qubit in range(n):
        circuit.hadamard(qubit)