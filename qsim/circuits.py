import numpy as np
from qsim.operators import Operator



class Circuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.gates = []


    def add_gate(self, operator, target_qubits):
        if not isinstance(operator, Operator):
            raise TypeError("Operator must be an instance of the Operator class.")
        if any(q < 0 or q >= self.num_qubits for q in target_qubits):
            raise ValueError("Target qubits must be within the range of the circuit's qubits.")

        self.gates.append((operator, target_qubits))