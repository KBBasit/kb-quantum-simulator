import numpy as np
from qsim.operators import Operator



class Circuit:
    def __init__(self, num_qubits):
        if not isinstance(num_qubits, int) or num_qubits < 1:
            raise ValueError("Number of qubits must be a positive integer.")

        self.num_qubits = num_qubits
        self.gates = []


    def add_gate(self, operator, target_qubits):
        if not isinstance(operator, Operator):
            raise TypeError("Operator must be an instance of the Operator class.")
        if not isinstance(target_qubits, tuple):
            raise TypeError("Target qubits must be a tuple.")
        if not all(isinstance(q, int) for q in target_qubits):
            raise TypeError("Target qubits must be integers.")
        if len(set(target_qubits)) != len(target_qubits):
            raise ValueError("Target qubits must be unique.")
        if len(target_qubits) != operator.operator_qubits:
            raise ValueError("Number of target qubits must match the operator's qubit count.")
        if any(q < 0 or q >= self.num_qubits for q in target_qubits):
            raise ValueError("Target qubits must be within the range of the circuit's qubits.")

        self.gates.append((operator, target_qubits))