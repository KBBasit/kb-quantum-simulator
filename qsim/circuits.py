'''
This module defines the Circuit class, which represents a
quantum circuit composed of quantum gates.
'''

from . import gates
from .operators import Operator



class Circuit:
    def __init__(self, num_qubits):
        if not isinstance(num_qubits, int) or num_qubits < 1:
            raise ValueError("Number of qubits must be a positive integer.")

        self.num_qubits = num_qubits
        self.operations = []


    def __len__(self):
        return len(self.operations)


    def __repr__(self):
        return f"Circuit(num_qubits={self.num_qubits}, num_gates={len(self)})"


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

        self.operations.append(("gate", operator, target_qubits))


    def add_measurement(self, key, qubits=None):
        if not isinstance(qubits, (tuple, type(None))):
            raise TypeError("Qubits must be a tuple of indices or None.")
        if qubits is not None:
            if not isinstance(qubits, tuple):
                raise TypeError("Target qubits must be a tuple or None.")
            if not all(isinstance(q, int) for q in qubits):
                raise TypeError("All target qubit indices must be integers.")
            if len(set(qubits)) != len(qubits):
                raise ValueError("Target qubit indices must be unique.")
            if any(q < 0 or q >= self.num_qubits for q in qubits):
                raise ValueError("Target qubit indices must be within the range of the circuit's qubits.")

        self.operations.append(("measure", qubits, key))


    def pauli_x(self, target_qubit):
        self.add_gate(Operator(gates.pauli_x()), (target_qubit,))


    def pauli_y(self, target_qubit):
        self.add_gate(Operator(gates.pauli_y()), (target_qubit,))


    def pauli_z(self, target_qubit):
        self.add_gate(Operator(gates.pauli_z()), (target_qubit,))


    def hadamard(self, target_qubit):
        self.add_gate(Operator(gates.hadamard()), (target_qubit,))


    def s_gate(self, target_qubit):
        self.add_gate(Operator(gates.s_gate()), (target_qubit,))


    def t_gate(self, target_qubit):
        self.add_gate(Operator(gates.t_gate()), (target_qubit,))


    def cnot(self, control_qubit, target_qubit):
        self.add_gate(Operator(gates.cnot()), (control_qubit, target_qubit))


    def controlled_z(self, control_qubit, target_qubit):
        self.add_gate(Operator(gates.controlled_z()), (control_qubit, target_qubit))


    def swap(self, qubit1, qubit2):
        self.add_gate(Operator(gates.swap()), (qubit1, qubit2))


    def r_x(self, target_qubit, theta):
        self.add_gate(Operator(gates.r_x(theta)), (target_qubit,))


    def r_y(self, target_qubit, theta):
        self.add_gate(Operator(gates.r_y(theta)), (target_qubit,))


    def r_z(self, target_qubit, theta):
        self.add_gate(Operator(gates.r_z(theta)), (target_qubit,))

    
    def phase_gate(self, target_qubit, theta):
        self.add_gate(Operator(gates.phase_gate(theta)), (target_qubit,))
