'''
This module defines the Circuit class, which represents a
quantum circuit composed of quantum gates and measurements.
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
        return f"Circuit(num_qubits={self.num_qubits}, num_operations={len(self)})"


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


    def c_apply(self, operator, measured_qubits, measurement_condition, target_qubits):
        '''
        Apply a quantum operator conditional on the measurement results of specified qubits.

        Parameters:
            operator (Operator): The quantum operator to apply.
            measured_qubits (tuple): A tuple of qubit indices whose measurement results will determine whether the operator is applied.
            measurement_condition (str): The bitstring condition for applying the operator.
            target_qubits (tuple): A tuple of qubit indices where the operator will be applied.
        Returns:
            None
        Raises:
            TypeError: If operator is not an instance of Operator.
            TypeError: If measured_qubits or target_qubits are not tuples of integers.
            ValueError: If the number of target qubits does not match the operator's qubit count.
            ValueError: If measured_qubits is empty.
            ValueError: If qubit indices are out of bounds or not unique.
            TypeError: If measurement_condition is not a string.
            ValueError: If measurement_condition is not a valid bitstring for the number of measured qubits.
        '''
        if not isinstance(operator, Operator):
            raise TypeError("Operator must be an instance of the Operator class.")
        if not isinstance(measured_qubits, tuple) or not isinstance(target_qubits, tuple):
            raise TypeError("Measured qubits and target qubits must be tuples.")
        if not all(isinstance(q, int) for q in measured_qubits) or not all(isinstance(q, int) for q in target_qubits):
            raise TypeError("All qubit indices must be integers.")
        if len(target_qubits) != operator.operator_qubits:
            raise ValueError("Number of target qubits must match the operator's qubit count.")
        if len(measured_qubits) == 0:
            raise ValueError("At least one measured qubit is required.")
        if len(set(measured_qubits)) != len(measured_qubits) or len(set(target_qubits)) != len(target_qubits):
            raise ValueError("Qubit indices must be unique.")
        if any(q < 0 or q >= self.num_qubits for q in measured_qubits + target_qubits):
            raise ValueError("Qubit indices must be within the range of the circuit's qubits.")
        if not isinstance(measurement_condition, str):
            raise TypeError("Measurement condition must be a string.")
        if not all(c in "01" for c in measurement_condition):
            raise ValueError("Measurement condition must be a valid bitstring.")
        if len(measurement_condition) != len(measured_qubits):
            raise ValueError("Measurement condition must have the same length as the number of measured qubits.")

        self.operations.append(("c_apply", operator, measured_qubits, measurement_condition, target_qubits))


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
