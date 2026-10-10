"""
A module for simulating quantum circuits using state 
vectors.
"""

import numpy as np
from .gates import hadamard, swap, controlled_phase
from .states import StateVector
from .operators import Operator
from .circuits import Circuit
from .measurements import measure



class StatevectorSimulator:
    """
    A class for simulating quantum circuits using state
    vectors.
    """
    def run(self, circuit, initial_state=None):
        """
        Run the quantum circuit on the simulator.

        Parameters
        ----------
        circuit : Circuit
            The quantum circuit to simulate.
        initial_state : StateVector, optional
            The initial state of the quantum system. If None, the zero state is used. Default is None.

        Returns
        -------
        tuple
            A tuple containing the final state of the quantum system and the measurement results.
        """
        if not isinstance(circuit, Circuit):
            raise TypeError("Circuit must be an instance of the Circuit class.")
        if initial_state is None:
            initial_state = StateVector.zero_state(circuit.num_qubits)
        if not isinstance(initial_state, StateVector):
            raise TypeError("Initial state must be an instance of the StateVector class.")
        if initial_state.num_qubits != circuit.num_qubits:
            raise ValueError("Initial state must have the same number of qubits as the circuit.")

        state = initial_state.copy()
        measurement_results = {}

        for operation in circuit.operations:
            operation_type = operation[0]
            if operation_type == "measure":
                _, qubits, key = operation
                measurement_results[key] = measure(state, qubits)
    
            elif operation_type == "gate":
                _, operator, target_qubits = operation
                operator.apply_gate(state, target_qubits, inplace=True)

            elif operation_type == "c_apply":
                _, operator, measurement_key, measurement_condition, target_qubits = operation
                if measurement_results[measurement_key] == measurement_condition:
                    operator.apply_gate(state, target_qubits, inplace=True)

            elif operation_type == "qft":
                _, target_qubits = operation
                controlled_phases = [Operator(controlled_phase(np.pi / (2 ** i))) for i in range(1, len(target_qubits))]
                hadamard_gate = Operator(hadamard())
                swap_gate = Operator(swap())

                qubit_index = 0
                while qubit_index < len(target_qubits):
                    qubit = target_qubits[qubit_index]
                    hadamard_gate.apply_gate(state, (qubit,), inplace=True)

                    for index in range(len(target_qubits) - qubit_index - 1):
                        controlled_phases[index].apply_gate(state, (qubit, target_qubits[qubit_index + index + 1]), inplace=True)

                    qubit_index += 1

                left = 0
                right = len(target_qubits) - 1
                while left < right:
                    swap_gate.apply_gate(state, (target_qubits[left], target_qubits[right]), inplace=True)
                    left += 1
                    right -= 1

            elif operation_type == "inverse_qft":
                _, target_qubits = operation
                inv_controlled_phases = [Operator(controlled_phase(-np.pi / (2 ** i))) for i in range(1, len(target_qubits))]
                hadamard_gate = Operator(hadamard())
                swap_gate = Operator(swap())

                left = 0
                right = len(target_qubits) - 1
                while left < right:
                    swap_gate.apply_gate(state, (target_qubits[left], target_qubits[right]), inplace=True)
                    left += 1
                    right -= 1

                qubit_index = len(target_qubits) - 1
                number_of_control_phase_gates_to_apply = 1
                while qubit_index >= 0:
                    qubit = target_qubits[qubit_index]
                    hadamard_gate.apply_gate(state, (qubit,), inplace=True)

                    if number_of_control_phase_gates_to_apply < len(target_qubits):
                        for application in range(number_of_control_phase_gates_to_apply):
                            inv_controlled_phases[application].apply_gate(state, (target_qubits[qubit_index - 1], target_qubits[qubit_index + application]), inplace=True)

                    number_of_control_phase_gates_to_apply += 1
                    qubit_index -= 1

            else:
                raise ValueError(f"Unknown operation type: {operation_type}")

        return state, measurement_results