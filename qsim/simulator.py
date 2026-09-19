from .gates import hadamard, swap, controlled_phase
from .states import StateVector
from .operators import Operator
from .circuits import Circuit
from .measurements import measure
import numpy as np



class StatevectorSimulator:
    '''
    A class for simulating quantum circuits using state vectors.
    '''
    def run(self, circuit, initial_state=None):
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

                qubit_index = 0
                controlled_phases = [Operator(controlled_phase(np.pi / (2 ** i))) for i in range(1, len(target_qubits))]
                hadamard = Operator(hadamard())
                while qubit_index < len(target_qubits):
                    qubit = target_qubits[qubit_index]
                    hadamard.apply_gate(state, (qubit), inplace=True)

                    for index in range(len(qubit_index + 1, len(target_qubits))):
                        controlled_phases[index].apply_gate(state, (target_qubits[qubit_index], target_qubits[qubit_index + index + 1]), inplace=True)

                    qubit_index += 1

                left = 0
                right = len(target_qubits) - 1
                while left < right:
                    swap.apply_gate(state, (target_qubits[left], target_qubits[right]), inplace=True)
                    left += 1
                    right -= 1

            elif operation_type == "inverse_qft":
                _, target_qubits = operation


            else:
                raise ValueError(f"Unknown operation type: {operation_type}")

        return state, measurement_results