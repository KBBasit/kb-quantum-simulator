from .states import StateVector
from .circuits import Circuit
from .measurements import measure



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
            else:
                raise ValueError(f"Unknown operation type: {operation_type}")

        return state, measurement_results