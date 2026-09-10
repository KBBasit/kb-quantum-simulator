from qsim.states import StateVector
from qsim.circuits import Circuit



class StatevectorSimulator:
    '''
    A class for simulating quantum circuits using state vectors.
    '''
    def run(self, circuit, initial_state=None):
        if not isinstance(circuit, Circuit):
            raise TypeError("Circuit must be an instance of the Circuit class.")

        if initial_state is None:
            initial_state = StateVector.zero_state(circuit.num_qubits)
        elif not isinstance(initial_state, StateVector):
            raise TypeError("Initial state must be an instance of the StateVector class.")
        elif initial_state.num_qubits != circuit.num_qubits:
            raise ValueError("Initial state must have the same number of qubits as the circuit.")