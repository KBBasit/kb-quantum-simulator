# kb_quantum_simulator
## Overview
KB quantum simulator is a Python-based quantum circuit simulator built using NumPy. It supports the construction and simulation of multi-qubit quantum circuits. The project also includes implementations of five quantum algorithms and protocols: Shor's factoring algorithm, Grover's search, quantum phase estimation, Deutsch-Jozsa and quantum teleportation.


## Installation
Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/KBBasit/kb-quantum-simulator.git
cd kb-quantum-simulator
```
Install the package and its dependencies:
```bash
python -m pip install .
```


## Usage
The following example demonstrates how to use the simulator to prepare the Bell state $|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$ from the initial state $`|00\rangle`$, measure the first qubit and print the measuremement result and final state.
```python
from kb_quantum_simulator.circuits import Circuit
from kb_quantum_simulator.simulator import StatevectorSimulator

sim = StatevectorSimulator()
circuit = Circuit(num_qubits=2)

circuit.hadamard(target_qubit=0)
circuit.cnot(control_qubit=0, target_qubit=1)
circuit.add_measurement(key="alice", qubits=(0,))

final_state, measurement_result = sim.run(circuit)
print(f"Measurement result: {measurement_result['alice']}")
print(f"Final state: {final_state.amplitudes}")
```






