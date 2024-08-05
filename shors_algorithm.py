import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, circuit_drawer, plot_state_qsphere, plot_bloch_multivector
from qiskit.quantum_info import Statevector
from math import gcd
import numpy as np
from fractions import Fraction

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_histogram(counts, filename):
    logging.info(f"Saving histogram to {filename}.")
    try:
        fig = plot_histogram(counts)
        fig.savefig(filename)
        plt.close(fig)
    except Exception as e:
        logging.error(f"An error occurred while saving the histogram: {e}")
        raise

def save_circuit_diagram(circuit, filename):
    logging.info(f"Saving circuit diagram to {filename}.")
    try:
        fig = circuit_drawer(circuit, output='mpl', style={'backgroundcolor': '#EEEEEE'})
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        logging.error(f"An error occurred while saving the circuit diagram: {e}")
        raise

def save_qsphere(statevector, filename):
    logging.info(f"Saving Q-sphere visualization to {filename}.")
    try:
        fig = plot_state_qsphere(statevector)
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        logging.error(f"An error occurred while saving the Q-sphere visualization: {e}")
        raise

def save_bloch_multivector(statevector, filename):
    logging.info(f"Saving Bloch multivector visualization to {filename}.")
    try:
        fig = plot_bloch_multivector(statevector)
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        logging.error(f"An error occurred while saving the Bloch multivector visualization: {e}")
        raise

def save_counts_to_file(counts, filename):
    logging.info(f"Saving counts to {filename}.")
    try:
        with open(filename, 'w') as f:
            f.write(str(counts))
    except Exception as e:
        logging.error(f"An error occurred while saving the counts to file: {e}")
        raise

def print_ascii_circuit(circuit, name):
    logging.info(f"Printing ASCII representation of {name}.")
    try:
        print(f"\nASCII representation of {name}:")
        print(circuit.draw(output='text'))
    except Exception as e:
        logging.error(f"An error occurred while printing the ASCII circuit: {e}")
        raise

def qpe_amod15(a):
    n_count = 8
    qc = QuantumCircuit(4 + n_count)
    
    # Initialize counting qubits in superposition
    for q in range(n_count):
        qc.h(q)
    
    # Initialize auxiliary register to |1>
    qc.x(3 + n_count)
    
    # Apply controlled-U operations
    for q in range(n_count):
        qc.append(c_amod15(a, 2**q), [q] + [i+n_count for i in range(4)])
    
    # Apply inverse QFT
    qc.append(qft_dagger(n_count), range(n_count))
    
    return qc

def c_amod15(a, power):
    """Controlled multiplication by a mod 15"""
    if a not in [2, 4, 7, 8, 11, 13]:
        raise ValueError("'a' must be 2, 4, 7, 8, 11 or 13")
    U = QuantumCircuit(4)
    for iteration in range(power):
        if a in [2, 13]:
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        if a in [7, 8]:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        if a in [4, 11]:
            U.swap(1, 3)
            U.swap(0, 2)
        if a in [7, 11, 13]:
            for q in range(4):
                U.x(q)
    U = U.control(1)
    U.name = f"C-U{power}"
    return U

def qft_dagger(n):
    qc = QuantumCircuit(n)
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc

def run_shors_algorithm(number_to_factor):
    logging.info(f"Running Shor's algorithm to factor the number {number_to_factor}.")
    try:
        backend = AerSimulator()
        a = 7

        qc = qpe_amod15(a)
        
        # Save the circuit diagram
        save_circuit_diagram(qc, 'shor_circuit_diagram.png')
        
        # Print ASCII representation of the circuit
        print_ascii_circuit(qc, "Shor's Algorithm Circuit")
        
        # Get the statevector before measurement
        statevector = Statevector.from_instruction(qc)
        
        # Save Q-sphere visualization
        save_qsphere(statevector, 'shor_qsphere.png')
        
        # Save Bloch multivector visualization
        save_bloch_multivector(statevector, 'shor_bloch_multivector.png')
        
        # Add measurement operations
        qc.measure_all()
        
        transpiled_qc = transpile(qc, backend)
        job = backend.run(transpiled_qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        save_histogram(counts, 'shor_histogram.png')
        save_counts_to_file(counts, 'shor_counts.txt')
        
        measured_phases = [int(k[::-1], 2)/(2**8) for k in counts.keys()]
        fractions = [Fraction.from_float(x).limit_denominator(15) for x in measured_phases]
        print("Measured fractions:", fractions)

        factors = set()
        for fraction in fractions:
            if fraction.denominator > 1:
                guess = gcd(a**(fraction.denominator//2) - 1, 15)
                if guess not in [1, 15]:
                    factors.add(guess)
        return factors
    except Exception as e:
        logging.error(f"An error occurred while running Shor's algorithm: {e}")
        raise

def main():
    logging.info("Starting the quantum circuit processing.")

    # Shor's algorithm
    number_to_factor = 15
    factors = run_shors_algorithm(number_to_factor)
    print(f"Factors of {number_to_factor}:", factors)

    logging.info("All plots, visualizations, and counts have been saved as PNG and text files in the current directory.")
    print("\nAll plots, circuit diagram, visualizations, and counts have been saved as PNG and text files in the current directory.")

if __name__ == "__main__":
    main()