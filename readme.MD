# Shor's Algorithm Quantum Circuit Simulation

This repository contains a Python implementation of Shor's algorithm using Qiskit, a quantum computing framework. Shor's algorithm is a quantum algorithm for integer factorization, which performs significantly faster than the best-known classical factoring algorithm.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [License](#license)
- [Contributing](#contributing)

## Introduction

Shor's algorithm, developed by Peter Shor in 1994, is a quantum algorithm for integer factorization. This implementation demonstrates the algorithm's capability to factor the number 15 using a quantum circuit simulation.

## Features

- Implementation of Shor's algorithm for factoring 15
- Quantum circuit generation and simulation using Qiskit
- Visualization of the quantum circuit
- Generation of various quantum state visualizations (Q-sphere, Bloch multivector)
- Histogram of measurement results
- Logging of the execution process

## Requirements

- Python 3.7+
- Qiskit
- Matplotlib
- NumPy

## Installation

1. Clone this repository:

git clone https://github.com/SuperSonnix71/Shors-quantum-algorithm.git

2. Navigate to the project directory:

cd Shors-quantum-algorithm

3. Install the required packages:

pip install -r requirements.txt

## Usage

Run the main script:


python shors_algorithm.py

This will execute Shor's algorithm to factor the number 15 and generate various output files.

## Outputs

The script generates the following outputs:

- `shor_circuit_diagram.png`: Visual representation of the quantum circuit
- `shor_qsphere.png`: Q-sphere visualization of the quantum state
- `shor_bloch_multivector.png`: Bloch multivector representation of the quantum state
- `shor_histogram.png`: Histogram of measurement results
- `shor_counts.txt`: Text file containing the measurement counts

Additionally, the script prints the ASCII representation of the circuit and the factors found for the number 15.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

MIT License

Copyright (c) 2024 SuperSonnix71

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

Contributions to improve the implementation or extend its functionality are welcome. Please feel free to submit pull requests or open issues for discussion.
