# Recurrent Neural Network (RNN) Example

This example demonstrates using an RNN to learn a number series.

## Description
-   **Task**: Learn the sequence $n, n+1, ... n+9$.
-   **Data**: Generated multiplication series (for `gen_mult_ser`) or number sequence (`gen_num_seq`).
-   **Model**: RNN (Hidden State Size: 10, Tanh Activation) + Softmax Output.
-   **Optimization**: Adam Optimizer, CrossEntropy Loss.
-   **Visualization**: Plots Training Error.

## Usage
Run the script from the root directory:
```bash
python -m mlfromscratch.examples.recurrent_neural_network
```

## Output
-   Prints Model Summary.
-   Prints example Input and Target sequences.
-   Prints Predicted sequences for test data.
-   Prints Accuracy.
-   Displays Error Plot.
