# Restricted Boltzmann Machine (RBM) Example

This example demonstrates RBM for generative modeling (reconstruction).

## Description
-   **Dataset**: MNIST (Filtered to digit '2').
-   **Task**: Learn the distribution of handwritten '2's and reconstruct them.
-   **Model**: RBM (50 Hidden Units).
-   **Visualization**: Plots Error, Reconstructed Images (First Iteration vs Last Iteration).

## Usage
Run the script from the root directory:
```bash
python -m mlfromscratch.examples.restricted_boltzmann_machine
```

## Output
-   Displays Error Plot.
-   Saves/Displays `rbm_first.png` (Reconstruction at start).
-   Saves/Displays `rbm_last.png` (Reconstruction at end).
