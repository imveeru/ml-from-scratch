# Particle Swarm Optimization (PSO) Example

This example demonstrates optimizing a Neural Network using Particle Swarm Optimization.

## Description
-   **Dataset**: Sklearn `iris`.
-   **Task**: Multi-class Classification.
-   **Optimization**:
    -   Population: 100 particles (Neural Networks).
    -   Generations: 10.
    -   Update Parameters: Inertia, Cognitive Weight, Social Weight.
-   **Model**: Neural Network (16 hidden units).

## Usage
Run the script from the root directory:
```bash
python -m mlfromscratch.examples.particle_swarm_optimization
```

## Output
-   Prints Population stats.
-   Prints Final Accuracy.
-   Displays classification plot.
