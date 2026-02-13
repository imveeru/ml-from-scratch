# Neuroevolution Example

This example demonstrates evolving a Neural Network using a Genetic Algorithm (Neuroevolution) instead of Gradient Descent.

## Description
-   **Dataset**: Sklearn `digits`.
-   **Task**: Multi-class Classification.
-   **Evolution**:
    -   Population: 100 Neural Networks.
    -   Generations: 3000.
    -   Mutation Rate: 0.01.
    -   Fitness: Accuracy on training set.
-   **Model Architecture**: 16 hidden units -> Output (Softmax).

## Usage
Run the script from the root directory:
```bash
python -m mlfromscratch.examples.neuroevolution
```

## Output
-   Prints Model Summary.
-   Prints Population stats.
-   Progress bar showing generations.
-   Final Accuracy.
-   Prediction Plot.
