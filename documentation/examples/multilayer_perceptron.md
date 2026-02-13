# Multilayer Perceptron (MLP) Example

This example demonstrates a Deep Neural Network (MLP) on the Digits dataset.

## Description
-   **Dataset**: Sklearn `digits`.
-   **Task**: Multi-class Classification.
-   **Model**: 5-layer MLP (512 units each) with LeakyReLU, Dropout, and Softmax.
-   **Optimization**: Adam Optimizer, CrossEntropy Loss.
-   **Visualization**: Training/Validation Error plot + Prediction Scatter Plot.

## Usage
Run the script from the root directory:
```bash
python -m mlfromscratch.examples.multilayer_perceptron
```

## Output
-   Model Summary.
-   Error Plot.
-   Accuracy.
-   Prediction Plot.
