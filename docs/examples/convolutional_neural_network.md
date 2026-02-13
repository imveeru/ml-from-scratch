# Convolutional Neural Network (CNN) Example

This example demonstrates how to build and train a CNN using the `deep_learning` module.

## Description
-   **Dataset**: Sklearn `digits` dataset (8x8 images).
-   **Task**: Multi-class Classification (10 digits).
-   **Model Architecture**:
    1.  Conv2D (16 filters)
    2.  ReLU + Dropout + BatchNorm
    3.  Conv2D (32 filters)
    4.  ReLU + Dropout + BatchNorm
    5.  Flatten
    6.  Dense (256) -> ReLU -> Dropout -> BatchNorm
    7.  Dense (10) -> Softmax
-   **Optimization**: Adam Optimizer, CrossEntropy Loss.

## Usage
Run the script from the root directory:
```bash
python -m mlfromscratch.examples.convolutional_neural_network
```

## Output
-   Prints model summary (layer shapes, parameters).
-   Plots Training and Validation Error over epochs.
-   Prints final Accuracy.
-   Displays a 2D PCA plot of the classification results.
