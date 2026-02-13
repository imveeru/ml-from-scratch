# Polynomial Regression Example

This example demonstrates Polynomial Ridge Regression.

## Description
-   **Dataset**: `TempLinkoping2016.txt`.
-   **Task**: Regression.
-   **Model**: Polynomial Ridge Regression.
-   **Process**:
    -   Uses **K-Fold Cross Validation** (k=10) to find the optimal regularization factor.
    -   Trains final model with best factor.
-   **Visualization**: Plots prediction line.

## Usage
Run the script from the root directory:
```bash
python -m mlfromscratch.examples.polynomial_regression
```

## Output
-   Prints Cross-Validation results (MSE for different regularization factors).
-   Prints Final MSE.
-   Displays Regression Plot.
