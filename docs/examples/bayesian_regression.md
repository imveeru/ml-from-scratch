# Bayesian Regression Example

This example demonstrates Bayesian Linear Regression on temperature data.

## Description
-   **Dataset**: `TempLinkoping2016.txt` (Daily temperature data).
-   **Task**: Regression (Predict Temperature given Day of Year).
-   **Model**: Bayesian Regression with Polynomial Features (Degree 4).
-   **Visualization**: Plots the training data, test data, prediction line, and the **Credible Interval** (uncertainty).

## Usage
Run the script from the root directory:
```bash
python -m mlfromscratch.examples.bayesian_regression
```

## Output
-   Prints Mean Squared Error (MSE).
-   Displays a plot showing the regression curve and the uncertainty band.
