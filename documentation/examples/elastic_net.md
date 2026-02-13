# Elastic Net Example

This example demonstrates `ElasticNet` regression on temperature data.

## Description
-   **Dataset**: `TempLinkoping2016.txt`.
-   **Task**: Regression.
-   **Model**: Elastic Net (L1 + L2 Regularization).
-   **Features**: Polynomial Features (Degree 13).
-   **Visualization**: Plots Training Error over iterations and the final Regression Line.

## Usage
Run the script from the root directory:
```bash
python -m mlfromscratch.examples.elastic_net
```

## Output
-   Displays Error Plot (MSE vs Iterations).
-   Prints Final MSE.
-   Displays Regression Plot.
