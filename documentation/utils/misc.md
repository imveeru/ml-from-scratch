# Misc / PlotUtils

This file contains plotting utilities to visualize data and model performance using `matplotlib`.

## 1. Classes Overview

### Plot
A class wrapper for various plotting functions.

-   **`plot_regression(lines, title, axis_labels, mse, scatter)`**:
    -   Plots regression lines and scatter points (dataset).
    -   Useful for Linear Regression, Polynomial Regression etc.

-   **`plot_in_2d(X, y_pred, title, accuracy, legend_labels)`**:
    -   Plots high-dimensional data in 2D by first reducing it using **PCA** (Prinicipal Component Analysis).
    -   Colors points based on their class label (ground truth or prediction).
    -   Used for Classification tasks.

-   **`plot_in_3d(X, y)`**:
    -   Similar to `plot_in_2d` but reduces data to 3 dimensions and plots a 3D scatter plot.

## 2. Helper Methods

-   **`_transform(X, dim)`**:
    -   Internal method used to project data $X$ onto its top `dim` principal components.
    -   Calculates Covariance Matrix -> Eigen Decomposition -> Project.

## 3. Usage Guidelines
-   **Dimensionality Reduction**: The plotting class automatically handles dimensionality reduction. You can pass raw $N$-dimensional data, and it will compute the 2D/3D projection for visualization.
