# Data Operation Utils

This file contains fundamental mathematical and statistical operations used throughout the library.

## 1. Functions Overview

### Statistics
-   **`calculate_entropy(y)`**: Computes the entropy of a label distribution (used in Decision Trees).
    $$ H(y) = - \sum_{i} p_i \log_2(p_i) $$
-   **`calculate_variance(X)`**: Computes the variance of features.
    $$ Var(X) = \frac{1}{N} \sum (x_i - \bar{x})^2 $$
-   **`calculate_std_dev(X)`**: Standard Deviation ($ \sqrt{Var(X)} $).
-   **`calculate_covariance_matrix(X, Y)`**: Computes the covariance matrix.
    $$ \Sigma = \frac{1}{N-1} (X - \bar{X})^T (Y - \bar{Y}) $$
-   **`calculate_correlation_matrix(X, Y)`**: Computes the Pearson correlation matrix.
    $$ Corr(X, Y) = \frac{Cov(X, Y)}{\sigma_X \sigma_Y} $$

### Metrics
-   **`mean_squared_error(y_true, y_pred)`**: MSE loss.
-   **`accuracy_score(y_true, y_pred)`**: Classification accuracy ($ \frac{\text{Correct}}{\text{Total}} $).
-   **`euclidean_distance(x1, x2)`**: L2 distance between two vectors.
    $$ d(x_1, x_2) = \sqrt{\sum (x_{1i} - x_{2i})^2} $$

## 2. Usage Guidelines
-   **Covariance vs Correlation**: Covariance indicates the direction of the linear relationship between variables. Correlation measures both the strength and direction of the linear relationship between two variables.
