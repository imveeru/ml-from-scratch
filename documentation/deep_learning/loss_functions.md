# Loss Functions

This file implements loss functions used to measure the error between predicted estimates and true values.

## 1. Mathematical Formulas

### Square Loss (Mean Squared Error)
Used for Regression problems.

$$ L(y, \hat{y}) = \frac{1}{2} (y - \hat{y})^2 $$
**Gradient:** $-(y - \hat{y})$

### Cross Entropy (Binary / Categorical)
Used for Classification problems. It measures the dissimilarity between the ground truth distribution and the predicted distribution.

$$ L(y, p) = - y \log(p) - (1 - y) \log(1 - p) $$
**Gradient:** $-\frac{y}{p} + \frac{1-y}{1-p}$

## 2. Usage Guidelines

### Industry Best Practices
-   **Numerical Stability**: The implementation clips predictions to $[1e-15, 1 - 1e-15]$ to avoid `log(0)` errors.
-   **Softmax + CrossEntropy**: In many frameworks, these are combined into a single computationally stable operation ("LogSumExp" trick), though here they are separate.

## 3. Implementation Details

Each class implements:
-   `loss(y_true, y_pred)`: Returns the scalar loss.
-   `gradient(y_true, y_pred)`: Returns gradient w.r.t predictions.
-   `acc(y_true, y_pred)`: Helper to calculate accuracy (for classification).
