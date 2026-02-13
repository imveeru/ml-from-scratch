# Logistic Regression

## 1. Pseudocode

```text
Fit(X, y, n_iterations):
    Initialize parameters (theta) randomly
    
    For i = 1 to n_iterations:
        Calculate linear prediction: z = X . theta
        Apply Sigmoid activation: y_pred = 1 / (1 + e^-z)
        
        If Gradient Descent:
            Calculate Gradient: grad = -(y - y_pred) . X
            Update parameters: theta = theta - learning_rate * grad
        Else (Batch Optimization / IRLS):
            Calculate Diagonal Gradient Matrix: diag = diag(sigmoid_grad(z))
            # Newton-Raphson / Iteratively Reweighted Least Squares step
            theta = inverse(X.T . diag . X) . X.T . (diag . X . theta + y - y_pred)

Predict(X):
    z = X . theta
    y_pred = 1 / (1 + e^-z)
    Return Round(y_pred)
```

## 2. Algorithm Explanation

**Logistic Regression** is a statistical model used for binary classification. Although named "regression", it is a classification algorithm. It models the probability that a given input point belongs to a certain class (e.g., class 1 vs. class 0) using the **Sigmoid function**.

The model finds the best-fitting parameters ($\theta$) that minimize the error between the predicted probabilities and the actual labels.

This implementation supports two optimization methods:
1.  **Gradient Descent**: Iteratively follows the negative gradient of the loss function.
2.  **Batch Optimization**: Uses a method similar to Newton-Raphson (Iteratively Reweighted Least Squares) to converge faster, but is computationally more expensive per step (requires matrix inversion).

## 3. Math Formulas

**Sigmoid Function:**
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

**Prediction:**
$$ \hat{y} = \sigma(X\theta) $$

**Gradient Descent Update:**
$$ \theta = \theta - \alpha \cdot X^T (\hat{y} - y) $$

**Hessian (for Newton's Method):**
$$ H = X^T \cdot \text{diag}(\hat{y}(1-\hat{y})) \cdot X $$

## 4. Inputs Required

-   **X**: Training features (`n_samples`, `n_features`).
-   **y**: Binary target labels (`n_samples`,).
-   **learning_rate**: Step size for gradient descent.
-   **gradient_descent**: Boolean flag to toggle between Gradient Descent and Batch Optimization.

## 5. Usage Guidelines

### When to use:
-   **Binary Classification**: Default choice for binary problems.
-   **Linearly Separable Data**: Works best when classes can be separated by a linear boundary.
-   **Probabilistic Output**: When you need not just the class, but the probability of class membership (e.g., credit scoring).

### When not to use:
-   **Non-Linear Data**: Cannot solve non-linear problems (like XOR) without feature engineering or kernels.
-   **Correlated Features**: Multicollinearity can affect coefficient interpretation and stability.
-   **Missing Data**: Requires complete data (or imputation).

### Industry Best Practices:
-   **Regularization**: L1 (Lasso) or L2 (Ridge) regularization is almost always used to prevent overfitting, especially with many features.
-   **Feature Scaling**: Essential for Gradient Descent convergence.
-   **Class Imbalance**: Use class weights or resampling techniques if classes are highly imbalanced.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: Matrix multiplications ($X \cdot \theta$) are parallelized by linear algebra libraries.
-   **Memory**: Standard gradient descent is memory efficient ($O(features)$). Batch optimization requires storing and inverting the Hessian matrix ($O(features^2)$), which is expensive for high-dimensional data.

## 7. Underlying Data Structure

-   **Numpy Arrays**: Used for parameters (theta), data matrices, and vectorized math operations.
