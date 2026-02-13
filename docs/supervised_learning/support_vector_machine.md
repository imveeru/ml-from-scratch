# Support Vector Machine (SVM)

## 1. Pseudocode

```text
Fit(X, y):
    # Setup Quadratic Programming Problem
    Compute Kernel Matrix K[i, j] = Kernel(x_i, x_j)
    
    P = Outer_Product(y, y) * K
    q = Vector of -1s
    A = y.T (Equality constraint)
    b = 0
    G, h = Inequality constraints (0 <= alpha <= C)
    
    # Solve QP
    alphas = Solve_QP(P, q, G, h, A, b)
    
    # Extract Support Vectors
    Support_Vectors = samples where alpha > threshold
    
    Calculate Intercept b using Support Vectors

Predict(sample):
    prediction = 0
    For each Support Vector sv:
        prediction += alpha_sv * y_sv * Kernel(sv, sample)
    
    Return Sign(prediction + b)
```

## 2. Algorithm Explanation

**Support Vector Machine (SVM)** is a powerful supervised learning model used for classification and regression. It finds the optimal hyperplane that separates the data points of different classes with the **maximum margin**.

This implementation solves the **Dual Form** of the SVM optimization problem using Quadratic Programming (`cvxopt` library).

It supports the **Kernel Trick**, allowing it to solve non-linear problems by mapping input data into high-dimensional feature spaces where they become linearly separable. Supported kernels include:
-   Linear
-   Polynomial
-   RBF (Radial Basis Function)

## 3. Math Formulas

**Optimization Problem (Dual Form):**
$$ \max_{\alpha} \sum_{i} \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j) $$

**Subject to:**
$$ 0 \le \alpha_i \le C $$
$$ \sum_{i} \alpha_i y_i = 0 $$

**Decision Function:**
$$ f(x) = \text{sign}\left(\sum_{i \in SV} \alpha_i y_i K(x_i, x) + b\right) $$

## 4. Inputs Required

-   **X**: Training features (`n_samples`, `n_features`).
-   **y**: Binary Training labels (`n_samples`,). Labels must be `-1` or `1`.
-   **C**: Regularization parameter. Controls trade-off between maximizing margin and minimizing classification error.
-   **kernel**: Kernel function (`linear`, `poly`, `rbf`).
-   **gamma**: Kernel coefficient for `rbf`, `poly`.
-   **power**: Degree of the polynomial kernel.

## 5. Usage Guidelines

### When to use:
-   **High Dimensionality**: Effective in high dimensional spaces (even where $features > samples$).
-   **Clear Margin**: Works best when there is a clear margin of separation.
-   **Non-Linear**: Can efficienty model complex non-linear boundaries using Kernels.

### When not to use:
-   **Large Datasets**: The training time complexity is between $O(N^2)$and$O(N^3)$, making it slow for datasets with$>100k $ samples.
-   **Noisy Data**: Overlap between classes can cause issues, though soft-margin ($C$) helps.
-   **Probability Output**: Does not directly provide probability estimates.

### Industry Best Practices:
-   **Scaling**: **Critical**. SVM is very sensitive to feature scaling. Always standardize data.
-   **Hyperparameter Tuning**: Performance depends heavily on `C` and `gamma` (for RBF). Grid Search is commonly used to find optimal values.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: The QP solver (`cvxopt`) handles the optimization. Predicting is parallelizable.
-   **Memory**: Storing the Kernel Matrix of size $(N \times N)$can be memory prohibitive for large$N$.

## 7. Underlying Data Structure

-   **Numpy Arrays**: For matrix operations.
-   **CVXOPT Matrices**: Used for the optimization solver.
