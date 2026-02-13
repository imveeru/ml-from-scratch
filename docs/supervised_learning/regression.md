# Regression Algorithms

This file contains implementations of various linear regression models.

## 1. Pseudocode

```text
Fit(X, y, n_iterations):
    Add bias column to X
    Initialize parameters theta randomly
    
    For i = 1 to n_iterations:
        Prediction = X . theta
        
        # Calculate Loss Gradient
        # Loss = 0.5 * Mean((y - Prediction)^2) + Regularization(theta)
        
        Gradient = -(y - Prediction) . X + Regularization_Gradient(theta)
        
        # Update Parameters
        theta -= learning_rate * Gradient

Predict(X):
    Add bias column to X
    Return X . theta
```

**Closed Form Solution (for Standard Linear Regression):**
```text
Fit(X, y):
    Add bias column to X
    theta = inverse(X.T . X) . X.T . y
```

## 2. Algorithm Explanation

**Linear Regression** models the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables) using linear predictor functions.
$$ y = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n $$

This implementation includes several variants:
1.  **Linear Regression**: Standard OLS (Ordinary Least Squares). Supports both Gradient Descent and Closed-Form (Least Squares) solutions.
2.  **Lasso Regression (L1)**: Adds L1 regularization term ($\lambda \sum |\theta_i|$). Encourages sparsity (some coefficients become exactly 0).
3.  **Ridge Regression (L2)**: Adds L2 regularization term ($\lambda \sum \theta_i^2$). Prevents large parameters, handling multicollinearity better.
4.  **Elastic Net**: Combines L1 and L2 regularization.
5.  **Polynomial Regression**: Transforms features into polynomial terms ($x, x^2, x^3...$) before applying linear regression to model non-linear relationships.

## 3. Math Formulas

**Model:**
$$ \hat{y} = X\theta $$

**Loss Functions:**
-   **OLS**: $J(\theta) = \frac{1}{2} ||y - X\theta||^2$-   **Ridge**:$J(\theta) = \frac{1}{2} ||y - X\theta||^2 + \frac{\lambda}{2} ||\theta||_2^2$-   **Lasso**:$J(\theta) = \frac{1}{2} ||y - X\theta||^2 + \lambda ||\theta||_1$

**Gradients:**
-   **OLS**: $-(y - \hat{y})X$-   **Ridge**:$-(y - \hat{y})X + \lambda \theta$-   **Lasso**:$-(y - \hat{y})X + \lambda \cdot \text{sign}(\theta)$

## 4. Inputs Required

-   **X**: Training features.
-   **y**: Target values.
-   **n_iterations**: Number of gradient descent steps.
-   **learning_rate**: Step size.
-   **reg_factor**: Regularization strength ($\lambda$).
-   **degree**: Degree of polynomial polynomials features (e.g., degree=2 $\rightarrow x, x^2$).

## 5. Usage Guidelines

### When to use:
-   **Linear Relationships**: Predicting sales, prices, usually good first baseline.
-   **Interpretability**: Coefficients give direct indication of feature importance and direction of effect.
-   **Lasso**: If you suspect only a few features are important (Feature Selection).
-   **Ridge**: If features are highly correlated (Multicollinearity).

### When not to use:
-   **Highly Non-Linear Data**: Fails unless using Polynomial features (which can get expensive).
-   **Outliers**: Sensitive to outliers (Squared error penalizes large errors heavily).

### Industry Best Practices:
-   **Scaling**: Features MUST be scaled/normalized for Gradient Descent/Regularization to work correctly.
-   **Regularization**: Almost always verify with Ridge/Lasso rather than plain OLS to prevent overfitting.
-   **Closed Form vs GD**: Use Closed Form (Normal Equation) for small datasets ($< 10k$ samples). Use Gradient Descent (`SGDRegressor`) for large datasets.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: Matrix operations are optimized.
-   **Memory**: Closed form solution requires $O(features^2)$memory to invert the matrix. Gradient descent is$O(features)$.

## 7. Underlying Data Structure

-   **Numpy Arrays**: Parameters and data.
