# Gradient Boosting

## 1. Pseudocode

```text
GradientBoosting(X, y, n_estimators, learning_rate):
    Initialize F_0(x) with a constant value (e.g., mean of y)
    
    For m = 1 to n_estimators:
        Compute pseudo-residuals:
            r_im = - gradient(Loss(y_i, F_{m-1}(x_i)))
            
        Train a Regression Tree h_m(x) to predict r_im from X
        
        Update model:
            F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
            
    Return F_n_estimators(x)
```

## 2. Algorithm Explanation

**Gradient Boosting** is an ensemble technique that builds a model in a stage-wise fashion. Unlike Adaboost, which updates instance weights, Gradient Boosting fits a new weak learner (typically a regression tree) to the *residual errors* (specifically, the negative gradient of the loss function) of the previous model.

This allows the algorithm to optimize arbitrary differentiable loss functions.
- **Regression**: Uses Square Loss. The negative gradient is simply $(y - \hat{y})$.
- **Classification**: Uses Cross-Entropy ("Log Loss"). The algorithm fits trees to the gradients of the log-likelihood.

In this implementation, `RegressionTree` is used as the base learner for both classification and regression tasks because the algorithm is essentially regressing on the gradients.

## 3. Math Formulas

**Model Update:**
$$ F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x) $$
Where $\nu$ is the `learning_rate`.

**Square Loss (Regression):**
$$ L(y, F(x)) = \frac{1}{2}(y - F(x))^2 $$
$$ -\frac{\partial L}{\partial F(x)} = y - F(x) $$ (Residual)

**Cross Entropy (Classification/Logistic):**
$$ L(y, p) = -y \log(p) - (1-y) \log(1-p) $$
Where $p = \sigma(F(x))$ (Sigmoid of the raw prediction).

## 4. Inputs Required

-   **X**: Training features (`n_samples`, `n_features`).
-   **y**: Target labels/values (`n_samples`,).
-   **n_estimators**: Number of boosting stages (trees) to perform.
-   **learning_rate**: Shrinkage parameter $\nu$ ($0 < \nu \le 1$) that scales the contribution of each tree.
-   **min_samples_split**, **min_impurity**, **max_depth**: Parameters for the underlying decision trees.

## 5. Usage Guidelines

### When to use:
-   **Structured/Tabular Data**: State-of-the-art performance on tabular data (along with variants like XGBoost/LightGBM).
-   **Heterogeneous Features**: Handles mix of numerical and categorical features well (via the decision trees).
-   **Prediction Accuracy**: Often provides higher accuracy than Random Forests.

### When not to use:
-   **High-Dimensional/Sparse Data**: Can be slow and prone to overfitting on very high-dimensional sparse data (like text). Linear models might be better.
-   **Real-time Constraints**: Prediction can be slow because it requires evaluating $N$ trees sequentially.
-   **Noise**: Can overfit to noise if number of trees is too large.

### Industry Best Practices:
-   **Regularization**: Tune `learning_rate` and `n_estimators` together. Lower learning rate with more estimators is generally better but slower.
-   **Early Stopping**: Stop training when validation error stops improving to prevent overfitting.
-   **Subsampling**: Stochastic Gradient Boosting (using a subset of data for each tree) often improves generalization.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency/Parallelism**: Like Adaboost, training is **sequential** (Step $m$ needs Step $m-1$), so trees cannot be trained in parallel. However, tree construction itself can be parallelized.
-   **Memory Management**: Stores `n_estimators` tree objects. Can be memory-intensive if trees are deep or numerous.

## 7. Underlying Data Structure

-   **List**: Stores the sequence of `RegressionTree` objects (`self.trees`).
-   **Numpy Arrays**: Used for vectorized gradient calculations and data manipulation.
