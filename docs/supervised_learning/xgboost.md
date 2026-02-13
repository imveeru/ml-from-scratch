# XGBoost

## 1. Pseudocode

```text
Fit(X, y, n_estimators):
    Initialize Predictions y_pred = 0
    
    For i = 1 to n_estimators:
        # Calculate Gradient (g) and Hessian (h) of Loss w.r.t y_pred
        # Train Tree to maximize Gain based on g and h
        Tree = XGBoostRegressionTree()
        Tree.fit(X, g, h)
        
        # Update Predictions
        y_pred -= learning_rate * Tree.predict(X)
        
Predict(X):
    y_pred = 0
    For Tree in Trees:
        y_pred -= learning_rate * Tree.predict(X)
        
    Return Softmax(y_pred)
```

## 2. Algorithm Explanation

**XGBoost** (eXtreme Gradient Boosting) is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.

It improves upon standard Gradient Boosting by:
1.  **Second-Order Approximation**: Uses the second-order Taylor expansion of the loss function (involving both Gradient and Hessian) to calculate the value of the leaf nodes and the split quality.
2.  **Regularization**: Explicitly adds regularization terms (L1/L2 on parameters, number of leaves) to the objective function to control overfitting.
3.  **Split Finding**: Uses a specialized Gain formula derived from the regularized objective.

This implementation focuses on the core mathematical principles (Taylor expansion and custom Gain calculation) for Classification.

## 3. Math Formulas

**Objective Function (at step t):**
$$ Obj^{(t)} \approx \sum_{i=1}^{n} [g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \Omega(f_t) $$
Where $g_i$is first derivative (gradient) and $h_i$ is second derivative (Hessian) of loss.

**Leaf Parameter Calculation:**
$$ \theta^* = -\frac{\sum g_i}{\sum h_i + \lambda} $$

**Split Gain:**
$$ Gain = \frac{1}{2} \left[ \frac{(\sum_L g_i)^2}{\sum_L h_i + \lambda} + \frac{(\sum_R g_i)^2}{\sum_R h_i + \lambda} - \frac{(\sum g_i)^2}{\sum h_i + \lambda} \right] - \gamma $$

## 4. Inputs Required

-   **X**: Training features.
-   **y**: Training labels.
-   **n_estimators**: Number of trees.
-   **learning_rate**: Shrinkage parameter.
-   **min_impurity**: Minimum gain required to make a split.

## 5. Usage Guidelines

### When to use:
-   **Tabular Data**: Simply the best general-purpose algorithm for tabular data competitions (Kaggle) and production systems.
-   **Speed and Performance**: constant optimizations make it faster and more accurate than standard GBM.

### When not to use:
-   **Unstructured Data**: Deep Learning is better for Images/Audio/Text.
-   **Very Small Data**: Linear models might be more robust and interpretable.

### Industry Best Practices:
-   **Hyperparameter Tuning**: Critical. Tune `max_depth`, `subsample`, `colsample_bytree`, `eta` (learning rate).
-   **Early Stopping**: Always use early stopping with a validation set.
-   **Missing Values**: XGBoost handles missing values internally (learns default direction), so imputation is not strictly necessary but often helpful.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**:
    -   Level 1: Iterative training of trees (Sequential).
    -   Level 2: **Block Structure**. Parallelizes the split finding process at each node across all features.
-   **Memory**: Efficient, but dataset size influences memory.

## 7. Underlying Data Structure

-   **Numpy Arrays**: Used for gradients and hessians.
-   **Trees**: Stores a custom `XGBoostRegressionTree` structure.
