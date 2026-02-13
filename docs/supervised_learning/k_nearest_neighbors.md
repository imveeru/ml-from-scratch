# K-Nearest Neighbors (KNN)

## 1. Pseudocode

```text
Predict(X_test, X_train, y_train, k):
    predictions = []
    
    For each test_sample in X_test:
        distances = []
        
        For each train_sample in X_train:
            Calculate Euclidean distance between test_sample and train_sample
            Add distance to distances list
        
        Sort distances in ascending order
        Get indices of the first k elements (k nearest neighbors)
        Get labels of these k neighbors from y_train
        
        # Majority Vote
        predicted_label = MostCommon(neighbor_labels)
        Add predicted_label to predictions
        
    Return predictions
```

## 2. Algorithm Explanation

**K-Nearest Neighbors (KNN)** is a non-parametric, **lazy learning** algorithm.
-   **Non-parametric**: It makes no assumptions about the underlying data distribution.
-   **Lazy Learning**: It does not "learn" a discriminative function from the training data during a training phase. Instead, it stores the entire training dataset and performs the computation (distance calculation) only at prediction time.

To classify a new data point, the algorithm:
1.  Calculates the distance from the new point to all other points in the dataset.
2.  Selects the `k` closest points (neighbors).
3.  Assigns the class label that is most common among those `k` neighbors (Majority Vote).

## 3. Math Formulas

**Euclidean Distance:**
$$ d(x, x') = \sqrt{\sum_{i=1}^{n} (x_i - x'_i)^2} $$
Where $x$is the test sample and $x'$is a training sample, and$n$ is the number of features.

## 4. Inputs Required

-   **X_test**: Test features of shape `(n_test_samples, n_features)`.
-   **X_train**: Training features of shape `(n_train_samples, n_features)`.
-   **y_train**: Training labels of shape `(n_train_samples,)`.
-   **k**: Hyperparameter, the number of neighbors to consider.

## 5. Usage Guidelines

### When to use:
-   **Small Datasets**: Works well and is very simple to implement/understand on small datasets.
-   **Non-Linear Data**: Capable of learning complex, non-linear decision boundaries.
-   **Baseline**: Good baseline model before trying complex algorithms.

### When not to use:
-   **Large Datasets**: Computationally expensive at prediction time ($O(N)$ for each test sample). Efficient search structures like KD-Trees or Ball Trees are needed for large data.
-   **High Dimensionality**: Suffers from the "Curse of Dimensionality". Distance metrics become less meaningful as dimensions increase.
-   **Noisy Data**: Sensitive to outliers/noise, especially with small `k`.

### Industry Best Practices:
-   **Feature Scaling**: **Crucial**. Since KNN relies on distance, features with larger scales will dominate the distance calculation. Always normalize/standardize data (e.g., to mean 0, variance 1) before using KNN.
-   **Choosing k**: 
    -   Small `k` (e.g., 1) $\rightarrow$ High variance (overfitting), jagged decision boundaries.
    -   Large `k` $\rightarrow$ High bias (underfitting), smooth boundaries.
    -   Select `k` using Cross-Validation. odd numbers are preferred for binary classification to avoid ties.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency/Parallelism**: Prediction is **embarrassingly parallel**. The distance calculation for each test sample is independent of others.
-   **Memory Management**: **High memory usage**. Requires storing the entire training dataset `X_train` and `y_train` in memory.

## 7. Underlying Data Structure

-   **Numpy Arrays**: Used to store the training data and perform vectorized distance calculations.
-   **Sorting**: `np.argsort` is used to find the indices of the nearest neighbors.
