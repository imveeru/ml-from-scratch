# Random Forest

## 1. Pseudocode

```text
Fit(X, y, n_estimators):
    Initialize list of Trees
    
    For i = 1 to n_estimators:
        # Bootstrap Aggregation (Bagging)
        X_subset, y_subset = Get_Random_Subset_With_Replacement(X, y)
        
        # Feature Subsampling
        Select random subset of feature indices -> feature_idx
        X_subset_features = X_subset[:, feature_idx]
        
        # Train Tree
        Tree = ClassificationTree()
        Tree.fit(X_subset_features, y_subset)
        
        Save Tree and feature_idx

Predict(X):
    predictions = []
    For each Tree in Trees:
        # Use only the features this tree was trained on
        X_features = X[:, Tree.feature_idx]
        pred = Tree.predict(X_features)
        Add pred to predictions
        
    # Majority Vote
    Return Mode(predictions) for each sample
```

## 2. Algorithm Explanation

**Random Forest** is an ensemble learning method that constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

It combines two key concepts to reduce overfitting and variance:
1.  **Bagging (Bootstrap Aggregation)**: Each tree is trained on a random subset of the data sampled with replacement.
2.  **Random Subspace Method (Feature Bagging)**: In this specific implementation, each tree is trained on a random subset of *features*. (Note: Standard Random Forest implementations typically select random features at *each split* of the tree, whereas this implementation selects them once per tree).

## 3. Math Formulas

**Final Prediction (Classification):**
$$ \hat{y} = \text{mode} \{ h_1(x), h_2(x), \dots, h_N(x) \} $$
Where $h_i(x)$ is the prediction of the $i$-th tree.

## 4. Inputs Required

-   **X**: Training features.
-   **y**: Training labels.
-   **n_estimators**: Number of trees in the forest.
-   **max_features**: Number of features to consider when looking for the best split.
-   **min_samples_split**, **min_gain**, **max_depth**: Constraints for the individual Decision Trees.

## 5. Usage Guidelines

### When to use:
-   **High Accuracy**: Generally produces very high accuracy on tabular data.
-   **Overfitting**: Less prone to overfitting than a single Decision Tree.
-   **Feature Importance**: Can be used to estimate feature importance (by seeing which features are used most).

### When not to use:
-   **Interpretability**: Harder to interpret than a single Decision Tree (Black Box).
-   **Latency**: Prediction can be slow (must run $N$ trees).
-   **Sparse Data**: Doesn't work well on very sparse data (like text) compared to linear models.

### Industry Best Practices:
-   **Number of Trees**: More trees are generally better (stabilizes the prediction) but increase computation cost. 100-500 is a common range.
-   **Feature Ratio**: A common default for `max_features` is $\sqrt{n\_features}$ for classification.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: **Embarrassingly Parallel**. Each tree can be trained completely independently of the others.
-   **Memory**: High memory usage. Must store $N$ full decision trees.

## 7. Underlying Data Structure

-   **List**: Stores the list of `ClassificationTree` objects.
-   **Numpy Arrays**: Data handling.
