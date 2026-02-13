# Decision Tree

## 1. Pseudocode

```text
BuildTree(X, y, current_depth):
    If current_depth >= max_depth OR n_samples < min_samples_split:
        Return LeafNode(value = CalculateLeafValue(y))

    best_split = None
    max_gain = 0
    
    For each feature in X:
        For each unique value (threshold) in feature:
            Split X, y into (X_left, y_left) and (X_right, y_right) based on threshold
            
            Calculate Impurity Gain (Information Gain or Variance Reduction)
            
            If Gain > max_gain:
                max_gain = Gain
                best_split = {feature, threshold, left_sets, right_sets}
                
    If max_gain > min_impurity:
        left_branch = BuildTree(best_split.leftX, best_split.lefty, current_depth + 1)
        right_branch = BuildTree(best_split.rightX, best_split.righty, current_depth + 1)
        Return DecisionNode(feature, threshold, left_branch, right_branch)
        
    Return LeafNode(value = CalculateLeafValue(y))
```

## 2. Algorithm Explanation

**Decision Trees** are predictive models that map features to target values by recursively splitting the data. The tree structure consists of **Decision Nodes** (which split data based on a feature threshold) and **Leaf Nodes** (which provide the final prediction).

This implementation includes three variants:
1.  **Classification Tree**: Used for categorical targets.
    -   **Splitting Criterion**: Information Gain (based on Entropy).
    -   **Leaf Value**: Majority vote (most common class label).
2.  **Regression Tree**: Used for continuous targets.
    -   **Splitting Criterion**: Variance Reduction.
    -   **Leaf Value**: Mean of the target values.
3.  **XGBoost Regression Tree**: Specialized tree for Gradient Boosting.
    -   **Splitting Criterion**: Approximate gain based on Second-Order Taylor Expansion (Gradient and Hessian of loss function).

The tree grows recursively (Greedy Approach). At each step, it finds the best feature and threshold that separates the data most effectively (maximizing "purity").

## 3. Math Formulas

**Entropy (for Classification):**
$$ H(y) = - \sum_{c} p(c) \log_2 p(c) $$

**Information Gain:**
$$ IG = H(S) - \sum_{v \in \{left, right\}} \frac{|S_v|}{|S|} H(S_v) $$

**Variance (for Regression):**
$$ Var(y) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \bar{y})^2 $$

**Variance Reduction:**
$$ VR = Var(S) - \sum_{v \in \{left, right\}} \frac{|S_v|}{|S|} Var(S_v) $$

## 4. Inputs Required

-   **X**: Training features of shape `(n_samples, n_features)`.
-   **y**: Target values of shape `(n_samples,)`.
-   **min_samples_split**: Min number of samples required to split an internal node.
-   **min_impurity**: Min impurity decrease required for a split.
-   **max_depth**: Max depth of the tree.

## 5. Usage Guidelines

### When to use:
-   **High Interpretability**: When you need to understand *why* a decision was made (rules are explicit).
-   **Mixed Data Types**: Handles both numerical and categorical data well (though this implementation treats all as numerical thresholds).
-   **Non-Linear Relationships**: Can capture non-linear patterns without heavy feature engineering.

### When not to use:
-   **High Dimensionality**: Can easily overfit on sparse, high-dimensional data.
-   **Linear Relationships**: If data is strictly linear, Linear Regression is better.
-   **Stability**: Small changes in data can result in a completely different tree (high variance). Random Forests solve this.

### Industry Best Practices:
-   **Pruning**: Use `max_depth`, `min_samples_split`, or `min_samples_leaf` to prevent the tree from growing too complex and overfitting.
-   **Ensembling**: Rarely used in isolation in production. Almost always used within **Random Forests** or **Gradient Boosting Machines (XGBoost/LightGBM)** for better accuracy and stability.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency/Parallelism**: Building a single tree is difficult to parallelize effectively due to its sequential, recursive nature. However, the calculation of impurity for different features at a specific node can be parallelized. (This implementation is single-threaded).
-   **Memory Management**: Recursive structures can consume stack space. If the tree is very deep (`max_depth=inf`), it might hit recursion limits or memory issues on large datasets.

## 7. Underlying Data Structure

-   **Tree (N-ary Tree / Binary Tree)**: The core structure is a linked set of `DecisionNode` objects. Each node points to `true_branch` (Left) and `false_branch` (Right).
-   **Recursion**: The `_build_tree` method uses recursion to construct the graph.
-   **Numpy Arrays**: Efficiently handles data filtering and calculations at each node.
