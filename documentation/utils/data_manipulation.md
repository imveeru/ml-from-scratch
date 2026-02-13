# Data Manipulation Utils

This file contains utility functions for handling, transforming, and preprocessing data.

## 1. Functions Overview

### Splitting & Shuffling
-   **`shuffle_data(X, y, seed)`**: Randomly shuffles two arrays in unison.
-   **`train_test_split(X, y, test_size, shuffle, seed)`**: Splits data into training and testing sets.
-   **`k_fold_cross_validation_sets(X, y, k, shuffle)`**: Splits data into $k$ folds for cross-validation.
-   **`batch_iterator(X, y, batch_size)`**: Generates mini-batches from the dataset.

### Normalization
-   **`normalize(X, axis, order)`**: L2 normalization (scales vectors to unit length).
    $$ X_{norm} = \frac{X}{||X||_2} $$
-   **`standardize(X)`**: Z-score normalization (scales features to 0 mean and 1 variance).
    $$ X_{std} = \frac{X - \mu}{\sigma} $$

### Feature Engineering
-   **`polynomial_features(X, degree)`**: Generates polynomial combinations of features (e.g., $x_1^2, x_1 x_2$). Used in Polynomial Regression.
-   **`to_categorical(x, n_col)`**: One-hot encoding for nominal targets.
    -   Input: `[0, 1]` -> Output: `[[1, 0], [0, 1]]`
-   **`to_nominal(x)`**: Converts one-hot encoding back to indices (argmax).

### Misc
-   **`divide_on_feature(X, feature_i, threshold)`**: Splits a dataset into two branches based on a feature threshold (used in Decision Trees).
-   **`get_random_subsets(X, y, n_subsets, replacements)`**: Bootstrapping method (used in Random Forest).
-   **`make_diagonal(x)`**: Constructs a diagonal matrix from a vector.

## 2. Usage Guidelines
-   **Standardization**: Essential for algorithms based on distance (KNN, SVM, K-Means) or gradients (Neural Networks, Logistic Regression). Tree-based models (Random Forest, Decision Tree) generally don't require it.
