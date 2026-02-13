# Adaboost Example

This example demonstrates how to use the `Adaboost` classifier from `supervised_learning` to classify handwritten digits.

## Description
-   **Dataset**: Sklearn `digits` dataset (filtered to only digits 1 and 8 for binary classification).
-   **Task**: Binary Classification (-1 vs 1).
-   **Model**: Adaboost with 5 weak classifiers (Decision Stumps).
-   **Visualization**: Projects the test data to 2D using PCA and plots the decision boundary/predictions.

## Usage
Run the script from the root directory:
```bash
python -m mlfromscratch.examples.adaboost
```

## Output
-   Prints accuracy score.
-   Displays a 2D scatter plot of the classification results.
