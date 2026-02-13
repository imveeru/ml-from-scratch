# Benchmarking Demo

This example benchmarks multiple classifiers on the Digits dataset.

## Description
-   **Dataset**: Sklearn `digits` (1 vs 8).
-   **Task**: Binary Classification.
-   **Models Compared**:
    -   Adaboost
    -   Decision Tree
    -   Gradient Boosting
    -   LDA
    -   Logistic Regression
    -   Multilayer Perceptron (MLP)
    -   Naive Bayes
    -   Perceptron
    -   Random Forest
    -   Support Vector Machine (SVM)
    -   XGBoost
-   **Visualization**: Plots the dataset in 2D PCA.

## Usage
Run the script from the root directory:
```bash
python -m mlfromscratch.examples.demo
```

## Output
-   Prints training progress for models.
-   Prints comparative **Accuracy** for all models.
-   Displays a scatter plot of the dataset.
