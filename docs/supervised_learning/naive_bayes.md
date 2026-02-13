# Naive Bayes (Gaussian)

## 1. Pseudocode

```text
Fit(X, y):
    For each class c in y:
        Calculate Prior P(c) = count(c) / total_samples
        For each feature f in X:
            Calculate Mean of feature f for class c -> mu_{c,f}
            Calculate Variance of feature f for class c -> sigma^2_{c,f}
            Store parameters (mu, sigma^2)

Predict(sample):
    For each class c:
        Initialize Posterior = P(c)
        For each feature value x_i in sample:
            # Calculate Likelihood using Gaussian PDF
            Likelihood = Gaussian(x_i, mu_{c,i}, sigma^2_{c,i})
            Posterior *= Likelihood
            
    Return Class with Maximum Posterior Probability
```

## 2. Algorithm Explanation

**Naive Bayes** is a probabilistic classifier based on Bayes' Theorem with the "naive" assumption of conditional independence between every pair of features given the value of the class variable.

This implementation is **Gaussian Naive Bayes**, which assumes that the continuous values associated with each class are distributed according to a Gaussian (Normal) distribution.

It calculates the posterior probability $P(Class | Data)$ for each class and selects the one with the highest probability.

## 3. Math Formulas

**Bayes' Theorem:**
$$ P(c|x) = \frac{P(x|c) P(c)}{P(x)} $$
Since $P(x)$ is constant for all classes, we ignore it:
$$ P(c|x) \propto P(c) \prod_{i=1}^{n} P(x_i|c) $$

**Gaussian Likelihood:**
$$ P(x_i|c) = \frac{1}{\sqrt{2\pi\sigma^2_{c,i}}} \exp\left(-\frac{(x_i - \mu_{c,i})^2}{2\sigma^2_{c,i}}\right) $$

## 4. Inputs Required

-   **X**: Training features (`n_samples`, `n_features`).
-   **y**: Target labels (`n_samples`,).

## 5. Usage Guidelines

### When to use:
-   **Text Classification**: Very effective for spam filtering, sentiment analysis (using Multinomial/Bernoulli variants, though Gaussian is for continuous data).
-   **High Dimensions**: Works well with high-dimensional data because of the independence assumption (less parameters to estimate).
-   **Small Data**: Needs less training data to estimate the parameters (mean and variance) compared to discriminative models.

### When not to use:
-   **Correlated Features**: If features are highly correlated, the independence assumption is violated, leading to poor performance.
-   **Complex Boundaries**: Can only learn linear decision boundaries (in the log-space).

### Industry Best Practices:
-   **Log-Probabilities**: In practice, compute sums of log-probabilities ($\log(P(c)) + \sum \log(P(x_i|c))$) instead of products of probabilities to prevent numerical underflow.
-   **Feature Engineering**: Remove highly correlated features to satisfy the independence assumption.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: 
    -   Training is $O(N \cdot D)$ and can be parallelized by class.
    -   Prediction is parallelizable per sample.
-   **Memory**: Very low memory footprint. Only needs to store mean and variance for each feature per class ($C \times D \times 2$ parameters).

## 7. Underlying Data Structure

-   **Numpy Arrays**: Stores class statistics (means and variances) efficiently.
-   **Dictionary/List**: Used to organize parameters by class.
