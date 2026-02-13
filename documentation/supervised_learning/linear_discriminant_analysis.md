# Linear Discriminant Analysis (LDA)

## 1. Pseudocode

```text
Fit(X, y):
    Split X into X1 (class 0) and X2 (class 1)
    
    Calculate Covariance Matrix for X1 -> cov1
    Calculate Covariance Matrix for X2 -> cov2
    Total Covariance cov_tot = cov1 + cov2
    
    Calculate Mean vector for X1 -> mean1
    Calculate Mean vector for X2 -> mean2
    Mean Difference mean_diff = mean1 - mean2
    
    # Calculate parameters allowing for best separation
    theta = inverse(cov_tot) * mean_diff

Predict(X):
    predictions = []
    For sample in X:
        projection = sample.dot(theta)
        If projection < 0:
            y = 1
        Else:
            y = 0
        Add y to predictions
    Return predictions
```

## 2. Algorithm Explanation

**Linear Discriminant Analysis (LDA)**, also known as Fisher's Linear Discriminant, is a method used for both classification and dimensionality reduction.

The goal of LDA is to find a linear combination of features that best separates two or more classes. It does this by maximizing the ratio of **between-class variance** to the **within-class variance** in the data.

In this implementation (Binary Classification):
1.  Computes the mean vectors and covariance matrices for each class.
2.  Computes the direction $\theta$ that maximizes the separation between the class means while minimizing the spread (covariance) within each class.
3.  Projects data onto this vector $\theta$.
4.  Classifies samples based on their position on this projection line.

## 3. Math Formulas

**Optimal Parameter Vector ($\theta$):**
$$ \theta \propto S_W^{-1} (\mu_1 - \mu_2) $$
Where:
-   $S_W = \Sigma_1 + \Sigma_2$ is the total within-class scatter (sum of covariance matrices).
-   $\mu_1, \mu_2$ are the mean vectors of the two classes.

**Prediction (Projection):**
$$ y_{pred} = \theta^T x $$

## 4. Inputs Required

-   **X**: Training features (`n_samples`, `n_features`).
-   **y**: Binary target labels (`n_samples`,). This implementation assumes labels are 0 and 1.

## 5. Usage Guidelines

### When to use:
-   **Dimensionality Reduction**: Excellent for reducing dimensions while preserving class discriminability before applying another classifier.
-   **Well-Separated Classes**: Works better than Logistic Regression when classes are well-separated.
-   **Gaussian Distribution**: Optimal if the features are normally distributed and classes have identical covariance matrices.

### When not to use:
-   **Non-Linear Boundaries**: LDA assumes a linear decision boundary. Fails on complex, non-linear data (unless used with kernels).
-   **Unequal Covariances**: If classes have significantly different covariance matrices, Quadratic Discriminant Analysis (QDA) is better.

### Industry Best Practices:
-   **Preprocessing**: Sensitive to outliers. Robust scaling or outlier removal is recommended.
-   **Assumption Check**: Check if features are roughly Gaussian. If not, consider transforming them (e.g., Box-Cox).
-   **Multicollinearity**: If features are highly correlated, the covariance matrix inversion can be unstable. Regularization (Shrinkage LDA) can help.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: Matrix operations (`np.linalg.pinv`, `dot`) are optimized and often parallelized by the underlying linear algebra libraries (BLAS/LAPACK).
-   **Memory**: Requires storing and operating on covariance matrices of size $(n_{features} \times n_{features})$. High memory usage if feature count is very large.

## 7. Underlying Data Structure

-   **Numpy Arrays**: Extensive use of numpy for Linear Algebra operations (Covariance, Pseudo-Inverse, Dot Product).
