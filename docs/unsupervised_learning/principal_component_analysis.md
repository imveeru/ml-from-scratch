# Principal Component Analysis (PCA)

## 1. Pseudocode

```text
Transform(X, n_components):
    # 1. Covariance Matrix
    Cov = Calculate_Covariance_Matrix(X)
    
    # 2. Eigendecomposition
    Eigenvalues, Eigenvectors = Eig(Cov)
    
    # 3. Sort & Select
    Sort Eigenvalues descending
    Select top n_components Eigenvectors -> Theta
    
    # 4. Project
    Return X.dot(Theta)
```

## 2. Algorithm Explanation

**Principal Component Analysis (PCA)** is a dimensionality reduction technique. It transforms the data into a new coordinate system such that the greatest variance by some scalar projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.

It is basically a linear transformation that rotates the axes to align with the directions of maximum variance in the data.

## 3. Math Formulas

**Covariance Matrix:**
$$ \Sigma = \frac{1}{n-1} (X - \bar{X})^T (X - \bar{X}) $$

**Projection:**
$$ T = X \Theta_k $$
Where $\Theta_k$ is the matrix of the top $k$ eigenvectors.

## 4. Inputs Required

-   **X**: Data points.
-   **n_components**: Target dimensionality.

## 5. Usage Guidelines

### When to use:
-   **Dimensionality Reduction**: Preprocessing step to reduce features before training a supervised model (avoids curse of dimensionality).
-   **Visualization**: Projecting high-dimensional data to 2D or 3D for plotting.
-   **Noise Reduction**: Removing components with small eigenvalues (which usually correspond to noise).

### When not to use:
-   **Non-Linear Manifolds**: PCA assumes linear patterns. If data lies on a Swiss Roll or other non-linear manifold, use t-SNE, UMAP, or Autoencoders.
-   **Interpretability**: The new features (Principal Components) are linear combinations of original features and often hard to interpret physically.

### Industry Best Practices:
-   **Standardization**: PCA is scale-invariant? NO. It is **highly sensitive** to scale. You **MUST** standardize data (mean=0, variance=1) before running PCA, or large magnitude features will dominate the eigenvalues.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: Matrix multiplication is parallelized.
-   **Memory**: Computing Covariance Matrix ($D \times D$) and Eigendecomposition can be expensive for very high $D$. Randomized PCA is often used for large datasets.

## 7. Underlying Data Structure

-   **Numpy Arrays**: Linear algebra.
