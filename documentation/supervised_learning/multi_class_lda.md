# Multi-Class Linear Discriminant Analysis (LDA)

## 1. Pseudocode

```text
Transform(X, y, n_components):
    # Calculate Scatter Matrices
    Compute Global Mean of X -> mean_total
    Initialize Within-Class Scatter Matrix SW = 0
    Initialize Between-Class Scatter Matrix SB = 0
    
    For each class c:
        X_c = samples belonging to class c
        mean_c = mean of X_c
        
        # Add to Within-Class Scatter
        SW += (n_samples_c - 1) * Covariance(X_c)
        
        # Add to Between-Class Scatter
        mean_diff = mean_c - mean_total
        SB += n_samples_c * (mean_diff . mean_diff.T)
        
    # Solve Generalized Eigenvalue Problem
    Compute Matrix A = inverse(SW) . SB
    Calculate Eigenvalues and Eigenvectors of A
    
    Sort Eigenvectors by Eigenvalues (descending)
    Select top n_components Eigenvectors -> Theta
    
    # Project Data
    X_transformed = X . Theta
    Return X_transformed
```

## 2. Algorithm Explanation

**Multi-Class LDA** generalizes Fisher's Linear Discriminant to problems with more than two classes. Ideally, it finds a lower-dimensional subspace that maximizes the separation between multiple classes.

It works by analyzing two measures of scatter:
1.  **Within-Class Scatter ($S_W$)**: How spread out the data is within each individual class. We want to *minimize* this.
2.  **Between-Class Scatter ($S_B$)**: How spread out the class means are from the total mean. We want to *maximize* this.

The algorithm finds the projection matrix $\Theta$ that maximizes the ratio $\text{det}(\Theta^T S_B \Theta) / \text{det}(\Theta^T S_W \Theta)$. This solution is given by the eigenvectors of $S_W^{-1} S_B$.

## 3. Math Formulas

**Within-Class Scatter Matrix:**
$$ S_W = \sum_{c} S_c = \sum_{c} \sum_{i \in c} (x_i - \mu_c)(x_i - \mu_c)^T $$

**Between-Class Scatter Matrix:**
$$ S_B = \sum_{c} N_c (\mu_c - \mu)(\mu_c - \mu)^T $$
Where $\mu$ is the overall mean and $\mu_c$ is the class mean.

**Optimization Problem:**
$$ \Theta_{opt} = \text{argmax}_\Theta \frac{|\Theta^T S_B \Theta|}{|\Theta^T S_W \Theta|} $$
Solution corresponds to the eigenvectors of $S_W^{-1} S_B$ with the largest eigenvalues.

## 4. Inputs Required

-   **X**: Training features (`n_samples`, `n_features`).
-   **y**: Target labels (`n_samples`,).
-   **n_components**: Number of dimensions to project data into (must be $< n_{classes} - 1$).

## 5. Usage Guidelines

### When to use:
-   **Dimensionality Reduction**: Great for reducing dimensions ($N \rightarrow C-1$) while keeping classes separated.
-   **Visualization**: projecting data to 2D for plotting class clusters.
-   **Preprocessing**: As a feature extraction step before a classifier (like KNN or Naive Bayes).

### When not to use:
-   **n_components limitation**: You can only project to at most $C-1$ dimensions (where $C$ is number of classes). If you need more dimensions, use PCA.
-   **Non-Linearity**: Fails if class separation is non-linear.

### Industry Best Practices:
-   **Scaling**: Standardize data before application.
-   **Regularization**: If $S_W$ is singular (not invertible), add a small constant to the diagonal (Regularized LDA).

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: Matrix operations are optimized by numpy.
-   **Memory**: Storing $S_W$ and $S_B$ ($Features \times Features$) can be expensive for high-dimensional data.

## 7. Underlying Data Structure

-   **Numpy Arrays**: Essential for handling matrices and eigendecomposition (`np.linalg.eigh`).
