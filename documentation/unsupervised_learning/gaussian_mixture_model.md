# Gaussian Mixture Model (GMM)

## 1. Pseudocode

```text
Run_EM(X, k, max_iterations):
    # Initialization
    Initialize Means, Covariances, and Priors randomly
    
    For i = 1 to max_iterations:
        # E-Step (Expectation)
        For each sample x_i, calculate probability it belongs to cluster k:
            Likelihood = P(x_i | Mean_k, Cov_k)
            Responsibility[i, k] = Prior_k * Likelihood
        Normalize Responsibilities so they sum to 1 per sample
        
        # M-Step (Maximization)
        For each cluster k:
            Total_Resp = Sum(Responsibility[:, k])
            
            # Update Parameters
            Mean_k = Sum(Responsibility[i, k] * x_i) / Total_Resp
            Cov_k = Sum(Responsibility[i, k] * (x_i - Mean_k)^2) / Total_Resp
            Prior_k = Total_Resp / N
            
        # Check Convergence
        If Change_in_Likelihood < Tolerance:
            Break
            
    Return Cluster Assignments (Argmax Responsibility)
```

## 2. Algorithm Explanation

**Gaussian Mixture Model (GMM)** is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.

It is a **Soft Clustering** method, meaning each data point belongs to every cluster with a certain probability (responsibility), rather than a hard assignment like K-Means.

It uses the **Expectation-Maximization (EM)** algorithm to iteratively optimize the parameters:
1.  **E-Step**: Estimate the probability (responsibility) of each point belonging to each cluster using current parameters.
2.  **M-Step**: Update the parameters (Mean, Covariance, Mixing Coefficients) to maximize the likelihood of the data given the responsibilities found in E-Step.

## 3. Math Formulas

**Multivariate Gaussian PDF:**
$$ \mathcal{N}(x | \mu, \Sigma) = \frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right) $$

**Responsibility (E-Step):**
$$ r_{nk} = \frac{\pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_n | \mu_j, \Sigma_j)} $$

**Parameter Updates (M-Step):**
$$ \mu_k^{new} = \frac{1}{N_k} \sum_{n=1}^{N} r_{nk} x_n $$
$$ \Sigma_k^{new} = \frac{1}{N_k} \sum_{n=1}^{N} r_{nk} (x_n - \mu_k^{new})(x_n - \mu_k^{new})^T $$

## 4. Inputs Required

-   **X**: Data points.
-   **k**: Number of clusters.
-   **max_iterations**: Limit on EM steps.
-   **tolerance**: Threshold for convergence.

## 5. Usage Guidelines

### When to use:
-   **Soft Assignments**: When you need probabilities of cluster membership (e.g., "This user is 80% Segment A, 20% Segment B").
-   **Elliptical Clusters**: Unlike K-Means (spherical), GMM can model elliptical clusters because each cluster has its own covariance matrix.

### When not to use:
-   **Non-Gaussian Data**: If the underlying clusters are definitely not Gaussian (e.g., concentric circles).
-   **Initialization**: Sensitive to initialization (can get stuck in local optima), often initialized with K-Means.
-   **Singularities**: Covariance matrix can become singular if a cluster collapses to a single point.

### Industry Best Practices:
-   **Covariance Type**: In libraries like Scikit-Learn, you can constrain the covariance matrix (`full`, `tied`, `diag`, `spherical`) to reduce parameters and overfitting.
-   **BIC/AIC**: Use Bayesian Information Criterion to determine the optimal number of clusters ($k$).

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: E-Step is fully parallelizable across samples.
-   **Memory**: Requires storing full covariance matrices ($K \times D^2$).

## 7. Underlying Data Structure

-   **Numpy Arrays**: For matrices.
-   **Linear Algebra**: Determinants and Inverses (`np.linalg.det`, `np.linalg.pinv`).
