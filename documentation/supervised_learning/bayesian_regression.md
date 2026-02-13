# Bayesian Regression

## 1. Pseudocode

```text
Fit(X, y, n_draws):
    # Calculate Posterior Hyperparameters (based on Conjugate Priors)
    Omega_n = X.T.dot(X) + Omega_0
    Mu_n = Inverse(Omega_n) . (X.T.dot(X).dot(theta_hat) + Omega_0.dot(Mu_0))
    nu_n = nu_0 + n_samples
    sigma_sq_n = ... (Scale update)
    
    # Sampling (Monte Carlo Simulation)
    For i = 1 to n_draws:
        # 1. Draw variance from Scaled Inverse Chi-Squared
        sigma_sq = Draw_Scaled_Inv_Chi2(df=nu_n, scale=sigma_sq_n)
        
        # 2. Draw parameters from Multivariate Normal
        theta = Draw_Multivariate_Normal(mean=Mu_n, cov=sigma_sq * Inverse(Omega_n))
        
        Store theta
    
    theta_mean = Mean(theta_draws)
    Calculate Credible Intervals (ETI) from theta_draws

Predict(X):
    Return X.dot(theta_mean)
    # Optionally return lower/upper bounds based on ETI
```

## 2. Algorithm Explanation

**Bayesian Regression** takes a probabilistic approach to linear regression. Instead of finding a single "best" set of parameters (like OLS or Ridge), it computes the **Posterior Probability Distribution** of the parameters, given the data and **Prior** beliefs.

It assumes:
1.  **Prior**: The parameters ($\theta$) follow a Normal Distribution, and the variance ($\sigma^2$) follows a Scaled Inverse Chi-Squared distribution.
2.  **Likelihood**: The data follows a Normal distribution.

Since these are **Conjugate Priors**, the Posterior distribution is also of the same family (Normal-Inverse-Chi-Squared). This allows us to analytically solve for the posterior parameters and then sample specific parameter vectors to form a distribution of possible models.

## 3. Math Formulas

**Posterior Parameters:**
$$ \Omega_n = X^T X + \Omega_0 $$
$$ \mu_n = \Omega_n^{-1} (X^T X \hat{\theta} + \Omega_0 \mu_0) $$
$$ \nu_n = \nu_0 + n $$

**Sampling:**
$$ \sigma^2 \sim \text{Scale-Inv-}\chi^2(\nu_n, \sigma_n^2) $$
$$ \theta \sim \mathcal{N}(\mu_n, \sigma^2 \Omega_n^{-1}) $$

## 4. Inputs Required

-   **X**: Training features.
-   **y**: Target values.
-   **n_draws**: Number of Monte Carlo samples to draw from the posterior.
-   **mu0**: Prior mean of parameters.
-   **omega0**: Prior precision matrix of parameters.
-   **nu0**: Prior degrees of freedom.
-   **sigma_sq0**: Prior scale.

## 5. Usage Guidelines

### When to use:
-   **Uncertainty Quantification**: When you need to know how confident the model is about its predictions (Credible Intervals).
-   **Prior Knowledge**: When you have strong prior beliefs about the parameters (e.g., from previous experiments).
-   **Small Data**: Bayesian methods are robust against overfitting on small datasets because they average over many possible models.

### When not to use:
-   **Large Data**: Computationally expensive due to matrix inversions and sampling. For large N, the posterior converges to the OLS solution anyway, so OLS is preferred for speed.
-   **High Dimensions**: Tuning priors for many dimensions is difficult.

### Industry Best Practices:
-   **Uninformative Priors**: If you don't have prior knowledge, use uninformative priors (e.g., $\mu_0=0, \Omega_0 \approx 0$) which makes the results similar to OLS/Ridge.
-   **Credible Intervals**: Use the output intervals to make risk-aware decisions (e.g., "We are 95% sure the value is between A and B").

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: Matrix operations are parallelized. Sampling is sequential here but could be parallelized.
-   **Memory**: Standard matrix requirements ($O(Features^2)$).

## 7. Underlying Data Structure

-   **Numpy Arrays**: Linear algebra.
-   **Scipy Stats**: Used for random variable sampling (`chi2`, `multivariate_normal`).
