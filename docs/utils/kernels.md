# Kernels

This file implements kernel functions used in Support Vector Machines (SVM) and other kernel-based algorithms. Kernels allow algorithms to operate in a high-dimensional implicit feature space without computing the coordinates of the data in that space.

## 1. Mathematical Formulas

### Linear Kernel
$$ K(x_1, x_2) = x_1^T x_2 $$
-   **Parameters**: None.

### Polynomial Kernel
Represents the similarity of vectors in a training set in a feature space over polynomials of the original variables.
$$ K(x_1, x_2) = (x_1^T x_2 + c)^d $$
-   **Parameters**:
    -   `power` ($d$): The degree of the polynomial.
    -   `coef` ($c$): Zero coefficient.

### RBF Kernel (Radial Basis Function)
Also known as the Gaussian Kernel. It is a popular kernel function used in various kernelized learning algorithms. In particular, it is commonly used in support vector machine classification.
$$ K(x_1, x_2) = \exp(-\gamma ||x_1 - x_2||^2) $$
-   **Parameters**:
    -   `gamma` ($\gamma$): Kernel coefficient. Determines how far the influence of a single training example reaches. Low values mean 'far' and high values mean 'close'.

## 2. Usage Guidelines

### Industry Best Practices

-   **Linear**: Use when the number of features is large (e.g., text classification) and data is likely linearly separable. Fast training.
-   **RBF**: The default choice for non-linear data. However, it requires careful tuning of $\gamma$(gamma) and $C$ (regularization).
-   **Polynomial**: Less common than RBF but useful in image processing.
