# Activation Functions

This file contains implementations of various activation functions used in Neural Networks, along with their gradients (derivatives) for backpropagation.

## 1. Mathematical Formulas

### Sigmoid
$`\sigma(x) = \frac{1}{1+e^{-x}}`$
**Derivative:** $\sigma(x)(1 - \sigma(x))$-   **Range**:$(0, 1)$. Good for binary classification output. Prone to vanishing gradient.

### Softmax
$$ \text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}} $$
-   **Range**: $(0, 1)$, sum = 1. Used for multi-class classification output.

### Tanh (Hyperbolic Tangent)
$$ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$
**Derivative:** $1 - \tanh^2(x)$-   **Range**:$(-1, 1)$. Zero-centered, generally better than Sigmoid for hidden layers.

### ReLU (Rectified Linear Unit)
$$ f(x) = \max(0, x) $$
**Derivative:** 1 if $x>0$ else 0
-   **Range**: $[0, \infty)$. Standard for modern deep learning. Solves vanishing gradient but suffers from dead ReLU.

### LeakyReLU
$$ f(x) = \max(\alpha x, x) $$
-   Allows small gradient when $x < 0$ to prevent dead neurons.

### ELU (Exponential Linear Unit)
$$ f(x) = x \text{ if } x \ge 0 \text{ else } \alpha(e^x - 1) $$
-   Smoother than ReLU, negative values allow pushing mean activation to 0.

### SELU (Scaled Exponential Linear Unit)
Self-normalizing properties to keep mean 0 and variance 1.

### SoftPlus
$$ f(x) = \ln(1 + e^x) $$
-   Smooth approximation of ReLU.

## 2. Usage Guidelines

### When to use:
-   **Hidden Layers**: **ReLU** is the default choice. Use **LeakyReLU/ELU/SELU** if the network is deep and dying ReLUs are a problem.
-   **Output Layer**:
    -   **Sigmoid**: Binary Classification (Probability).
    -   **Softmax**: Multi-class Classification (Probability Distribution).
    -   **Linear (None)**: Regression (Real values).
    -   **Tanh**: Regression (if range is -1 to 1) or specific generative models (GAN generator output).

## 3. Implementation Details

Each class implements:
-   `__call__(x)`: Forward pass.
-   `gradient(x)`: Backward pass (derivative w.r.t input).
