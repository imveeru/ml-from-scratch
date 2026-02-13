# Perceptron

## 1. Pseudocode

```text
Fit(X, y, n_iterations):
    Initialize parameters Theta, theta0 uniformly
    
    For i = 1 to n_iterations:
        # Forward Pass
        Linear_Output = X . Theta + theta0
        Prediction = Activation(Linear_Output)
        
        # Backward Pass (Gradient Calculation)
        # Gradient w.r.t Linear Output
        Error_Gradient = Loss_Gradient(y, Prediction) * Activation_Gradient(Linear_Output)
        
        # Gradient w.r.t Parameters
        Grad_Theta = X.T . Error_Gradient
        Grad_theta0 = Sum(Error_Gradient)
        
        # Update Parameters
        Theta -= learning_rate * Grad_Theta
        theta0 -= learning_rate * Grad_theta0

Predict(X):
    Return Activation(X . Theta + theta0)
```

## 2. Algorithm Explanation

**Perceptron** is the simplest type of artificial neural network. It consists of a single layer of output neurons connected to the input neurons.

Historically, the "Perceptron" referred to a specific algorithm with a Heaviside step function activation. However, this implementation is more general and acts as a **Single-Layer Neural Network** that supports various activation functions (Sigmoid, ReLU, Tanh, etc.) and loss functions (Square Loss, Cross Entropy).

It learns a linear decision boundary (hyperplane) that separates the data classes.

## 3. Math Formulas

**Forward Pass:**
$$ z = X\Theta + \theta_0 $$
$$ \hat{y} = \sigma(z) $$

**Gradient Descent Update:**
$$ \frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} $$
$$ \Theta \leftarrow \Theta - \alpha (X^T \frac{\partial L}{\partial z}) $$

## 4. Inputs Required

-   **X**: Training features.
-   **y**: Training labels.
-   **n_iterations**: Number of training steps.
-   **activation_function**: The activation function (e.g., Sigmoid, ReLU).
-   **loss**: The loss function (e.g., SquareLoss, CrossEntropy).
-   **learning_rate**: Step size.

## 5. Usage Guidelines

### When to use:
-   **Linearly Separable Data**: Works well if data can be separated by a straight line/plane.
-   **Simple Baselines**: Good checking if a simple linear model explains the data.

### When not to use:
-   **Non-Linear Data**: Cannot solve XOR problem or other non-linear data distributions (unless using a non-linear activation like ReLU/Sigmoid makes it a Logistic Regression equivalent, but it's still a single layer).
-   **Complex Patterns**: For complex patterns, use Multi-Layer Perceptron (MLP).

### Industry Best Practices:
-   **Feature Scaling**: Standardize input features.
-   **Bias Term**: Always include a bias term (handled automatically here as `theta0`).

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: Matrix operations are parallelized.
-   **Memory**: Low memory footprint ($O(Features \times Outputs)$).

## 7. Underlying Data Structure

-   **Numpy Arrays**: Used for parameters and vectorized calculations.
