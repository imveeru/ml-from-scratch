# Multilayer Perceptron (MLP)

## 1. Pseudocode

```text
Fit(X, y, n_iterations):
    Initialize parameters (Theta, theta0, Theta_out, theta0_out) uniformly
    
    For i = 1 to n_iterations:
        # Forward Pass
        Hidden_Input = X . Theta + theta0
        Hidden_Output = Sigmoid(Hidden_Input)
        
        Output_Input = Hidden_Output . Theta_out + theta0_out
        Prediction = Softmax(Output_Input)
        
        # Backward Pass (Backpropagation)
        Grad_Output = Loss_Gradient(y, Prediction) * Softmax_Gradient(Output_Input)
        Grad_Theta_out = Hidden_Output.T . Grad_Output
        Grad_theta0_out = Sum(Grad_Output)
        
        Grad_Hidden = (Grad_Output . Theta_out.T) * Sigmoid_Gradient(Hidden_Input)
        Grad_Theta = X.T . Grad_Hidden
        Grad_theta0 = Sum(Grad_Hidden)
        
        # Update Parameters
        Theta_out -= learning_rate * Grad_Theta_out
        theta0_out -= learning_rate * Grad_theta0_out
        Theta -= learning_rate * Grad_Theta
        theta0 -= learning_rate * Grad_theta0

Predict(X):
    # Forward pass only
    H = X.Theta + theta0
    H_out = Sigmoid(H)
    Out = H_out.Theta_out + theta0_out
    Return Softmax(Out)
```

## 2. Algorithm Explanation

**Multilayer Perceptron (MLP)** is a class of feedforward artificial neural networks. This implementation is a "vanilla" neural network with:
-   **Input Layer**: Takes the feature vector.
-   **One Hidden Layer**: Uses **Sigmoid** activation.
-   **Output Layer**: Uses **Softmax** activation (for classification probability).

The network learns using **Backpropagation**, which calculates the gradient of the loss function with respect to the parameters by applying the Chain Rule, moving from the output layer back to the input layer.

## 3. Math Formulas

**Forward Pass:**
$$ H = \sigma(X\Theta + \theta_0) $$
$$ \hat{y} = \text{Softmax}(H\Theta_{out} + \theta_{0,out}) $$

**Backpropagation (Gradients):**
Output Layer Error signal:
$$ \delta_{out} = \nabla_L \odot \sigma'(\text{out}_{in}) $$
Hidden Layer Error signal:
$$ \delta_{hidden} = (\delta_{out} \Theta_{out}^T) \odot \sigma'(\text{hidden}_{in}) $$

**Parameter Update:**
$$ \Theta_{out} \leftarrow \Theta_{out} - \alpha (H^T \delta_{out}) $$
$$ \Theta \leftarrow \Theta - \alpha (X^T \delta_{hidden}) $$

## 4. Inputs Required

-   **X**: Training features (`n_samples`, `n_features`).
-   **y**: Training labels (`n_samples`, `n_classes`). One-hot encoded.
-   **n_hidden**: Number of neurons in the hidden layer.
-   **n_iterations**: Number of training epochs.
-   **learning_rate**: Step size for parameter updates.

## 5. Usage Guidelines

### When to use:
-   **Complex Non-Linear Relationships**: Can approximate any continuous function (Universal Approximation Theorem).
-   **Feature Interaction**: Good at learning interactions between features without explicit feature engineering.

### When not to use:
-   **Small Datasets**: Prone to overfitting on small data.
-   **Structured Tabular Data**: Gradient Boosted Trees (XGBoost) often outperform simple MLPs on tabular data.
-   **Interpretability**: It is a "Black Box" model; hard to understand why a specific prediction was made.

### Industry Best Practices:
-   **Deep Learning Frameworks**: In production, use optimized frameworks like TensorFlow or PyTorch.
-   **Pre-processing**: Data **must** be normalized/standardized.
-   **Regularization**: Use Dropout or L2 regularization to prevent overfitting.
-   **Activation Functions**: ReLU is generally preferred over Sigmoid for hidden layers in modern deep learning to avoid vanishing gradients.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: Matrix multiplications are parallelizable.
-   **Memory**: Stores parameter matrices and intermediate activations for the backward pass. Memory usage scales with $O(Neurons^2)$.

## 7. Underlying Data Structure

-   **Numpy Arrays**: Used for parameters and activations.
