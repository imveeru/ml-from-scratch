# Neural Network

This file implements the `NeuralNetwork` class, which serves as a container for building and training deep learning models.

## 1. Pseudocode

```text
NeuralNetwork(optimizer, loss):
    Initialize layers list
    Initialize errors list

Add(layer):
    If not first layer:
        Set Input Shape = Previous Layer Output Shape
    Initialize Layer Parameters
    Add to layers list

Fit(X, y, epochs, batch_size):
    For epoch = 1 to epochs:
        For batch in (X, y):
            # 1. Forward Pass
            Prediction = Forward_Pass(batch)
            
            # 2. Loss & Gradient
            Loss = Loss_Function(y, Prediction)
            Gradient = Loss_Gradient(y, Prediction)
            
            # 3. Backward Pass
            Backward_Pass(Gradient)
            
        Evaluate Validation Set (if provided)
        
Forward_Pass(X):
    For layer in layers:
        X = layer.forward(X)
    Return X

Backward_Pass(Gradient):
    For layer in reversed(layers):
        Gradient = layer.backward(Gradient)
```

## 2. Usage Guidelines

### Building a Model
The framework follows a Sequential API style similar to Keras.
1.  Instantiate `NeuralNetwork`.
2.  Add layers using `.add()`.
3.  Call `.fit()` to train.

### Industry Best Practices
-   **Mini-batches**: Always use mini-batch training (batch_size < dataset size) for stochastic gradient descent to escape local minima and fit in memory.
-   **Validation**: Always provide a validation set to monitor for overfitting during training.
-   **Input Normalization**: Ensure input $X$is scaled (e.g., to $[0, 1]$ or standardized) for faster convergence.

## 3. Concurrency, Parallelism, Memory Management

-   **Memory**: The model stores intermediate activations for every layer during the forward pass to use in the backward pass. This means memory usage is proportional to `Batch Size * Sum(Layer Sizes)`.
-   **Concurrency**: Python's GIL limits threading, but Numpy operations (dot products) use BLAS/LAPACK which can use multiple cores.

## 4. Underlying Data Structure

-   **List**: Stores the sequence of layers.
-   **Dictionary**: Stores training history (errors).
