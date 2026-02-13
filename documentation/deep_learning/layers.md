# Deep Learning Layers

This file implements the building blocks of Neural Networks. Each layer inherits from the base `Layer` class and implements `forward_pass` (for prediction/training) and `backward_pass` (for backpropagation).

## 1. Layers Overview

### Dense (Fully Connected)
Standard layer where every input node is connected to every output node.
-   **Forward**: $Y = X\Theta + \theta_0$
-   **Backward**: Computes gradients for $\Theta$ and $\theta_0$ and propagates error to previous layer.

### RNN (Recurrent Neural Network)
Vanilla RNN layer for temporal data.
-   **Forward**: Compsoes hidden states over time: $h_t = \tanh(\Theta_x x_t + \Theta_h h_{t-1})$. Inputs are $(batch, timesteps, features)$.
-   **Backward**: Uses Backpropagation Through Time (BPTT).

### Conv2D (Convolutional)
2D Convolution layer for images.
-   **Implementation**: Uses `im2col` (image-to-column) transformation to convert convolution into a matrix multiplication for efficiency.
-   **Forward**: Convolves filters over input.
-   **Backward**: Propagates gradients to filters and input.

### BatchNormalization
Normalizes the input to have mean 0 and variance 1, then scales and shifts.
-   **Math**: $\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$
-   **Training vs Inference**: Uses current batch statistics during training, but running moving averages during inference.

### Pooling (MaxPooling2D, AveragePooling2D)
Downsamples the input.
-   **Max**: Takes the maximum value in the window.
-   **Average**: Takes the average value.
-   **Implementation**: Also uses `im2col`.

### Transformation Layers
-   **Flatten**: Flattens $N$-D tensor to 2D (batch\_size, features).
-   **Reshape**: Reshapes tensor to specified shape.
-   **UpSampling2D**: Repeats rows/cols to upsample image (nearest neighbor).
-   **ZeroPadding2D / ConstantPadding2D**: Adds padding to borders.

### Regularization
-   **Dropout**: Randomly sets input units to 0 with probability $p$ during training to prevent overfitting. scales inputs by $(1-p)$ (or uses inverted dropout mask) to maintain expected value.

### Activation
Wrapper layer to apply activation functions (from `activation_functions.py`).

## 2. Usage Guidelines

### Industry Best Practices

-   **Conv2D**:
    -   Use small filters ($3 \times 3$) and stack them deep.
    -   Use `Same` padding to keep spatial dimensions until pooling.
    -   Stride > 1 can replace pooling for downsampling.

-   **Initialization**:
    -   Parameters are initialized using **Xavier/Glorot** (uniform between $\pm \frac{1}{\sqrt{n}}$) or similar heuristics to keep variance stable.

-   **Batch Norm**:
    -   Place **after** Convolution/Dense and **before** Activation (though debated, this is the original paper's recommendation).
    -   Crucial for deep networks.

-   **Dropout**:
    -   Use in fully connected layers (e.g., $p=0.5$).
    -   Less common in Conv layers (Spatial Dropout is preferred there).

## 3. Concurrency, Parallelism, Memory Management

-   **Memory**: Storing `im2col` matrices can be memory intensive ($O(k^2)$ expansion). Modern frameworks use specialized CUDA kernels to avoid materializing this matrix.
-   **Cache**: Forward pass caches `layer_input` and intermediate values (like `col` matrices or `masks`) for the backward pass. This doubles memory usage during training.

## 4. Underlying Data Structure

-   **Numpy Arrays**: All parameters and data.
-   **Computed Graphs**: Implicitly defined by the sequence of layers in a model.
