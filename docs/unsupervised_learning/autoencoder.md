# Autoencoder

## 1. Pseudocode

```text
Build Network:
    Encoder: Input -> Dense(512) -> LeakyReLU -> BN -> Dense(256) -> LeakyReLU -> BN -> Latent(128)
    Decoder: Latent(128) -> Dense(256) -> LeakyReLU -> BN -> Dense(512) -> LeakyReLU -> BN -> Output(28x28) (Tanh)

Train(n_epochs, batch_size):
    Load MNIST Data
    Normalize Data to [-1, 1]
    
    For epoch = 1 to n_epochs:
        Get Random Batch of Images
        
        # Train Autoencoder to reconstruct the input
        # Input: Image, Target: Image
        Loss = NeuralNetwork.train_on_batch(Images, Images)
        
        If epoch % save_interval == 0:
            Save Reconstructed Images
```

## 2. Algorithm Explanation

An **Autoencoder** is a neural network designed to learn an efficient data coding in an unsupervised manner. It works by compressing the input into a latent-space representation and then reconstructing the output from this representation.

It consists of two parts:
1.  **Encoder**: Maps the input to the latent code ($\phi: \mathcal{X} \rightarrow \mathcal{F}$).
2.  **Decoder**: Maps the latent code back to the original input space ($\psi: \mathcal{F} \rightarrow \mathcal{X}$).

The network is trained to minimize the reconstruction error: $L(x, \psi(\phi(x)))$. This forces the network to learn the most important features of the data in the bottleneck (latent layer).

## 3. Math Formulas

**Reconstruction Loss (Square Loss):**
$$ L(x, \hat{x}) = \frac{1}{2} ||x - \hat{x}||^2 $$

**Latent Representation:**
$$ z = \sigma(\Theta_{enc} x + \theta_{enc\_0}) $$

**Reconstruction:**
$$ \hat{x} = \sigma(\Theta_{dec} z + \theta_{dec\_0}) $$

## 4. Inputs Required

-   **Data**: Images (MNIST in this implementation, flattened to 784 dimension vectors).
-   **n_epochs**: Number of training iterations.
-   **batch_size**: Size of mini-batches.

## 5. Usage Guidelines

### When to use:
-   **Dimensionality Reduction**: Non-linear alternative to PCA.
-   **Denoising**: Can be trained to remove noise from images (Denoising Autoencoder).
-   **GenAI**: Variational Autoencoders (VAEs) are used for generating new data.
-   **Feature Learning**: Pre-training inputs for supervised tasks.

### When not to use:
-   **Simple Compression**: Standard compression algorithms (JPEG/ZIP) are better for pure file size reduction.
-   **Linear Data**: If relationships are linear, PCA is faster and guaranteed to find the global optimum.

### Industry Best Practices:
-   **Bottleneck Size**: The latent dimension is a critical hyperparameter. Too small = high loss (underfitting). Too large = Identity function (no compression, overfitting).
-   **Regularization**: Use Sparse Autoencoders (L1 on activations) or Denoising criteria to learn robust features.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: Neural Network training is parallelized on GPUs.
-   **Memory**: Standard Deep Learning memory requirements (Forward/Backward pass activations).

## 7. Underlying Data Structure

-   **NeuralNetwork**: Uses the custom NN framework derived from `mlfromscratch.deep_learning`.
-   **Layers**: Dense, LeakyReLU, BatchNormalization.
