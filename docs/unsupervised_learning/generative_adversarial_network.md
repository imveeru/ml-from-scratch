# Generative Adversarial Network (GAN)

## 1. Pseudocode

```text
Build Generator():
    Input(Noise) -> Dense(256) -> LeakyReLU -> BN
    Dense(512) -> LeakyReLU -> BN
    Dense(1024) -> LeakyReLU -> BN
    Dense(Output_Dim) -> Tanh ("Image")

Build Discriminator():
    Input(Image) -> Dense(512) -> LeakyReLU -> Dropout
    Dense(256) -> LeakyReLU -> Dropout
    Dense(2) -> Softmax (Real vs Fake)

Train(epochs):
    For epoch = 1 to epochs:
        # 1. Train Discriminator
        Get Batch of Real Images (Label = 1)
        Generate Batch of Fake Images from Noise (Label = 0)
        
        Train Discriminator on Real
        Train Discriminator on Fake
        
        # 2. Train Generator
        Freeze Discriminator Parameters
        Generate Noise
        Target Labels = 1
        
        Train Combined Model on Noise with Target Labels
```

## 2. Algorithm Explanation

**GAN** (Generative Adversarial Network) consists of two neural networks, a **Generator** and a **Discriminator**, that contest with each other in a zero-sum game framework.

-   The **Generator ($G$)** takes random noise as input and tries to generate data samples (e.g., images) that resemble the real training data.
-   The **Discriminator ($D$)** takes a data sample as input and tries to predict whether it is real (from the dataset) or fake (from the generator).

As training progresses, $G$becomes better at fooling $D$, and $D$ becomes better at detecting fakes, leading to the generation of highly realistic data.

## 3. Math Formulas

**Objective Function:**
$$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log (1 - D(G(z)))] $$

## 4. Inputs Required

-   **Noise ($z$)**: Random latent vector.
-   **Training Data**: Images (MNIST).

## 5. Usage Guidelines

### When to use:
-   **Data Synthesis**: Generating synthetic data for privacy or augmentation.
-   **Simple Baselines**: Standard GANs are simpler to implement than DCGANs but less stable for images. Better for low-dimensional tabular data.

### When not to use:
-   **Mode Collapse**: Standard GANs suffer heavily from mode collapse (generating only one type of output).
-   **Stability**: Dense GANs are very hard to train on complex images compared to CNN-based GANs (DCGAN).

### Industry Best Practices:
-   **Normalization**: Scale inputs to $[-1, 1]$ to match `Tanh` output of generator.
-   **Dropout**: Used in the discriminator to prevent it from overpowering the generator too quickly.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: Parallelized training on GPU.
-   **Memory**: Standard NN memory usage.

## 7. Underlying Data Structure

-   **NeuralNetwork**: Custom implementation.
-   **Layers**: Dense, LeakyReLU, BatchNormalization, Dropout.
