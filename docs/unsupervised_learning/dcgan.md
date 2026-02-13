# DCGAN

## 1. Pseudocode

```text
Build Generator():
    Input(Noise) -> Dense -> Reshape
    UpSample -> Conv2D -> BN -> LeakyReLU
    UpSample -> Conv2D -> BN -> LeakyReLU
    Conv2D -> Tanh
    
Build Discriminator():
    Input(Image) -> Conv2D -> LeakyReLU -> Dropout
    Conv2D -> BN -> LeakyReLU -> Dropout
    Flatten -> Dense(2) -> Softmax

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
        Target Labels = 1 (Trick Discriminator to think they are real)
        
        Train Combined Model (Generator + Discriminator) on Noise with Target Labels
```

## 2. Algorithm Explanation

**DCGAN** (Deep Convolutional Generative Adversarial Network) is a class of CNNs used to generate realistic images.

It consists of two networks competing against each other:
1.  **Generator ($G$)**: Tries to generate images that look real to fool the discriminator.
2.  **Discriminator ($D$)**: Tries to distinguish between real images from the dataset and fake images from the generator.

The core innovation of DCGAN over standard GANs is the use of convolutional layers (specifically upsampling/transposed convolutions in the generator and strided convolutions in the discriminator) which makes it stable for image generation.

## 3. Math Formulas

**Objective Function (Minimax Game):**
$$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log (1 - D(G(z)))] $$

-   $D(x)$: Probability that$x$ is real.
-   $G(z)$: Image generated from noise$z$.

## 4. Inputs Required

-   **Noise ($z$)**: A vector of random numbers (usually from Normal distribution) acts as the seed for generation.
-   **Training Data**: Images (MNIST).

## 5. Usage Guidelines

### When to use:
-   **Image Generation**: Generating art, faces, or augmenting datasets.
-   **Style Transfer**: Can be adapted for image-to-image translation (Pix2Pix, CycleGAN).

### When not to use:
-   **Exact Representation**: GANs "hallucinate" details and do not preserve exact data properties like compression.
-   **Discrete Data**: Hard to train on text or discrete data due to differentiability issues.

### Industry Best Practices:
-   **Batch Normalization**: Essential for stability in both $G$and $D$.
-   **LeakyReLU**: Use LeakyReLU in Discriminator to prevent "dying ReLU".
-   **Strided Convolutions**: Use strides instead of Pooling layers.
-   **Label Smoothing**: Use soft labels (e.g., 0.9 instead of 1.0) to prevent the discriminator from becoming too confident too early.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: GPU acceleration is mandatory for reasonable training times.
-   **Memory**: High VRAM usage due to training two deep networks simultaneously.

## 7. Underlying Data Structure

-   **NeuralNetwork**: Custom implementation.
-   **Layers**: Conv2D, UpSampling2D, BatchNormalization.
