# Restricted Boltzmann Machine (RBM)

## 1. Pseudocode

```text
Fit(X, n_hidden, n_iterations):
    # Initialization
    Initialize Parameters Theta, visible_bias, hidden_bias
    
    For i = 1 to n_iterations:
        For each batch in X:
            # Positive Phase (Data dependent)
            pos_hidden_probs = Sigmoid(batch . Theta + h_bias)
            pos_hidden_states = Sample(pos_hidden_probs)
            pos_associations = batch.T . pos_hidden_probs
            
            # Negative Phase (Reconstruction / Model dependent)
            # Gibbs Sampling step
            neg_visible_probs = Sigmoid(pos_hidden_states . Theta.T + v_bias)
            neg_visible_states = Sample(neg_visible_probs)
            
            neg_hidden_probs = Sigmoid(neg_visible_states . Theta + h_bias)
            neg_associations = neg_visible_states.T . neg_hidden_probs
            
            # Update Parameters (Contrastive Divergence)
            Theta += learning_rate * (pos_associations - neg_associations)
            v_bias += learning_rate * (batch - neg_visible_states)
            h_bias += learning_rate * (pos_hidden_probs - neg_hidden_probs)
```

## 2. Algorithm Explanation

A **Restricted Boltzmann Machine (RBM)** is a generative stochastic artificial neural network that can learn a probability distribution over its set of inputs.

It consists of two layers:
1.  **Visible Layer**: Represents the input data.
2.  **Hidden Layer**: Represents latent variables (features).

It is "Restricted" because there are no connections between nodes within the same layer (bipartite graph).

The training algorithm, **Contrastive Divergence (CD)**, approximates the gradient of the log-likelihood by running a short Markov Chain (Gibbs Sampling) starting from the data.

## 3. Math Formulas

**Energy Function:**
$$ E(v, h) = -b^T v - c^T h - h^T \Theta v $$

**Conditional Probabilities:**
$$ P(h_j=1 | v) = \sigma(c_j + v^T \Theta_{:,j}) $$
$$ P(v_i=1 | h) = \sigma(b_i + h^T \Theta_{i,:}) $$

**Parameter Update (CD-k):**
$$ \Delta \Theta = \eta (\langle v h^T \rangle_{data} - \langle v h^T \rangle_{reconstruction}) $$

## 4. Inputs Required

-   **X**: Training data (Visible units).
-   **n_hidden**: Number of hidden units.
-   **n_iterations**: Number of training steps.

## 5. Usage Guidelines

### When to use:
-   **Collaborative Filtering**: famously used in the Netflix Prize.
-   **Dimensionality Reduction**: Learning binary latent codes.
-   **Pre-training**: Stacking RBMs creates a **Deep Belief Network (DBN)**, which was one of the first successful ways to train deep networks.

### When not to use:
-   **Modern Deep Learning**: Autoencoders and GANs have largely superseded RBMs for most generative tasks and feature learning because they are easier to train with backpropagation.
-   **Continuous Data**: The standard Bernoulli RBM assumes binary data. Gaussian-Bernoulli RBMs exist but are harder to tune.

### Industry Best Practices:
-   **Learning Rate**: Needs to be smaller than standard backprop.
-   **Momentum**: Often crucial for convergence.
-   **Monitoring**: Reconstruction error is a proxy for likelihood but not perfect.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: Matrix operations are parallelized.
-   **Memory**: Standard matrix storage ($Visible \times Hidden$).

## 7. Underlying Data Structure

-   **Numpy Arrays**: Parameters and biases.
