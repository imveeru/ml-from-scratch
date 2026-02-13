# Optimizers

This file contains implementations of various gradient-based optimization algorithms used to train neural networks.

## 1. Algorithms Overview

### Stochastic Gradient Descent (SGD)
The most basic optimizer. Updates parameters by moving in the opposite direction of the gradient.
$$ \theta = \theta - \eta \cdot \nabla J(\theta) $$
-   **Momentum**: Accelerates SGD in the relevant direction and dampens oscillations.
    $$ v_t = \gamma v_{t-1} + (1 - \gamma) \nabla J(\theta) $$
    $$ \theta = \theta - \eta v_t $$

### Nesterov Accelerated Gradient (NAG)
A variant of momentum that calculates the gradient at the "lookahead" position ($\theta - \gamma v_{t-1}$) rather than the current position. This makes it more responsive to changes in terrain.

### Adagrad
Adaptive learning rates. It adapts the learning rate to the parameters, performing smaller updates (i.e. low learning rates) for parameters associated with frequent features, and larger updates (i.e. high learning rates) for parameters associated with infrequent features.
$$ G_t = G_{t-1} + (\nabla J)^2 $$
$$ \theta = \theta - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla J $$

### Adadelta
Extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. It restricts the window of accumulated past gradients to some fixed size.

### RMSprop
Unpublished method (proposed by Geoff Hinton in Coursera class). Similar to Adadelta, it resolves Adagrad's radically diminishing learning rates by using a moving average of the squared gradient.
$$ E[g^2]_t = 0.9 E[g^2]_{t-1} + 0.1 g^2 $$
$$ \theta = \theta - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g $$

### Adam (Adaptive Moment Estimation)
Combines the advantages of two other extensions of stochastic gradient descent: **AdaGrad** (Adaptive Gradient Algorithm) and **RMSProp** (Root Mean Square Propagation).
It computes adaptive learning rates for each parameter.
$$ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g $$
$$ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g^2 $$
$$ \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $$
$$ \theta = \theta - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t $$

## 2. Usage Guidelines

### Industry Best Practices

-   **SGD + Momentum**: Often generalizes better than adaptive methods for computer vision (ConvNets), though requires more tuning of learning rate schedules.
-   **Adam**: The default "go-to" optimizer for most problems. Good starting point ($lr=3e-4$ or $1e-3$). Fast convergence.
-   **Learning Rate Decay**: Crucial for all optimizers. Reduce LR when loss plateaus.

## 3. Inputs Required

-   `learning_rate`: Step size (alpha).
-   `rho`, `beta1`, `beta2`: Decay rates for moving averages.
-   `epsilon`: Small constant for numerical stability (div by zero).

## 4. Underlying Data Structure

-   **Numpy Arrays**: Storing velocity/moment vectors same shape as parameters.
