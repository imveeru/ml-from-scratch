# Adaboost

## 1. Pseudocode

```text
Initialize sample weights w = 1/N for all N samples
For t = 1 to T (number of classifiers):
    Train a weak classifier h_t using data samples weighted by w
    Calculate the error rate ε_t of h_t:
        ε_t = sum(w_i for misclassified samples) / sum(w)
    
    If ε_t > 0.5:
        Flip polarity of h_t
        ε_t = 1 - ε_t
        
    Calculate classifier weight α_t:
        α_t = 0.5 * ln((1 - ε_t) / ε_t)
        
    Update sample weights:
        For each sample i:
            w_i = w_i * exp(-α_t * y_i * h_t(x_i))
            
    Normalize weights w so they sum to 1
    Save h_t and α_t

Final Prediction H(x):
    H(x) = sign(sum(α_t * h_t(x)) for all t)
```

## 2. Algorithm Explanation

**Adaboost (Adaptive Boosting)** is an ensemble learning method that combines multiple "weak" classifiers to form a stronger classifier. A weak classifier is one that performs slightly better than random guessing (e.g., a Decision Stump).

The core idea is to train classifiers sequentially. After each classifier is trained, the algorithm increases the weights of the misclassified samples. This forces the next classifier in the sequence to focus more on the difficult samples that previous classifiers got wrong.

In this implementation:
- **Weak Classifier**: A `DecisionStump`, which is a one-level decision tree that classifies based on a single feature threshold.
- **Training**: 
    1.  The algorithm initiates by assigning equal weights to all training samples.
    2.  In each iteration, it finds the best `DecisionStump` that minimizes the weighted classification error.
    3.  It calculates the stump's performance (`alpha`) based on its error. Lower error leads to a higher alpha.
    4.  It updates the sample weights: weights of misclassified samples are increased, and correct ones are decreased.
- **Prediction**: The final prediction is a weighted sum of the predictions from all weak classifiers. The sign of this sum determines the class label (-1 or 1).

## 3. Math Formulas

**Classifier Weight ($\alpha_t$):**
$$ \alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right) $$
Where $\epsilon_t$ is the weighted error rate of classifier $t$.

**Weight Update:**
$$ w_i^{(t+1)} = w_i^{(t)} \cdot \exp\left(-\alpha_t \cdot y_i \cdot h_t(x_i)\right) $$
Where $y_i$ is the true label (-1 or 1) and $h_t(x_i)$ is the predicted label.

**Final Prediction:**
$$ H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right) $$

## 4. Inputs Required

- **X**: Training dataset of shape `(n_samples, n_features)`.
- **y**: Target labels of shape `(n_samples,)`. Labels must be encoded as `{-1, 1}`.
- **n_clf**: (Hyperparameter) The number of weak classifiers to use in the ensemble.

## 5. Usage Guidelines

### When to use:
- **Binary Classification**: classic Adaboost is designed for binary classification.
- **Improving Weak Learners**: When you have a simple model (like a decision stump) and want to boost its performance.
- **Low Noise Data**: Works well on clean datasets where the relationship between features and target is complex but not noisy.

### When not to use:
- **Noisy Data**: Adaboost is sensitive to outliers and noise because it tries hard to fit them (by increasing their weights), which can lead to overfitting.
- **Complex Base Classifiers**: If your base classifier is already complex (like a deep neural network), boosting might lead to overfitting or negligible gains.

### Industry Best Practices:
- **Early Stopping**: Monitor validation error and stop adding classifiers when performance plateaus to prevent overfitting.
- **Shrinkage (Learning Rate)**: Often combined with a learning rate parameter (0 < $\nu$ < 1) scaling the contribution of each classifier: $H(x) = \text{sign}(\sum \nu \alpha_t h_t(x))$.
- **Outlier Handling**: Pre-process data to remove outliers before training.

## 6. Concurrency, Parallelism, Memory Management

- **Concurrency/Parallelism**: The training process is inherently **sequential**. Each classifier depends on the weights updated by the previous one, making standard parallelization (at the classifier level) impossible. However, the search for the best split within the `DecisionStump` (iterating over features and thresholds) could be parallelized.
- **Memory Management**: The algorithm stores `n_clf` weak classifier objects. For decision stumps, this is very lightweight. The main memory usage comes from storing the dataset `X` and the weights `w`.

## 7. Underlying Data Structure

- **Numpy Arrays**: Used for storing data features (`X`), labels (`y`), sample weights (`w`), and predictions. This allows for efficient vectorized operations.
- **List**: A Python list `self.clfs` is used to store the sequence of trained `DecisionStump` objects.
