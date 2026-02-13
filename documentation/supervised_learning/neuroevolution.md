# Neuroevolution

## 1. Pseudocode

```text
Evolve(X, y, n_generations):
    Initialize Population of N Neural Network models with random parameters
    
    For generation = 1 to n_generations:
        # Evaluation
        For each model in Population:
            Calculate Fitness = 1 / (Loss(model(X), y) + epsilon)
            
        Sort Population by Fitness (Descending)
        
        # Selection
        Select top 40% as "Winners" -> Next_Population
        Select top 60% as "Parent Pool"
        
        # Reproduction (Crossover & Mutation)
        While len(Next_Population) < Population_Size:
            Select Parent1, Parent2 from Parent Pool (prob proportional to fitness)
            
            Child1, Child2 = Crossover(Parent1, Parent2)
            
            Mutate(Child1, mutation_rate)
            Mutate(Child2, mutation_rate)
            
            Add Children to Next_Population
            
        Population = Next_Population
        
    Return Fittest Individual from Population
```

## 2. Algorithm Explanation

**Neuroevolution** refers to the application of **Evolutionary Algorithms** (like Genetic Algorithms) to optimize Neural Networks. Instead of using gradient-based methods like Backpropagation, Neuroevolution evolves the parameters (and sometimes topology) of the network.

This implementation uses a standard Genetic Algorithm approach:
1.  **Population**: Maintained a set of Neural Networks.
2.  **Fitness**: Evaluated based on how well the network performs (Accuracy/Loss) on the dataset.
3.  **Selection**: The "fittest" networks survive and reproduce.
4.  **Crossover**: Offspring are created by mixing the parameters of two parents.
5.  **Mutation**: Random Gaussian noise is added to the parameters of the offspring to maintain diversity and explore the search space.

## 3. Math Formulas

**Fitness Function:**
$$ Fitness = \frac{1}{Loss + \epsilon} $$
Where $\epsilon$ is a small constant to prevent division by zero.

**Mutation (Gaussian):**
$$ W_{new} = W_{old} + \mathcal{N}(0, \sigma^2) \cdot \text{mask} $$
Where mask is sampled from a Binomial distribution $B(1, mutation\_rate)$.

## 4. Inputs Required

-   **X**: Training features.
-   **y**: Training labels.
-   **population_size**: Number of models in the population.
-   **mutation_rate**: Probability of a weight value being mutated.
-   **model_builder**: Function that returns a new instance of a Neural Network.

## 5. Usage Guidelines

### When to use:
-   **Non-Differentiable Loss**: When the objective function is not differentiable (step functions, discrete rewards), making Backpropagation impossible.
-   **Reinforcement Learning**: Often used in RL tasks where gradients are sparse or delayed.
-   **Avoiding Local Minima**: Can escape local minima better than Gradient Descent in some complex landscapes due to its stochastic nature.

### When not to use:
-   **Supervised Learning Standard Tasks**: For standard classification/regression with differentiable loss, Gradient Descent (Backprop) is significantly faster and more efficient.
-   **Large Networks**: Optimizing millions of parameters (Deep Learning) via evolution is computationally prohibitive compared to SGD.

### Industry Best Practices:
-   **Hybrid Approaches**: Use Neuroevolution to find good architectures or initial parameters, then fine-tune with Gradient Descent.
-   **Parallelization**: Evaluation of population fitness is perfectly parallelizable across CPU/GPU cores.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: Highest potential for parallelism. Each individual in the population can be evaluated independently on separate threads/machines.
-   **Memory**: High memory consumption. Requires storing `population_size` distinct copies of the model parameters.

## 7. Underlying Data Structure

-   **List**: Stores the population of Neural Network objects.
-   **Numpy Arrays**: Stores the parameters of each network.
