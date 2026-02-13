# Genetic Algorithm

## 1. Pseudocode

```text
Run(target_string, iterations):
    Population = Initialize_Random_Strings()
    
    For epoch = 1 to iterations:
        # 1. Evaluation
        For each individual in Population:
            Calculate Fitness = 1 / (Distance(individual, target) + epsilon)
            
        If Best_Individual == Target:
            Return Best_Individual
            
        # 2. Selection
        Calculate Selection Probabilities based on Fitness (Roulette Wheel)
        
        New_Population = []
        For i = 1 to Population_Size / 2:
            Parent1 = Select(Population, Probabilities)
            Parent2 = Select(Population, Probabilities)
            
            # 3. Crossover
            Child1, Child2 = Crossover(Parent1, Parent2)
            
            # 4. Mutation
            Child1 = Mutate(Child1)
            Child2 = Mutate(Child2)
            
            Add Children to New_Population
            
        Population = New_Population
```

## 2. Algorithm Explanation

**Genetic Algorithms (GAs)** are search heuristics that mimic the process of natural selection. They are used to generate high-quality solutions to optimization and search problems.

This implementation demonstrates a GA solving the "Infinite Monkey Theorem" problem: trying to evolve a random string into a specific target string (e.g., "Hello World").

Key components:
1.  **Selection**: Preferentially selecting "fitter" individuals to be parents.
2.  **Crossover**: Combining parts of two parents to create offspring (mixing genes).
3.  **Mutation**: Randomly altering genes to maintain diversity and prevent premature convergence.

## 3. Inputs Required

-   **target_string**: The string to be evolved.
-   **population_size**: Number of individuals.
-   **mutation_rate**: Probability of a character mutation.

## 4. Usage Guidelines

### When to use:
-   **Combinatorial Optimization**: Problems where the search space is vast and discrete (e.g., Traveling Salesman, Scheduling).
-   **Black Box Optimization**: When the objective function is not differentiable or no gradient information is available.

### When not to use:
-   **Simple Problems**: For problems with known efficient algorithms (e.g., sorting, shortest path), GAs are much slower.
-   **Gradient Information**: If gradients are available, gradient-based methods are almost always faster.

### Industry Best Practices:
-   **Elitism**: Keeping the best $N$ individuals from the previous generation unchanged to ensure the best solution is never lost (not implemented here but common).
-   **Diversity Maintenance**: Critical to prevent the population from converging to a local optimum too early.

## 5. Concurrency, Parallelism, Memory Management

-   **Concurrency**: Fitness evaluation is embarrassingly parallel.
-   **Memory**: Proportional to `population_size * individual_size`.

## 6. Underlying Data Structure

-   **Strings/Lists**: Representing the DNA (genome) of individuals.
