# Particle Swarm Optimization (PSO)

## 1. Pseudocode

```text
Evolve(X, y, n_generations):
    Initialize Population of N Neural Network models (Particles)
    Initialize Velocity for all particles to 0
    Initialize Personal_Best for each particle
    Initialize Global_Best
    
    For epoch = 1 to n_generations:
        For each particle in Population:
            # Update Velocity
            r1, r2 = random(0, 1)
            Velocity = (inertia * Velocity) + 
                       (cognitive_w * r1 * (Personal_Best_Theta - Current_Theta)) +
                       (social_w * r2 * (Global_Best_Theta - Current_Theta))
                       
            # Clamp Velocity to [-max_v, max_v]
            
            # Update Position (Parameters)
            Current_Theta += Velocity
            
            # Evaluate
            Fitness = 1 / (Loss(particle(X), y) + epsilon)
            
            # Update Personal Best
            If Fitness > Personal_Best_Fitness:
                Personal_Best_Theta = Current_Theta
                Personal_Best_Fitness = Fitness
                
            # Update Global Best
            If Fitness > Global_Best_Fitness:
                Global_Best = particle
```

## 2. Algorithm Explanation

**Particle Swarm Optimization (PSO)** is a computational method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality.

It solves a problem by having a population of candidate solutions, here dubbed **particles**, and moving these particles around in the search-space according to simple mathematical formulae over the particle's position and velocity. Each particle's movement is influenced by its local best known position, but is also guided toward the best known positions in the search-space, which are updated as better positions are found by other particles.

In this implementation, PSO is used to train a Neural Network, where the "position" of a particle corresponds to the parameters (theta) of the network.

## 3. Math Formulas

**Velocity Update:**
$$ v_{id}(t+1) = w \cdot v_{id}(t) + c_1 r_1 (p_{id} - x_{id}(t)) + c_2 r_2 (p_{gd} - x_{id}(t)) $$

Where:
-   $v_{id}$: Velocity of particle$i $in dimension$d $.
-   $x_{id}$: Position (weight) of particle$i$.
-   $w$: **Inertia weight**. Controls impact of previous velocity.
-   $c_1$: **Cognitive weight**. Pulls particle towards its own best position ($p_{id}$).
-   $c_2$: **Social weight**. Pulls particle towards the swarm's best known position ($p_{gd}$).
-   $r_1, r_2$: Random numbers in $[0, 1]$.

**Position Update:**
$$ x_{id}(t+1) = x_{id}(t) + v_{id}(t+1) $$

## 4. Inputs Required

-   **X**: Training features.
-   **y**: Training labels.
-   **population_size**: Number of particles in the swarm.
-   **inertia_weight**: Parameter controlling momentum ($0 \le w < 1$).
-   **cognitive_weight**: Parameter controlling individual memory ($c_1$).
-   **social_weight**: Parameter controlling swarm influence ($c_2$).
-   **max_velocity**: Maximum allowed velocity to prevent explosion.

## 5. Usage Guidelines

### When to use:
-   **Global Optimization**: Good at finding global optima in complex, multi-modal landscapes where Gradient Descent might get stuck in local minima.
-   **Non-Differentiable Problems**: Does not require gradients.
-   **Simplicity**: Easier to implement and tune than some other evolutionary algorithms.

### When not to use:
-   **Gradient-Friendly Problems**: If the loss function is differentiable and convex, Gradient Descent is much faster and more precise.
-   **High Dimensionality**: PSO struggles to converge in very high-dimensional spaces (like modern Deep Neural Networks with millions of parameters).

### Industry Best Practices:
-   **Hybridization**: Use PSO to find a good initialization region, then switch to Gradient Descent for fine-tuning.
-   **Parameter Tuning**: Convergence is highly sensitive to $w, c_1, c_2$. Common values are $w=0.729, c_1=c_2=1.49$.
-   **Adaptive Parameters**: Decaying inertia weight ($w$ decreases over time) helps exploration early on and exploitation later.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: Particle evaluations are independent and can be fully parallelized.
-   **Memory**: Requires storing current parameters, velocity, and personal best parameters for each particle. Memory usage = $3 \times Population \times Parameters$.

## 7. Underlying Data Structure

-   **List** and **Dictionaries**: Used to manage the swarm and the different weight matrices (W, w0) for each layer.
