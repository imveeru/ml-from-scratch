# Partitioning Around Medoids (PAM) / K-Medoids

## 1. Pseudocode

```text
Run(X, k):
    # Initialization
    Initialize Medoids randomly (select k distinct samples from X)
    Assign each sample to nearest Medoid
    Calculate Current_Cost (Sum of distances)
    
    Loop:
        Best_Cost = Current_Cost
        Best_Swap = None
        
        For each Medoid m:
            For each Non-Medoid o:
                # Tentative Swap
                Swap m and o
                Reassign samples to nearest Medoid
                New_Cost = Calculate_Cost()
                
                If New_Cost < Best_Cost:
                    Best_Cost = New_Cost
                    Best_Swap = (m, o)
                
                Revert Swap
                
        If Best_Cost < Current_Cost:
            Perform Best_Swap
            Current_Cost = Best_Cost
        Else:
            Break (Local Optimum reached)
            
    Return Cluster Assignments
```

## 2. Algorithm Explanation

**Partitioning Around Medoids (PAM)**, also known as **K-Medoids**, is a clustering algorithm related to K-Means.

The key difference is that in K-Medoids, the center of a cluster (the **medoid**) is always an actual data point from the dataset, whereas in K-Means, the center (centroid) is the average of the points and might not be a real data point.

This makes K-Medoids more robust to noise and outliers because medoids are less influenced by extreme values than means.

## 3. Math Formulas

**Cost Function:**
$$ J = \sum_{j=1}^{k} \sum_{x \in C_j} d(x, m_j) $$
Where $m_j$ is the medoid of cluster $j$, and $d$ is an arbitrary distance metric (usually Euclidean or Manhattan).

## 4. Inputs Required

-   **X**: Data points.
-   **k**: Number of clusters.

## 5. Usage Guidelines

### When to use:
-   **Robustness**: When data has outliers or noise.
-   **Arbitrary Distance**: When you have a custom distance matrix but cannot easily calculate a "mean" (e.g., strings, graphs). K-Means requires a valid mean; K-Medoids only requires pairwise distances.
-   **Interpretability**: When you need the cluster representative to be an actual example (e.g., "The representative user for Segment A is User #123").

### When not to use:
-   **Large Datasets**: PAM is computationally expensive ($O(k(n-k)^2)$ per iteration). For large data, algorithms like **CLARA** (Clustering Large Applications), which applies PAM to samples, are used.

### Industry Best Practices:
-   **Initialization**: Like K-Means, good initialization matters. **K-Medoids++** exists.
-   **Distance Metric**: Choosing the right distance metric (Manhattan, Cosine, etc.) is often more important than the algorithm itself.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: Calculating swap costs can be parallelized.
-   **Memory**: Storing distance matrix is $O(N^2)$.

## 7. Underlying Data Structure

-   **Numpy Arrays**: Data handling.
