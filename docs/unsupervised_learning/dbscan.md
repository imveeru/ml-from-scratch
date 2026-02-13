# DBSCAN

## 1. Pseudocode

```text
Predict(X, eps, min_samples):
    Initialize all samples as unvisited
    Initialize Clusters = []
    
    For each sample p in X:
        If p is visited: Continue
        
        Mark p as visited
        Neighbors = Get_Neighbors(p, eps)
        
        If len(Neighbors) < min_samples:
            Mark p as Noise (initially)
        Else:
            # p is a Core Point
            Create new Cluster C
            Expand_Cluster(p, Neighbors, C, eps, min_samples)
            Add C to Clusters
            
Expand_Cluster(p, Neighbors, C, eps, min_samples):
    Add p to C
    For each point q in Neighbors:
        If q is unvisited:
            Mark q as visited
            q_Neighbors = Get_Neighbors(q, eps)
            If len(q_Neighbors) >= min_samples:
                Neighbors = Neighbors + q_Neighbors
        
        If q is not not member of any cluster:
            Add q to C
```

## 2. Algorithm Explanation

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) allows clustering of data of arbitrary shapes and is robust to outliers.

It classifies points into three categories:
1.  **Core Points**: Points that have at least `min_samples` within radius `eps`.
2.  **Border Points**: Points that are reachable from a Core Point but do not have enough neighbors themselves to be core.
3.  **Noise Points**: Points that are not reachable from any Core Point.

Unlike K-Means, DBSCAN does not require specifying the number of clusters in advance.

## 3. Inputs Required

-   **X**: Data points.
-   **eps**: The maximum distance between two samples for them to be considered as in the same neighborhood.
-   **min_samples**: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

## 4. Usage Guidelines

### When to use:
-   **Arbitrary Shapes**: When specific clusters are not spherical (e.g., ring shapes, moons).
-   **Unknown K**: When you don't know the number of clusters.
-   **Outliers**: When the data contains noise/outliers.

### When not to use:
-   **Varying Density**: DBSCAN struggles if clusters have significantly different densities (cannot find a single valid `eps`).
-   **High Dimensions**: The "Curse of Dimensionality" makes Euclidean distance less meaningful, making it hard to find a good `eps`.

### Industry Best Practices:
-   **Scaling**: Data distances are critical, so normalization/standardization is usually required.
-   **K-Distance Graph**: A common heuristic to find `eps` is to plot the distance to the k-th nearest neighbor (sorted) and look for the "elbow".

## 5. Concurrency, Parallelism, Memory Management

-   **Concurrency**: Simple implementation is sequential. Efficient implementations use spatial index structures (k-d trees) to speed up neighbor search ($O(N \log N)$ instead of $O(N^2)$).
-   **Memory**: Storing distance matrix is $O(N^2)$, so efficient index is preferred.

## 6. Underlying Data Structure

-   **Lists/Sets**: To track visited points and cluster memberships.
-   **Numpy Arrays**: Distance calculations.
