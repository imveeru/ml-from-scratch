# K-Means Clustering

## 1. Pseudocode

```text
Run(X, k, max_iterations):
    # Initialization
    Initialize Centroids randomly from X
    
    For i = 1 to max_iterations:
        # 1. Assignment Step
        For each sample x:
            Find nearest Centroid c_j (min Euclidean Distance)
            Assign x to Cluster j
            
        # 2. Update Step
        For each Cluster j:
            New_Centroid_j = Mean of all samples assigned to Cluster j
            
        # Check Convergence
        If Centroids did not change (or change < tolerance):
            Break
            
    Return Cluster Assignments
```

## 2. Algorithm Explanation

**K-Means** is one of the simplest and most popular unsupervised machine learning algorithms. Ideally, the algorithm partitions the data into $K$ clusters such that data points in the same cluster are similar and data points in different clusters are dissimilar.

The objective is to minimize the **Within-Cluster Sum of Squares (WCSS)**, also known as Inertia.

## 3. Math Formulas

**Euclidean Distance:**
$$ d(x, c) = \sqrt{\sum_{i=1}^{n} (x_i - c_i)^2} $$

**Centroid Update:**
$$ c_j = \frac{1}{|C_j|} \sum_{x \in C_j} x $$

## 4. Inputs Required

-   **X**: Data points.
-   **k**: Number of clusters.
-   **max_iterations**: Limit on iterations.

## 5. Usage Guidelines

### When to use:
-   **General Purpose**: Good default choice for clustering.
-   **Vector Quantization**: Reducing the number of colors in an image.
-   **Preprocessing**: As a feature engineering step (distance to cluster centers).

### When not to use:
-   **Non-Globular Shapes**: Fails on complex shapes (like concentric circles) where density-based methods (DBSCAN) work better.
-   **Varying Sizes/Densities**: Assumes clusters are roughly spherical and of similar size.
-   **Outliers**: Sensitive to outliers (mean is pulled by them).

### Industry Best Practices:
-   **Initialization**: Use **K-Means++** initialization (not implemented here but standard in libraries) to choose initial centroids that are far apart, speeding up convergence.
-   **Elbow Method**: Run for different $K$ and plot WCSS to find the "elbow" point where adding more clusters gives diminishing returns.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: Distance calculations can be parallelized.
-   **Memory**: $O(N \times K)$. Very memory efficient compared to hierarchical clustering $O(N^2)$.

## 7. Underlying Data Structure

-   **Numpy Arrays**: Data and Centroids.
