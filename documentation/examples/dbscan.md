# DBSCAN Example

This example demonstrates Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

## Description
-   **Dataset**: Sklearn `make_moons` (Non-linear, crescent shapes).
-   **Task**: Clustering.
-   **Model**: DBSCAN.
-   **Visualization**: Plots the clusters found by DBSCAN side-by-side with the actual labels using PCA (though PCA is redundant for 2D data, the Plot utility uses it).

## Usage
Run the script from the root directory:
```bash
python -m mlfromscratch.examples.dbscan
```

## Output
-   Displays two plots: Predicted Clusters vs Actual Clusters.
