# Apriori Example

This example demonstrates the Apriori algorithm for Association Rule Mining.

## Description
-   **Dataset**: Hardcoded dummy transaction dictionary.
-   **Task**: Find frequent itemsets and generate association rules.
-   **Parameters**:
    -   `min_sup` (Minimum Support): 0.25
    -   `min_conf` (Minimum Confidence): 0.8

## Usage
Run the script from the root directory:
```bash
python -m mlfromscratch.examples.apriori
```

## Output
-   Prints the input transactions.
-   Prints found Frequent Itemsets.
-   Prints generated Rules with their Support and Confidence.
