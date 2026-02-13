# Genetic Algorithm Example

This example demonstrates a Genetic Algorithm evolving a string to match a target.

## Description
-   **Target**: "Genetic Algorithm".
-   **Population Size**: 100.
-   **Mutation Rate**: 0.05.
-   **Process**:
    -   Calculate fitness (character match).
    -   Select parents (proportional to fitness).
    -   Crossover (single point).
    -   Mutate.

## Usage
Run the script from the root directory:
```bash
python -m mlfromscratch.examples.genetic_algorithm
```

## Output
-   Prints the fittest candidate in each generation until the target is reached.
