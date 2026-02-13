# Apriori

## 1. Pseudocode

```text
Find_Frequent_Itemsets(transactions, min_sup):
    L1 = {Items with support >= min_sup}
    k = 2
    
    While L_{k-1} is not empty:
        # Generate Candidates C_k from L_{k-1}
        # Join step: combine two itemsets if they share first k-2 items
        C_k = Join(L_{k-1})
        
        # Prune step: remove candidate if any subset is not in L_{k-1}
        C_k = Prune(C_k)
        
        # Count support for candidates
        L_k = {c in C_k if Support(c) >= min_sup}
        
        k += 1
        
    Return Union(L1, L2, ...)

Generate_Rules(F, min_conf):
    Rules = []
    For each frequent itemset f in F:
        For each subset s of f:
            Antecedent = s
            Consequent = f - s
            Confidence = Support(f) / Support(s)
            
            If Confidence >= min_conf:
                Add (Antecedent -> Consequent) to Rules
```

## 2. Algorithm Explanation

**Apriori** is a classic algorithm used for **Association Rule Mining** on transactional databases. It identifies frequent itemsets (items that appear together often) and derives association rules from them.

It uses a "bottom-up" approach, where frequent subsets are extended one item at a time (candidate generation), and groups of candidates are tested against the data.

**Key Rule (Apriori Principle):**
> All non-empty subsets of a frequent itemset must also be frequent.
> Conversely, if an itemset is infrequent, all its supersets will be infrequent.

This principle allows the algorithm to prune the search space significantly.

## 3. Math Formulas

**Support:**
$$ \text{Support}(X) = \frac{\text{Number of transactions containing } X}{\text{Total number of transactions}} $$

**Confidence:**
$$ \text{Confidence}(A \rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)} = P(B|A) $$

## 4. Inputs Required

-   **transactions**: List of lists, where each inner list is a transaction containing items.
-   **min_sup**: Minimum support threshold (0 to 1).
-   **min_conf**: Minimum confidence threshold (0 to 1).

## 5. Usage Guidelines

### When to use:
-   **Market Basket Analysis**: Finding patterns in customer purchases (e.g., "People who buy Bread and Milk also buy Eggs").
-   **Recommender Systems**: Simple product recommendations based on co-occurrence.

### When not to use:
-   **Very Large Datasets**: Generates a huge number of candidates. Algorithms like **FP-Growth** are faster and more memory conservative because they don't generate candidates.
-   **Low Support Thresholds**: If `min_sup` is too low, the number of frequent itemsets explodes exponentially.

### Industry Best Practices:
-   **Parameter Tuning**: Start with high `min_sup` and lower it gradually until you find enough interesting rules without crashing memory.
-   **Lift**: Often used alongside Confidence. Lift > 1 implies positive correlation. Lift = 1 implies independence. $Lift(A \rightarrow B) = \frac{Conf(A \rightarrow B)}{Sup(B)}$.

## 6. Concurrency, Parallelism, Memory Management

-   **Concurrency**: Counting support for candidates can be parallelized.
-   **Memory**: Can be very memory intensive. Storing all candidates $C_k$ can exhaust RAM if $k$ is large and min_sup is low.

## 7. Underlying Data Structure

-   **Lists/Sets**: Used to store itemsets and candidates.
-   **Itertools**: Used for generating combinations.
