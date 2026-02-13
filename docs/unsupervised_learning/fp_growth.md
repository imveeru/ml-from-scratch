# FP-Growth

## 1. Pseudocode

```text
Find_Frequent_Itemsets(transactions, min_sup):
    # 1. Count Frequencies and Filter
    Freq_Items = Count_Items(transactions)
    Filter items where Count < min_sup
    
    # 2. Build FP-Tree
    Root = New_Node()
    For each transaction:
        Sort transaction by item frequency
        Insert_Transaction(Root, transaction)
        
    # 3. Mine Tree
    Mine_Tree(Root, null_suffix)

Mine_Tree(Tree, suffix):
    For each item 'i' in Header_Table(Tree) (from low support to high):
        New_Freq_Set = suffix + {i}
        Add New_Freq_Set to Global_Frequent_Itemsets
        
        # Construct Conditional Pattern Base
        Conditional_Paths = Get_Prefix_Paths(i)
        
        # Build Conditional FP-Tree
        Cond_Tree = Build_Tree(Conditional_Paths)
        
        If Cond_Tree is not empty:
            Mine_Tree(Cond_Tree, New_Freq_Set)
```

## 2. Algorithm Explanation

**FP-Growth** (Frequent Pattern Growth) is a fast and efficient algorithm for mining frequent itemsets. Unlike Apriori, it does **not** generate candidates.

Instead, it uses a divide-and-conquer strategy:
1.  **Compresses** the database representing frequent items into a frequent-pattern tree, or **FP-tree**, which retains the itemset association information.
2.  **Divides** the compressed database into a set of conditional databases (a special kind of projected database), each associated with one frequent item, and mines each such database separately.

This structure allows it to avoid the costly candidate generation and repeated database scans of Apriori.

## 3. Inputs Required

-   **transactions**: List of lists containing items.
-   **min_sup**: Minimum support count/fraction.

## 4. Usage Guidelines

### When to use:
-   **Large Databases**: Much faster than Apriori for large datasets because it only passes over the database twice.
-   **Dense Datasets**: Works well when there are many long frequent patterns.

### When not to use:
-   **Memory Constraints**: The FP-Tree must fit in memory. For extremely massive datasets (Big Data), distributed versions or disk-based algorithms are needed.

### Industry Best Practices:
-   **Preprocessing**: Sort items in transactions by global frequency (descending) before inserting into the tree to maximize compactness/sharing of nodes.

## 5. Concurrency, Parallelism, Memory Management

-   **Concurrency**: Conditional FP-Trees can be mined in parallel.
-   **Memory**: Uses a "Trie"-like structure to compress data. Memory usage depends on the sparsity/overlap of transactions.

## 6. Underlying Data Structure

-   **FP-Tree**: A specialized prefix tree.
-   **Linked Lists**: Used in the header table to link all occurrences of the same item in the tree for fast traversal.
