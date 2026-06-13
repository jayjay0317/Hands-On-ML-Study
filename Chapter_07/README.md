# Chapter 7: Dimensionality Reduction

## 💡 Key Takeaways


## 🧠 Self-Reflection & Insights

1. NumPy Dimension (ndim)
Definition: Refers to the structural depth or the number of axes in the array.

Core Concept: It represents the number of nested containers (brackets []) in the data structure. It is purely about how the data is organized in computer memory.

Perspective: Computational (Data Structure).

Example: A 2D matrix (like a table with 5 rows and 3 columns) has ndim=2, regardless of the data it holds.

2. PCA Dimension (Feature Space)
Definition: Refers to the number of features (variables) or the degrees of freedom that define each data point.

Core Concept: It represents the number of dimensions in the vector space where the data points exist. It is about the informational complexity of the data.

Perspective: Mathematical (Vector Space).

Example: If each data point consists of 3 features (e.g., Height, Weight, Age), the data is 3-dimensional, even if it is stored in a 2D NumPy array.


- Why does preserving more variance preserve more information?

- the optimal number of dimensions also depends on which model will be used after dimensionality reduction
    - smaller number of dimensions might be enough for a more powerful model
