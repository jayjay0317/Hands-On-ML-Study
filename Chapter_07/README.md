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

- The recovered data is not exactly the same as the original data because PCA dimensionality reduction is a lossy process. When PCA reduces the data from 784 dimensions to fewer dimensions, such as 154, it discards some information. Since many different original 784-dimensional points can be projected to the same lower-dimensional point, the original data cannot be perfectly reconstructed.
- If all 784 principal components are used, the PCA transformation matrix is square and invertible, so the original data can be recovered exactly. However, when only some components are used, the matrix is not square, so a true inverse does not exist. Therefore, `inverse_transform()` only reconstructs an approximation of the original data by projecting the reduced data back into the original feature space.
