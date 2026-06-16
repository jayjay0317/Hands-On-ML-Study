# Chapter 7: Dimensionality Reduction

## 💡 Key Takeaways

- **Dimensionality in NumPy vs PCA:** In NumPy, dimension often refers to `ndim`, which describes the number of axes or structural depth of an array. In PCA, dimension refers to the number of features or variables that define each data point in feature space. For example, a dataset with 3 features is mathematically 3-dimensional, even if it is stored as a 2D NumPy array with shape `(m, 3)`.

- **The Curse of Dimensionality:** As the number of features increases, the feature space grows rapidly and the data becomes increasingly sparse. This can make models slower, harder to visualize, and more prone to overfitting because the training data covers only a small portion of the full feature space.

- **Projection vs Manifold Learning:** Projection methods reduce dimensionality by mapping data onto a lower-dimensional space, while manifold learning methods try to uncover a lower-dimensional structure that may be curved. This distinction explains why PCA may work well on flat structures but fail on nonlinear datasets such as the Swiss roll.

- **PCA and Variance Preservation:** PCA reduces dimensionality by finding directions that preserve as much variance as possible. Preserving more variance usually means keeping more of the meaningful structure in the data, since high-variance directions often capture the main patterns of variation between instances.

- **PCA and Feature Scaling:** PCA is sensitive to feature scales because it focuses on directions with the largest variance. If one feature has a much larger scale than others, PCA may treat it as more important, so standardization is often necessary before applying PCA.

- **Choosing the Number of Dimensions:** The optimal number of dimensions is not fixed. It can be chosen by preserving a target amount of variance, such as 95%, or by selecting the value of `n_components` that gives the best validation performance for the downstream model.

- **PCA as Lossy Compression:** PCA compression is lossy when only some principal components are kept. When PCA reduces data to fewer dimensions, `inverse_transform()` can only reconstruct an approximation of the original data because a true inverse no longer exists.

- **Randomized PCA:** Full SVD becomes expensive when the data matrix $X$ has many instances $m$ and features $n$. Randomized PCA speeds this up by approximating only the leading $d$ principal components, which is much cheaper when $d \ll n$.

- **Random Projection:** Random Projection reduces dimensionality by multiplying the original data matrix by a randomly generated projection matrix. Unlike PCA, it does not search for optimal variance-preserving axes, but it can approximately preserve distances between instances when the target dimension is large enough.

- **Random Projection vs Randomized PCA:** Random Projection is a dimensionality reduction method on its own, where random directions become the final reduced space. Randomized PCA is still PCA; it uses randomness only as a computational shortcut to approximate the top principal components faster.

- **Sparse Random Projection:** Sparse random projection is more memory efficient because the random projection matrix contains mostly zeros. Instead of storing every entry in a dense matrix, sparse representations store only the nonzero values and their positions.

- **LLE and Manifold Learning:** Locally Linear Embedding, or LLE, is a nonlinear dimensionality reduction technique. It tries to preserve local neighbor relationships rather than projecting the data onto a flat subspace, making it useful for data that lies on a curved manifold.

- **Other Dimensionality Reduction Techniques:** Techniques such as `MDS`, `Isomap`, `t-SNE`, and `LDA` serve different purposes. `t-SNE` is mainly used for visualization, while `LDA` is supervised and uses class labels to find axes that separate classes well.

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

- The best value for n_components is selected based on the model’s validation performance.

During the search, different parameter combinations are tested, and each one is evaluated using cross-validation. The combination that gives the highest average validation score is chosen as the best.

In general, n_components controls how much dimensionality reduction PCA performs:


Too small: may lose important information
Too large: may keep unnecessary noise or reduce the benefit of PCA

So the best value is the one that provides the best balance and leads to the strongest model performance.

- Full SVD becomes expensive when the data matrix $X$ has many instances $m$ and features $n$, because it decomposes the full $m \times n$ matrix. Randomized PCA speeds this up by approximating only the leading $d$ principal components, which is much cheaper when $d \ll n$.

- Randomized PCA uses random projection to quickly find a smaller subspace that likely contains the most important principal components. Instead of computing the full SVD of the entire data matrix, it performs SVD inside this smaller subspace, making it much faster when the target dimension `d` is much smaller than the original number of features.

- Random Projection reduces dimensionality by multiplying the original data matrix by a randomly generated projection matrix. It does not search for the optimal variance-preserving axes like PCA, but it can approximately preserve distances between instances when the target dimension is large enough.

- Random Projection is a dimensionality reduction method on its own. It directly projects the original data onto randomly generated lower-dimensional axes. It does not try to find the best variance-preserving directions like PCA, but it can approximately preserve distances between instances when the target dimension is large enough.

Randomized PCA is still PCA. Its goal is still to find principal components that preserve as much variance as possible. The difference is that it uses randomness to approximate the top principal components faster instead of computing the full SVD exactly.

In short, Random Projection uses random directions as the final reduced space, while Randomized PCA uses randomness only as a computational shortcut to approximate PCA.

- Sparse random projection is more memory efficient because the random projection matrix contains mostly zeros. Instead of storing every entry in a dense matrix, sparse representations store only the nonzero values and their positions. This is especially useful for very high-dimensional datasets where most values are zero, such as text data.

- LLE does not assign absolute positions to the low-dimensional points. The embedding is only meaningful up to translation, rotation, reflection, and scaling. To choose one valid coordinate system, LLE applies constraints such as centering the embedding around zero and fixing its scale. Therefore, the exact coordinates of each point are not important by themselves; what matters is whether the local neighbor relationships are preserved.
