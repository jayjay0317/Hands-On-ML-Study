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

- **Clarifying the Meaning of Dimension:** I realized that the word dimension can mean different things depending on context. In NumPy, dimension describes the structure of the array, such as whether it is 1D, 2D, or 3D. In PCA, dimension refers to the number of features in the mathematical vector space where each data point exists.

- **Why Variance Represents Information:** I initially questioned why preserving more variance means preserving more information. The key intuition is that variance measures how much the data changes along a direction. If a direction has very low variance, most points are similar along that axis, so removing it usually loses less information than removing a high-variance direction.

- **The Importance of Scaling Before PCA:** Since PCA chooses directions based on variance, feature scale can strongly affect the result. A feature with a large scale may dominate the principal components even if it is not truly more important, so standardization is often necessary before applying PCA.

- **The Model-Dependent Nature of Dimensionality Reduction:** I learned that the best number of dimensions depends not only on the dataset, but also on the model used after dimensionality reduction. A more powerful model may perform well with fewer dimensions, while a simpler model may need more features to preserve enough predictive information.

- **Understanding PCA Reconstruction:** PCA reconstruction helped me understand why dimensionality reduction is lossy. Once data is projected from 784 dimensions to fewer dimensions, some information is discarded. Because many original points can share the same lower-dimensional representation, `inverse_transform()` can only map the reduced data back into the original feature space approximately.

- **Validation-Based Choice of `n_components`:** I realized that choosing `n_components` is similar to tuning any other hyperparameter. During cross-validation, different values are tested, and the best value is the one that leads to the strongest validation performance. Too few components may lose important information, while too many may keep unnecessary noise or reduce the benefit of PCA.

- **Randomized PCA as Approximate PCA:** Randomized PCA is not a different dimensionality reduction goal from PCA. It still tries to find variance-preserving principal components. The difference is computational: it uses random projection to find a smaller subspace that likely contains the most important components, then performs SVD inside that subspace.

- **Random Projection as a Different Idea:** Random Projection felt surprising because it does not search for the best axes at all. Instead, it relies on the Johnson-Lindenstrauss lemma, which says that distances can be approximately preserved after random projection if the target dimension is large enough.

- **Random Projection vs Randomized PCA:** I initially found the names confusing because both involve randomness. The distinction is that Random Projection uses random directions as the final representation, while Randomized PCA uses randomness only to make PCA faster.

- **Sparse Matrix Memory Efficiency:** Sparse random projection made me think more carefully about memory usage. A sparse matrix is efficient because it does not store all the zeros. Instead, it stores only the nonzero values and their positions, which is especially useful for very high-dimensional datasets such as text data.

- **LLE and Coordinate Ambiguity:** LLE clarified that low-dimensional embeddings do not have meaningful absolute positions. The result is meaningful only up to translation, rotation, reflection, and scaling. LLE also does not randomly initialize points and move them iteratively; after computing reconstruction weights, it solves an eigenvalue problem to find all low-dimensional coordinates at once.

- **t-SNE as a Visualization Tool:** I learned that not every dimensionality reduction technique is meant to be used as preprocessing for a predictive model. For example, `t-SNE` is mainly useful for visualization and exploration because it focuses on preserving local similarity in a low-dimensional plot.
