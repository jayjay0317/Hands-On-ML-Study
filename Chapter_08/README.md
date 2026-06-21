# Chapter 8: Unsupervised Learning Techniques

## 💡 Key Takeaways


## 🧠 Self-Reflection & Insights

- how does a $k$-dimensional dataset from `KMeans` class's `transform()` method be used as a nonlinear dimensionality reduction technique?
    - how can these distances be used as extra features to train another model? (example)

- if there are only few labels and clustering is used to propagate the labels to all the instances in the same cluster, doesn't it create more bias compared to the actual label? (for example, two instances in same clusters can have different actual labels and the model wouldn't be able to learn this and might end up with an overconfident model) Does this technique always lead to improvement in performance?
    - **Partial Label Propagation:** Instead of propagating representative labels to every instance in the same cluster, a safer strategy is to propagate labels only to instances that are close to their cluster centroids. Instances far from the centroid may be near cluster boundaries or may not fit the cluster well, so they are more likely to receive incorrect propagated labels. Partial label propagation can reduce label noise by keeping only the more reliable pseudo-labeled examples.
    
    - **Risk of Label Propagation:** Label propagation assumes that instances in the same cluster likely share the same class label. This assumption can be useful, but it is not always true. If a cluster contains mixed classes, propagated labels can introduce label noise and make the model overconfident in incorrect labels. This helped me understand that semi-supervised learning does not automatically improve performance; it depends on how well the cluster structure aligns with the true class structure.
    
    - **Active Learning:** Active learning is another way to reduce labeling cost. Instead of labeling random instances, the model asks a human expert to label the most useful examples, often the ones it is most uncertain about. This can make the labeling process more efficient because the model focuses on examples that are likely to improve performance the most.


- is it necessary to specify `random_state` in k-means clustering when initial centroid positions are given?

- I expected mini-batch k-means to always have higher inertia than regular k-means because it uses approximate updates. However, this is not guaranteed. Since k-means can converge to local optima depending on initialization and update paths, mini-batch k-means can sometimes find a solution with lower inertia.

- lower inertia doesn't always mean it has the most optimal number of clusters. The number of clusters that gives more clusters of similar sizes might be optimal despite higher inertia value. Silhouette diagram can be used to visually analyze this.

- several variants of image segmentation: color segmentation, semantic segmentation, instance segmentation
      - The state of the art in semantic or instance segmentation today is achieved using complex architectures based on convolutional neural networks or vision              transformers

- Why does labeling representative instances instead of just random instances improve the model performance?

- is there a different way to find an optimal eps for DBSCAN other than looking at the plots like in the book?

- why is the computational complexity of a `GaussianMixture` model greater when `covariance_type` is "tied" compared to when it is "spherical" or "diag"?
      - Wouldn't it be less complex because all clusters have the same covariance matrix?

- gaussian mixture model assumes that the instances were generated from a mixture of several Gaussian distributions whose parameters are unknown. How can we make this assumption in practice?

- **Diagonal Covariance in GMM:** When `covariance_type="diag"`, each cluster uses a diagonal covariance matrix. The diagonal values represent the variance of each feature, while the off-diagonal values are fixed at `0`.

- **Why Diagonal Covariance Cannot Model Rotated Clusters:** Setting the off-diagonal covariance values to `0` assumes that the features do not vary together. As a result, the model can create horizontally or vertically stretched ellipses, but it cannot represent tilted or rotated ellipses.

- **Comparison with Full Covariance:** With `covariance_type="full"`, the model can learn nonzero covariance values between features. This allows it to model correlations between features and capture rotated elliptical clusters.

- **Mixture Density vs. Cluster Membership:** For a point located between two Gaussian components, GMM combines the density contributions from all components, weighted by their cluster weights, to calculate the overall density. Therefore, a point can still have a relatively high total density if it lies in an overlap region between clusters.

- **`score_samples()` vs. `predict_proba()`:** The `score_samples()` method returns the log of the overall mixture density at each point. In contrast, `predict_proba()` returns the relative probability, or responsibility, that each Gaussian component has for that point.

- **Points Between Clusters Are Not Always Anomalies:** A point between two clusters is not automatically an anomaly. If the Gaussian distributions overlap, both components may contribute enough density for the point to be considered normal. However, if the clusters are well separated and the region between them has low density, the point is more likely to be flagged as an anomaly.

- **Why GMM Is More Flexible than k-means:** K-means represents each cluster only by a centroid and assigns instances based on the nearest centroid. In contrast, GMM also learns each cluster's relative weight and covariance structure. This allows GMM to model clusters with different sizes, densities, elliptical shapes, and orientations.
