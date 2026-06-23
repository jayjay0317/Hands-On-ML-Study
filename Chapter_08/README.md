# Chapter 8: Unsupervised Learning Techniques

## 💡 Key Takeaways

- **K-Means and Cluster Evaluation:** K-means assigns instances to the nearest centroid and works best when clusters are roughly spherical and similarly sized. Since inertia always decreases as the number of clusters increases, lower inertia alone does not guarantee the best value of $k$. Silhouette scores and silhouette diagrams can provide additional evidence about cluster quality.

- **Cluster Distances and Semi-Supervised Learning:** `KMeans.transform()` represents each instance using its distances to the centroids. These distances can be used as cluster-based features, either as a lower-dimensional representation or as extra features for another supervised model. Clustering can also help select representative instances for labeling, but label propagation may introduce noise when clusters contain mixed classes.

- **Mini-Batch K-Means and Image Segmentation:** Mini-batch k-means uses small batches to scale clustering to larger datasets. It may sometimes achieve lower inertia than regular k-means because both methods can converge to different local optima. K-means can also be used for color-based image segmentation by clustering pixel RGB values.

- **DBSCAN:** DBSCAN identifies dense regions as clusters and labels isolated points as anomalies. It can detect arbitrarily shaped clusters, but the results depend strongly on `eps` and `min_samples`.

- **Gaussian Mixture Models:** GMMs model data as a mixture of Gaussian distributions. Unlike k-means, they learn cluster centers, relative weights, and covariance structures, allowing clusters to have different sizes, densities, elliptical shapes, and orientations.

- **GMM Covariance and Density:** With `covariance_type="diag"`, GMM can model axis-aligned ellipses but not rotated ones because the off-diagonal covariance values are fixed at `0`. `predict_proba()` returns cluster membership probabilities, while `score_samples()` returns the overall mixture density, which can be used for anomaly detection.

- **Bayesian GMM and Anomaly Detection:** GMM-based anomaly detection identifies instances in low-density regions. Bayesian GMM can start with more components than necessary and assign near-zero weights to unnecessary components, helping estimate the effective number of clusters.

## 🧠 Self-Reflection & Insights

- **Cluster Distances as a Nonlinear Representation:** I learned that `KMeans.transform()` can create a cluster-based representation by converting each instance into distances from $k$ centroids. When $k$ is smaller than the original number of features, this can act as dimensionality reduction. These distances can also be appended to the original features so that another model can learn both the raw inputs and the instance's relationship to learned clusters.

- **Label Propagation Is Not Always Reliable:** I questioned whether assigning one label to every instance in a cluster could create bias or overconfidence. If a cluster contains instances from different true classes, propagated labels can introduce label noise. Partial label propagation can reduce this risk by assigning labels only to instances close to cluster centroids.

- **Representative Labels Can Be More Useful Than Random Labels:** Labeling representative instances can improve performance because the selected examples cover different regions of the data distribution. Randomly selected labeled examples may be redundant or concentrated in only a small part of the dataset.

- **DBSCAN Parameter Selection Requires More Than Visual Inspection:** The book uses plots to show the effect of different `eps` values, but in practice a k-distance plot can help identify a reasonable neighborhood radius. The final choice should still depend on the scale and expected density of the data.

- **GMM Assumptions Should Be Evaluated:** Assuming Gaussian clusters is a modeling choice, not a guaranteed fact. In practice, I should check whether the data appears reasonably elliptical, whether the learned density structure makes sense, and whether the model performs well for the final task.

- **Understanding Density and Anomalies:** A point between two Gaussian components is not automatically anomalous. If the components overlap, both can contribute density and the point may still be considered normal. This helped me distinguish overall mixture density from cluster membership probability.



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

- difference between probability and likelihood
