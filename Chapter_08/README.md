# Chapter 8: Unsupervised Learning Techniques

## 💡 Key Takeaways

- **K-Means and Cluster Evaluation:** K-means assigns instances to the nearest centroid and works best when clusters are roughly spherical and similarly sized. Since inertia always decreases as the number of clusters increases, lower inertia alone does not guarantee the best value of $k$. Silhouette scores and silhouette diagrams can provide additional evidence about cluster quality.

- **Cluster Distances and Semi-Supervised Learning:** `KMeans.transform()` represents each instance using its distances to the centroids. These distances can be used as cluster-based features, either as a lower-dimensional representation or as extra features for another supervised model. Clustering can also help select representative instances for labeling, but label propagation may introduce noise when clusters contain mixed classes.

- **Mini-Batch K-Means and Image Segmentation:** Mini-batch k-means uses small batches to scale clustering to larger datasets. It may sometimes achieve lower inertia than regular k-means because both methods can converge to different local optima. K-means can also be used for color-based image segmentation by clustering pixel RGB values.

- **DBSCAN:** DBSCAN identifies dense regions as clusters and labels isolated points as anomalies. It can detect arbitrarily shaped clusters, but the results depend strongly on `eps` and `min_samples`.

- **Gaussian Mixture Models:** GMMs model data as a mixture of Gaussian distributions. Unlike k-means, they learn cluster centers, relative weights, and covariance structures, allowing clusters to have different sizes, densities, elliptical shapes, and orientations.

- **GMM Covariance and Density:** With `covariance_type="diag"`, GMM can model axis-aligned ellipses but not rotated ones because the off-diagonal covariance values are fixed at `0`. `predict_proba()` returns cluster membership probabilities, while `score_samples()` returns the overall mixture density, which can be used for anomaly detection.

- **Bayesian GMM and Anomaly Detection:** GMM-based anomaly detection identifies instances in low-density regions. Bayesian GMM can start with more components than necessary and assign near-zero weights to unnecessary components, helping estimate the effective number of clusters.

- **Probability vs. Likelihood:** Probability asks how likely different data outcomes are when the model parameters are fixed. Likelihood asks which parameter values make the observed data most plausible. GMM training uses likelihood-based optimization to estimate the Gaussian weights, means, and covariance matrices that best explain the observed data.

## 🧠 Self-Reflection & Insights

- **Cluster Distances as a Nonlinear Representation:** I learned that `KMeans.transform()` can create a cluster-based representation by converting each instance into distances from $k$ centroids. When $k$ is smaller than the original number of features, this can act as dimensionality reduction. These distances can also be appended to the original features so that another model can learn both the raw inputs and the instance's relationship to learned clusters.

- **Label Propagation Is Not Always Reliable:** I questioned whether assigning one label to every instance in a cluster could create bias or overconfidence. If a cluster contains instances from different true classes, propagated labels can introduce label noise. Partial label propagation can reduce this risk by assigning labels only to instances close to cluster centroids.

- **Representative Labels Can Be More Useful Than Random Labels:** Labeling representative instances can improve performance because the selected examples cover different regions of the data distribution. Randomly selected labeled examples may be redundant or concentrated in only a small part of the dataset.

- **DBSCAN Parameter Selection Requires More Than Visual Inspection:** The book uses plots to show the effect of different `eps` values, but in practice a k-distance plot can help identify a reasonable neighborhood radius. The final choice should still depend on the scale and expected density of the data.

- **GMM Assumptions Should Be Evaluated:** Assuming Gaussian clusters is a modeling choice, not a guaranteed fact. In practice, I should check whether the data appears reasonably elliptical, whether the learned density structure makes sense, and whether the model performs well for the final task.

- **Understanding Density and Anomalies:** A point between two Gaussian components is not automatically anomalous. If the components overlap, both can contribute density and the point may still be considered normal. This helped me distinguish overall mixture density from cluster membership probability.
