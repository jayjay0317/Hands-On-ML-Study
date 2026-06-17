# Chapter 8: Unsupervised Learning Techniques

## 💡 Key Takeaways


## 🧠 Self-Reflection & Insights

- how does a $k$-dimensional dataset from `KMeans` class's `transform()` method be used as a nonlinear dimensionality reduction technique?
    - how can these distances be used as extra features to train another model? (example)

- if there are only few labels and clustering is used to propagate the labels to all the instances in the same cluster, doesn't it create more bias compared to the actual label? (for example, two instances in same clusters can have different actual labels and the model wouldn't be able to learn this and might end up with an overconfident model) Does this technique always lead to improvement in performance?

- is it necessary to specify `random_state` in k-means clustering when initial centroid positions are given?

- I expected mini-batch k-means to always have higher inertia than regular k-means because it uses approximate updates. However, this is not guaranteed. Since k-means can converge to local optima depending on initialization and update paths, mini-batch k-means can sometimes find a solution with lower inertia.

- lower inertia doesn't always mean it has the most optimal number of clusters. The number of clusters that gives more clusters of similar sizes might be optimal despite higher inertia value. Silhouette diagram can be used to visually analyze this.

- several variants of image segmentation: color segmentation, semantic segmentation, instance segmentation
      - The state of the art in semantic or instance segmentation today is achieved using complex architectures based on convolutional neural networks or vision              transformers 
