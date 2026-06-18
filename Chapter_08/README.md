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
