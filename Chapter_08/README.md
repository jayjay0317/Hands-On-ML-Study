# Chapter 8: Unsupervised Learning Techniques

## 💡 Key Takeaways


## 🧠 Self-Reflection & Insights

- how does a $k$-dimensional dataset from `KMeans` class's `transform()` method be used as a nonlinear dimensionality reduction technique?
    - how can these distances be used as extra features to train another model? (example)

- if there are only few labels and clustering is used to propagate the labels to all the instances in the same cluster, doesn't it create more bias compared to the actual label? (for example, two instances in same clusters can have different actual labels and the model wouldn't be able to learn this and might end up with an overconfident model) Does this technique always lead to improvement in performance?

- is it necessary to specify `random_state` in k-means clustering when initial centroid positions are given?
