# Chapter 5: Decision Trees
## 💡 Key Takeaways

## 🧠 Self-Reflection & Insights

- Scikit-Learn uses the CART algorithm, which produces only binary trees, meaning trees where split nodes
always have exactly two children

- Decision trees are intuitive, and their decisions are easy to interpret. Such models are
often called white box models.
  -  The field of interpretable ML aims at creating ML systems that can
  explain their decisions in a way humans can understand. This is important in many
  domains, for example in healthcare, to let a doctor review the diagnosis; in finance, to
  let analysts understand the risks; in a judicial system, to let a human make the final
  call; or in human resources, to ensure decisions aren’t biased.

- A decision tree can also estimate the probability that an instance belongs to a particu!
lar class k
  - First it traverses the tree to find the leaf node for this instance, and then it returns the proportion of
    instances of class k among the training instances that would also reach this leaf node.

- How CART algorithm splits the training set into two subsets:
  - using a single feature k and a threshold tk
  - searches for the pair (k, tk) that produces the purest subsets, weighted by their size
    -> A child node can have higher impurity than the parent node, but the weighted impurity of the two child nodes is lower than that of the parent node.

-  the CART algorithm is a greedy algorithm:  it greedily searches for an optimum split at the top level, then repeats the process at each subsequent level.
    - does not check whether the split will lead to the lowest possible impurity several levels down.
    - often produces a solution that’s reasonably good but not guaranteed to be optimal.
    - what about ensemble algorithms like randomforest? is it a greedy algorithm?

- decision trees is a nonparametric model
    - the number of parameters is not determined prior to training, which increases the risk of overfitting
     - regularize the model by:
         - increasing min_* hyperparameters or ccp_alpha or
         - decreasing max_* hyperparameters
      - ccp_alpha is the complexity parameter that controls post-pruning by removing branches whose improvement is smaller than the penalty value.

- Both thermodynamic entropy and decision tree entropy measure uncertainty or disorder using similar logarithmic ideas.
The difference is that thermodynamic entropy describes physical states in nature, while decision tree entropy measures uncertainty in data classification and information.

- Decision trees are also capable of performing regression tasks. While linear regres!
sion only works well with linear data, decision trees can fit all sorts of complex
datasets.

- In decision tree regressions, predicted value for each region is always the average target value of the instances in that region.
  - The CART algorithm works as described earlier, except that instead of trying to split
    the training set in a way that minimizes impurity, it now tries to split the training
    set in a way that minimizes the MSE.

- decision trees love orthogonal decision boundaries, which makes them sensitive to the data’s orientation.
  - One way to limit this problem is to scale the data, then apply a principal component analysis (PCA)
    transformation.
  - it rotates the data in a way that reduces the correlation between the features, which often (not always) makes
    things easier for trees.

- the main issue with decision trees is that they have quite a high
variance: small changes to the hyperparameters or to the data may produce very
different models.
  - since the training algorithm used by Scikit-Learn is stochastic, even retraining
the same decision tree on the exact same data may produce a very different model unless you set the random_state hyperparameter
  - by averaging predictions over many trees, it’s possible to reduce variance
significantly. Such an ensemble of trees is called a random forest
