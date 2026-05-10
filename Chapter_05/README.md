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

