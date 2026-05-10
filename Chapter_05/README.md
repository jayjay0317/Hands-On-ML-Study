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

