# Chapter 3: Classification
## 💡 Key Takeaways

## 🧠 Self-Reflection & Insights

- Difference between Classification vs Regression
- Characteristics of MNIST dataset
- sgd deals with training instances independently, one at a time, which makes it well suited for online learning
- accuracy is generally not the preferred performance measure for classifiers, especially when you are dealing with skewed datasets (why?)
  - confusion matrix (not only 2x2)
- precision is typically used along with recall
    - precision/recall trade off: increasing the threshold increases precision but reduces recall, and vice versa.
        - high precision or high recall is preferred depending on contexts
- f1 score: convenient when you need a single metric to compare two classifiers
- precision_recall_curve:
    - The precision and recall arrays contain one more value than the thresholds because they include the values at both        ends of the threshold range.
        - The thresholds array has n values (decision boundaries where the model’s prediction changes).
          But these thresholds create n + 1 intervals
    - Thresholds come from the model’s decision scores
        - The function precision_recall_curve collects the decision scores produced by the model.
          It then sorts these scores and uses them as potential threshold values.
          Each unique score represents a point where the model’s predictions can change.
- the ROC curve plots the true positive rate against the false positive rate
    - again there is a trade off between TPR and FPR
- one way to compare classifiers is to measure the auc
