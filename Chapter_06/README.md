# Chapter 6: Ensemble Learning and Random Forests
## 💡 Key Takeaways


## 🧠 Self-Reflection & Insights

- for an ensemble methods to work best, all predictors must be as independent from one another as possible, making uncorrelated errors and make uncorrelated errors
    - cannot be true because they are trained on the same data
    - one way is to get diverse classifiers and train them using very different algorithms
    - you can also play with the model hyperparameters to get diverse models, or train the models on different subsets of the data
 
- Soft voting often performs better than hard voting because it considers not only the predicted class but also the confidence (probability) of each model.
  - However, this also means that very low probabilities can strongly affect the average. In other words:
        - High confidence predictions have more influence.
        - But extremely low confidence predictions also pull the average down.
        - So soft voting works well only when the models produce reliable probability estimates (well-calibrated
            probabilities). If a model is overconfident or poorly calibrated, soft voting can even hurt performance.
            That is why techniques such as weighted soft voting or probability calibration are often used in practice.

- How does bagging introduce more diversity in the subsets that each predictor is trained on compared to pasting?
      - extra diversity means the predictors end up being less correlated, so the ensemble's variance is reduced
      - overall, bagging often results in better models so its generally preferred

- in extra-trees, random thresholds are used for each feature rather than searching for the best possible threshold. What distribution do these random threshold values follow?

- boosting primarily reduces bias, whereas bagging primarily reduces variance.
- Predictor weights can become negative when a weak learner performs worse than random guessing. However, if $ \alpha_j $ is negative, then $ \exp(\alpha_j) < 1 $, so according to the instance weight update rule, wouldn't the weights of misclassified instances actually decrease rather than increase?
