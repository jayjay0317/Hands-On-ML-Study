# Chapter 6: Ensemble Learning and Random Forests
## 💡 Key Takeaways

- Ensemble learning combines multiple predictors to build a model that is often more accurate and robust than any individual model.

- For ensemble methods to work well, the individual predictors should make different kinds of errors. This can be encouraged by using different algorithms, different hyperparameters, or different subsets of the training data.

- Voting classifiers combine predictions from multiple models. Hard voting uses majority vote, while soft voting averages predicted probabilities.

- Soft voting often performs better than hard voting because it uses confidence information, but it depends on the models producing reliable probability estimates.

- Bagging trains the same type of model on different random subsets of the training data. This is especially effective for high-variance models such as decision trees.

- Bagging generally reduces variance, while boosting primarily reduces bias by sequentially correcting previous errors.

- Random Forests improve upon individual decision trees by training many trees on random subsets of data and features.

- Extra-Trees add even more randomness by using random thresholds for splits instead of searching for the best possible threshold.

- AdaBoost increases the weights of misclassified instances so that later predictors focus more on difficult examples.

- Gradient Boosting trains each new predictor on the residual errors made by the previous ensemble.

- Shrinkage is a regularization technique in Gradient Boosting. A lower `learning_rate` requires more trees but often improves generalization.

- Histogram-Based Gradient Boosting speeds up training by binning numerical features into a fixed number of discrete intervals, reducing the number of split thresholds that need to be evaluated.

- HGB is especially useful for large tabular datasets because it is fast, memory efficient, and supports missing values and categorical features natively.

- Stacking converts the predictions of multiple base models into new meta-features. A final estimator then learns how to combine these meta-features to make the final prediction.

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
- Predictor weights can become negative when a weak learner performs worse than random guessing. However, if $\alpha_j$ is negative, then $\exp(\alpha_j) < 1$, so according to the instance weight update rule, wouldn't the weights of misclassified instances actually decrease rather than increase?

-A more accurate predictor receives a larger predictor weight $\alpha_j$. This means that when a strong predictor misclassifies an instance, AdaBoost treats that instance as especially difficult and increases its weight more aggressively. In other words, if a reliable predictor gets an example wrong, the next predictor should pay much more attention to that example.

- shrinkage: a regularization technique for gradient boosting
      - If you set the `learning_rate` to a low value, such as 0.05, you will need more trees in the ensemble to fit the
         training set, but the predictions will usually generalize better.
      - adding too many trees can cause overfitting
      -  you can use cross-validation to find the optimal learning rate, using `GridSearchCV` or `RandomizedSearchCV`

- To find the optimal number of trees: if you set the n_iter_no_change hyperparameter to an integer value, say 10, then the GradientBoostingRegressor will automatically stop adding more trees during training if it sees that the last 10 trees didn’t help.
      - How does judge that the last 10 trees didn't help? (How does it measure performance?)

- In regular Gradient Boosting, the decision trees may need to search through many possible split thresholds.
    Histogram-Based Gradient Boosting simplifies this process by replacing continuous values with integer bin indices.
    For example, instead of evaluating many exact values of a feature, the model may group them into bins such as:
    low values  
    medium values  
    high values  
    This makes the split search much faster, especially for large datasets.

- The $\log(m)$ term in regular GBRT comes from sorting feature values while searching for the best split. HGB avoids much of this cost by binning continuous features into a fixed number of integer bins before training.
- HGB speeds up training mainly by binning numerical features, but it is also convenient for real-world tabular datasets because Scikit-Learn's HGB implementation supports categorical features and missing values natively. This means it often requires less preprocessing, such as scaling, imputation, or one-hot encoding.

- In stacking, the final estimator does not learn directly from the original input features. Instead, it learns from the predictions generated by the base models.
- Each base model produces a prediction or probability estimate, and these outputs become new meta-features for the final estimator.
- For example, if there are three base models, the final estimator may receive three meta-features, one from each model.
- This allows stacking to learn how to combine different models more flexibly than hard voting or soft voting.
- Stacking converts the predictions of multiple base models into new meta-features. The final estimator then learns how to combine these meta-features to make the final prediction.
