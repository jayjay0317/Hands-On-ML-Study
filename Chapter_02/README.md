# Chapter 2: End-to-End Machine Learning Project
## 💡 Key Takeaways
- **Automating Data Pipelines:** I learned the importance of fetching data through code rather than manual downloads. This approach ensures reproducibility and makes the entire modelling pipeline easily updatable when new data arrives.

- **Statistical Foundations in Model Evaluation:** Choosing between RMSE and MAE is a mathematical decision rather than a simple coding step. Recognizing how RMSE heavily penalizes outliers (L2 norm) compared to MAE (L1 norm) reinforced the necessity of analyzing the underlying data distribution before evaluating model performance.

## 🧠 Self-Reflection & Insights

- test set generation - stratified sampling (add more later ex. why it reduces bias)
- Each splitter has a split() method that returns an iterator over different training/
test splits of the same data. (cross-validation)
- data snooping (put the test set aside and onlyh explore the train set)
- When creating new combined features, make sure they are not
too linearly correlated with existing features: collinearity can cause
issues with some models, such as linear regression. In particular,
avoid simple weighted sums of existing features.
- imputation: filling in missing values (can use the SimpleImputer in sklearn)
- drop() creates a copy of the data and does not affect the original dataset
- scikit-learn transformers output numpy arrays (why?)
- ordinal encoder: gives numbers 1, 2, 3, ... to each category. Ml algorithms will assume that two nearby values are more silimar than two distant values -> use one-hot encoding to resolve this issue by creating one binary attribute per category (called dummy attributes)
- the output of a onehotencoder is a SciPy sparse matrix, instead of a NumPy array
- onehotencoder remembers which categories it was trained on
- fit the scalers to the training data only. Once you have a trained scaler, you can then use it to transform any other set
- bucketizing: handles heavy-tailed features
- handling multimodal distribution:
    - method 1: bucketizing
    - method 2: use similarity measure such as a RBF
- When the target variable is scaled, the model’s predictions need to be transformed back to the original scale
    - Instead of manually applying the inverse transformation, you can use 'TransformedTargetRegressor' to handle this automatically
