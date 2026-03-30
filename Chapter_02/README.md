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
    - method 2: use similarity measure such as a RBF (can also measure 2d distance)
- RBF is useful for capturing similarity based on the distance between data points (in a scale of 0~1)
- When the target variable is scaled, the model’s predictions need to be transformed back to the original scale
    - Instead of manually applying the inverse transformation, you can use 'TransformedTargetRegressor' to handle this automatically
- write a custom class to make my custom transformer to be trainable, learn some parameters in the 'fit()' method and using them later in the 'transform()' method
- pipeline helps data transformation steps to be executed in the right order
    - support indexing (how can this be useful?)
- most models prefer features with roughly uniform or normal distributions
- cross-validation features expect a utility function (greater is better) rather than a cost function (lower is better), so the scoring function is actually the opposite of the RMSE
    - it's a negative value, so you need to switch the sign of the output to get the RMSE scores
- Optimization: Model and Evaluation Caching

Performing 10 fold cross validation on a Random Forest model is a highly computationally expensive and time         consuming process. To resolve this and maximize development efficiency, I implemented a serialization and caching     pipeline utilizing the joblib library.

Problem: Rerunning the entire Jupyter Notebook repeatedly executes the heavy pipeline training and validation steps, which significantly disrupts the development workflow.

Solution: During the initial execution, the fully trained model object and the cross validation results array are saved directly to the hard drive as .pkl files using the joblib.dump function.

Impact: For all subsequent runs, a conditional statement verifies the existence of these saved files. If present, they are instantly loaded into memory via joblib.load. This completely eliminates unnecessary computation wait times, allowing for an immediate transition into the hyperparameter tuning phase.

- Wrapping preprocessing steps in a Scikit-Learn pipeline allows you
 to tune the preprocessing hyperparameters along with the model
 hyperparameters.

- can set the pipeline's memory parameter to the path of a caching directory if fitting the pipeline transformers is computationally expensive

- important to check model fairness

- **Bidirectional Nature of RMSE:**
  The final RMSE of approximately $41,500 represents the absolute magnitude of the error. Objectively, this means our predictions carry a bidirectional volatility, potentially overestimating or underestimating the actual house prices by an average of $40,000.

- **Risk Control via Confidence Intervals:**
  Rather than relying on a single point estimate, a 95% confidence interval (approx. $39K to $43K) was computed using the bootstrap method. This mathematically bounds and verifies the best and worst error scenarios the model might face in a production environment.

- **Business Limitations & Critical Evaluation:**
  Although the test set evaluation demonstrated stable generalization without overfitting, a $40,000 variance poses a significant financial risk in real estate transactions. From a critical business perspective, this current error margin limits the feasibility of immediate commercial deployment.

- **Future Work:**
  Future iterations will incorporate residual analysis to visually verify directional bias (e.g., whether the model consistently overprices or underprices homes). Furthermore, implementing advanced feature engineering and more powerful algorithms like XGBoost or LightGBM will be necessary to narrow the error margin to practical industry standards.

- Once your model is transferred to production, you can load it and use it. For this you
must first import any custom classes and functions the model relies on (which means
transferring the code to production), then load the model using joblib and use it to
make predictions

-  you can wrap the model within a dedicated web service that your web application can query through a REST API
    - This makes
it easier to upgrade your model to new versions without interrupting the main appli!
cation. It also simplifies scaling, since you can start as many web services as needed
and load-balance the requests coming from your web application across these web
services. Moreover, it allows your web application to use any programming language,
not just Python.
