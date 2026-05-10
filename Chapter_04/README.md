# Chapter 4: Training Models
## 💡 Key Takeaways

## 🧠 Self-Reflection & Insights
- two different ways to train a linear regression model:
  1. using a closed-form equation (normal equation)
  2. using gradient descent
- scikit-learn LinearRegression solves the least-squares problem using an SVD-based method (lstsq), which is
  mathematically equivalent to using the Moore–Penrose pseudoinverse.
  - This approach always works (even when the matrix is singular or non-invertible), unlike the normal equation which can fail when X^T X is not invertible.
- The Normal Equation becomes computationally expensive as the number of features increases because it requires inverting an (n+1)×(n+1) matrix, with complexity around O(n^2.4 to n^3), meaning doubling features can increase computation time by about 5–8×, while Scikit-Learn’s SVD-based LinearRegression is more efficient with O(n^2) complexity, so doubling features increases time by about 4×
- gradient descent:
  - better suited for cases where there are a large number of features or too many training instances to fit in memory
  - There may be holes, ridges, plateaus, and all sorts of irregular terrain in the cost function, making convergence to     the minimum difficult.
  - ensure that all features have a similar scale, or else it will take much longer to converge
  - you can use grid search to find a good learning rate
  - batch gradient descent is terribly slow on very large training sets but gradient descent scales well with the number     of features
  - set a very large number of epochs and interrupt the algorithm when the gradient vector's norm becomes smaller than       the tolerance
  - stochastic gradient descent can jump out of local minima
  - Q: why use number of epochs when sgd picks a random instance every step? (each epoch doesn't mean every instance has been used exactly once)
  - Without shuffling, SGD processes correlated (e.g., label-sorted) data in sequence, producing biased gradients that prevent stable convergence to the global optimum.
  - mini-batch GD will end up walking around a bit closer to the minimum than stochastic GD—but it may be harder for it to escape from local minima
- fit vs partial_fit
- Polynomial regression: Adding powers of features creates nonlinear patterns in the input space, but the model remains linear because it is still a linear combination of the transformed features (linear in the parameters).
- It is critical to scale the data using StandardScaler before applying any regularization because these algorithms are highly sensitive to the scale of the input features
- Lasso regression automatically performs feature selection and outputs a sparse model
with few nonzero feature weights
- Lagrange multiplier in understanding the difference between ridge, lasso regression

### Engineering Insight Feature Engineering Over Algorithm Tuning

During the evaluation of our models a critical discrepancy was observed when predicting the value for X 1.5. The purely linear regularized models Ridge Lasso and Elastic Net consistently predicted values around 5.0. However the Early Stopping model predicted approximately 4.12. 

Given that the true underlying data generation function was quadratic $y = 0.5x^2 + x + 2$ the actual expected value mathematically is 4.625. The 4.12 prediction was significantly closer to reality.

This mathematically proves a fundamental machine learning principle. The linear models were strictly constrained by their one dimensional perspective resulting in high bias regardless of which sophisticated regularization algorithm was applied. Conversely the Early Stopping model utilized polynomial features allowing it to understand the true curvature of the data before optimization even began.

The ultimate takeaway for production environments is clear. Algorithm selection and hyperparameter tuning can only optimize the information provided. Transforming the data space to reflect reality through rigorous feature engineering is fundamentally more impactful for reducing error than merely swapping algorithms.

- Logistic regression is a special case of softmax regression for binary classification.
    - The logistic function is a special case of the softmax function when the number of classes is two.
