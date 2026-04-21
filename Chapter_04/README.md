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
