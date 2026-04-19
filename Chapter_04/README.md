# Chapter 4: Training Models
## 💡 Key Takeaways

## 🧠 Self-Reflection & Insights
- two different ways to train a linear regression model:
  1. using a closed-form equation (normal equation)
  2. using gradient descent
- scikit-learn LinearRegression solves the least-squares problem using an SVD-based method (lstsq), which is
  mathematically equivalent to using the Moore–Penrose pseudoinverse.
  - This approach always works (even when the matrix is singular or non-invertible), unlike the normal equation which can fail when X^T X is not invertible.
