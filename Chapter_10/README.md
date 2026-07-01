# Chapter 10: Building Neural Networks with PyTorch

## 💡 Key Takeaways



## 🧠 Self-Reflection & Insights

- It’s generally better to use 32 bits in deep learning because this takes half the RAM and speeds up computations, and neural nets do not actually need the extra precision offered by 64-bit floats.

- `from_numpy()` creates a tensor on the CPU that just uses the NumPy array’s data directly, without copying it, but modifying the NumPy array will also modify the tensor, and vice versa.

- In-place operations can save memory, but they should be used carefully when working with autograd because modifying tensors needed for backpropagation can cause errors
