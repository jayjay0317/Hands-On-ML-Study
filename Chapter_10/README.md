# Chapter 10: Building Neural Networks with PyTorch

## 💡 Key Takeaways



## 🧠 Self-Reflection & Insights

- It’s generally better to use 32 bits in deep learning because this takes half the RAM and speeds up computations, and neural nets do not actually need the extra precision offered by 64-bit floats.

- `from_numpy()` creates a tensor on the CPU that just uses the NumPy array’s data directly, without copying it, but modifying the NumPy array will also modify the tensor, and vice versa.

- In-place operations can save memory, but they should be used carefully when working with autograd because modifying tensors needed for backpropagation can cause errors

- GPUs are most effective for large parallel computations. For small operations, the overhead of moving data and coordinating GPU work can reduce or eliminate the speed advantage.

- It’s best to use a with torch.no_grad() context during inference:
  PyTorch will consume less RAM and run faster since it won’t have to keep track of the computation graph.
