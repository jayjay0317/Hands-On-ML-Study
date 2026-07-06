# Chapter 10: Building Neural Networks with PyTorch

## 💡 Key Takeaways



## 🧠 Self-Reflection & Insights

- It’s generally better to use 32 bits in deep learning because this takes half the RAM and speeds up computations, and neural nets do not actually need the extra precision offered by 64-bit floats.

- `from_numpy()` creates a tensor on the CPU that just uses the NumPy array’s data directly, without copying it, but modifying the NumPy array will also modify the tensor, and vice versa.

- In-place operations can save memory, but they should be used carefully when working with autograd because modifying tensors needed for backpropagation can cause errors

- GPUs are most effective for large parallel computations. For small operations, the overhead of moving data and coordinating GPU work can reduce or eliminate the speed advantage.

- It’s best to use a with torch.no_grad() context during inference:
  PyTorch will consume less RAM and run faster since it won’t have to keep track of the computation graph.

- **Averaging Mini-Batch Losses:** When the last mini-batch is smaller than the other batches, averaging batch losses with `total_loss / len(train_loader)` gives that smaller batch the same weight as a full batch. This slightly changes the reported epoch loss. For an exact instance-level mean loss, multiply each batch loss by its batch size, sum these values, and divide by the total number of instances. This affects only the displayed epoch loss, not the gradient updates themselves.

- **Recreating the Optimizer for a New Model:** An optimizer is connected to the specific parameters passed through `model.parameters()` when it is created. Therefore, after creating a new model or replacing its layers, create a new optimizer so that the new model's weights and biases are updated during training. A loss function such as `nn.MSELoss()` can usually be reused, but redefining it in each independent training section can make the notebook easier to run from top to bottom.

- connecting inputs directly to the output layer makes it possible for the neural network to learn both deep patterns and simple rules.
    - a regular MLP forces all the data to flow through the full stack of layers; thus, simple patterns in the data may end up being distorted by this sequence of transformations

- **Why PyTorch Layers Can Be Called Like Functions:** A layer such as `nn.Linear` is an object created from a class, not a regular function. However, PyTorch modules implement Python's special `__call__()` method, which allows an object to be used with function-like syntax such as `layer(X)`. Internally, this calls the layer's `forward()` computation while also allowing PyTorch to manage features such as autograd and hooks.

- **Starred Unpacking for Multiple Inputs:** Starred unpacking collects a variable number of input tensors into one list. In a batch such as `(X_wide, X_deep, y)`, the statement `for *X_batch_inputs, y_batch in train_loader:` stores the input tensors in `X_batch_inputs` and keeps the final target tensor in `y_batch`.

- **Unpacking Inputs When Calling a Model:** The expression `model(*X_batch_inputs)` unpacks the list of input tensors into separate positional arguments. For example, if `X_batch_inputs` contains `[X_batch_wide, X_batch_deep]`, then `model(*X_batch_inputs)` is equivalent to `model(X_batch_wide, X_batch_deep)`.

- **When Starred Unpacking Is Useful:** This approach is useful when a training loop should support models with different numbers of inputs. Instead of writing separate loops for two-input, three-input, or multimodal models, the loop can move every input tensor to the device and pass them to the model dynamically.

- **Choosing the Number of Output Neurons:** The output layer depends on the prediction task. For binary classification, one output neuron is usually enough because it represents the score or probability of the positive class. For single-label multiclass classification, use one output neuron per class. For multilabel classification, use one output neuron per possible label because each label is predicted independently. For regression, the number of output neurons matches the number of continuous target values predicted for each instance.

- **Multi-Task Multiclass Classification:** Some problems require several classification tasks at once, where each task has its own mutually exclusive classes. In this case, a model can share lower layers and use a separate output head for each task. Each head has one output neuron per class and typically uses its own `CrossEntropyLoss()`. The final training loss is usually a weighted sum of the task-specific losses.

- **Handling Class Imbalance:** In an imbalanced classification dataset, frequent classes can dominate the loss and bias the model toward predicting them more often. `nn.CrossEntropyLoss()` supports a `weight` argument that assigns larger loss weights to rare classes and smaller weights to common classes. A common approach is to use weights inversely proportional to class frequency, then normalize them so they sum to `1`.
