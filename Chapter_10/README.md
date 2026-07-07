# Chapter 10: Building Neural Networks with PyTorch

## 💡 Key Takeaways

- **PyTorch Tensors and Devices:** PyTorch tensors support automatic differentiation and hardware acceleration. `float32` is commonly used in deep learning because it uses less memory and is generally faster than `float64`. GPUs are most useful for large parallel computations, while small operations may not benefit much from GPU acceleration because of transfer and coordination overhead.

- **Training, Evaluation, and Autograd:** Training involves a forward pass, loss calculation, backward pass, parameter update, and gradient reset. Use `model.train()` during training and `model.eval()` during evaluation because layers such as dropout and batch normalization behave differently in each mode. During inference, `with torch.no_grad():` avoids building a computation graph and reduces memory use.

- **Mini-Batch Training:** `DataLoader` makes mini-batch training scalable and convenient. When logging epoch loss, averaging batch losses gives a smaller final batch the same weight as full batches. This affects the displayed loss slightly but does not affect gradient updates.

- **Custom Modules and Flexible Architectures:** Custom classes that inherit from `nn.Module` make it possible to build nonsequential architectures such as Wide & Deep models, models with multiple inputs, and models with multiple outputs. PyTorch layers can be called like functions because `nn.Module` supports `layer(X)` syntax through its internal `__call__()` behavior.

- **Multiple Inputs and Outputs:** Starred unpacking and named input dictionaries help training loops support multiple model inputs. Multiple-output models can use shared layers with separate output heads, and the final loss is typically a weighted combination of the losses from each output.

- **Choosing Outputs and Loss Functions:** The output layer depends on the task. Regression predicts continuous values and commonly uses `nn.MSELoss()`. Multiclass classification uses one output neuron per class and `nn.CrossEntropyLoss()`, while binary and multilabel classification commonly use `nn.BCEWithLogitsLoss()`.

- **Classification Details:** Multiclass models output logits rather than probabilities. `nn.CrossEntropyLoss()` works directly with logits, while `F.softmax()` can be applied later when class probabilities are needed. For imbalanced data, class weights can help prevent common classes from dominating training.

- **Hyperparameter Tuning:** Optuna can search for hyperparameters such as learning rate and hidden-layer size by training multiple models and comparing validation scores. Pruning becomes useful when trials are expensive because it can stop weak trials early.

- **Saving and Optimizing Models:** Saving a model's `state_dict()` is generally safer and more portable than saving the full model object. PyTorch also supports inference optimization through TorchScript and `torch.compile()`, which can improve execution efficiency for repeated inference.

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

- **Practical Optuna Extensions:** Optuna also supports pruning unpromising trials and running trials in parallel to reduce tuning time. These features become more useful when each trial is expensive or when the search space is large.

- **Regression vs. Multiclass Classification MLPs:** The hidden-layer structure can be very similar, but the output layer, targets, loss function, and evaluation metric depend on the task. Regression predicts continuous values, so the number of output neurons matches the target dimension and `nn.MSELoss()` is commonly used. Multiclass classification uses one output neuron per class, returns logits, and commonly uses `nn.CrossEntropyLoss()` with integer class indices. Regression is often evaluated with MSE or RMSE, while classification is often evaluated with accuracy.

- **Pruning Unpromising Trials:** Optuna can stop trials early when their intermediate performance is unlikely to beat previous trials. This becomes useful when each trial requires many epochs or when the hyperparameter search is large.
