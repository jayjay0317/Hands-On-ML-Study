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

- **Device Consistency Matters:** I learned that model parameters and input tensors must be on the same device. When loading a model, recreating it on the CPU while using GPU inputs causes device mismatch errors.

- **The Output Layer Defines the Problem:** The number of output neurons, target format, loss function, and evaluation metric all depend on what the model is being asked to predict. This made regression, multiclass classification, multilabel classification, and multi-task learning feel more connected.

- **Reusable Training Loops Are Valuable:** Multiple-input examples showed how Python features such as starred unpacking and dictionary unpacking can make training loops more flexible. This will be useful when working with more complex projects containing tabular data, images, text, or several input sources.

- **Logits and Probabilities Have Different Roles:** I learned that logits are useful for efficient and numerically stable training, while probabilities are mainly useful for interpretation and downstream decisions.

- **Practical Deep Learning Includes Engineering Decisions:** Training models is not only about defining layers and losses. Device placement, saving weights, tuning hyperparameters, evaluation mode, and inference optimization are all important parts of building reliable PyTorch workflows.
