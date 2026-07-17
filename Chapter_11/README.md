# Chapter 11: Training Deep Neural Networks

## 💡 Key Takeaways

- **Stable Deep Network Training:** Deep neural networks can suffer from vanishing or exploding gradients. Better initialization, non-saturating activation functions, normalization layers, and gradient clipping can make training more stable.

- **Initialization and Activation Functions:** PyTorch layers initialize parameters automatically, but explicitly matching initialization to the activation function can help deeper networks. For example, He/Kaiming initialization works well with `ReLU`, while activation functions such as ELU, GELU, and Swish can improve gradient flow.

- **Normalization Layers:** Batch normalization and layer normalization help stabilize training. Batch normalization depends on mini-batch statistics during training and running estimates during evaluation, so using `model.train()` and `model.eval()` correctly is important.

- **Transfer Learning and Pretraining:** Reusing pretrained layers can help when the source and target tasks are similar. When labeled data is limited, unsupervised pretraining or pretraining on an auxiliary task can also provide useful representations.

- **Optimizers and Learning Rate Schedules:** Momentum, Nesterov momentum, RMSProp, Adam, and AdamW can speed up training compared to plain gradient descent. Learning rate schedules such as performance scheduling, cosine annealing, warm-up, and 1Cycle scheduling adjust the learning rate to balance fast progress and careful fine-tuning.

- **Regularization Techniques:** $\ell_1$ regularization, $\ell_2$ regularization, weight decay, dropout, and max-norm regularization help reduce overfitting. Stronger regularization can improve generalization, but too much can cause underfitting.

- **Practical DNN Defaults:** A reasonable starting setup is He initialization, `ReLU` for shallow networks or `Swish` for deep networks, batch normalization or layer normalization for deep networks, early stopping with weight decay if needed, Nesterov momentum or `AdamW`, and performance scheduling or 1Cycle scheduling.

- **AdamW and Weight Decay:** `AdamW` is often preferred over Adam when using weight decay. The `weight_decay` hyperparameter controls the strength of weight decay and plays a role similar to the regularization factor $\lambda$.

## 🧠 Self-Reflection & Insights

- **Training vs. Evaluation Mode:** I learned that `model.train()` and `model.eval()` are especially important when using layers such as batch normalization and dropout, because these layers behave differently during training and evaluation.

- **Understanding Normalization:** I initially thought `nn.BatchNorm2d` computed separate statistics for each pixel location, but I learned that it computes one mean and variance per channel across the batch and spatial dimensions. I also better understood how `keepdim=True` helps preserve dimensions for broadcasting.

- **Adaptive Optimizers Are Not Always Better:** Adaptive optimizers such as Adam and RMSProp often require less learning rate tuning and converge quickly, but they do not always generalize best. SGD with momentum or Nesterov momentum can still be worth trying.

- **Learning Rate Scheduling Trade-Off:** I noticed that some schedulers reduce the learning rate when progress stalls, while others periodically raise it again. This helped me understand the trade-off between fine-tuning near a good solution and encouraging renewed exploration.

- **Dropout and MC Dropout:** Dropout reduces overfitting by randomly dropping activations during training and scaling the remaining values by the keep probability $1 - p$. MC dropout extends this idea by keeping dropout active during evaluation and averaging multiple predictions, making it similar in spirit to an ensemble method.

- **Learning Rate Finder Intuition:** I learned that the maximum learning rate for 1Cycle scheduling can be chosen using a learning rate finder. The goal is to find a range where the loss decreases quickly, but before the learning rate becomes so large that training becomes unstable.

- **Choosing Training Techniques:** Chapter 11 made me realize that training deep neural networks is not about using every technique at once. The right choices depend on the problem, such as using regularization for overfitting, normalization for deeper networks, learning rate schedules for faster convergence, and MC dropout when uncertainty estimates matter.
