# Chapter 11: Training Deep Neural Networks

## 💡 Key Takeaways

## 🧠 Self-Reflection & Insights

- **PyTorch Initialization:** PyTorch layers automatically initialize their parameters, so manual initialization is not always necessary. However, for deeper networks, explicitly matching the initialization method to the activation function, such as He/Kaiming initialization for `ReLU`, can make training more stable.

- **Centered Activations and Gradient Flow:** Activation functions such as ELU can produce negative outputs, which helps keep the average activation closer to `0`. This gives the next layer a more centered input distribution, reducing the chance that activations are pushed into saturated regions. As a result, derivatives are less likely to become extremely small, helping gradients flow more reliably through deep networks.

- **Batch Normalization During Training and Evaluation:** During training, batch normalization uses mini-batch statistics and updates running estimates of the mean and variance. During evaluation, it uses these running estimates instead, so using `model.train()` and `model.eval()` correctly is important.

- **Understanding `BatchNorm2d`:** I initially thought `nn.BatchNorm2d` computed separate means and variances for each pixel location. I learned that it actually computes one mean and variance per channel, using all values across the batch and spatial dimensions. For an input shaped `[batch_size, channels, height, width]`, each channel is normalized using statistics from all images and all spatial positions in that channel.

- **LayerNorm vs. Tabular Feature Scaling:** Layer normalization normalizes across feature dimensions within each instance, so it works naturally when those dimensions belong to the same representation space, such as image spatial values or transformer embeddings. For tabular datasets like California housing, where features have different meanings and scales, standard feature preprocessing such as `StandardScaler` is usually the first step. `BatchNorm` or `LayerNorm` can still be tested inside a neural network, but they do not replace thoughtful tabular preprocessing.

- **Broadcasting in Normalization:** In tensor operations, broadcasting allows tensors with compatible shapes to be combined without manually copying values. For example, if `inputs` has shape `[32, 3, 100, 200]` and `means` has shape `[32, 3, 1, 1]`, PyTorch can subtract each image-channel mean from all corresponding height and width positions. Using `keepdim=True` preserves reduced dimensions as size `1`, making this kind of broadcasting easier and less error-prone.

-  adaptive learning rate algorithm, like Adam, AdaGrad, and RMSProp, requires less tuning of the learning rate hyperparameter $\eta$

- **Adam Optimizer:** Adam combines momentum and adaptive learning rates. Momentum helps speed up movement in consistent gradient directions, while the squared-gradient average adjusts each parameter's learning rate by reducing updates in steep directions and allowing relatively larger updates in flatter directions. This often makes training faster and more stable, but it does not guarantee reaching the global minimum.

- **Optimizer Choice:** Adaptive optimizers such as RMSProp, Adam, AdaMax, NAdam, and AdamW often converge quickly, but they do not always generalize best on every dataset. Adam or AdamW are strong defaults, while SGD with momentum or Nesterov momentum can still be worth trying when generalization is disappointing.

- **Reducing vs. Restarting the Learning Rate:** Some schedulers, such as performance scheduling, reduce the learning rate when validation performance stops improving so the optimizer can take smaller, more careful steps near a good solution. Other schedules, such as cosine annealing with warm restarts, periodically raise the learning rate again to encourage renewed exploration. These strategies are not contradictory: reducing the learning rate focuses on fine-tuning, while restarting the learning rate encourages exploration.

- **Learning Rate Scheduling Trade-Off:** I noticed that different schedulers can respond to stagnation in opposite ways. Performance scheduling reduces the learning rate when validation performance stops improving, encouraging careful fine-tuning. In contrast, cosine annealing with warm restarts periodically raises the learning rate again to encourage renewed exploration. This helped me understand that learning rate schedules balance two different goals: fine-tuning near a good solution and exploring new regions of the loss surface.

- The maximum learning rate is chosen using a learning rate finder. Starting from a very small learning rate, we gradually increase it and observe how the loss changes. Very small learning rates make the loss decrease too slowly, while overly large learning rates make the loss unstable. A good maximum learning rate is usually chosen in the range where the loss decreases quickly but before it starts to explode.

- `weight_decay` controls the strength of weight decay and usually plays a role similar to the regularization factor $\lambda$.
