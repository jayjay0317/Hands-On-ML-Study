# Chapter 11: Training Deep Neural Networks

## 💡 Key Takeaways

## 🧠 Self-Reflection & Insights

- **PyTorch Initialization:** PyTorch layers automatically initialize their parameters, so manual initialization is not always necessary. However, for deeper networks, explicitly matching the initialization method to the activation function, such as He/Kaiming initialization for `ReLU`, can make training more stable.

- **Centered Activations and Gradient Flow:** Activation functions such as ELU can produce negative outputs, which helps keep the average activation closer to `0`. This gives the next layer a more centered input distribution, reducing the chance that activations are pushed into saturated regions. As a result, derivatives are less likely to become extremely small, helping gradients flow more reliably through deep networks.

- **Batch Normalization During Training and Evaluation:** During training, batch normalization uses mini-batch statistics and updates running estimates of the mean and variance. During evaluation, it uses these running estimates instead, so using `model.train()` and `model.eval()` correctly is important.

- **Understanding `BatchNorm2d`:** I initially thought `nn.BatchNorm2d` computed separate means and variances for each pixel location. I learned that it actually computes one mean and variance per channel, using all values across the batch and spatial dimensions. For an input shaped `[batch_size, channels, height, width]`, each channel is normalized using statistics from all images and all spatial positions in that channel.
