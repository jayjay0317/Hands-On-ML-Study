# Chapter 9: Introduction to Artificial Neural Networks

## 💡 Key Takeaways

- **From Perceptrons to MLPs:** A perceptron is a linear classifier because it separates classes using a linear decision boundary. Stacking multiple perceptrons into a multilayer perceptron (MLP) introduces hidden layers and nonlinear activation functions, allowing the network to learn more complex patterns such as XOR.

- **Backpropagation and Weight Initialization:** Backpropagation computes how each weight and bias contributes to the loss, then gradient descent updates the parameters to reduce the error. Hidden-layer weights must be initialized randomly to break symmetry. If neurons in the same layer start with identical weights and biases, they receive the same inputs, produce the same outputs, and receive identical gradient updates, so they continue learning the same feature.

- **Activation Functions and ReLU:** Nonlinear activation functions are necessary because stacking only linear layers is still equivalent to one linear transformation. ReLU is widely used because it is simple and fast to compute, helps gradients flow for positive inputs, and avoids the bounded-output behavior of sigmoid and `tanh`. However, ReLU can produce zero gradients for negative inputs, which may cause some neurons to stop learning.

- **Feature Scaling for Image MLPs:** `MinMaxScaler` rescales pixel values to the `0` to `1` range, which is compatible with image intensities and often works well with the default settings of `MLPClassifier`. In contrast, `StandardScaler` can amplify small variations in low-variance background pixels and give them unnecessary importance.

- **Neural Network Confidence:** Neural networks can assign very high probabilities to incorrect predictions. Cross-entropy training encourages the model to make the correct class more likely, but it does not guarantee that predicted probabilities are well calibrated.

- **Transfer Learning:** Lower layers often learn general patterns, while upper layers learn more task-specific patterns. Transfer learning reuses useful pretrained layers and adapts the remaining layers to a new but related task.

- **Optimization Challenges:** Large neural networks often do not suffer much from poor local minima because many local optima can perform similarly to the global optimum. However, training can still slow down on plateaus or poorly conditioned regions of the loss surface.

## 🧠 Self-Reflection & Insights

- perceptron is a linear model so it has limitations, some of which can be eliminated by stacking multiple perceptrons (mlp)

- why neurons in the same layer would learn the same features and the network would behave as if it had only one neuron per layer if all weights were initialized to the same value?
    - if only two neurons out of every neuron in the same layer were initialized to the same value, is it equivalent to having one less neuron?

- what makes ReLU so practical and powerful?
      - Does the binary nature of the ReLU derivative (0 or 1) have any negative impact on Gradient Descent? (especially derivative being 0 for z < 0)

- **Why MinMaxScaler Can Work Better for Images:** `MinMaxScaler` rescales each pixel feature to the `0` to `1` range, which is usually compatible with image intensity values and the default settings of `MLPClassifier`. `StandardScaler` forces every pixel feature to have unit variance, so low-variance background pixels can be amplified relative to their original variation. This may give unnecessary importance to pixels that contain little useful visual information.

- What make neural networks overconfident even on wrong predictions?

- how do we choose how many layers to reuse in transfer learning?

- large neural networks rarely get stuck in local minima, and even when they do, these local optima are often almost as good as the global optimum. However, they can still get stuck on long plateaus for a long time.
