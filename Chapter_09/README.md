# Chapter 9: Introduction to Artificial Neural Networks

## 💡 Key Takeaways

- **From Perceptrons to MLPs:** A perceptron is a linear classifier because it separates classes using a linear decision boundary. Stacking multiple perceptrons into a multilayer perceptron (MLP) introduces hidden layers and nonlinear activation functions, allowing the network to learn more complex patterns such as XOR.

- **Backpropagation and Weight Initialization:** Backpropagation computes how each weight and bias contributes to the loss, then gradient descent updates the parameters to reduce the error. Hidden-layer weights must be initialized randomly to break symmetry. If neurons in the same layer start with identical weights and biases, they receive the same inputs, produce the same outputs, and receive identical gradient updates, so they continue learning the same feature.

- **Activation Functions and ReLU:** Nonlinear activation functions are necessary because stacking only linear layers is still equivalent to one linear transformation. ReLU is widely used because it is simple and fast to compute, helps gradients flow for positive inputs, and avoids the bounded-output behavior of sigmoid and `tanh`. However, ReLU can produce zero gradients for negative inputs, which may cause some neurons to stop learning.

- **Feature Scaling for Image MLPs:** `MinMaxScaler` rescales pixel values to the $0$ to $1$ range, which is compatible with image intensities and often works well with the default settings of `MLPClassifier`. In contrast, `StandardScaler` can amplify small variations in low-variance background pixels and give them unnecessary importance.

- **Neural Network Confidence:** Neural networks can assign very high probabilities to incorrect predictions. Cross-entropy training encourages the model to make the correct class more likely, but it does not guarantee that predicted probabilities are well calibrated.

- **Transfer Learning:** Lower layers often learn general patterns, while upper layers learn more task-specific patterns. Transfer learning reuses useful pretrained layers and adapts the remaining layers to a new but related task.

- **Optimization Challenges:** Large neural networks often do not suffer much from poor local minima because many local optima can perform similarly to the global optimum. However, training can still slow down on plateaus or poorly conditioned regions of the loss surface.

## 🧠 Self-Reflection & Insights

- **Symmetry Breaking Is About Learning Diversity:** I learned that random initialization is not just used to make training random. Its main purpose is to ensure that neurons can learn different features. If two neurons begin with exactly the same weights and biases, they remain identical throughout training. If only two neurons are initialized identically, they effectively act like duplicate neurons, so the layer loses some of its useful capacity.

- **ReLU Has a Trade-Off:** ReLU is practical because its derivative is simple: $1$ for positive inputs and $0$ for negative inputs. For positive inputs, this allows gradients to pass through without shrinking. However, the zero derivative for negative inputs can create “dead” neurons that stop updating. I want to learn later how variants such as Leaky ReLU address this issue.

- **Scaling Depends on the Meaning of Features:** I learned that preprocessing is not only about putting all features on similar scales. For images, pixel intensity has meaningful structure, so scaling values from $0$ – $255$ to $0$–$1$ can preserve useful information better than forcing every pixel position to have the same variance.

- **High Confidence Does Not Mean High Reliability:** The Fashion MNIST example showed that an MLP can be extremely confident even when its prediction is wrong. This connects to my earlier interest in calibration: model probabilities should be evaluated rather than automatically trusted.

- **Transfer Learning Requires a Similarity Judgment:** I wondered how many pretrained layers should be reused in transfer learning. My current understanding is that more lower layers can be reused when the new task and data are similar to the original task, while more layers may need fine-tuning when the tasks or input distributions are different.

- **Loss Landscapes Are More Complex Than Local Minima:** I initially focused on local minima as the main optimization problem. However, plateaus, saddle points, and poorly conditioned regions can also slow training because gradients become very small or progress becomes inefficient.
