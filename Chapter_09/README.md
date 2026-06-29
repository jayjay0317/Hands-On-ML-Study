# Chapter 9: Introduction to Artificial Neural Networks

## 💡 Key Takeaways

## 🧠 Self-Reflection & Insights

- perceptron is a linear model so it has limitations, some of which can be eliminated by stacking multiple perceptrons (mlp)

- why neurons in the same layer would learn the same features and the network would behave as if it had only one neuron per layer if all weights were initialized to the same value?
    - if only two neurons out of every neuron in the same layer were initialized to the same value, is it equivalent to having one less neuron?

- what makes ReLU so practical and powerful?
      - Does the binary nature of the ReLU derivative (0 or 1) have any negative impact on Gradient Descent? (especially derivative being 0 for z < 0)

- **Why MinMaxScaler Can Work Better for Images:** `MinMaxScaler` rescales each pixel feature to the `0` to `1` range, which is usually compatible with image intensity values and the default settings of `MLPClassifier`. `StandardScaler` forces every pixel feature to have unit variance, so low-variance background pixels can be amplified relative to their original variation. This may give unnecessary importance to pixels that contain little useful visual information.

- What make neural networks overconfident even on wrong predictions?

- how do we choose how many layers to reuse in transfer learning?
