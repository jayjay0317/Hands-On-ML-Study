# Chapter 12: Deep Computer Vision Using Convolutional Neural Networks

## 💡 Key Takeaways

## 🧠 Self-Reflection & Insights

- **Filters and Feature Maps:** One filter is applied across all spatial locations to produce one 2D feature map. Each position in that feature map is the output of one neuron. A convolutional layer contains multiple filters, so it outputs multiple stacked feature maps. Neurons within the same feature map share the same filter weights and bias.

- **Filters Across Input Channels:** Each neuron in an output feature map applies one shared filter to a local receptive field across all feature maps from the previous layer. The filter contains a separate weight for each spatial position within the receptive field and for each input feature map. All neurons in the same output feature map share this entire filter.

- **Weights in a Convolutional Neuron:** Each neuron in an output feature map uses a filter with $f_h \times f_w \times f_{n'}$ weights, where $f_h$ and $f_w$ are the kernel dimensions and $f_{n'}$ is the number of input feature maps. Neurons in the same output feature map share this filter and differ only in the input region they process.

Depthwise max pooling takes the maximum activation across several feature maps at the same spatial location, allowing the network to ignore which specific feature variant, such as a rotation, produced the strongest response.

- **Typical CNN Architecture:** CNNs usually stack convolutional layers and activation functions, followed by pooling layers. As the spatial dimensions decrease, the number of feature maps often increases. Near the output, the feature maps are flattened and passed to fully connected layers for classification.

- **Using `functools.partial()`:** `functools.partial()` creates a new callable with selected arguments already fixed. It is useful for reducing repeated code when the same function or class is called many times with shared default arguments.

- **Data Augmentation:** Data augmentation reduces overfitting by creating realistic variations of training images, helping CNNs become more robust to changes in position, orientation, size, and lighting.

- **GoogLeNet and Inception Modules:** Inception modules process the same input through multiple parallel branches with different kernel sizes, then concatenate the resulting feature maps. `1 × 1` convolutions act as bottlenecks that reduce computational cost while learning relationships across channels.

- **Why Inception Modules Use Parallel Branches:** A sequential stack of convolutional layers transforms features through a single path, with each layer processing only the previous layer’s output. In contrast, an inception module applies different operations to the same input in parallel, such as `1 × 1`, `3 × 3`, and `5 × 5` convolutions and max pooling. This allows the network to capture features at multiple spatial scales, then concatenate them along the channel dimension. `1 × 1` convolutions are also used as bottlenecks to reduce the number of channels before more expensive convolutions, limiting the computational cost.

- How does skipping a random set of residual units not compromise accuracy while speeding up training? Does it have a regularization effect like dropout?

- A residual unit splits the input into two paths:
  - The residual branch learns a transformation $F(x)$.
  - The shortcut branch passes the original input $x$ forward.  
  The two paths are added element-wise to produce $F(x) + x$.

- **Global Average Pooling:** Global average pooling replaces each feature map with the average of all its spatial activations, reducing an input of shape `[batch_size, channels, height, width]` to `[batch_size, channels, 1, 1]`. This greatly reduces the number of features and parameters before the output layer, lowers computational cost and overfitting risk, and emphasizes whether a feature is present rather than its exact location. It is therefore especially useful for image classification, where precise spatial information is often less important.
