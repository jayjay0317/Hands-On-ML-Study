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
