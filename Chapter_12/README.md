# Chapter 12: Deep Computer Vision Using Convolutional Neural Networks

## 💡 Key Takeaways

## 🧠 Self-Reflection & Insights

- **Filters and Feature Maps:** One filter is applied across all spatial locations to produce one 2D feature map. Each position in that feature map is the output of one neuron. A convolutional layer contains multiple filters, so it outputs multiple stacked feature maps. Neurons within the same feature map share the same filter weights and bias.

- **Filters Across Input Channels:** Each neuron in an output feature map applies one shared filter to a local receptive field across all feature maps from the previous layer. The filter contains a separate weight for each spatial position within the receptive field and for each input feature map. All neurons in the same output feature map share this entire filter.
