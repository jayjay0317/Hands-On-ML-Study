# Chapter 12: Deep Computer Vision Using Convolutional Neural Networks

## 💡 Key Takeaways

- **Filters and Feature Maps:** One filter is applied across all spatial locations to produce one feature map. Each filter spans all input channels, and neurons within the same output feature map share the same filter weights and bias.

- **Typical CNN Architecture:** CNNs usually stack convolutional layers, activation functions, and pooling layers. As spatial dimensions decrease, the number of feature maps often increases, allowing deeper layers to learn higher-level features.

- **Global Average Pooling:** Global average pooling reduces each feature map to one average activation, greatly reducing the number of features and parameters before the output layer.

- **Data Augmentation:** Random transformations such as flips, rotations, crops, and color adjustments reduce overfitting by exposing the model to realistic variations of training images.

- **Modern CNN Architectures:** Inception modules capture features at multiple spatial scales using parallel branches, residual units use shortcut connections to learn $F(x) + x$, and EfficientNet scales depth, width, and input resolution together.

- **Transfer Learning:** A pretrained CNN can be adapted to a smaller target dataset by replacing its classification head, initially freezing the pretrained layers, and later fine-tuning the full model with a smaller learning rate.

- **Fully Convolutional Networks:** FCNs replace dense layers with convolutional layers, allowing the network to process different spatial input sizes and produce predictions across multiple image regions in a single forward pass.

- **Semantic and Instance Segmentation:** Semantic segmentation assigns a class to every pixel, while instance segmentation additionally distinguishes separate objects of the same class.

## 🧠 Self-Reflection & Insights

- **Why Shared Filters Matter:** Reusing the same filter across spatial locations allows CNNs to detect the same feature anywhere in an image while using far fewer parameters.

- **Why CNNs Downsample:** Strides and pooling reduce computation and increase receptive fields, allowing deeper layers to learn more abstract features while the number of channels often increases.

- **Why Inception Uses Parallel Branches:** Applying different operations to the same input in parallel allows the network to capture patterns at multiple spatial scales before combining them.

- **Why Segmentation Downsamples and Then Upsamples:** Downsampling helps the network understand **what** is present, while transposed convolutions and skip connections recover **where** it is by restoring spatial resolution and finer details.

- **Transfer Learning Depends on Similarity:** Pretrained features are most useful when the original and target datasets contain related visual patterns, since lower and intermediate layers can then transfer more effectively.

- **Memory–Computation Trade-Off in RevNets:** RevNets reduce memory usage by reconstructing intermediate activations during backpropagation instead of storing them, trading additional computation for lower memory usage.
