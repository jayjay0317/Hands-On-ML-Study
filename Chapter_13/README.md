# Chapter 13: Processing Sequences Using RNNs and CNNs

## 💡 Key Takeaways

## 🧠 Self-Reflection & Insights

- **Why RNNs Can Handle Different Sequence Lengths:** An RNN has a fixed `input_size`, meaning each time step must contain the same number of features. However, the same recurrent weights are reused at every time step, so the network can process sequences with different numbers of time steps. In other words, the feature dimension is fixed, while the sequence length can vary.
