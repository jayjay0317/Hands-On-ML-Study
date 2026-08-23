# Chapter 13: Processing Sequences Using RNNs and CNNs

## 💡 Key Takeaways

## 🧠 Self-Reflection & Insights

- **Why RNNs Can Handle Different Sequence Lengths:** An RNN has a fixed `input_size`, meaning each time step must contain the same number of features. However, the same recurrent weights are reused at every time step, so the network can process sequences with different numbers of time steps. In other words, the feature dimension is fixed, while the sequence length can vary.

- **RNN Memory:** Since each output depends on the previous output, the output at time $t$ indirectly depends on all earlier inputs, giving the network a form of memory.

- **Memory Cells and Hidden State:** An RNN preserves information through a hidden state $h_{(t)}$, which is updated from the current input $x_{(t)}$ and the previous state $h_{(t-1)}$. In a basic RNN cell, the hidden state is also used directly as the output.
