# Chapter 3: Classification
## 💡 Key Takeaways

- **Classification Fundamentals:** Differentiating between categorical predictions and continuous value regression using the MNIST dataset as the foundational baseline.
- **Training Dynamics:** Utilizing Stochastic Gradient Descent (SGD) which processes training instances independently making it highly suitable for online learning systems.
- **Multiclass Strategies:** Understanding how algorithms handle multiple classes. OvO trains multiple binary classifiers on small data subsets which is ideal for algorithms that scale poorly like SVM. OvR trains one model per class and serves as the default for most other algorithms.
- **Advanced Classification:** Expanding beyond single targets to Multilabel classification (outputting multiple binary tags) and Multioutput classification (predicting multiple multiclass labels such as denoising individual image pixels).
- **Establishing Baselines:** Implementing Dummy Classifiers to set a performance floor ensuring the trained model is genuinely learning patterns rather than simply guessing the most frequent class.

## 🧠 Self-Reflection & Insights

- **The Illusion of Accuracy:** Accuracy is fundamentally flawed when dealing with skewed datasets. A model predicting "not 5" on MNIST easily achieves 90 percent accuracy. True performance evaluation strictly requires the Confusion Matrix, Precision, and Recall.
- **Navigating the Precision Recall Trade off:** Increasing the decision threshold inherently boosts precision while degrading recall. The optimal balance depends entirely on the business context whether prioritizing high precision (safe content filtering) or high recall (fraud detection). The F1 score offers a harmonic mean for convenient model comparison.
- **Choosing the Right Evaluation Curve:** While the ROC curve plots the True Positive Rate against the False Positive Rate it can be dangerously misleading if the negative class is massive. The PR curve is the objectively superior choice whenever the positive class is rare or when false positives carry more weight than false negatives.
- **Mathematical Stability via Scaling:** Applying StandardScaler to pixel brightness drastically improves the mathematical convergence of gradient descent algorithms allowing linear models to find optimal solutions faster.
- **Preventing Error Propagation in Classifier Chains:** When building multilabel models using ClassifierChain training with true labels creates a mismatch with testing conditions. Using cross validation predictions during training forces the model to learn how to handle its own errors leading to significantly better real world generalization.
- **Probability Calibration:** A model's raw output scores might be overconfident. Utilizing tools like CalibratedClassifierCV ensures these estimated probabilities closely reflect actual real world likelihoods.
- **Robustness through Data Augmentation:** Artificially shifting or rotating training images forces the model to learn spatial tolerance reducing overfitting and improving performance on poorly centered data.
