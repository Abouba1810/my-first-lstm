# My First LSTM: Character-Level Sequence Modeling with PyTorch

A clean, modular, and production-ready implementation of a Long Short-Term Memory (LSTM) network built from scratch using PyTorch. This project demonstrates foundational mastery of recurrent architectures, sequence modeling, and tracking complex tensor dimensions.

---

## Theoretical Background

Introduced by Sepp Hochreiter and Jürgen Schmidhuber (1997), the LSTM architecture effectively solves the vanishing gradient problem inherent in standard Recurrent Neural Networks (RNNs). It achieves this by maintaining a persistent Cell State ($c_t$) regulated by three distinct, learnable gating mechanisms:

* **Forget Gate ($f_t$):** Evaluates the current input and previous hidden state to decide what information to discard from the cell state.
* **Input Gate ($i_t$):** Identifies which new information should be integrated into the cell state.
* **Output Gate ($o_t$):** Computes the next hidden state ($h_t$), determining what context is passed forward.

### Core Mathematical Formulation
At each sequence step $t$, the cell transitions are governed by:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(c_t)$$

---

## Engineering Best Practices Applied

* **Strict Shape Tracking:** Every core operation includes inline code comments tracing tensor transitions across standard deep learning dimensions: `(B, T, C)`—Batch size, Sequence length, Embedding/Channel size.
* **Modular Pipeline:** Clean, decoupled separation between data processing, neural network architecture definitions, and the training evaluation loops.
* **Robust Error Prevention:** Implements dynamic dataset masking and boundary limits during batch generation to prevent index errors.

---

## Getting Started

### Prerequisites
Ensure you have Python 3.8+ and PyTorch installed on your system:
```bash
pip install torch numpy
```
# Installation & Setup
# Clone the repository
```bash
git clone [https://github.com/Abouba1810/my-first-lstm.git](https://github.com/Abouba1810/my-first-lstm.git)
cd my-first-lstm
```
# Running the Pipeline
```bash
python 'train and inference.py'
```
