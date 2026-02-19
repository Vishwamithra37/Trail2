# PyTorch Guide: Concepts You Know → Code

You already understand neural networks, hidden layers, curve fitting, and embeddings.
This guide maps those concepts directly to PyTorch.

---

## 1. Tensors — The Universal Data Container

Everything in PyTorch is a **tensor**. Think of it as a numpy array that can live on a GPU and track gradients.

```python
import torch

# Scalars, vectors, matrices, higher-dimensional arrays — all tensors
scalar = torch.tensor(3.14)                  # 0-D  — just a number (like a Python float)
vector = torch.tensor([1.0, 2.0, 3.0])      # 1-D  — 1 row, 3 columns
matrix = torch.tensor([[1, 2], [3, 4]])      # 2-D  — 2 rows, 2 columns
batch  = torch.randn(32, 784)               # 2-D  — 32 rows, 784 columns
```

**`randn` value range:** Values come from a standard normal distribution (mean=0, std=1).
~68% fall in [-1, +1], ~95% in [-2, +2], ~99.7% in [-3, +3]. Technically ±infinity but extremely rare beyond ±3.

```python
# Other random generators:
torch.rand(32, 784)              # uniform between 0 and 1
torch.randint(0, 100, (32, 784)) # random integers between 0 and 99
torch.randn(32, 784) * 5 + 10   # normal distribution, mean=10, std=5
```

**Key operations you'll use constantly:**

```python
x.shape          # dimensions of the tensor
x.dtype          # data type (float32, int64, etc.)
x.to('cuda')     # move to GPU
x.requires_grad  # is autograd tracking this tensor?
```

| Attribute       | Primary use                  | In real model code?                          |
|-----------------|------------------------------|----------------------------------------------|
| `x.shape`       | Debugging shape mismatches   | Occasionally (reshaping, dynamic sizes)      |
| `x.dtype`       | Debugging type mismatches    | Rarely (mixed precision, casting)            |

Both are **properties, not methods** — no parentheses, no cost to access. Sprinkle `print(x.shape)` liberally when building a new model, then remove them once things work.

---

## 2. The Training Loop — Curve Fitting in PyTorch

You know curve fitting: you have data, a model with parameters, a loss function, and you adjust parameters to minimize loss. PyTorch does exactly this:

```python
# THE CORE LOOP — memorize this pattern
for epoch in range(num_epochs):
    # 1. Forward pass: feed data through model, get predictions
    predictions = model(inputs)

    # 2. Compute loss: how wrong are we?
    loss = loss_fn(predictions, targets)

    # 3. Backward pass: compute gradients (∂loss/∂each_parameter)
    loss.backward()

    # 4. Update parameters: nudge them in the direction that reduces loss
    optimizer.step()

    # 5. Zero gradients: reset for next iteration (PyTorch accumulates by default)
    optimizer.zero_grad()
```

| Concept you know          | PyTorch equivalent                          |
|---------------------------|---------------------------------------------|
| Model/function            | `model = nn.Module` subclass                |
| Parameters (weights)      | `nn.Parameter`, auto-created by layers      |
| Loss function             | `nn.MSELoss()`, `nn.CrossEntropyLoss()` etc |
| Gradient descent          | `torch.optim.SGD`, `torch.optim.Adam` etc   |
| Computing gradients       | `loss.backward()` (autograd does the math)  |
| Learning rate             | `optim.Adam(model.parameters(), lr=0.001)`  |

---

## 3. Autograd — Automatic Differentiation

This is PyTorch's killer feature. You don't manually compute gradients. PyTorch records every operation on tensors and builds a computation graph, then walks it backwards to compute all gradients at once.

```python
x = torch.tensor(2.0, requires_grad=True)  # "track this variable"
y = x ** 3 + 2 * x                          # y = x³ + 2x
y.backward()                                 # compute dy/dx
print(x.grad)                                # 14.0  (3x² + 2 = 3(4) + 2)
```

**When you call `loss.backward()`**, it computes ∂loss/∂w for EVERY parameter `w` in your model. The optimizer then uses these gradients to update the weights. That's it. That's backpropagation in PyTorch.

---

## 4. nn.Module — Building Networks

Every neural network in PyTorch is a class that inherits from `nn.Module`. You define layers in `__init__` and wire them together in `forward`.

### A Simple FFNN (Feedforward Neural Network)

```python
import torch.nn as nn

class SimpleFFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Define layers (each has its own weights and biases)
        self.layer1 = nn.Linear(input_size, hidden_size)   # input → hidden
        self.layer2 = nn.Linear(hidden_size, output_size)  # hidden → output
        self.relu = nn.ReLU()                               # activation function

    def forward(self, x):
        # Define how data flows through the layers
        x = self.layer1(x)   # linear transform: x @ W.T + b
        x = self.relu(x)     # activation: max(0, x)
        x = self.layer2(x)   # linear transform to output
        return x

# Create model
model = SimpleFFNN(input_size=784, hidden_size=128, output_size=10)
```

**What `nn.Linear(in, out)` actually does:**
- Creates a weight matrix W of shape `[out, in]` and a bias vector b of shape `[out]`
- On `forward`: computes `x @ W.T + b`
- That's it — it's one layer of your neural network

### Concept mapping:

| Concept                   | PyTorch                                      |
|---------------------------|----------------------------------------------|
| A layer                   | `nn.Linear(in_features, out_features)`       |
| Activation function       | `nn.ReLU()`, `nn.Sigmoid()`, `nn.Tanh()`    |
| Stack of layers           | Define multiple `nn.Linear` in `__init__`    |
| Forward pass              | `def forward(self, x)` — you define the flow |
| All model parameters      | `model.parameters()` — iterator over weights |

---

## 5. nn.Sequential — Quick Way to Stack Layers

If your network is just "layer → activation → layer → activation → ...", use `nn.Sequential` instead of writing a full class:

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Same as building a class, but less boilerplate
output = model(input_tensor)
```

Use `nn.Module` subclass when you need custom logic (skip connections, branching, etc.). Use `nn.Sequential` for simple stacks.

---

## 6. Loss Functions — Measuring Error

| Task                      | Loss function                | What it measures                    |
|---------------------------|------------------------------|-------------------------------------|
| Regression (predict a #)  | `nn.MSELoss()`               | Mean squared error                  |
| Binary classification     | `nn.BCEWithLogitsLoss()`     | Binary cross-entropy                |
| Multi-class classification| `nn.CrossEntropyLoss()`      | Cross-entropy (includes softmax!)   |

```python
loss_fn = nn.CrossEntropyLoss()

predictions = model(inputs)         # raw scores (logits), shape [batch, classes]
loss = loss_fn(predictions, labels) # labels are class indices, shape [batch]
```

**Important:** `nn.CrossEntropyLoss` applies softmax internally. Do NOT put softmax in your model's `forward` if you use this loss.

---

## 7. Optimizers — Updating Weights

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

The optimizer holds a reference to your model's parameters. After `loss.backward()` fills in the `.grad` for each parameter, `optimizer.step()` updates them.

| Optimizer          | When to use                                |
|--------------------|--------------------------------------------|
| `optim.SGD`        | Simple, good baseline                      |
| `optim.Adam`       | Good default for most tasks                |
| `optim.AdamW`      | Adam + proper weight decay (use for LLMs)  |

---

## 8. DataLoader — Batching Your Data

You don't feed the entire dataset at once. You split it into **batches**.

```python
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(input_tensors, label_tensors)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_inputs, batch_labels in loader:
    predictions = model(batch_inputs)
    loss = loss_fn(predictions, batch_labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

- `TensorDataset` wraps your tensors into a dataset
- `DataLoader` handles batching, shuffling, and iteration
- `batch_size=32` means 32 samples per forward pass

---

## 9. Embeddings — Looking Up Vectors

You know embeddings: mapping discrete tokens (words, IDs) to dense vectors. In PyTorch:

```python
embed = nn.Embedding(num_embeddings=10000, embedding_dim=256)
# This creates a lookup table of shape [10000, 256]
# 10000 vocabulary items, each represented as a 256-dimensional vector

token_ids = torch.tensor([42, 7, 1337])  # 3 token IDs
vectors = embed(token_ids)                # shape: [3, 256]
```

- Input: integer IDs → Output: dense vectors
- The embedding vectors are **learnable parameters** — they get updated during training
- This is literally a matrix where row `i` is the vector for token `i`

---

## 10. Shapes — The #1 Source of Bugs

Most PyTorch bugs are shape mismatches. Always track shapes mentally:

```python
batch = torch.randn(32, 784)        # 32 samples, 784 features
# Pass through nn.Linear(784, 256)
# Output: [32, 256]                  # 32 samples, 256 features

# Pass through nn.Linear(256, 10)
# Output: [32, 10]                   # 32 samples, 10 class scores
```

**Rule:** `nn.Linear(in, out)` takes `[batch, in]` and returns `[batch, out]`.

Use `x.shape` liberally while debugging. Print it after every layer if confused.

---

## 11. GPU — Moving Things to CUDA

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)   # move model parameters to GPU
inputs = inputs.to(device)  # move data to GPU
labels = labels.to(device)

# Now training runs on GPU — same code, just faster
```

**Rule:** Model and data must be on the same device. You'll get an error if they're not.

---

## 12. Train vs Eval Mode

```python
model.train()  # enable dropout, batch norm updates — use during training
model.eval()   # disable dropout, freeze batch norm — use during inference

with torch.no_grad():       # don't track gradients — saves memory during inference
    predictions = model(test_inputs)
```

---

## 13. Transformers — The Architecture

You said you understand transformers to an extent. Here's how the pieces map to PyTorch.

### 13.1 The Big Picture

A transformer processes **sequences** (e.g., sentences). The core insight: instead of processing tokens one-by-one (like RNNs), transformers let every token look at every other token simultaneously via **self-attention**.

```
Input tokens → Embedding → [Transformer Block × N] → Output
```

Each Transformer Block contains:
1. **Self-Attention** — "which other tokens should I pay attention to?"
2. **Feed-Forward Network** — a plain FFNN applied to each token independently
3. **Layer Norm + Residual connections** — for stable training

### 13.2 Self-Attention: The Core Mechanism

For each token, we compute three vectors from its embedding:
- **Q (Query):** "What am I looking for?"
- **K (Key):** "What do I contain?"
- **V (Value):** "What information do I provide?"

```python
# Attention(Q, K, V) = softmax(Q @ K.T / √d_k) @ V

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)  # learns to produce Q
        self.key   = nn.Linear(embed_dim, embed_dim)  # learns to produce K
        self.value = nn.Linear(embed_dim, embed_dim)  # learns to produce V
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        # x shape: [batch, seq_len, embed_dim]
        Q = self.query(x)   # [batch, seq_len, embed_dim]
        K = self.key(x)     # [batch, seq_len, embed_dim]
        V = self.value(x)   # [batch, seq_len, embed_dim]

        # Attention scores: how much should each token attend to every other?
        scores = (Q @ K.transpose(-2, -1)) / self.scale  # [batch, seq_len, seq_len]
        weights = torch.softmax(scores, dim=-1)           # normalize to probabilities

        # Weighted sum of values
        output = weights @ V  # [batch, seq_len, embed_dim]
        return output
```

**Intuition:** The `scores` matrix is `[seq_len, seq_len]` — entry `[i][j]` says "how much should token `i` attend to token `j`." After softmax, these become weights. Each token's output is a weighted average of all token Values.

### 13.3 Multi-Head Attention

Instead of one set of Q, K, V, we use multiple "heads" that each learn different attention patterns (one head might learn syntax, another might learn coreference, etc.):

```python
# PyTorch provides this built-in:
attn = nn.MultiheadAttention(embed_dim=256, num_heads=8)

# Under the hood: splits embed_dim into 8 heads of 32 dims each,
# runs attention independently, then concatenates and projects back.

output, attention_weights = attn(query, key, value)
```

### 13.4 Positional Encoding

Attention is order-agnostic (it treats "dog bit man" same as "man bit dog"). Positional encodings add position information:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        self.register_buffer('pe', pe.unsqueeze(0))    # [1, max_len, embed_dim]

    def forward(self, x):
        # x shape: [batch, seq_len, embed_dim]
        return x + self.pe[:, :x.size(1)]  # add position info to embeddings
```

`register_buffer` stores the tensor on the model (moves to GPU with model) but doesn't treat it as a trainable parameter.

### 13.5 A Full Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual connection and layer norm
        attn_out, _ = self.attention(x, x, x)   # Q=K=V=x (self-attention)
        x = self.norm1(x + self.dropout(attn_out))  # residual + normalize

        # Feed-forward with residual connection and layer norm
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))     # residual + normalize
        return x
```

**Why residual connections (`x + ...`)?** Without them, gradients vanish in deep networks. The skip connection gives gradients a direct path backwards.

**Why LayerNorm?** Normalizes activations to prevent them from exploding or vanishing. Stabilizes training.

### 13.6 Causal Masking (for GPT-style models)

In language models, token 5 should only attend to tokens 1-5, not future tokens. This is done with a **causal mask**:

```python
# Create an upper-triangular mask (True = "block this position")
seq_len = 10
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
# mask[i][j] = True if j > i (can't attend to future)

attn_out, _ = self.attention(x, x, x, attn_mask=mask)
```

### 13.7 Encoder vs Decoder

| Component    | Attention type    | Use case                        | Example       |
|-------------|-------------------|---------------------------------|---------------|
| **Encoder** | Full (see all)    | Understanding/classification    | BERT          |
| **Decoder** | Causal (past only)| Generation (next token)         | GPT           |
| **Both**    | Cross-attention   | Seq-to-seq (translation, etc.)  | Original T5   |

---

## 14. Putting It All Together — Complete FFNN Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- 1. Create dummy data ---
X = torch.randn(1000, 20)          # 1000 samples, 20 features
y = (X[:, 0] + X[:, 1] > 0).long() # binary label based on first 2 features

# --- 2. DataLoader ---
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# --- 3. Define model ---
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 2)    # 2 classes
)

# --- 4. Loss and optimizer ---
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 5. Training loop ---
for epoch in range(20):
    total_loss = 0
    for batch_x, batch_y in loader:
        predictions = model(batch_x)
        loss = loss_fn(predictions, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

# --- 6. Inference ---
model.eval()
with torch.no_grad():
    test_input = torch.randn(5, 20)
    output = model(test_input)
    predicted_classes = output.argmax(dim=1)
    print(f"Predictions: {predicted_classes}")
```

---

## Quick Reference Cheat Sheet

| You want to...                  | PyTorch code                                    |
|---------------------------------|-------------------------------------------------|
| Create a layer                  | `nn.Linear(in, out)`                            |
| Activation function             | `nn.ReLU()`, `torch.relu(x)`                    |
| Stack layers simply             | `nn.Sequential(...)`                             |
| Custom network                  | Subclass `nn.Module`                             |
| Embedding lookup                | `nn.Embedding(vocab_size, dim)`                  |
| Compute loss                    | `loss_fn(predictions, targets)`                  |
| Compute gradients               | `loss.backward()`                                |
| Update weights                  | `optimizer.step()`                               |
| Clear gradients                 | `optimizer.zero_grad()`                          |
| Batch your data                 | `DataLoader(dataset, batch_size=32)`             |
| Move to GPU                     | `.to('cuda')`                                    |
| Save model                      | `torch.save(model.state_dict(), 'model.pth')`   |
| Load model                      | `model.load_state_dict(torch.load('model.pth'))` |
| Self-attention                  | `nn.MultiheadAttention(embed_dim, num_heads)`    |
| Transformer block               | `nn.TransformerEncoderLayer(d_model, nhead)`     |
| Positional encoding             | Add sin/cos vectors or use learned embeddings    |
| Causal mask                     | `torch.triu(ones, diagonal=1).bool()`            |
