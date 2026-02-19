# How `backward()` Actually Works

## It's NOT solving one big equation

A common misconception: you might think `backward()` takes something like `y = 3x² + 2x` and computes `dy/dx`. That's not what's happening.

Instead, your model is a **chain of simple operations**, and PyTorch records every one of them into a computation graph.

## The Computation Graph

When you write:

```python
x = input_data                  # [32, 784]
h = x @ W1 + b1                # linear layer 1
h = relu(h)                     # activation
out = h @ W2 + b2              # linear layer 2
loss = cross_entropy(out, y)   # scalar loss value
```

PyTorch builds this graph behind the scenes:

```
input → matmul(W1) → add(b1) → relu → matmul(W2) → add(b2) → cross_entropy → loss
```

## Walking the Graph Backwards (Chain Rule)

When you call `loss.backward()`, it walks this graph **backwards** using the chain rule:

```
∂loss/∂W2 = ∂loss/∂out  ×  ∂out/∂W2
∂loss/∂W1 = ∂loss/∂out  ×  ∂out/∂h  ×  ∂h/∂W1
```

Each individual derivative is simple (derivative of matmul, derivative of relu, etc.). The chain rule just multiplies them together along the path.

**That's why it works for EVERY parameter** — each parameter appears somewhere in the graph, and PyTorch traces the path from `loss` back to that parameter, multiplying derivatives along the way.

## Concrete Example

```python
# 2 parameters: W1 and W2
# Forward: loss = (x * W1 * W2 - target)²

x = torch.tensor(2.0)
W1 = torch.tensor(3.0, requires_grad=True)
W2 = torch.tensor(4.0, requires_grad=True)
target = torch.tensor(10.0)

out = x * W1 * W2              # 2 * 3 * 4 = 24
loss = (out - target) ** 2     # (24 - 10)² = 196

loss.backward()

print(W1.grad)  # ∂loss/∂W1 = 2(out-target) * x * W2 = 2(14) * 2 * 4 = 224
print(W2.grad)  # ∂loss/∂W2 = 2(out-target) * x * W1 = 2(14) * 2 * 3 = 168
```

One `backward()` call computed gradients for **both** W1 and W2 by tracing the graph back from `loss` to each of them.

A real model has millions of parameters, but the principle is identical — just a longer chain.

## The `@` Operator — Matrix Multiplication

In the examples above, `@` is Python's **matrix multiplication operator**. It's equivalent to `torch.matmul()`:

```python
# These are identical:
result = A @ B
result = torch.matmul(A, B)

# NOT the same as element-wise multiplication:
A * B      # multiplies matching elements: [1,2] * [3,4] = [3, 8]
A @ B      # matrix multiply: dot products of rows × columns
```

Quick comparison:

| Operator | What it does             | Example                              |
|----------|--------------------------|--------------------------------------|
| `*`      | Element-wise multiply    | `[1,2] * [3,4]` → `[3, 8]`         |
| `@`      | Matrix multiplication    | `[1,2] @ [3,4]` → `11` (dot product)|

In neural networks, every `nn.Linear` layer does `x @ W.T + b` under the hood — that's the core math of a layer.
