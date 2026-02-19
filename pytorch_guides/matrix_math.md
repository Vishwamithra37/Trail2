# Matrix Math Quick Reference

## 1. Scalars, Vectors, Matrices

```
Scalar:  5                      → just a number (0-D)
Vector:  [1, 2, 3]             → a row of numbers (1-D)
Matrix:  [[1, 2],              → a grid of numbers (2-D)
          [3, 4]]
```

---

## 2. Basic Arithmetic (Element-wise)

These operate on matching elements. Both sides must have the same shape (or be broadcastable).

```python
A = [1, 2, 3]
B = [4, 5, 6]

A + B = [5, 7, 9]        # addition
A - B = [-3, -3, -3]     # subtraction
A * B = [4, 10, 18]      # element-wise multiply (Hadamard product)
A / B = [0.25, 0.4, 0.5] # element-wise division
```

**Scalar broadcast** — a scalar applies to every element:

```python
A = [1, 2, 3]
A * 2 = [2, 4, 6]
A + 10 = [11, 12, 13]
```

---

## 3. Dot Product (Vectors)

Multiply matching elements, then sum. Both vectors must have the **same length**.

```
[1, 2, 3] · [4, 5, 6] = 1×4 + 2×5 + 3×6 = 32
```

```python
torch.dot(a, b)   # only for 1-D vectors
# or
a @ b              # also works for 1-D vectors (returns scalar)
```

---

## 4. Matrix Multiplication (`@`)

### The Rule: (m × n) @ (n × p) → (m × p)

The **inner dimensions must match**. The result takes the outer dimensions.

```
(2×3) @ (3×4) → (2×4)  ✅  inner 3 matches
(2×3) @ (4×3) → error   ❌  inner 3 ≠ 4
```

### How it works — row × column dot products:

```
A = [[1, 2, 3],      B = [[7, 8],
     [4, 5, 6]]           [9, 10],
                           [11, 12]]

A is (2×3), B is (3×2) → result is (2×2)

result[0][0] = [1,2,3] · [7,9,11]  = 1×7 + 2×9 + 3×11  = 58
result[0][1] = [1,2,3] · [8,10,12] = 1×8 + 2×10 + 3×12 = 64
result[1][0] = [4,5,6] · [7,9,11]  = 4×7 + 5×9 + 6×11  = 139
result[1][1] = [4,5,6] · [8,10,12] = 4×8 + 5×10 + 6×12 = 154

A @ B = [[58,  64],
         [139, 154]]
```

```python
result = A @ B
result = torch.matmul(A, B)   # identical
```

---

## 5. Transpose

Flip rows and columns. Shape `(m × n)` becomes `(n × m)`.

```
A = [[1, 2, 3],        A.T = [[1, 4],
     [4, 5, 6]]               [2, 5],
                               [3, 6]]
(2×3)                   (3×2)
```

```python
A.T                    # shorthand
A.transpose(0, 1)     # explicit dims
```

**Why this matters:** `nn.Linear` stores weights as `(out, in)` and computes `x @ W.T + b` — the transpose makes the shapes line up.

---

## 6. Shape Compatibility Cheat Sheet

| Operation            | Requirement                          | Result shape |
|----------------------|--------------------------------------|--------------|
| `A + B`, `A * B`     | Same shape (or broadcastable)        | Same shape   |
| `A @ B`              | A is (m×**n**), B is (**n**×p)       | (m × p)      |
| `torch.dot(a, b)`   | Both 1-D, same length               | scalar       |
| `A.T`               | A is (m×n)                           | (n × m)      |

---

## 7. Batched Matrix Multiply

In neural networks, you work with **batches**. PyTorch handles the batch dimension automatically:

```python
# 32 samples, each a (10×5) matrix
A = torch.randn(32, 10, 5)
B = torch.randn(32, 5, 3)

result = A @ B   # shape: (32, 10, 3)
# Does 32 independent (10×5) @ (5×3) multiplications
```

---

## 8. Common PyTorch Operations

```python
# Element-wise
A + B                     # add
A * B                     # element-wise multiply
A ** 2                    # square every element

# Matrix
A @ B                     # matrix multiply
A.T                       # transpose

# Reductions
x.sum()                   # sum all elements → scalar
x.sum(dim=0)              # sum along rows → collapse to 1 row
x.sum(dim=1)              # sum along columns → collapse to 1 column
x.mean()                  # average all elements
x.max()                   # max element

# Shape manipulation
x.view(32, -1)            # reshape (-1 = infer this dim)
x.unsqueeze(0)            # add a dim: [3] → [1, 3]
x.squeeze()               # remove dims of size 1: [1, 3, 1] → [3]
```

---

## 9. Neural Network Context

Every `nn.Linear(in, out)` is just:

```
output = input @ W.T + b
```

Where:
- `input` is `(batch, in)`
- `W` is `(out, in)` — stored transposed for efficiency
- `b` is `(out)` — broadcast-added to every sample
- `output` is `(batch, out)`

That's it. A "deep neural network" is just this operation repeated with activations in between.
