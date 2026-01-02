# Embeddings in Part 2 (MLP Context Model)

This note explains what embeddings are, why they are used in Part 2 of makemore, and what expressions like `C[X]` or `C[Xb]` mean mathematically and computationally.

---

## 1. What is an embedding?

An embedding is a learned mapping from a discrete symbol (such as a character index) to a continuous vector.

For a vocabulary of size $V$ and embedding dimension $d$, the embedding matrix is:

$$
C \in \mathbb{R}^{V \times d}
$$

Each row of $C$ corresponds to one token.  
For a token index $x \in \{0, \dots, V-1\}$, the embedding lookup is:

$$
\text{embedding}(x) = C[x]
$$

---

## 2. Why embeddings instead of one-hot vectors?

A one-hot encoding represents a token as a vector:

$$
\mathbf{x} \in \{0,1\}^V
$$

with exactly one non-zero entry.

Limitations of one-hot encodings:
- Very high dimensional
- No notion of similarity between tokens
- Inefficient for learning structure

Embeddings solve this by mapping tokens into a dense, low-dimensional space where $d \ll V$.

---

## 3. What does `C[X]` mean?

In PyTorch, indexing a tensor performs row selection.

If:
- $C \in \mathbb{R}^{V \times d}$
- $X \in \mathbb{Z}^{N}$

then:

```python
emb = C[X]
```

produces a tensor:

$$
\text{emb} \in \mathbb{R}^{N \times d}
$$

Each element of `X` selects one row of `C`.

---

## 4. Batched context lookup: `C[Xb]`

In Part 2, inputs are contexts:

$$
X_b \in \mathbb{Z}^{B \times T}
$$

where:
- $B$ is the batch size
- $T$ is the context length

Then:

```python
emb = C[Xb]
```

produces:

$$
\text{emb} \in \mathbb{R}^{B \times T \times d}
$$

This is a vectorized gather operation.

---

## 5. Mathematical interpretation of embedding lookup

Embedding lookup is equivalent to applying a linear layer to a one-hot vector.

For a token $x$:
1. One-hot encode: $\mathbf{e}_x \in \{0,1\}^V$
2. Apply a linear transformation: $\mathbf{e}_x^\top C = C[x]$

Thus, embedding lookup is equivalent to a linear layer with no bias applied to a one-hot input.

---

## 6. Explicit equivalent without indexing

The same operation can be written explicitly as:

```python
x_onehot = F.one_hot(X, num_classes=V).float()  # (N, V)
emb = x_onehot @ C                              # (N, d)
```

This is mathematically identical to `C[X]`, but slower and more memory-intensive.

---

## 7. Embeddings are learnable parameters

The embedding matrix $C$:
- is initialized randomly
- participates in backpropagation
- is updated via gradient descent

There is nothing special about embeddings mathematically; they are ordinary model parameters.

---

## 8. Why embeddings matter in language models

Embeddings allow models to learn similarity relationships between tokens and to represent discrete symbols in a continuous space that neural networks can exploit.

As models scale, embeddings become richer, while the loss function remains the same.

---

## 9. Mental model to remember

An embedding layer is a linear layer applied to a one-hot vector, implemented efficiently via indexing.

Pipeline:

```text
token indices
   ↓
embedding lookup (C[X])
   ↓
concatenation / flattening
   ↓
MLP
   ↓
logits
```