# Embeddings in Part 2 (MLP Context Model) — with a full **context** example

This note explains what embeddings are, why they are used in Makemore Part 2 (the MLP model), and what expressions like `C[X]` or `C[Xb]` mean **when the input is a context window** (e.g., 3 characters) and the output is the **next** character (Karpathy’s setup).

---

## 1) What is an embedding?

An embedding is a learned mapping from a discrete symbol (character/token ID) to a continuous vector.

For vocabulary size $V$ and embedding dimension $d$:

$$
C \in \mathbb{R}^{V \times d}
$$

- Row `C[i]` is the embedding vector for token ID `i`.

---

## 2) Why embeddings instead of one-hot vectors?

A one-hot encoding represents a token as a vector $\mathbf{e}_x \in \{0,1\}^V$ with a single 1.

Limitations of one-hot:
- Very high dimensional (size $V$)
- No notion of similarity (all different tokens are equally “far”)
- Inefficient compared to a dense lookup

Embeddings solve this by learning a dense vector of size $d \ll V$.

---

## 3) What does $\mathbb{R}^{V \times d}$ mean? (numerical example)

Example vocabulary (5 tokens):

```text
vocab = ['@', 'a', 'b', 'c', '.']
ids:      0    1    2    3    4
```

Here:
- $V = 5$
- choose $d = 2$

A concrete embedding matrix $C \in \mathbb{R}^{5 \times 2}$:

$$
C =
\begin{bmatrix}
 0.10 & -0.20 \\
 0.00 &  0.30 \\
-0.40 &  0.20 \\
 0.50 & -0.10 \\
-0.20 &  0.40
\end{bmatrix}
$$

---

## 4) What does `C[x]` mean numerically? (single token)

If $x = 3$ (token `'c'`), then:

$$
C[3] = [0.50, -0.10]
$$

---

## 5) Batched lookup: `C[X]`

If you have a batch of token IDs:

$$
X \in \{0,\dots,V-1\}^{N}
$$

then `C[X]` returns a matrix of shape $(N, d)$.

---

## 6) Context window lookup (the key idea): `C[Xb]` with $B=3$

In Part 2, each training example is a **context window** of length $B$
and the target is the **next** character.

Input:
$$
X_b \in \{0,\dots,V-1\}^{N \times B}
$$

Target:
$$
Y \in \{0,\dots,V-1\}^{N}
$$

Embedding lookup:

- `C`: $(V, d)$
- `X_b`: $(N, B)$
- `E = C[X_b]`: $(N, B, d)$

---

## 7) Full numerical context example ($B=3$ → predict next char)

Vocabulary and embedding table are the same as above.

### Step A — Context batch

```text
Xb =
[[0, 1, 2],     # ['@', 'a', 'b']
 [3, 4, 1]]     # ['c', '.', 'a']
```

Targets:

```text
Y = [3, 2]      # ['c', 'b']
```

### Step B — Embedding lookup

$$
E = C[X_b] =
\begin{bmatrix}
\begin{bmatrix}
 C[0] \\
 C[1] \\
 C[2]
\end{bmatrix}
\\
\begin{bmatrix}
 C[3] \\
 C[4] \\
 C[1]
\end{bmatrix}
\end{bmatrix}
$$

Numerically:

$$
E =
\begin{bmatrix}
\begin{bmatrix}
 0.10 & -0.20 \\
 0.00 &  0.30 \\
-0.40 &  0.20
\end{bmatrix}
\\
\begin{bmatrix}
 0.50 & -0.10 \\
-0.20 &  0.40 \\
 0.00 &  0.30
\end{bmatrix}
\end{bmatrix}
$$

Shape: $(N, B, d) = (2, 3, 2)$.

### Step C — Concatenation

Flatten the context embeddings:

$$
x = \text{reshape}(E) \in \mathbb{R}^{N \times (B d)}
$$

Here $B d = 6$:

$$
x =
\begin{bmatrix}
 0.10 & -0.20 & 0.00 & 0.30 & -0.40 & 0.20 \\
 0.50 & -0.10 & -0.20 & 0.40 & 0.00 & 0.30
\end{bmatrix}
$$

---

## 8) MLP to logits for the next character

Hidden layer:

$$
h = \tanh(x W_1 + b_1)
$$

Output logits:

$$
\text{logits} = h W_2 + b_2
$$

Where:

- $W_1 \in \mathbb{R}^{(B d) \times H}$
- $W_2 \in \mathbb{R}^{H \times V}$

Logits shape:

$$
\text{logits} \in \mathbb{R}^{N \times V}
$$

Softmax converts logits to probabilities, and cross-entropy compares them with $Y$.

---

## 9) Why embedding lookup equals one-hot + linear layer

For a single token $x$:

- One-hot vector $\mathbf{e}_x \in \{0,1\}^V$
- Linear embedding:

$$
\mathbf{e}_x^T C = C[x]
$$

For a whole context:

$$
O \in \{0,1\}^{N \times B \times V}
$$

$$
E = O C \in \mathbb{R}^{N \times B \times d}
$$

This is exactly what `C[Xb]` computes efficiently.

---

## 10) Final mental picture

- `Xb`: context **IDs**
- `C[Xb]`: embeddings for each context position
- concatenate → one vector per example
- MLP → logits over vocabulary
- cross-entropy trains next-character prediction