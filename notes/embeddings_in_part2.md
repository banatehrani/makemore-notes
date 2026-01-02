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

## 3. What does $\mathbb{R}^{V \times d}$ mean? (numerical example)

Consider a small vocabulary:

Vocab = # Embeddings in Part 2 (MLP Context Model)

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

## 3. What does $\mathbb{R}^{V \times d}$ mean? (numerical example)

Consider a small vocabulary:

vocab = ['.', 'a', 'b', 'c', 'd']

Here:
- Vocabulary size: $V = 5$
- Embedding dimension: $d = 4$

Then the embedding matrix has shape:

$$
C \in \mathbb{R}^{5 \times 4}
$$

A concrete numerical example of such a matrix is:

$$
C =
\begin{bmatrix}
 0.10 & -0.30 &  0.25 &  0.90 \\
 1.20 &  0.50 & -0.10 &  0.40 \\
-0.70 &  1.10 &  0.60 & -0.20 \\
 0.30 & -0.80 &  1.50 &  0.70 \\
-1.10 &  0.20 & -0.40 &  0.05
\end{bmatrix}
$$

Each row is the embedding vector for one token.

---

## 4. What does `C[x]` mean numerically?

If the input token index is: x = 3 # token 'c'

Then the embedding lookup selects row 3:

$$
C[3] = [0.30, -0.80, 1.50, 0.70]
$$

The discrete symbol `'c'` is now represented as a 4-dimensional real vector.

---

## 5. Batched lookup: `C[X]`

If we have a batch of token indices: 
then the embedding lookup selects row 3:

$$
C[3] = [0.30, -0.80, 1.50, 0.70]
$$

The discrete symbol `'c'` is now represented as a 4-dimensional real vector.

---

## 5. Batched lookup: `C[X]`

If we have a batch of token indices: X = [0, 4, 1]


then:

$$
C[X] =
\begin{bmatrix}
 0.10 & -0.30 &  0.25 &  0.90 \\
-1.10 &  0.20 & -0.40 &  0.05 \\
 1.20 &  0.50 & -0.10 &  0.40
\end{bmatrix}
$$

---

## 6. Context window lookup: `C[Xb]`

Suppose we use a context length $T = 3$ and batch size $B = 2$:

Xb = [[0, 1, 2],
[3, 4, 1]]


Then:

$$
C[Xb] \in \mathbb{R}^{2 \times 3 \times 4}
$$

---

## 7. Mathematical interpretation of embedding lookup

Embedding lookup is equivalent to applying a linear layer to a one-hot vector.

For a token $x$:
1. One-hot encode:
   $$
   \mathbf{e}_x \in \{0,1\}^V
   $$
2. Apply a linear transformation:
   $$
   \mathbf{e}_x^\top C = C[x]
   $$

---

## 8. Embeddings are learnable parameters

The embedding matrix $C$:
- is initialized randomly
- participates in backpropagation
- is updated via gradient descent

---

## 9. Mental model to remember

An embedding layer is a linear layer applied to a one-hot vector, implemented efficiently via indexing.