# Embeddings (Makemore Part 2)

This note explains **what embeddings are** at a conceptual level and then shows **two concrete examples**:
1) embedding lookup (the Makemore way), and  
2) one-hot encoding + linear layer, showing they are **equivalent**.

The goal is clarity, not heavy math.

---

## 1) What are embeddings? (conceptual explanation)

In language models, characters (or words) are **discrete symbols**:
- `'a'`, `'b'`, `'c'`, `'.'`, etc.

Neural networks, however, work with **numbers**, not symbols.

An **embedding** is simply a learned table that maps each symbol ID to a small vector of numbers.

Think of it as a **lookup table**:
- input: an integer ID (e.g. `2`)
- output: a vector (e.g. `[-0.4, 0.2]`)

Why this is useful:
- Much smaller than one-hot vectors
- Learns similarity (similar characters can get similar vectors)
- Fast and efficient

Nothing magical is happening; we are just **replacing IDs with vectors** before feeding them to a neural network.

---

## 2) Example 1 — Embedding lookup (Makemore style)

### Vocabulary

```text
vocab = ['@', 'a', 'b', 'c', '.']
ids:      0    1    2    3    4
```

### Embedding table

Suppose we choose embedding dimension `d = 2`:

```text
C =
[
 [ 0.10, -0.20 ],   # '@'
 [ 0.00,  0.30 ],   # 'a'
 [-0.40,  0.20 ],   # 'b'
 [ 0.50, -0.10 ],   # 'c'
 [-0.20,  0.40 ]    # '.'
]
```

Each row is the vector representation of one character.

### Single-character lookup

If the input character is `'b'` (ID = 2):

```text
C[2] = [-0.40, 0.20]
```

That’s the embedding.

### Context example (like Makemore)

Suppose block size `B = 3` and the context is:

```text
['@', 'a', 'b'] → IDs [0, 1, 2]
```

Embedding lookup:

```text
C[[0, 1, 2]] =
[
 [ 0.10, -0.20 ],   # '@'
 [ 0.00,  0.30 ],   # 'a'
 [-0.40,  0.20 ]    # 'b'
]
```

Then Makemore **concatenates** these vectors:

```text
[ 0.10, -0.20, 0.00, 0.30, -0.40, 0.20 ]
```

This single vector is what goes into the MLP to predict the **next** character.

### PyTorch version

```python
import torch

C = torch.tensor([
    [ 0.10, -0.20],
    [ 0.00,  0.30],
    [-0.40,  0.20],
    [ 0.50, -0.10],
    [-0.20,  0.40]
])

Xb = torch.tensor([0, 1, 2])   # ['@', 'a', 'b']

emb = C[Xb]        # shape: (3, 2)
x   = emb.view(-1) # shape: (6,)
```

---

## 3) Example 2 — One-hot encoding (equivalent idea)

Now let’s do the **same thing** using one-hot vectors.

### One-hot vectors

Vocabulary size = 5.

```text
'@' → [1, 0, 0, 0, 0]
'a' → [0, 1, 0, 0, 0]
'b' → [0, 0, 1, 0, 0]
'c' → [0, 0, 0, 1, 0]
'.' → [0, 0, 0, 0, 1]
```

### Linear layer weights

Use the **same numbers** as the embedding table:

```text
W =
[
 [ 0.10, -0.20 ],
 [ 0.00,  0.30 ],
 [-0.40,  0.20 ],
 [ 0.50, -0.10 ],
 [-0.20,  0.40 ]
]
```

### Multiply one-hot by `W`

For `'b'` (ID = 2):

```text
[0, 0, 1, 0, 0] @ W = [-0.40, 0.20]
```

This is **exactly the same** vector as `C[2]`.

### Context example

Context `['@', 'a', 'b']`:

```text
one-hot('@') @ W → [ 0.10, -0.20 ]
one-hot('a') @ W → [ 0.00,  0.30 ]
one-hot('b') @ W → [-0.40,  0.20 ]
```

Concatenate → same input vector as before.

### PyTorch version (one-hot)

```python
import torch
import torch.nn.functional as F

W = torch.tensor([
    [ 0.10, -0.20],
    [ 0.00,  0.30],
    [-0.40,  0.20],
    [ 0.50, -0.10],
    [-0.20,  0.40]
])

Xb = torch.tensor([0, 1, 2])        # IDs
one_hot = F.one_hot(Xb, num_classes=5).float()

emb = one_hot @ W                  # shape: (3, 2)
x   = emb.view(-1)                  # shape: (6,)
```

---

## 4) Key takeaway

- **Embeddings are just learned lookup tables**
- `C[x]` is mathematically the same as `one_hot(x) @ W`
- We use embeddings because they are:
  - simpler
  - faster
  - more memory efficient

Embedding lookup = one-hot encoding + linear layer (done efficiently).