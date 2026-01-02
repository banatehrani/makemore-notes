# Cross-Entropy (from scratch)

This note explains what `torch.nn.functional.cross_entropy` computes, why it is used in language modeling (and multi-class classification in general), and how to reproduce it **from first principles**. We begin with the naive mathematical definition, show why it is numerically unsafe in practice, and then derive the **stable formulation** used by PyTorch and other deep-learning libraries.

The setting is character-level language modeling (as in *makemore*), but everything here applies to any multi-class classification problem.

---

## Step 1 — Logits vs probabilities

Neural networks usually output **logits**, not probabilities.

- Logits are unnormalized real-valued scores (one per class).
- They can be any real number (negative or positive, small or large).
- They do **not** sum to 1 and are not constrained to the interval \([0, 1]\).

Why logits are used:
- Optimization is easier in unconstrained real space.
- The network does not need to enforce probability constraints internally.
- Probabilities can be derived later in a controlled and stable way.

Notation used throughout this note:

- `logits`: shape `(N, C)`
  - `N` = number of examples
  - `C` = number of classes (e.g., vocabulary size)
- `ys`: shape `(N,)`
  - `ys[i]` is the integer index of the correct class for example `i`

For a single example, the logits vector is:


$\mathbf{z} = (z_0, z_1, \dots, z_{C-1})$

---

## Step 2 — Softmax: converting logits to probabilities

Softmax converts logits into a probability distribution:

$p_j = \frac{e^{z_j}}{\sum_{k=0}^{C-1} e^{z_k}}$

Properties:
- $\(p_j \ge 0\)$
- $\(\sum_j p_j = 1\)$

Softmax answers the question:

> “Given these scores, how likely is each class?”

---

## Step 3 — Negative log-likelihood (NLL)

For training, we only care about the probability assigned to the **correct** class.

If the true class index is `y`, the model assigns probability $\(p_y\)$.
The per-example loss is defined as:

$\ell = -\log(p_y)$

Why use the logarithm?
- It strongly penalizes confident wrong predictions.
- It turns products of probabilities into sums.
- It produces a smooth and well-behaved objective for optimization.

The dataset loss is the mean:

$\text{loss} = \frac{1}{N}\sum_{i=1}^{N} -\log(p_{y^{(i)}})$

This is the same objective used earlier in the count-based bigram model.

---

## Step 4 — Naive cross-entropy (explicit but fragile)

The following is a **direct translation of the mathematics** using only `exp`, `sum`, and `log`.

```python
import torch

# logits: (N, C)
# ys:     (N,)

exp_logits = torch.exp(logits)                        # (N, C)
sum_exp = exp_logits.sum(dim=1, keepdim=True)         # (N, 1)
probs = exp_logits / sum_exp                          # (N, C)

correct_probs = probs[torch.arange(len(ys)), ys]      # (N,)
loss = -torch.log(correct_probs).mean()
```

This is mathematically correct and should match (up to floating-point noise):

```python
import torch.nn.functional as F
loss = F.cross_entropy(logits, ys)
```

So why doesn’t PyTorch implement cross-entropy exactly like the naive code above?

---

## Step 5 — Why PyTorch does NOT compute it this way

The naive implementation explicitly computes exponentials of logits.

Exponentials grow extremely fast:

- `exp(50) ≈ 5e21`
- `exp(100) ≈ 3.7e43`
- In `float32`, large logits overflow to `inf`

When overflow occurs:
- `exp_logits` can contain `inf`
- `sum_exp` can become `inf`
- `probs = inf / inf` becomes `nan`
- `log(probs)` becomes `nan`
- Training breaks, often silently

Underflow also occurs:
- Very negative logits produce `exp(logit) ≈ 0`
- Probabilities can become exactly zero
- `log(0)` becomes `-inf`

Because logits can grow large during training, **explicit softmax followed by log is unsafe in practice**.

---

## Step 6 — The key algebraic identity (log-softmax)

Starting from softmax for the correct class $\(y\)$:

$p_y = \frac{e^{z_y}}{\sum_k e^{z_k}}$

Taking the logarithm:

$\log(p_y) = \log(e^{z_y}) - \log\left(\sum_k e^{z_k}\right)$

Since $\(\log(e^{z_y}) = z_y\)$:

$\log(p_y) = z_y - \log\left(\sum_k e^{z_k}\right)$

So the per-example loss becomes:

$\ell = -z_y + \log\left(\sum_k e^{z_k}\right)$

This formulation avoids explicitly computing probabilities.

---

## Step 7 — Stability trick: subtracting the maximum logit

The remaining challenge is computing:

$\log\left(\sum_k e^{z_k}\right)$

without overflow.

Let:

$m = \max_k z_k$

Define shifted logits:

$z'_k = z_k - m$

Then:

$\sum_k e^{z_k} = \sum_k e^{z'_k + m} = e^m \sum_k e^{z'_k}$

Taking the logarithm:

$\log\left(\sum_k e^{z_k}\right) = m + \log\left(\sum_k e^{z'_k}\right)$

Why this works:
- At least one shifted logit equals 0
- All others are $\(\le 0\)$
- $\(e^{z'_k} \in (0, 1]\)$
- No overflow occurs

Softmax is invariant to constant shifts, so subtracting \(m\) does **not** change the resulting probabilities.

---

## Step 8 — Stable manual cross-entropy (no softmax, no logsumexp)

Below is a numerically stable implementation using only basic operations.

```python
import torch

# logits: (N, C)
# ys:     (N,)

# 1) subtract row-wise maximum
m = logits.max(dim=1, keepdim=True).values
logits_shifted = logits - m

# 2) compute log(sum(exp(.))) safely
exp_logits = torch.exp(logits_shifted)
sum_exp = exp_logits.sum(dim=1, keepdim=True)
log_sum_exp = torch.log(sum_exp)

# 3) log-probabilities
log_probs = logits_shifted - log_sum_exp

# 4) negative log-likelihood
loss = -log_probs[torch.arange(len(ys)), ys].mean()
```

This computes the same objective as the naive version, but safely.

---

## Step 9 — What `F.cross_entropy` does internally

Conceptually, PyTorch computes:

1. A **stable log-softmax** of the logits
2. The **negative log-likelihood** of the correct class indices
3. The **mean over the batch**

In compact form:

```text
cross_entropy(logits, ys) = NLL(log_softmax(logits), ys)
```

PyTorch fuses these steps for numerical stability, performance, and memory efficiency.

---

## Step 10 — Final takeaway

Cross-entropy answers a simple question:

> **How surprised is the model, on average, by the correct answer?**

A useful mental pipeline:

```text
logits → stable normalization → log(prob correct) → negate → mean
```

As models evolve from:
- bigram models
- to MLPs with embeddings
- to deeper architectures

the **loss function remains the same**.  
Only the **model producing the logits** changes.