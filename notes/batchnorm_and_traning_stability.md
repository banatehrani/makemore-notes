# Part 3 — Batch Normalization & Training Stability (Conceptual Notes)

These notes summarize and clarify **all conceptual topics discussed by Karpathy in Part 3**, independent of any specific notebook implementation.  
The goal is intuition, not heavy math.

---

## 1. Why deep neural networks are hard to train

As neural networks get deeper, several issues arise:

### Vanishing / exploding gradients
During backpropagation, gradients are repeatedly multiplied by weights and activation derivatives.  
If these factors are smaller than 1, gradients shrink exponentially (vanishing).  
If larger than 1, they grow uncontrollably (exploding).

This makes early layers:
- learn very slowly, or
- become numerically unstable

---

## 2. Activation functions and dead neurons

### Saturating activations (e.g. `tanh`)
`tanh(x)` squashes inputs into $(-1, 1)$.

For large $|x|$, its derivative is near zero:

$$
\frac{d}{dx} \tanh(x) \approx 0
$$

This causes **vanishing gradients**.

### Dead neurons (mostly ReLU-related)
With ReLU, neurons can permanently output zero if weights push inputs negative.
Once dead, gradients stop flowing.

**Key insight:**  
Activation choice affects gradient flow and trainability.

---

## 3. What Batch Normalization does (high level)

BatchNorm normalizes intermediate activations to have:
- mean ≈ 0
- variance ≈ 1

For a pre-activation $h$:

$$
\hat{h} = \frac{h - \mu}{\sqrt{\sigma^2 + \varepsilon}}
$$

Then applies a **learned affine transform**:

$$
y = \gamma \hat{h} + \beta
$$

This:
- stabilizes activation distributions
- keeps gradients in a healthy range
- allows higher learning rates

---

## 4. Why BatchNorm helps optimization

BatchNorm:
- reduces sensitivity to initialization
- reduces internal covariate shift
- keeps activations out of saturation zones
- smooths the loss landscape

Result: **faster and more stable training**

---

## 5. Train vs inference behavior in BatchNorm

During training:
- statistics ($\mu$, $\sigma^2$) are computed per mini-batch

During inference:
- **running averages** are used instead

Running statistics are updated via exponential moving average:

$$
\mu_{run} \leftarrow (1 - m) \mu_{run} + m \mu_{batch}
$$

This is why BatchNorm behaves differently in train vs eval mode.

---

## 6. Momentum and affine parameters

### Momentum
Controls how fast running statistics adapt:
- small momentum → smoother, slower updates
- large momentum → faster adaptation, more noise

### Affine parameters
$\gamma$ (scale) and $\beta$ (shift) allow the network to:
- undo normalization if needed
- preserve representational power

Without affine parameters, BatchNorm would be overly restrictive.

---

## 7. Temperature (sampling, not training)

Temperature rescales logits before softmax:

$$
p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

- $T < 1$: sharper, more confident predictions
- $T > 1$: smoother, more diverse predictions

Temperature affects **generation**, not training.

---

## 8. Initialization: past vs modern practice

Historically:
- training deep nets was extremely sensitive to initialization
- bad init = no learning

Modern improvements:
- careful initializations (Xavier, Kaiming)
- BatchNorm / LayerNorm
- residual connections
- better optimizers

Result: **nets today are far more robust**

---

## 9. How neural networks are initialized today

Common practice:
- scale weights based on fan-in
- zero biases
- rely on normalization layers

Example principle (Xavier-style):

$$
\text{Var}(W) \propto \frac{1}{\text{fan-in}}
$$

This keeps activations well-scaled across layers.

---

## 10. Why results may differ from Karpathy’s

Differences can come from:
- random seeds
- learning rate schedules
- number of steps
- batch size
- exact architecture
- no hyperparameter tuning

**Important:**  
Different loss values do NOT imply conceptual misunderstanding.

---

## 11. Coupling in BatchNorm and its limitations

BatchNorm couples samples **within the same batch**:
- small batch sizes → noisy statistics
- correlations between examples

This motivates alternatives:
- LayerNorm (normalize per sample)
- GroupNorm
- RMSNorm

---

## 12. PyTorch abstractions (`Linear`, `BatchNorm1d`)

PyTorch modules encapsulate:
- parameters (`weight`, `bias`)
- buffers (running mean/variance)
- train/eval behavior

Understanding the math first (as we did) makes these abstractions transparent.

---

## 13. Neural network diagnostics (why they matter)

Useful diagnostics:
- activation histograms
- gradient norms
- % of saturated neurons
- weight distributions

These tools help:
- debug training
- understand failures
- build intuition

They are **diagnostic**, not required for training.

---

## Final takeaway

BatchNorm is not magic.
It is a carefully designed statistical tool that:
- stabilizes activations
- improves gradient flow
- reduces sensitivity to initialization

Understanding *why* it works is far more important than memorizing how to use it.