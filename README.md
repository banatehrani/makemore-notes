# makemore-notes

This repository is a **from-scratch, step-by-step reimplementation** of Andrej Karpathy’s **makemore** project, following the *Zero-to-Hero* neural networks series.

The goal is **deep understanding**, not copy-paste:
- every model is implemented incrementally
- every design choice is explained
- every commit reflects a real learning step

This repo complements:
- **Karpathy’s original makemore** (reference implementation)
- my own **micrograd-notes**, where backpropagation and autograd were built from first principles

---

## What is makemore?

**makemore** is a character-level autoregressive language model.  
Given a sequence of characters, the model predicts the **next character**, and this process is repeated to generate complete strings (e.g., names).

Despite its simplicity, makemore gradually introduces many core ideas behind modern deep learning:
- probabilistic modeling
- embeddings
- multilayer perceptrons
- normalization layers
- manual backpropagation
- convolutional / hierarchical architectures

---

## Structure of this repository

```text
makemore-notes/
│
├── notebooks/          # exploratory notebooks (one per major concept)
├── src/
│   └── makemore_notes/ # reusable components and utilities
│
├── data/               # datasets (e.g. names.txt)
├── README.md
└── .gitignore