# Kohn–Sham Spectral Embedding (KSSE)

**Kohn–Sham Spectral Embedding (KSSE)** is an energy‑based model that replaces the dense
top‑layer classifier of convolutional neural networks with a sparse‑graph spectral embedding
evaluated at the Nishimori temperature.  By mapping pre‑trained visual features onto
quasi‑cyclic low‑density parity‑check graphs, constructing a regularised Laplacian that acts as an
effective Kohn–Sham Hamiltonian and solving independent single‑channel eigenproblems in  
**O(N log N)** time via FFT on circulant blocks, KSSE delivers state‑of‑the‑art accuracy with
dramatically fewer parameters.

> **Result** – On ImageNet‑1000 (1 300 000 training / 50 000 test), KSSE achieves  
> **88.93 % Top‑1 accuracy** using only ~21 M parameters, outperforming Swin‑L (197 M) and matching the
> lower end of ViT‑H/14 (632 M) while reducing model size by 10×–30×.

---

## Table of Contents

| Section | Description |
|---------|-------------|
| [Overview](#overview) | High‑level idea and motivation |
| [Key Concepts](#key-concepts) | QC‑LDPC graphs, Nishimori temperature, star‑domain surgery |
| [Model Pipeline](#model-pipeline) | From features to spectral embedding |
| [Training & Evaluation](#training--evaluation) | How the logistic regression head is trained and accuracy measured |
| [Results & Ablations](#results--ablations) | Accuracy tables, ablation studies, comparison with other methods |
| [Citation](#citation) | How to reference KSSE in your work |

---

## Overview

Large‑scale image classification usually relies on deep convolutional networks or vision transformers.
While these models excel at feature extraction (e.g. EfficientNet‑B4), their fully‑connected heads
suffer from poor scalability with the number of classes and require a fixed input size.

KSSE offers a fundamentally different approach:

1. **Graph representation** – Visual features are mapped to Ising couplings on a sparse,
   quasi‑cyclic LDPC graph.
2. **Energy‑based inference** – Classification is performed by finding low‑energy configurations
   of the Random‑Bond Ising Model (RBIM) at the Nishimori temperature, where community structure
   becomes maximally detectable.
3. **Spectral embedding** – The regularised Laplacian \(L_\beta\) plays the role of a Kohn–Sham Hamiltonian;
   its minimum eigenvector for each feature channel provides an embedding that is then fed to a very small
   logistic regression head.

The resulting system has only ~21 M trainable parameters (the backbone is frozen) and requires no back‑propagation
through the spectral layer.

---

## Key Concepts

| Concept | What it means in KSSE |
|---------|-----------------------|
| **QC‑LDPC graph** | A sparse, toroidal graph built from circulant permutation matrices.  Its structure fixes where Ising couplings live; the edge shifts are the only tunable parameters. |
| **Affinity tensor** | For each edge and feature channel, a similarity score (derived from Manhattan distance) is turned into an Ising coupling \(J_{ij}\).  Positive scores → ferromagnetic, negative → antiferromagnetic. |
| **Regularised Laplacian** | \(L_\beta = I - SWS\), where \(W_{ij}=\tanh(\beta J_{ij})/(1-\tanh^2(\beta J_{ij}))\) and \(S\) normalises by the vertex degrees.  This operator is the analogue of a Kohn–Sham Hamiltonian. |
| **Nishimori temperature** | The inverse temperature \(\beta_N\) at which the minimum eigenvalue of \(L_\beta\) becomes zero.  At this point, class information is maximally encoded in the energy landscape. |
| **Star‑domain surgery** | Edge‑shift modifications that create local convexity around codewords (class representatives) while bounding residual frustration.  This removes harmful trapping sets without destroying useful codewords. |
| **Fractal analysis** | The correlation dimension \(D_2\) of the Bethe free‑energy landscape is used to certify that star‑domain surgery has produced smooth basins (\(D_2<1\)).  It also justifies using only a handful of Fourier modes for Rayleigh refinement. |

---

## Model Pipeline

The inference pipeline consists of the following stages (conceptually, no implementation code shown):

1. **Feature extraction** – Feed all images through a frozen EfficientNet‑B4 backbone to obtain
   \(D=1792\)-dimensional feature vectors.
2. **Affinity tensor construction** – For each edge \((i,j)\) in the QC‑LDPC graph and for every channel,
   compute a similarity score from the Manhattan distance of the two corresponding feature vectors;
   normalise by z‑scoring to obtain signed Ising couplings \(J_{ij}^{(k)}\).
3. **Nishimori temperature search** – For each channel, perform a binary search over \(\beta\) until the
   minimum eigenvalue of the regularised Laplacian is zero.  All channels converge to nearly the same
   temperature (within 0.5 %).
4. **Star‑domain surgery** – Modify edge shifts so that every surviving codeword \(TS(a,0)\) becomes the centre
   of a star domain; residual frustration is bounded by \(\rho(B_\gamma)\leq 1+\delta\), with \(\delta\to 0\).
5. **FFT‑based eigenvalue computation** – Because each Laplacian block is circulant,
   compute its spectrum via a single FFT.  After surgery, only the first five Fourier modes
   are needed for accurate Rayleigh refinement.
6. **Spectral embedding** – Concatenate the minimum eigenvectors from all \(D\) channels to obtain an
   \((N\times D)\) embedding matrix \(\mathbf{E}\).
7. **Classification head** – Train a simple logistic regression layer (single fully‑connected
   layer with L2 regularisation) on the embeddings of frozen training representatives.
8. **Transductive inference** – For each test block, embed the test images together with a small set
   of frozen representatives in the same graph; compute logits via the trained logistic head and
   assign class labels.

---

## Training & Evaluation

* The only learnable part is the logistic regression head (≈1.8 M parameters).  
  It is trained for 10 epochs using AdamW (learning rate 1e‑3, weight decay 1e‑5) on the frozen
  embeddings of training representatives.
* Accuracy is evaluated over all 50 000 test images.  Because KSSE embeds test and training nodes jointly,
  it operates under a **transductive** protocol: each inference batch contains both frozen and thawed
  nodes in the same graph.

---

## Results & Ablations

| Graph Size (N) | Frozen / Thawed per class | Column Weight | Top‑1 Accuracy |
|----------------|---------------------------|---------------|----------------|
| 20 000         | 10 / 10                   | 28            | 80.20 %        |
| 25 000         | 10 / 10                   | 34            | 81.40 %        |
| 30 000         | 5 / 15                    | 48            | 83.00 %        |
| **45 000**     | **5 / 5**                 | **48**        | **88.93 %**    |

### Ablation Highlights

* **Column weight** – Increasing the QC‑LDPC column weight enlarges the code distance \(d_{\min}\) and
  suppresses frustrated cycles, improving accuracy.
* **Graph size** – Larger graphs provide more frozen representatives per class, stabilising the Nishimori temperature (quasi‑stationarity).
* **Thawed nodes** – Fewer test images per batch lead to smaller perturbations of \(\beta_N\) and higher accuracy.

### Comparison with State‑of‑the‑Art

| Model | Parameters | FLOPs | Top‑1 |
|-------|------------|-------|-------|
| Swin‑L (ImageNet‑21K) | 197 M | 34–104 G | 86.4–87.3 % |
| ViT‑H/14 (ImageNet‑21K) | 632 M | 167–616 G | 88.0–89.5 % |
| EfficientNet‑B7 | 66 M | – | 84.3 % |
| **KSSE + EfficientNet‑B4** | **≈21 M** | **≈4.2 G** | **88.93 %** |

> *Note*: KSSE is evaluated under a transductive protocol (joint graph embedding of test and training representatives).  
> A matched \(k\)-NN baseline on the same frozen features achieves only ~78.8 %, indicating that the
> spectral embedding contributes roughly 10 pp beyond nearest‑neighbour transduction.

---

## Citation

If you use KSSE in your research, please cite:

```bibtex
@misc{usatyuk2026kohnsham,
  title         = {Kohn--Sham Spectral Embedding on Sparse Graphs at the Nishimori Temperature for Image Classification}, 
  author        = {Vasiliy S. Usatyuk and Denis A. Sapozhnikov and Sergei I. Egorov},
  year          = {2026},
  eprint        = {2607.28428},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2607.28428}
}
```

(Work was presented at The 10th International Conference in Deep Learning in Computational Physics {https://theory.sinp.msu.ru/doku.php/dlcp2026/program}. Publication is currently under review.)

---
