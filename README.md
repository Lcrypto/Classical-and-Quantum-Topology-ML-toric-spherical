# Classical and Quantum Topology Machine Learning: RBIM Spectral Embedding via Toric & Hyperbolic LDPC Codes for CNN, Transformer, and Diffusion Models

> **Unifying Principle.** Sparse, structured graphs from coding theory—operated at the Nishimori temperature of a Random-Bond Ising Model (RBIM)—yield spectral embeddings whose geometry is controlled by trapping-set elimination, enabling both high accuracy and hardware efficiency.

## 1. Classical Machine Learning: Marginalization on Sparse LDPC Tanner Graphs

In classical ML, marginalization is naturally visualized as inference on a graph. A parity–check matrix $H$ defines a Tanner graph $G=(V,E)$ that encodes the constraints of an LDPC code; performing inference on this graph corresponds to MAP/ML estimation.

Formally, let $G=(V,E)$ be an undirected graph with $|V|=n$ vertices and $|E|=m$ edges. Each vertex carries a binary spin variable and each edge $\{i,j\}\in E$ carries a random coupling weight drawn from a distribution $\mathcal{P}$. The Random-Bond Ising Model (RBIM) on $G$ is specified by the Edwards–Anderson Hamiltonian *(Citation 1)*:

$$
H(s) = -\sum_{\langle i,j\rangle \in E} J_{ij}\, s_i s_j,\qquad s_i\in\{-1,+1\}.
$$

At the **Nishimori temperature**, this Gibbs measure coincides with the Bayesian posterior over coupling parameters. Consequently, stochastic gradient Langevin dynamics becomes *exact* rather than approximate, and the spectral geometry of the associated operators (studied via the Bethe–Hessian) directly controls inference quality *(Citation 1)*.

## 2. Quantum System Representation

Consider a quantum system of $N$ qubits, i.e., a Hilbert space of dimension $2^{N}$. In an arbitrary basis $\{|\sigma\rangle\}$ the state vector is

$$
|\Psi\rangle = \sum_{\sigma} c_\sigma\, |\sigma\rangle,
$$

with associated probability distribution $p(\sigma)=|c_\sigma|^2$. For a chosen computational basis $\{|x\rangle\}$ we write

$$
|\Psi\rangle = \sum_{x} \Psi(x)\, |x\rangle,\qquad p(x)=|\Psi(x)|^{2}.
$$

## 3. Estimation Problem in Hilbert Space

The objective of quantum inference is to find the ground state (minimal eigenvector) and its energy for a Hamiltonian operator $\mathcal{H}$, which requires optimization over an exponentially large space—exactly analogous to MAP/ML estimation on the classical Tanner graph. The expected energy (Rayleigh quotient) is

$$
\frac{\langle \Psi | \mathcal{H} | \Psi\rangle}{\langle \Psi|\Psi\rangle}.
$$

## 4. Reparameterization & Trapping‑Set Elimination: A Gauge‑Theory Perspective

Classical ML and quantum ML exploit a common idea: **reparameterization**.  
In QML this appears as unitary operations or gauge changes that leave the Hamiltonian spectrum invariant while reshaping the state representation. In classical LDPC decoding, reparameterization is used to **eliminate trapping sets**—small subgraphs that trap iterative message-passing algorithms and cause decoding failures.

The key insight, spanning both regimes, is that **changing decoder parameters is equivalent to performing a gauge transformation on the Tanner graph**:

| Classical ML | Quantum ML | Information Theory (LDPC) |
|---|---|---|
| Variable/constraint reparameterization (e.g., belief propagation updates) | Unitary/gauge transformations of the Hamiltonian; parameter-shift rule for gradients *(Citation 3)* | Adjusting check‑node weights or local fields to reshape asymmetrical subgraphs |
| Simplifies inference and improves convergence | Preserves eigenvalues while altering wavefunction phase/amplitude distribution | Removes harmful trapping sets by modifying the graph’s local structure |

Thus, whether reparameterizing a quantum state or tuning decoder parameters in an LDPC code, we are effectively **changing the gauge of the underlying graphical model**, thereby eliminating problematic trapping sets and controlling spectral geometry.

### From Trapping Sets to Degenerate Ground Spaces *(Citation 3)*

In classical LDPC decoding, trapping sets create local minima. In quantum CSS codes, analogous structures appear as degenerate excited states. The spectral gap between logical and physical excitations is precisely the code distance $d_{\min}$; maximizing it simultaneously improves **classical adversarial robustness** and **quantum fault tolerance**.

## 5. Spectral Graph Embedding for Feature Clustering & Classification

Application of this framework to feature embedding is demonstrated in the Python repository **[RBIM Nishimori Clustering](https://github.com/Lcrypto/Classical-and-Quantum-Topology-ML-toric-spherical/tree/main/RBIM_Nishimori_Clustering)** (details in [10–12]). The method implements spectral clustering for Random-Bond Ising Models using **Bethe–Hessian matrices optimized at the Nishimori temperature**, applied to graph models such as Erdős–Rényi and LDPC codes (PEG, QC, Multi-Edge QC). It extends these methods to image clustering tasks, demonstrating enhanced accuracy with a cosine similarity metric on both synthetic data and GAN-generated image datasets [9].

These spectral embeddings are “self-correcting”: BP decoding errors and training failures can be diagnosed by the same spectral diagnostic—the negative eigenvalue count of the Bethe–Hessian *(Citation 2)*.

### Comparison of Clustering Accuracy Using Different Methods
*Table shows accuracy (overlap) with two edge weight metrics: original (Eq.1) and proposed cosine similarity (Eq.2).*  
*Best overall results highlighted in **bold**.*

| Graph Matrix | Nishimori | Spin Glass | Mean-field | Laplacian |
|---|---|---|---|---|
| **Size 2×2 (L=3000)** | | | | |
| E(H)₃ | 23.67%, 31.40% | 26.63%, 30.30% | 0.03%, 20.70% | 32.73%, 0.03% |
| E(H)₄ | 54.43%, 37.93% | 47.00%, 32.80% | 1.20%, 3.16% | 0.03%, 0.10% |
| E(H)₅ | 53.40%, 33.10% | 46.57%, 33.50% | 1.10%, 4.56% | 0.10%, 0.03% |
| E(H)₆ | 26.20%, 31.73% | 29.10%, 26.56% | 0.03%, 22.46% | 31.97%, 0.03% |
| E(H)₇ | 23.67%, 31.40% | 26.63%, 30.33% | 0.03%, 20.70% | 32.73%, 0.03% |
| E(H)₈ | 54.43%, 37.93% | 47.00%, 32.80% | 1.27%, 3.16% | 0.03%, 0.10% |
| E(H)₉ | 53.40%, 33.10% | 46.57%, 33.43% | 1.10%, 4.56% | 0.10%, 0.03% |
| E(H)₁₀ | 26.17%, 31.73% | 29.10%, 26.56% | 0.03%, 22.46% | 31.97%, 0.03% |
| E(H)₁₁ | 23.67%, 31.40% | 26.63%, 30.33% | 0.03%, 20.60% | 32.77%, 0.03% |
| E(H)₁₂ | 54.43%, 37.93% | 47.00%, 32.80% | 1.27%, 3.16% | 0.03%, 0.10% |
| E(H)₁₃ | 26.17%, 31.70% | 29.10%, 26.56% | 0.03%, 22.46% | 31.97%, 0.03% |
| **Size 4×4 (L=1500)** | | | | |
| E(H)₁₄ | 68.37%, 75.63% | 67.70%, 73.53% | 0.07%, 21.43% | 0.03%, 0.03% |
| E(H)₁₅ | 69.70%, 74.70% | 66.43%, 71.83% | 0.30%, 0.13% | 0.03%, 0.03% |
| E(H)₁ | 70.67%, 76.46% | 67.83%, 73.96% | 0.17%, 0.30% | 67.63%, 0.03% |
| E(H)₁₆ | 68.60%, 75.76% | 65.17%, 71.73% | 0.17%, 0.20% | 66.10%, 0.03% |
| E(H)₁₇ | 68.37%, 75.63% | 67.70%, 73.53% | 0.07%, 21.43% | 0.03%, 0.03% |
| E(H)₁₈ | 69.70%, 74.70% | 66.53%, 71.83% | 0.30%, 0.13% | 0.03%, 0.03% |
| E(H)₁₉ | 70.63%, 76.46% | 67.77%, 74.13% | 0.17%, 0.30% | 67.63%, 0.03% |
| E(H)₂₀ | 68.60%, 75.70% | 65.17%, 71.73% | 0.17%, 0.20% | 66.07%, 0.03% |
| E(H)₂₁ | 69.70%, 74.70% | 66.20%, 71.83% | 0.30%, 0.13% | 0.03%, 0.03% |
| E(H)₂₂ | 70.67%, 76.46% | 67.77%, 74.13% | 0.17%, 0.30% | 67.63%, 0.03% |
| E(H)₂₃ | 68.60%, 75.73% | 65.17%, 71.73% | 0.17%, 0.20% | 66.07%, 0.03% |
| E(H)₂₄ | 64.57%, 72.53% | 63.57%, 71.33% | 0.17%, 0.26% | 0.03%, 0.03% |
| E(H)₂₅ | 67.73%, 74.26% | 66.40%, 73.50% | 0.13%, 0.26% | 0.03%, 0.03% |
| **Size 16×16 (L=375)** | | | | |
| E(H)₂ | **90.60%**, **93.23%** | **90.00%**, **92.46%** | 0.20%, 0.30% | 0.00%, 0.03% |
| E(H)₂₆ | 90.57%, 92.30% | 89.47%, 92.16% | 0.23%, 78.53% | 0.03%, 0.03% |
| E(H)₂₇ | 89.43%, 92.56% | 89.03%, 92.20% | 0.13%, 72.03% | 0.03%, 0.03% |
| E(H)₂₈ | 74.27%, 82.63% | 62.33%, 75.60% | 0.87%, 17.50% | 0.03%, 0.03% |
| E(H)₂₉ | 74.40%, 82.16% | 62.90%, 76.20% | 0.17%, 24.76% | 0.03%, 0.03% |
| E(H)₃₀ | 77.83%, 84.83% | 65.80%, 80.26% | 0.20%, 24.20% | 0.03%, 0.03% |
| E(H)₃₁ | 77.57%, 83.80% | 65.57%, 78.20% | 0.23%, 19.30% | 0.03%, 0.03% |
| **Size 25×25 (L=240)** | | | | |
| E(H)₃₂ | 87.77%, 91.00% | 85.80%, 90.43% | 0.37%, 68.46% | 87.03%, 90.40% |
| E(H)₃₃ | 87.73%, 91.76% | 86.67%, 91.00% | 0.40%, 0.86% | 0.03%, 90.96% |
| E(H)₃₄ | 88.30%, 91.26% | 86.23%, 90.50% | 0.33%, 68.63% | 87.20%, 91.16% |
| E(H)₃₅ | 87.50%, 91.56% | 85.73%, 91.26% | 0.07%, 75.16% | 0.03%, **91.40%** |
| E(H)₃₆ | 87.20%, 91.00% | 85.70%, 90.43% | 0.27%, 69.50% | 85.87%, 90.70% |
| E(H)₃₇ | 86.67%, 90.80% | 86.43%, 90.86% | 0.03%, 0.03% | 0.03%, 91.03% |
| E(H)₃₈ | 86.50%, 90.60% | 85.20%, 89.83% | 0.27%, 68.76% | 85.00%, 90.33% |
| E(H)₃₉ | 86.83%, 90.73% | 86.07%, 89.83% | 0.03%, 0.06% | 0.03%, 89.86% |
| E(H)₄₀ | 87.37%, 91.56% | 85.97%, 90.56% | 0.53%, 0.56% | 86.63%, 91.00% |
| E(H)₄₁ | 86.93%, 91.03% | 85.57%, 89.66% | 0.30%, **79.16%** | 86.13%, 90.00% |
| **Size 48×48 (L=125)** | | | | |
| E(H)₄₂ | 88.60%, 91.86% | 87.73%, 90.46% | 0.20%, 78.73% | **87.23%**, 0.03% |

### Edge Weight Metrics
**Original Metric (Eq.1)** from Dall'Amico et al. [9]:  
$J_{ij} = |(z_i, z_j)| / l$

**Proposed Cosine Similarity Metric (Eq.2)** from our paper [10]:  
$J_{ij} = |(z_i, z_j)| / (|z_i| |z_j|)$

Where:
- $z_i, z_j$ = node feature vectors
- $(\cdot,\cdot)$ = scalar product
- $l$ = normalization constant (original)
- $|\cdot|$ = vector norm

### Recent Improvements
All results in the table above have been further improved by more careful Nishimori temperature estimation and feature selection. In particular, for **Quasi-Cyclic graph E(H)₂** (size 16×16, circulant permutation matrix $L=375$), clustering overlap values now exceed **99%** (up from 92.46% reported in [10]).

---

## 6. Quantum Machine Learning: QC‑LDPC Ansätze, Annealing & Training

The transition from classical to quantum computing fundamentally alters how information is represented and learned. The LDPC-based framework becomes not merely advantageous but **essential** in the quantum regime *(Citation 2)*.

### 6.1 LDPC‑Structured Quantum Ansätze and Annealing

Variational Quantum Algorithms (VQAs) employ parameterised circuits whose expressivity depends critically on entanglement structure. Random circuits suffer from barren plateaus, while highly structured circuits may be classically simulable. **QC‑LDPC graphs define an optimal middle ground**: the Tanner graph of a quantum LDPC code determines the entanglement pattern of the variational ansatz, with girth and EMD directly controlling entanglement spreading.

Quantum annealers physically implement Ising Hamiltonians via superconducting qubits; embedding problems onto hardware graphs is a major bottleneck. QC‑LDPC codes provide native problem structures whose sparse local interactions map directly onto Chimera or Pegasus topologies **without minor-embedding overhead**. Future quantum annealers could run BP-style algorithms in hardware, using transverse-field tunnelling to escape trapping-set basins at rates governed by the Nishimori temperature *(Citation 2)*.

### 6.2 Quantum Error Correction as Model Regularization

In quantum machine learning every gate is noisy; error correction is a prerequisite for scalable computation rather than an afterthought. The topological LDPC codes developed in our framework—particularly hyperbolic surface codes derived from toroidal QC‑LDPC structures—offer **constant-rate, linear-distance protection**.

Crucially, the **same graph structure serves dual purposes** *(Citation 2)*:
1. As a sparse prior for classical DNN inference (see Sections 5 & 7).
2. As a quantum error-correcting code protecting the variational quantum circuit.

This unification implies that hardware-optimised QC‑LDPC graphs are **“future-proof”**: they accelerate classical inference today and provide native error correction for quantum accelerators tomorrow *(Citation 2)*.

### 6.3 Conceptual Shifts in Moving to Quantum Training *(Citation 3)*

| Classical Regime | Quantum Regime |
|---|---|
| **Parameters** | Real parameters $\theta\in\mathbb{R}^d$ → unitary evolution angles $\phi\in[0,2\pi)^d$ parameterising $U(\phi)$. The loss landscape lives on the Lie algebra $\mathfrak{su}(2^n)$ rather than Euclidean space. |
| **Gradients** | Chain rule / back-propagation → **parameter-shift rule**: $$\frac{\partial\langle O\rangle}{\partial\phi_i}=\frac{1}{2}\Big(\langle O\rangle_{\phi_i+\pi/2}-\langle O\rangle_{\phi_i-\pi/2}\Big).$$ The sparse QC‑LDPC structure reduces overhead: because each parameter affects only a local neighbourhood (bounded degree), gradients can be estimated in parallel with query complexity $O(d_c)$ rather than $O(d)$. |
| **Noise** | Thermal noise at $T=1/\beta$ → quantum decoherence. The Nishimori condition generalises to the **quantum Nishimori temperature**, where the Kraus operator representation of noise commutes with code stabilizers, maximising the quantum Fisher information metric. |
| **Subgraph defects** | Trapping sets in classical LDPC → degenerate excited states in quantum CSS codes. The spectral gap between logical and physical excitations is exactly $d_{\min}$; maximising it simultaneously improves classical adversarial robustness and quantum fault tolerance. |

## 7. Large Language Models as LDPC Codes: Graph & Decoder Optimization

The proposed approach has direct implications for new embedding methods based on trapping sets. Statistical physics and number geometry are utilized to optimize error-correcting codes, leading to these embedding and sparse factorization methods. Our work establishes a direct connection between DNN architecture and error-correcting coding by demonstrating how state-of-the-art Transformer architectures (ChordMixer, Mega, Mega-chunk, CDIL, ...) from the Long-Range Arena can be equivalent to specific types of block and convolutional LDPC codes.

### 7.1 Mega & Mega‑chunk as GeIRA QC‑LDPC
![Mega architecture](https://github.com/Lcrypto/Classical-and-Quantum-Topology-ML-toric-spherical/blob/main/Mega_arch.png)

The Mega-chunk model with two chunks of length 3 is represented above. Its bipartite graph representation shows that such a neural network is equivalent to a parity-check matrix of protograph:

$$
H_{\text{MEGA}} = 
\begin{bmatrix}
1 & 1 & 1 & 1\\
0 & 1 & 1 & 1 \\
0 & 0 & 1 & 1 \\
0 & 0 & 0 & 1
\end{bmatrix}.
$$

Mega and Mega-chunk Attention models use Generalized Irregular Repeat Accumulate (GeIRA) protograph QC-LDPC codes (Repeat Accumulate (RA) [0] and GeIRA [0]); for details see the article https://arxiv.org/abs/2307.15778.

![RA embedding](https://github.com/Lcrypto/Classical-and-Quantum-Topology-ML-toric-spherical/blob/main/RA_embd.png)
![RA codes](https://github.com/Lcrypto/Topology-Signal-Processing/blob/master/RA_codes.png)

The dynamic approach combines convolutional and block codes by incorporating attention and convolution within a deep neural network (DNN) proposed by Schmidhuber in 1991. RA LDPC codes and GeIRA codes under DNN allow bypassing non-linear processing via fast weights [–1].

![Fast & slow weights](https://github.com/Lcrypto/Topology-Signal-Processing/blob/master/Fast_slow_weigth.png)

### 7.2 ChordMixer: Cage‑Graph Attention
Another state-of-the-art attention architecture from the long-range arena is presented in article [1], based on the P2P Chord protocol. ChordMixer utilizes **Cage graphs** as distance graphs to design its attention mechanism, yielding an attention structure equivalent to the parity-check matrix of cage/distance-graph LDPC codes (for details see https://arxiv.org/abs/2307.15778).

![Chord protocol & cage graph](https://github.com/Lcrypto/Classical-and-Quantum-Topology-ML-toric-spherical/blob/main/Chord_protocol_cage_distance_graph_parity-check_matrix.png)

### 7.3 CDIL: Column‑Weight‑3 Convolutional Code
The convolutional model “CDIL” uses the 2P Chord protocol as its basis and is equivalent to a **column-weight-3 convolutional code** (computation tree of weight 3).

![CDIL](https://github.com/Lcrypto/Classical-and-Quantum-Topology-ML-toric-spherical/blob/main/CDIL.png)

### 7.4 Chemical Analogy: QC Circulants as Electron Cloud Gauges
QC codes correspond to certain types of chemical elements; the carbon element is represented by the mixed-automorphism Shu-Lin-Fossorier QC-LDPC code. The feasibility of the original parity-check matrix serving as a representation of Nitrogen can be confirmed by employing circulant-shift logic:

$$
H=\left( \begin{array}{c} 
{\begin{array}{ccccc} {I_{17}} & {I_{17}} & {I_{17}} & {I_{17}} & {I_{17}} \end{array}} \\ 
{H_{1}} \\ {H_{2}} \\ {H_{3}} \\ {H_{4}} 
\end{array}\right) 
= 
\left( \begin{array}{c} 
{\begin{array}{ccccc} {I_{17}} & {I_{17}} & {I_{17}} & {I_{17}} & {I_{17}} \end{array}} \\ 
{C_{85}^{0}+C_{85}^{24}+C_{85}^{40}+C_{85}^{71}+C_{85}^{84}} \\ 
{C_{85}^{1}+C_{85}^{49}+C_{85}^{58}+C_{85}^{81}+C_{85}^{84}} \\ 
{C_{85}^{3}+C_{85}^{14}+C_{85}^{32}+C_{85}^{78}+C_{85}^{84}} \\ 
{C_{85}^{16}+ C_{85}^{33}+C_{85}^{50}+C_{85}^{67}+C_{85}^{84}} 
\end{array} \right).
$$

To determine spherical coordinates $\varphi, \theta$, we use the criterion for the presence of a cycle of length 6 (cycle-based gauge) in the quasi-cyclic check matrix [Fossorier04]. Based on the electron cloud associated with an atom, we refer to this as the **Schrödinger-Heisenberg-Bohr-Fossorier Electron Cloud Gauge (SHBF Cycle Gauge)**:

$$
\left( \sum_{i=1}^{N-1} \Delta_{ji,ji+1}(l_i) \right) \bmod k = 0.
$$

Collapsing the matrix along radii gives:

$$
H= \left( \begin{array}{c} 
{\begin{array}{cccccc} {C_{8}^{0}} & {C_{8}^{0}} & {C_{8}^{2}} & {C_{8}^{2}} & {C_{8}^{2}} & {C_{8}^{2}} \end{array}} \\ 
{C_{48}^{1}+C_{48}^{7}+C_{48}^{13}+C_{48}^{19}+C_{48}^{25}+C_{48}^{31}} \\ 
{C_{48}^{23}+C_{48}^{17}+C_{48}^{47}+C_{48}^{41}+C_{48}^{35}+C_{48}^{29}} 
\end{array} \right)
\to
\left( \begin{array}{c} 
{\begin{array}{cccccc} {I_{8}} & {I_{8}} & {I_{8}} & {I_{8}} & {I_{8}} & {I_{8}} \end{array}} \\ 
{C_{48}^{1}+C_{48}^{7}+C_{48}^{13}+C_{48}^{19}+C_{48}^{25}+C_{48}^{31}} \\ 
{C_{48}^{23}+C_{48}^{17}+C_{48}^{47}+C_{48}^{41}+C_{48}^{35}+C_{48}^{29}} 
\end{array} \right).
$$

The first collapsed row contains 6 circulants of size 8 and weight 1:
- 2 shift-$0$ circulants → $1s^2$
- 4 shift-$2$ circulants → $2s^2\,2p^2$

yielding the Carbon electron configuration:
$$1s^{2};\; 2s^{2}\,2p^{2}.$$

## 8. From Ising to QUBO: Quantum Annealing and QAOA

Numerous optimization problems—including those in quantum computer vision [Yu22]—can be converted into Quadratic Unconstrained Binary Optimization (QUBO) form [Ble23]. QUBO problems are typically NP-complete, requiring exploration of an exponentially growing solution space classically. The exponentially expanding Hilbert space of a quantum system naturally accommodates this combinatorial search.

The **Quantum Approximate Optimization Algorithm (QAOA)** is designed for QUBOs by utilizing a quantum circuit to find approximate solutions. When applied to the Sherrington-Kirkpatrick Ising model, QAOA can be seen as analogous to back-propagation loss landscapes in DNN training; both suffer from TS pseudo-codeword-like effects resembling belief propagation failures.

QAOA solves binary optimization problems with unknown vector $\mathbf{x} \in \{0,1\}^n$ and symmetric matrix $\mathbf{Q}\in\mathbb{R}^{n\times n}$:

$$
C(\mathbf{x}) = \mathbf{x}^T \mathbf{Q} \mathbf{x} = \sum_{i,j=1}^{n} Q_{ij}x_i x_j.
$$

QUBO maps one-to-one to Ising variables $\mathbf{z}\in\{-1,1\}^n$ via $z_i = 2x_i - 1$. The layer depth in QAOA correlates with the number of belief-propagation iterations in the Wiberg decoding tree.

Overall, this framework has the potential to advance multiple fields—from Information Theory and DNN architecture design (sparse structured prior graph topology) to efficient hardware design for Quantum and Classical DPU/TPU architectures, Materials Science, and beyond.

## 9. Matrix Factorization as Sparse Code-on-Graph Embedding

Matrix factorization can be viewed as a special case of Ising spin-glass embedding—i.e., low-dimension projection via codes on the graph.

Sparse Factorization (SF) is formulated [1] as:

$$
\min_{W^{(1)},\dots,W^{(M)}} \Bigl\| X - \prod_{m=1}^{M} W^{(m)} \Bigr\|_F^2,
$$

where each $W^{(m)}$ is a sparse square matrix with non-zero positions specified by:
- Chord protocol (SF Chord),
- LDPC codes parity-check using PEG+ACE,
- QC-LDPC and MET QC-LDPC codes with circulant weight $>1$,
- Product multigraph MET QC-LDPC codes via SA+EMD (Simulated Annealing with exact Extrinsic Message Degree optimization).

![Comparison](https://github.com/Lcrypto/Classical-and-Quantum-Topology-ML-toric-spherical/blob/main/images_compare_27_02_.jpg)
![Table 1 v2](https://github.com/Lcrypto/Classical-and-Quantum-Topology-ML-toric-spherical/blob/main/Table_1_v2.png)

We modified the Matlab platform from [1] as the base for Non-parametric Sparse Factorisation using LDPC codes, MET QC-LDPC codes and Multi-graph Product codes:  
https://github.com/RuslanKhalitov/SparseFactorization

Code constructions used:
- **PEG+ACE**: Progressive Edge Growth with Approximate Cycle Extrinsic message degree [2,3,4] — https://github.com/Lcrypto/classic-PEG-
- **QC-LDPC / MET QC-LDPC / Multigraph product (Chord-like)**: Simulated Annealing with EMD and code-distance sieving [5,6] — 
  - https://github.com/Lcrypto/Simulated-annealing-lifting-QC-LDPC
  - https://github.com/Lcrypto/Length-und-Rate-adaptive-code

### Parity-Check Matrix Notation Examples

**Quasi-cyclic multigraph product code (Chord-like)** `AntinegoMETProduct3.txt`:



1	1	205
0&154&3&2&65&85&70&97

*1 column, 1 row, QC circulant of size 205; weight-8 circulant with shifts 0, 154, 3, 2, 65, 85, 70, 97.*

**WebkbCornell factorization MET QC-LDPC** `3_3_65weight3.txt`:

3	3	65
50&1&26	2&49&19	13&5&42
5&58	5&60	60&4
18&4&48	28&23&61	4&53&1


**A99m factorization QC-LDPC** `13_13_18A99m.txt`:
*13 columns, 13 rows, QC circulant size 18; -1 denotes a zero circulant of size 19×19.*

13	13	18
14	1	1	-1	0	17	-1	-1	-1	12	17	-1	-1
3	11	6	17	-1	-1	-1	14	8	-1	16	-1	-1
-1	-1	-1	-1	9	2	-1	-1	-1	13	1	4	13
8	5	9	0	5	-1	1	14	-1	10	6	-1	-1
-1	-1	6	0	-1	12	-1	1	-1	-1	1	0	-1
-1	2	-1	-1	10	-1	0	9	4	11	-1	0	3
9	-1	-1	16	-1	9	16	9	9	3	2	2	-1
-1	1	-1	-1	16	-1	5	6	4	5	0	8	16
-1	11	16	1	-1	11	5	-1	15	-1	17	-1	6
17	-1	16	3	10	12	9	-1	6	10	-1	16	4
8	12	-1	-1	0	1	5	-1	17	-1	-1	-1	3
10	-1	1	7	4	9	15	15	-1	1	-1	5	6
8	10	6	1	-1	-1	-1	4	3	-1	-1	4	17



### Datasets Used in Factorization Experiments

| Dataset | $N$ | Description |
|---|---|---|
| **AuralSonar** | 100 | Aural Sonar data from Philips et al. (2006); presented in Chen et al. (2009). |
| **Protein** | 213 | RBF kernel between 213 proteins [Chen et al., 2009]. |
| **Voting** | 435 | Dissimilarities between voting records with 16 scaled attributes [Chen et al., 2009]. |
| **Yeast** | 200 | Pairwise Smith-Waterman similarities converted to dissimilarities. |
| **Sawmill** | 36 | Sparse matrix (124 nz) from the Pajek communication network dataset. |
| **Scotland** | 108 | Corporate interlocks in Scotland, 1904–5; sparse matrix (644 nz). |
| **A99m** | 234 | Character relations from German soap opera *Lindenstrasse* (510 nz). |
| **Mexican power** | 35 | Mexican political elite core network (117 nz). |
| **Strike** | 24 | Informal communication during a sawmill strike (38 nz). |
| **Webkb Cornell** | 195 | Citation network among 195 Cornell publications (304 nz). |
| **WorldTrade** | 80 | World trade in miscellaneous manufactures of metal, 1994 (998 nz). |
| **Mesh1e1** | 48 | NASA sparse matrix (306 nz) from Alex Pothen. |
| **Mesh2e1** | 306 | NASA sparse matrix (2018 nz). |
| **OrbitRaising** | 442 | Optimal control sparse matrix (2906 nz). |
| **Shuttle Entry** | 560 | Optimal control sparse matrix (6891 nz). |
| **AntiAngiogenesis** | 205 | Optimal control sparse matrix (1783 nz). |
| **Phoneme** | 256 | Covariance matrix of the Phoneme dataset [Hastie et al., 2001]. |
| **MiniBooNE** | 50 | Covariance matrix from UCI; original data has 130064 instances. |
| **Covertype** | 54 | Covariance matrix from UCI; original data has 581012 instances. |
| **Mfeat** | 649 | Covariance matrix of the Multiple Features dataset (UCI). |
| **OptDigits** | 64 | Optical handwritten digits covariance matrix (UCI). |
| **PenDigits** | 16 | Pen-based handwritten digits covariance matrix (UCI). |
| **Acoustic** | 50 | Vehicle sound signal features (LIBSVM). |
| **IJCNN** | 22 | Binary classification features from IJCNN (LIBSVM). |
| **Spam Ham** | 448 | Email classification dataset with 10000 instances. |
| **TIMIT** | 390 | Speech recognition; MFCCs over 10 consecutive windows. |
| **Votes** | 16 | US Congressional voting records benchmark (435 instances). |

---

# References

-1. Schmidhuber J. Learning to control fast-weight memories: An alternative to recurrent nets. Technical Report FKI-147-91, Institut für Informatik, Technische Universität München, 26 March 1991. https://people.idsia.ch/~juergen/FKI-147-91ocr.pdf; illustration https://people.idsia.ch/~juergen/fast-weight-programmer-1991-transformer.html

0. Divsalar D., H. Jin, and R. J. McEliece. "Coding theorems for ‘turbo-like’ codes." *Proc. 36th Allerton Conf.* 1998, pp. 201–210.

0. Divsalar D., Jones C., Dolinar S., Thorpe J. "Protograph based LDPC codes with minimum distance linearly growing with block size," *IEEE GLOBECOM*, St. Louis, MO, 2005.

0. Liva G. et al. "Simple Reconfigurable Low-Density Parity-Check Codes." *IEEE Communications Letters*, Vol. 9, No. 2, February 2005.

1. Ruslan Khalitov, Tong Yu, Lei Cheng, Zhirong Yang, **Sparse factorization of square matrices with application to neural attention modeling**, *Neural Networks*, Volume 152, 2022, pp. 160-168. https://github.com/RuslanKhalitov/SparseFactorization

2. Xiao-Yu Hu, Eleftheriou E., Arnold D.M. "Regular and irregular progressive edge-growth Tanner graphs," *IEEE Trans. Inform. Theory*, vol. 51, no. 1, pp. 386-398, Jan. 2005. Implementation: https://github.com/Lcrypto/classic-PEG-/

3. Diouf M., Declercq D., Fossorier M., Ouya S., Vasic B. "Improved PEG construction of large girth QC-LDPC codes," *ISTC*, pp. 146-150, 2016.

4. Usatyuk V., Minenkov A. Progressive edge growth for LDPC code construction (C++ / MATLAB PEG+ACE implementations). https://github.com/Lcrypto/classic-PEG-/

5. Usatyuk V., Vorobyev I. "Simulated Annealing Method for Construction of High-Girth QC-LDPC Codes," *TSP*, Athens, Greece, 2018. Implementation: https://github.com/Lcrypto/Simulated-annealing-lifting-QC-LDPC

6. Usatyuk V.S., Egorov S., Svistunov G. "Construction of Length and Rate Adaptive MET QC-LDPC Codes by Cyclic Group Decomposition," *EWDTS*, Batumi, Georgia, 2019. Implementation: https://github.com/Lcrypto/Length-und-Rate-adaptive-code

7. Usatyuk V.S., Sapozhnikov D., Egorov S.I. **Spherical and Hyperbolic Toric Topology-Based Codes On Graph Embedding for Ising MRF Models: Classical and Quantum Topology Machine Learning.** arXiv:2307.15778, 2023.

8. Usatyuk V.S., Egorov S.I. "Topology-Aware Sparse Factorization in Energy-Based Models: Tori QC-LDPC Codes and Circular Hyperboloid MET QC-LDPC Codes," *DSPA*, Moscow, 2024. https://ieeexplore.ieee.org/document/10510073

9. Dall'Amico, L. et al. (2021). *Nishimori meets Bethe: a spectral method for node classification in sparse weighted graphs.* J. Stat. Mech. 093405.

10. Usatyuk V.S., Sapozhnikov D.A., Egorov S.I. (2024). **Enhanced Image Clustering with Random-Bond Ising Models Using LDPC Graph Representations and Nishimori Temperature.** *Moscow Univ. Phys.* 79 (Suppl 2), S647-S665. https://link.springer.com/article/10.3103/S0027134924702102; Preprint: https://theory2.sinp.msu.ru/lib/exe/fetch.php/dlcp/dlcp2024/bphm647.pdf

11. Usatyuk V.S., Sapozhnikov D.A., Egorov S.I. "Flattening Images Manifold at the Nishimori Point: Exploring Linearity in Polysemantic Embedding for Human-Aware Bayesian CNN Image Classification," *DSPA*, Moscow, 2025. https://ieeexplore.ieee.org/document/10977938

12. Usatyuk V.S., Egorov S.I. "Boosting DNN Efficiency: Replacing FC Layers with Graph Embeddings for Hardware Acceleration," *DSPA*, Moscow, 2025. https://ieeexplore.ieee.org/document/10977895

---

# **Cite this research**

```bibtex
@article{Usatyuk2023TopoML,
       author = {{Usatyuk}, Vasiliy and {Sapozhnikov}, Denis and {Egorov}, Sergey},
        title = "{Spherical and Hyperbolic Toric Topology-Based Codes On Graph Embedding for Ising MRF Models: Classical and Quantum Topology Machine Learning}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Information Theory, Computer Science - Artificial Intelligence, Computer Science - Computer Vision and Pattern Recognition, Computer Science - Machine Learning, Mathematics - Dynamical Systems},
         year = 2023,
        month = jul,
          eid = {arXiv:2307.15778},
        pages = {arXiv:2307.15778},
          doi = {10.48550/arXiv.2307.15778},
archivePrefix = {arXiv},
       eprint = {2307.15778},
 primaryClass = {cs.IT},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230715778U},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
