# Spectral Clustering with Bethe-Hessian and Nishimori Temperature for RBIM on Graph Models

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lcrypto/Classical-and-Quantum-Topology-ML-toric-spherical/blob/main/LICENSE)

Python implementation of spectral graph clustering methods for Random-Bond Ising Models (RBIM) using Nishimori temperature, featuring:
- **Bethe-Hessian with Nishimori temperature**
- Spin Glass methods
- Mean-field approximations
- Laplacian techniques

Tested on diverse graph models:
- Erdős–Rényi (ER)
- Progressive Edge Growth (PEG)
- Multi-Edge QC-LDPC
- Quasi-cyclic (QC) LDPC

Developed as companion code for the research paper:  
*Enhanced Image Clustering with Random-Bond Ising Models Using LDPC Graph Representations and Nishimori Temperature* [1]. Preprint can be read from https://theory2.sinp.msu.ru/lib/exe/fetch.php/dlcp/dlcp2024/bphm647.pdf

## Repository Structure

### (`/Graph_matrix`)[https://github.com/Lcrypto/Classical-and-Quantum-Topology-ML-toric-spherical/tree/main/RBIM_Nishimori_Clustering/Graph%20Matrix]
Contains graph generation implementations for:
- Progressive Edge Growth (PEG) graphs
- Quasi-cyclic (QC) LDPC graphs  
*(Used for Figure 8 in [1])*

### `/BetheNishimori_ER`
Synthetic data clustering scripts under Erdős–Rényi graphs:
- Bethe-Hessian spectral clustering
- Nishimori temperature optimization
- Performance evaluation metrics

### `/NB_clustering_cat_dog_from_GAN_Manifold`
Image clustering application using QC graphs:
- Two-class GAN-generated image dataset (Cats vs Dogs) from [2]
- QC graph spectral decomposition

## Key Features
- Implementation of spectral methods from Dall'Amico et al. [2] extended to LDPC graphs
- Custom PEG and QC-LDPC graph generators
- Simulated annealing integration for spin glass optimization
- Comparative analysis framework for clustering methods

### Comparison of Clustering Accuracy Using Different Methods
*Table shows accuracy (overlap) with two edge weight metrics: original (Eq.1) and proposed cosine similarity (Eq.2)*  
*Best overall results highlighted in **bold***

| Graph Matrix                 | Nishimori         | Spin Glass        | Mean-field       | Laplacian        |
|------------------------------|-------------------|-------------------|------------------|------------------|
| **Size 2×2 (L=3000)**        |                   |                   |                  |                  |
| E(H)₃                        | 23.67%, 31.40%    | 26.63%, 30.30%    | 0.03%, 20.70%    | 32.73%, 0.03%    |
| E(H)₄                        | 54.43%, 37.93%    | 47.00%, 32.80%    | 1.20%, 3.16%     | 0.03%, 0.10%     |
| E(H)₅                        | 53.40%, 33.10%    | 46.57%, 33.50%    | 1.10%, 4.56%     | 0.10%, 0.03%     |
| E(H)₆                        | 26.20%, 31.73%    | 29.10%, 26.56%    | 0.03%, 22.46%    | 31.97%, 0.03%    |
| E(H)₇                        | 23.67%, 31.40%    | 26.63%, 30.33%    | 0.03%, 20.70%    | 32.73%, 0.03%    |
| E(H)₈                        | 54.43%, 37.93%    | 47.00%, 32.80%    | 1.27%, 3.16%     | 0.03%, 0.10%     |
| E(H)₉                        | 53.40%, 33.10%    | 46.57%, 33.43%    | 1.10%, 4.56%     | 0.10%, 0.03%     |
| E(H)₁₀                       | 26.17%, 31.73%    | 29.10%, 26.56%    | 0.03%, 22.46%    | 31.97%, 0.03%    |
| E(H)₁₁                       | 23.67%, 31.40%    | 26.63%, 30.33%    | 0.03%, 20.60%    | 32.77%, 0.03%    |
| E(H)₁₂                       | 54.43%, 37.93%    | 47.00%, 32.80%    | 1.27%, 3.16%     | 0.03%, 0.10%     |
| E(H)₁₃                       | 26.17%, 31.70%    | 29.10%, 26.56%    | 0.03%, 22.46%    | 31.97%, 0.03%    |
| **Size 4×4 (L=1500)**        |                   |                   |                  |                  |
| E(H)₁₄                       | 68.37%, 75.63%    | 67.70%, 73.53%    | 0.07%, 21.43%    | 0.03%, 0.03%     |
| E(H)₁₅                       | 69.70%, 74.70%    | 66.43%, 71.83%    | 0.30%, 0.13%     | 0.03%, 0.03%     |
| E(H)₁                        | 70.67%, 76.46%    | 67.83%, 73.96%    | 0.17%, 0.30%     | 67.63%, 0.03%    |
| E(H)₁₆                       | 68.60%, 75.76%    | 65.17%, 71.73%    | 0.17%, 0.20%     | 66.10%, 0.03%    |
| E(H)₁₇                       | 68.37%, 75.63%    | 67.70%, 73.53%    | 0.07%, 21.43%    | 0.03%, 0.03%     |
| E(H)₁₈                       | 69.70%, 74.70%    | 66.53%, 71.83%    | 0.30%, 0.13%     | 0.03%, 0.03%     |
| E(H)₁₉                       | 70.63%, 76.46%    | 67.77%, 74.13%    | 0.17%, 0.30%     | 67.63%, 0.03%    |
| E(H)₂₀                       | 68.60%, 75.70%    | 65.17%, 71.73%    | 0.17%, 0.20%     | 66.07%, 0.03%    |
| E(H)₂₁                       | 69.70%, 74.70%    | 66.20%, 71.83%    | 0.30%, 0.13%     | 0.03%, 0.03%     |
| E(H)₂₂                       | 70.67%, 76.46%    | 67.77%, 74.13%    | 0.17%, 0.30%     | 67.63%, 0.03%    |
| E(H)₂₃                       | 68.60%, 75.73%    | 65.17%, 71.73%    | 0.17%, 0.20%     | 66.07%, 0.03%    |
| E(H)₂₄                       | 64.57%, 72.53%    | 63.57%, 71.33%    | 0.17%, 0.26%     | 0.03%, 0.03%     |
| E(H)₂₅                       | 67.73%, 74.26%    | 66.40%, 73.50%    | 0.13%, 0.26%     | 0.03%, 0.03%     |
| **Size 16×16 (L=375)**       |                   |                   |                  |                  |
| E(H)₂                        | **90.60%**, **93.23%** | **90.00%**, **92.46%** | 0.20%, 0.30% | 0.00%, 0.03% |
| E(H)₂₆                       | 90.57%, 92.30%    | 89.47%, 92.16%    | 0.23%, 78.53%    | 0.03%, 0.03%     |
| E(H)₂₇                       | 89.43%, 92.56%    | 89.03%, 92.20%    | 0.13%, 72.03%    | 0.03%, 0.03%     |
| E(H)₂₈                       | 74.27%, 82.63%    | 62.33%, 75.60%    | 0.87%, 17.50%    | 0.03%, 0.03%     |
| E(H)₂₉                       | 74.40%, 82.16%    | 62.90%, 76.20%    | 0.17%, 24.76%    | 0.03%, 0.03%     |
| E(H)₃₀                       | 77.83%, 84.83%    | 65.80%, 80.26%    | 0.20%, 24.20%    | 0.03%, 0.03%     |
| E(H)₃₁                       | 77.57%, 83.80%    | 65.57%, 78.20%    | 0.23%, 19.30%    | 0.03%, 0.03%     |
| **Size 25×25 (L=240)**       |                   |                   |                  |                  |
| E(H)₃₂                       | 87.77%, 91.00%    | 85.80%, 90.43%    | 0.37%, 68.46%    | 87.03%, 90.40%   |
| E(H)₃₃                       | 87.73%, 91.76%    | 86.67%, 91.00%    | 0.40%, 0.86%     | 0.03%, 90.96%    |
| E(H)₃₄                       | 88.30%, 91.26%    | 86.23%, 90.50%    | 0.33%, 68.63%    | 87.20%, 91.16%   |
| E(H)₃₅                       | 87.50%, 91.56%    | 85.73%, 91.26%    | 0.07%, 75.16%    | 0.03%, **91.40%**|
| E(H)₃₆                       | 87.20%, 91.00%    | 85.70%, 90.43%    | 0.27%, 69.50%    | 85.87%, 90.70%   |
| E(H)₃₇                       | 86.67%, 90.80%    | 86.43%, 90.86%    | 0.03%, 0.03%     | 0.03%, 91.03%    |
| E(H)₃₈                       | 86.50%, 90.60%    | 85.20%, 89.83%    | 0.27%, 68.76%    | 85.00%, 90.33%   |
| E(H)₃₉                       | 86.83%, 90.73%    | 86.07%, 89.83%    | 0.03%, 0.06%     | 0.03%, 89.86%    |
| E(H)₄₀                       | 87.37%, 91.56%    | 85.97%, 90.56%    | 0.53%, 0.56%     | 86.63%, 91.00%   |
| E(H)₄₁                       | 86.93%, 91.03%    | 85.57%, 89.66%    | 0.30%, **79.16%**| 86.13%, 90.00%   |
| **Size 48×48 (L=125)**       |                   |                   |                  |                  |
| E(H)₄₂                       | 88.60%, 91.86%    | 87.73%, 90.46%    | 0.20%, 78.73%    | **87.23%**, 0.03%|


### Edge Weight Metrics

**Original Metric (Eq.1)** from Dall'Amico et al. [2]:  
J<sub>ij</sub> = |(z<sub>i</sub>, z<sub>j</sub>)| / l

**Proposed Cosine Similarity Metric (Eq.2)** from our paper [1]:  
J<sub>ij</sub> = |(z<sub>i</sub>, z<sub>j</sub>)| / (|z<sub>i</sub>| |z<sub>j</sub>|)

Where:
- $z_i, z_j$ = node feature vectors
- $(\cdot,\cdot)$ = scalar product
- $l$ = normalization constant (original)
- $|\cdot|$ = vector norm

### Key Differences
| Metric         | Convergence Property               | Graph Density Sensitivity |
|----------------|------------------------------------|---------------------------|
| **Original (Eq.1)** | Slower convergence            | Fails on dense graphs     |
| **Proposed (Eq.2)** | Faster convergence            | Robust to dense graphs    |

The cosine similarity metric (Eq.2) provides:
1. Better normalization for feature vectors
2. Improved convergence in spectral clustering
3. Reduced heterogeneity in adjacency matrices
4. Enhanced performance on dense graphs

## References
1. **Primary Study**  
   Usatyuk, V.S., Sapozhnikov, D.A. & Egorov, S.I. (2024).  
   *Enhanced Image Clustering with Random-Bond Ising Models Using LDPC Graph Representations and Nishimori Temperature*.  
   Moscow Univ. Phys. 79 (Suppl 2), S647-S665.   
  https://link.springer.com/article/10.3103/S0027134924702102

2. **Foundational Method**  
   Dall'Amico, L. et al. (2021).  
   *Nishimori meets Bethe: a spectral method for node classification in sparse weighted graphs*.  
   J. Stat. Mech. 093405.  
  

3. **Graph Implementations**  
   - [PEG Algorithm (Matlab/C/Python)](https://github.com/Lcrypto/classic-PEG-)
   - [Erdős–Rényi Graph Model](https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model)
   - [Simulated Annealing Optimization](https://ieeexplore.ieee.org/document/8441303)

## License
Distributed under **Apache License 2.0**.  
See [LICENSE](https://github.com/Lcrypto/Classical-and-Quantum-Topology-ML-toric-spherical/blob/main/LICENSE) for full terms.
