# Nishimori-Bethe Spectral Clustering for Quasi-Cyclic Graphs

## Overview
This repository implements spectral graph clustering (using data from two classes cat and dog GAN feature, [1]) using the **Nishimori Temperature Bethe-Hessian** method optimized for quasi-cyclic graph structures. The algorithm combines techniques from statistical physics and spectral graph theory to detect communities in graph-structured data with superior performance on quasi-cyclic codes, use as example graph from and methods inmplentation detailed described at [2]. Data file and features you can download from [googledisk](https://drive.google.com/drive/folders/1b-Csoy27ATPN--F6CHJV4vznCieFwNL7?usp=share_link), after download place both files to folder with `NB_clustering.py` script.

Key features:
- Bethe-Hessian operator with Nishimori temperature optimization
- Support for quasi-cyclic graph structures
- Multiple similarity metrics for affinity matrix construction
- Spectral embedding and clustering/classification
- Quantitative evaluation via clustering overlap metric

## Modules

### 1. Core Clustering (`NB_clustering.py`)
**Main class**: `NishimoriBetheSpectralClustering`  
Performs spectral clustering using regularized Laplacian and Nishimori temperature optimization.

Key methods:
- `fit(train, train_labels)`: Computes spectral embedding and clusters data
- `predict(sample)`: Predicts labels for new data
- `spectral_embedding()`: Generates low-dimensional embeddings
- `_find_beta_N()`: Optimizes Nishimori temperature via eigenvalue analysis

Parameters:
- `adjacency_matrix`: Sparse graph representation
- `n_clusters`: Number of clusters (default=2)
- `sim`: Similarity measure (`'cos'`, `'euclidean'`, etc.)
- `is_z_scoring`: Apply z-score normalization (default=True)
- `state`: `'cluster'` or `'classifier'` mode

### 2. Temperature Optimization (`_temperature_optimizations.py`)
Implements critical temperature estimation for spin-glass systems.

Functions:
- `find_beta_sg_dichotomy()`: Computes spin-glass temperature βₛg via bisection
- Solves: _c·φ·mean(tanh²(β·J)) - 1 = 0_

### 3. Similarity Metrics (`_similarity.py`)
Provides 7 similarity measures for affinity matrix calculation:

| Metric        | Function          | Key        |
|---------------|-------------------|------------|
| Cosine        | `cos_sim()`       | `'cos'`    |
| Euclidean     | `euclidean_sim()` | `'euclidean'` |
| Minkowski     | `minkowski_sim()` | `'minkowski'` |
| Manhattan     | `manhattan_sim()` | `'manhattan'` |
| Chebyshev     | `chebyshev_sim()` | `'chebyshev'` |
| Mahalanobis   | `mahalanobis_sim()`| `'mahalanobis'` |
| Pearson       | `pearson_sim()`   | `'pearson'` |

Accessed via `similarity_dict` mapping.

### 4. Quasi-Cyclic Graph  (`qc2sparse.py`)
Constructs sparse matrices from quasi-cyclic graph descriptions.

Functions:
- `build_sparse_matrix(filename)`: Main parser for QC description files
- `parse_input_file(filename)`: Reads QC code parameters
- `create_circulant_matrix(n, k)`: Generates permutation matrices
- `display_matrix_portrait(matrix)`: Visualizes sparse matrix structure

Supports multi-edge graphs with `&`-separated indices in input files.

## Usage Example

```python
from NB_clustering import NishimoriBetheSpectralClustering
from qc2sparse import build_sparse_matrix

# 1. Prepare graph and data
graph = build_sparse_matrix('qc_code.txt')
data = np.loadtxt('features.dat')
true_labels = np.concatenate([np.zeros(500), np.ones(500)])

# 2. Initialize and fit model
model = NishimoriBetheSpectralClustering(
    adjacency_matrix=graph,
    n_clusters=2,
    sim='cos',
    state='cluster'
)
pred_labels = model.fit(data)

# 3. Evaluate clustering
overlap = calculate_overlap(pred_labels, true_labels)
print(f"Clustering Overlap: {overlap:.3f}")
```

## Evaluation Metric
**Clustering Overlap**: Quantifies separation quality  
`Overlap = |2×(accuracy - 0.5)|` where:
- `0.0` = Random guessing
- `0.5` = Good separation (~75% accuracy)
- `1.0` = Perfect match

## Dependencies
- Python 3.7+
- NumPy
- SciPy
- scikit-learn
- Matplotlib (for visualization)

## References
1.  Lorenzo Dall'Amico et al. "Nishimori meets Bethe: a spectral method for node classification in sparse weighted graphs," J. Stat. Mech., 093405 (2021). 
2. Usatyuk, V.S., Sapozhnikov, D.A. & Egorov, S.I. Enhanced Image Clustering with Random-Bond Ising Models Using LDPC Graph Representations and Nishimori Temperature. Moscow Univ. Phys. 79 (Suppl 2), S647-S665 (2024).  
