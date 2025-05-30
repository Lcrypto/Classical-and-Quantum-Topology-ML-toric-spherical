We are given a Jupyter notebook (BetheNishimori.ipynb)[https://github.com/Lcrypto/Classical-and-Quantum-Topology-ML-toric-spherical/blob/main/RBIM_Nishimori_Clustering/BetheNishimori_ER/BetheNishimori.ipynb] that performs clustering on a synthetic graph using four different algorithms and compares their performance.

### Overview of the Notebook:

1. **Install Dependencies**: The notebook starts by installing necessary Python packages (networkx, numpy, scikit-learn, scipy) if they are not already installed.

2. **Import Libraries**: It then imports required libraries and custom modules (`Basic_functions`, `Clustering`, `NBNC`).

3. **Read Data**: Reads the graph data from files (`edge_list.txt` and `J_edge_list.txt`). The graph is an Erdős–Rényi graph with 3000 nodes and an average degree of 8. The edge weights are generated based on cluster labels (two clusters of equal size).

4. **Clustering Algorithms**: Applies four clustering algorithms:

- **Nishimori-Bethe Hessian (BHN)**: A spectral clustering method using the Bethe Hessian matrix with parameters derived from the Nishimori conditions.

- **Spin-Glass Bethe Hessian (SG)**: Uses the Bethe Hessian matrix with parameters set by the spin-glass model.

- **Mean Field Approximation (MF)**: A naive mean field approach for clustering.

- **Weighted Laplacian (LAP)**: Spectral clustering using the signed Laplacian matrix.

5. **Results**: Computes the overlap (accuracy) of each algorithm's cluster assignment compared to the true labels.

### Key Observations from the Results:

- The Nishimori-Bethe Hessian and Spin-Glass Bethe Hessian methods perform similarly well (overlap ~66%).

- The Mean Field and Laplacian methods perform poorly (overlap near 0%).

### GitHub README Description:

We can create a concise and informative README for the GitHub repository. The README should include:

- **Title**: Clustering on Synthetic Graph Data

- **Description**: A comparison of four clustering algorithms on an Erdős–Rényi graph with two balanced clusters.

- **Algorithms**:

1. Nishimori-Bethe Hessian

2. Spin-Glass Bethe Hessian

3. Mean Field Approximation

4. Weighted Laplacian

- **Results**: The Nishimori-Bethe Hessian and Spin-Glass Bethe Hessian methods significantly outperform the others.

### Proposed README.md:

```markdown

# Clustering on Synthetic Graph Data

This project compares the performance of four clustering algorithms on a synthetic Erdős–Rényi graph with two balanced clusters.

## Algorithms

1. **Nishimori-Bethe Hessian (BHN)**: A spectral method using the Bethe Hessian matrix with parameters derived from the Nishimori conditions.

2. **Spin-Glass Bethe Hessian (SG)**: Uses the Bethe Hessian matrix with parameters set by the spin-glass model.

3. **Mean Field Approximation (MF)**: A naive mean field approach for clustering.

4. **Weighted Laplacian (LAP)**: Spectral clustering using the signed Laplacian matrix.

## Results

The following overlaps (accuracy) were achieved on a graph of 3000 nodes with average degree 8:

| Algorithm          | Overlap |

|--------------------|---------|

| BH Nishimori       | 66.07%  |

| BH spin glass      | 66.00%  |

| Mean field         |  0.53%  |

| Laplacian          |  0.07%  |

The results show that the Bethe Hessian based methods (Nishimori and spin-glass) are effective for clustering in this setting, while the mean field and Laplacian methods perform poorly.

## Code

The main code is in the Jupyter notebook `BetheNishimori.ipynb`. The notebook relies on custom modules (`Basic_functions.py`, `Clustering.py`, `NBNC.py`) and data files (`edge_list.txt`, `J_edge_list.txt`).

## Dependencies

- networkx

- numpy

- scikit-learn

- scipy

```
