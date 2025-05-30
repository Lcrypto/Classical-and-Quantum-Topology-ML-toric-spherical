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

### `/Graph_matrix`
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
- Two-class GAN-generated image dataset (Cats vs Dogs)
- Manifold learning integration
- QC graph spectral decomposition

## Key Features
- Implementation of spectral methods from Dall'Amico et al. [2] extended to LDPC graphs
- Custom PEG and QC-LDPC graph generators
- Simulated annealing integration for spin glass optimization
- Comparative analysis framework for clustering methods

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
