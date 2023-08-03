# Classical and Quantum Topology Machine Learning: Spherical and Hyperbolic Toric Topologies
The GitHub repositories referenced in this paper, titled “Spherical and Hyperbolic Toric Topology-Based Codes On Graph Embedding for Ising MRF Models: Classical and Quantum Topology Machine Learning”, contain the source code related to the research [https://doi.org/10.48550/arXiv.2307.15778](https://arxiv.org/abs/2307.15778). 


Codes on the Graph based Sparse Matrix Factorization application represented bellow: 


The Sparse Factorization (SF) can thus be formulated as the following optimization problem:




$$
\mathop{\min }\limits_{W^{(1)} ,\ldots ,W^{(M)} } \left\| X-\prod _{m=1}^{M}W^{(M)}  \right\| _{F}^{2}
$$





where $W^{(M)} $'s are sparse square matrices with non-zero positions specified by the Chord protocol (SF Chord), LDPC codes parity-check  using PEG+ACE and QC-LDPC codes parity-check matrix, MET QC-LDPC codes with circulant more than 1 and product multigraph MET QC-LDPC codes parity-check using SA+EMD, Simulated Annealing with exact cycle extrinsic message degree optimization (EMD). 

![alt text](https://github.com/Lcrypto/Classical-and-Quantum-Topology-ML-toric-spherical/blob/main/Table_1_v1.png)



We modified the Matlab platform described in the paper “Sparse factorization of square matrices with application to neural attention modeling” by Ruslan Khalitov, Tong Yu, Lei Cheng, and Zhirong Yang, published in Neural Networks, Volume 152, 2022, Pages 160-168, as the base platform for Non-parametric Sparse Factorisation using LDPC codes, MET QC-LDPC codes and Multi-graph Product codes in our work https://github.com/RuslanKhalitov/SparseFactorization.  


# **Cite**
```
@article{Usatyuk2023TopoML,
  title={Spherical and Hyperbolic Toric Topology-Based Codes On Graph Embedding for Ising MRF Models: Classical and Quantum Topology Machine Learning},
  author={Usatyuk, Vasiliy and Sapozhnikov, Denis and Egorov, Sergey},
  year={2023},
  archivePrefix = {arXiv},
  eprint = {2307.15778},
   primaryClass = {cs.IT},
}
```
