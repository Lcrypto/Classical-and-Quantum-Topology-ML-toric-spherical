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


Here is a list of datasets with their respective sizes (N) and descriptions:

	**AuralSonar	** (N = 100): This dataset contains Aural Sonar data from the study by Philips et al. (2006) investigating human ability to distinguish different types of sonar signals by ear. The data has been presented in Chen et al. (2009).

	**Protein** (N = 213): This dataset, also presented in Chen et al. (2009), contains the radial basis function (RBF) kernel between 213 proteins.

	**Voting** (N = 435): This dataset, also presented in Chen et al. (2009), contains dissimilarities between 435 voting records with 16 scaled voting attributes.

	**Yeast** (N = 200): This dataset, from the same repository as AuralSonar in Chen et al. (2009), converts the pairwise Smith-Waterman similarities $s_{ij}$ (Lanckriet et al., 2004; Xu et al., 2014) to dissimilarities by $d_{ij}=\sqrt{s_{ii}+s_{jj}-s_{ji}-s_{ij}}.$ The data set converts the pairwise Smith-Waterman similarities $s_{ij}$ (Lanckriet et al., 2004; Xu et al., 2014) to dissimilarities by $d_{ij}=\sqrt{s_{ii}+s_{jj}-s_{ji}-s_{ij}}.$

Lanckriet, G., Deng, M., Cristianini, N., Jordan, M., Noble, W., 2004. Kernel-
based data fusion and its application to protein function prediction in yeast.
150 Biocomputing 2004, Proceedings of the Pacific Symposium, Hawaii, USA , 300--311.
Xu, W., Hancock, E.R., Wilson, R.C., 2014. Ricci flow embedding for rectifying
non-euclidean dissimilarity data. Pattern Recognition 47, 3709--3725.

	**Sawmill (N = 36): This dataset is a sparse matrix with 124 non-zero entries representing the Sawmill communication network from the Pajek data sets. Data available at http://vlado.fmf.uni-lj.si/pub/networks/data/

	**Scotland	** (N = 108): This dataset is about corporate interlocks in Scotland from 1904-5. It is a sparse matrix with 644 non-zero entries.

	**A99m** (N = 234): This dataset is about the characters and their relations in the long-running German soap opera called `Lindenstrasse'. It is a sparse matrix with 510 non-zero entries.

	**Mexican power	** (N = 35): This dataset contains the core of the Mexican political elite: the presidents 40 and their closest collaborators. It is a sparse matrix with 117 non-zero entries.

	**Strike	** (N = 24): This dataset is a social network about informal communication within a sawmill on strike. It is a sparse matrix with 38 non-zero entries.

	**Webkb Cornell** (N = 195): This dataset is about citations among 195 publications from Cornell. It is a sparse matrix with 304 non-zero entries. Data available at https://linqs.soe.ucsc.edu/data

	**WorldTrade** (N = 80): This dataset is about world trade in miscellaneous manufactures of metal, 1994. It is a sparse matrix with 998 non-zero entries.

	**Mesh1e1** (N = 48): This dataset is originally from NASA, collected by Alex Pothen. It is a sparse matrix with 306 non-zero entries.  Data available at   https://sparse.tamu.edu/

	**Mesh2e1	** (N = 306): This dataset is also originally from NASA, collected by Alex Pothen. It is a sparse matrix with 2018 non-zero entries.

	**OrbitRaising** (N = 442): This dataset was from an optimal control problem. It is a sparse matrix with 2906 non-zero entries.

	**Shuttle Entry** (N = 560): This dataset was also from an optimal control problem. It is a sparse matrix with 6891 non-zero entries.

	**AntiAngiogenesis** (N = 205): This dataset was also from an optimal control problem. It is a sparse matrix with 1783 non-zero entries.

	**Phoneme	** (N = 256): This dataset contains the covariance matrix of the Phoneme data set accompanied with the Elements of Machine Learning book (Hastie et al., 2001). The original data has 4508 instances of 256 dimensions.  Data available at  https://web.stanford.edu/~hastie/ElemStatLearn/data.html


    Hastie, T., Tibshirani, R., Friedman, J., 2001. The Elements of Statistical Learning. Springer New York Inc.

	**MiniBooNE	** (N = 50): This dataset contains the covariance matrix of the MiniBooNE particle identification data set in the UCI Repository. The original data has 130064 instances of 50 dimensions. Data available at  https://archive.ics.uci.edu/ml/

	**Covertype	** (N = 54): This dataset contains the covariance matrix of the Covertype data set in the UCI Repository. The original data has 581012 instances of 54 dimensions.

	**Mfeat** (N = 649): This dataset contains the covariance matrix of the Multiple Features data set in the UCI Repository. The original data has 2000 instances of 649 dimensions.

	**OptDigits** (N = 64): This dataset contains the covariance matrix of the Optical Recognition of Handwritten Digits data set in the UCI Repository. The original data has 5620 instances of 64 dimensions.

	**PenDigits** (N = 16): This dataset contains the covariance matrix of the Pen-Based Recognition of Handwritten Digits data set in the UCI Repository. The original data has 10992 instances of 16 dimensions

	**Acoustic** (N = 50): This dataset contains acoustic features from a vehicle sound signal, which can be used to classify the type of vehicle. It is a dataset commonly used in machine learning research, and has been made available by the LIBSVM Classification data collection. Data available at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.
html

	**IJCNN** (N = 22): This dataset contains features from the ijcnn data set, which is also commonly used in machine learning research. It consists of binary classification problems with 22 features, and has been made available by the LIBSVM Classification data collection. Data available at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

	**Spam Ham** (N = 448): This dataset is used for email classification practice, with the goal of determining whether an email is spam or ham. It contains 10000 instances with 448 features.

	**TIMIT** (N = 390): This dataset is used in speech recognition research, with the goal of identifying spoken words. It contains 151290 instances, each with 390 features that represent Mel Frequency Cepstral Coefficients (MFCCs) calculated over 10 consecutive 30ms windows. Data available at https://catalog.ldc.upenn.edu/LDC93S1

	**Votes** (N = 16): This dataset contains voting records from the US Congress, and is often used as a benchmark for supervised learning algorithms. It contains 435 instances with 16 dimensions representing different attributes related to congressional voting.




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
