# Classical and Quantum Topology Machine Learning: Spherical and Hyperbolic Toric Topologies
The GitHub repositories referenced in this paper, titled “Spherical and Hyperbolic Toric Topology-Based Codes On Graph Embedding for Ising MRF Models: Classical and Quantum Topology Machine Learning”, contain the source code related to the research [https://doi.org/10.48550/arXiv.2307.15778](https://arxiv.org/abs/2307.15778). 
The paper introduces the application of information geometry to describe the ground states of Ising models. This is achieved by utilizing parity-check matrices of cyclic and quasi-cyclic codes on toric and spherical topologies. The approach establishes a connection between machine learning and error-correcting coding, specifically in terms of automorphism and the size of the circulant of the quasi-cyclic code. This proposed approach has implications for the development of new embedding methods based on trapping sets. Statistical physics and number geometry are utilized to optimize error-correcting codes, leading to these embedding and sparse factorization methods. The paper establishes a direct connection between DNN architecture and error-correcting coding by demonstrating how state-of-the-art DNN architectures (ChordMixer, Mega, Mega-chunk, CDIL, ...) from the long-range arena can be equivalent to specific types (Cage-graph, Repeat Accumulate) of block and convolutional LDPC codes. QC codes correspond to certain types of chemical elements, with the carbon element being represented by the mixed automorphism Shu-Lin-Fossorier QC-LDPC code. The Quantum Approximate Optimization Algorithm (QAOA) used in the Sherrington-Kirkpatrick Ising model can be seen as analogous to the back-propagation loss function landscape in training DNNs. This similarity creates a comparable problem with TS pseudo-codeword, resembling the belief propagation method. Additionally, the layer depth in QAOA correlates to the number of decoding belief propagation iterations in the Wiberg decoding tree. Overall, this work has the potential to advance multiple fields, from Information Theory, DNN architecture design (sparse and structured prior graph topology), efficient hardware design for Quantum and Classical DPU/TPU (graph, quantize and shift register architect.) to Materials Science and beyond.



Matrix factorization can be considered as a special case of (Ising spin-glass) embedding, low dimension projection. Codes on the Graph based Sparse Matrix Factorization application represented bellow: 


The Sparse Factorization (SF) can thus be formulated as the following optimization problem, paper [1]:




$$
\mathop{\min }\limits_{W^{(1)} ,\ldots ,W^{(M)} } \left\| X-\prod _{m=1}^{M}W^{(M)}  \right\| _{F}^{2}
$$





where $W^{(M)} $'s are sparse square matrices with non-zero positions specified by the Chord protocol (SF Chord), LDPC codes parity-check  using PEG+ACE and QC-LDPC codes parity-check matrix, MET QC-LDPC codes with circulant more than 1 and product multigraph MET QC-LDPC codes parity-check using SA+EMD, Simulated Annealing with exact cycle extrinsic message degree optimization (EMD). 

![alt text](https://github.com/Lcrypto/Classical-and-Quantum-Topology-ML-toric-spherical/blob/main/Table_1_v1.png)



We modified the Matlab platform described in the paper “Sparse factorization of square matrices with application to neural attention modeling” by Ruslan Khalitov, Tong Yu, Lei Cheng, and Zhirong Yang, published in Neural Networks, Volume 152, 2022, Pages 160-168, as the base platform for Non-parametric Sparse Factorisation using LDPC codes, MET QC-LDPC codes and Multi-graph Product codes in our work https://github.com/RuslanKhalitov/SparseFactorization for using Code on the Graph: LDPC Codes constructed using Progressive Edge Grown method with ACE optimization [2,3]; QC-LDPC Codes, Multi-Edge QC-LDPC Code, Multigraph product code (Chord like) using Simulated Annealing method with EMD and code distance sieving optimization [4, 5]. 





Parity-check matrix of code on the graph use the following notation:


For example Quasi-cyclic (QC) multigraph product code (Chord like) AntinegoMETProduct3.txt:

     1	1	205


     0&154&3&2&65&85&70&97	


1 column 1 row and QC  circulant of size 205


circulant have weight 8 with shifts:  0, 154, 3, 2, 65, 85, 70, 97	





WebkbCornell factorization Multi-edge Type (MET) QC-LDPC parity-check matrix  "3_3_65weight3.txt" :


3 columns 3 rows and QC circulant of size 65


3	3	65

each circulant represented by following shifts:


     50&1&26	2&49&19	13&5&42	


     5&58	5&60	60&4	


     18&4&48	28&23&61	4&53&1	


A99m factorization QC-LDPC parity-check matrix  "13_13_18A99m.txt" :


consist of 13 columns 13 rows with QC circulant of size 18 of weigth 1 (where -1, it zero circulat of size 19x19):


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



Here is a list of datasets with their respective sizes (N) and descriptions:

**AuralSonar** (N = 100): This dataset contains Aural Sonar data from the study by Philips et al. (2006) investigating human ability to distinguish different types of sonar signals by ear. The data has been presented in Chen et al. (2009).


**Protein** (N = 213): This dataset, also presented in Chen et al. (2009), contains the radial basis function (RBF) kernel between 213 proteins.


**Voting** (N = 435): This dataset, also presented in Chen et al. (2009), contains dissimilarities between 435 voting records with 16 scaled voting attributes.


**Yeast** (N = 200): This dataset, from the same repository as AuralSonar in Chen et al. (2009), converts the pairwise Smith-Waterman similarities $s_{ij}$ (Lanckriet et al., 2004; Xu et al., 2014) to dissimilarities by $d_{ij}=\sqrt{s_{ii}+s_{jj}-s_{ji}-s_{ij}}.$ The data set converts the pairwise Smith-Waterman similarities $s_{ij}$ (Lanckriet et al., 2004; Xu et al., 2014) to dissimilarities by $d_{ij}=\sqrt{s_{ii}+s_{jj}-s_{ji}-s_{ij}}.$
Lanckriet, G., Deng, M., Cristianini, N., Jordan, M., Noble, W., 2004. Kernel-based data fusion and its application to protein function prediction in yeast.
150 Biocomputing 2004, Proceedings of the Pacific Symposium, Hawaii, USA , 300--311.  Xu, W., Hancock, E.R., Wilson, R.C., 2014. Ricci flow embedding for rectifying
non-euclidean dissimilarity data. Pattern Recognition 47, 3709--3725.


**Sawmill** (N = 36): This dataset is a sparse matrix with 124 non-zero entries representing the Sawmill communication network from the Pajek data sets. Data available at http://vlado.fmf.uni-lj.si/pub/networks/data/


**Scotland** (N = 108): This dataset is about corporate interlocks in Scotland from 1904-5. It is a sparse matrix with 644 non-zero entries.


**A99m** (N = 234): This dataset is about the characters and their relations in the long-running German soap opera called `Lindenstrasse'. It is a sparse matrix with 510 non-zero entries.


**Mexican power** (N = 35): This dataset contains the core of the Mexican political elite: the presidents 40 and their closest collaborators. It is a sparse matrix with 117 non-zero entries.


**Strike** (N = 24): This dataset is a social network about informal communication within a sawmill on strike. It is a sparse matrix with 38 non-zero entries.
**Webkb Cornell** (N = 195): This dataset is about citations among 195 publications from Cornell. It is a sparse matrix with 304 non-zero entries. Data available at https://linqs.soe.ucsc.edu/data


**WorldTrade** (N = 80): This dataset is about world trade in miscellaneous manufactures of metal, 1994. It is a sparse matrix with 998 non-zero entries.


**Mesh1e1** (N = 48): This dataset is originally from NASA, collected by Alex Pothen. It is a sparse matrix with 306 non-zero entries.  Data available at   https://sparse.tamu.edu/


**Mesh2e1** (N = 306): This dataset is also originally from NASA, collected by Alex Pothen. It is a sparse matrix with 2018 non-zero entries.


**OrbitRaising** (N = 442): This dataset was from an optimal control problem. It is a sparse matrix with 2906 non-zero entries.


**Shuttle Entry** (N = 560): This dataset was also from an optimal control problem. It is a sparse matrix with 6891 non-zero entries.


**AntiAngiogenesis** (N = 205): This dataset was also from an optimal control problem. It is a sparse matrix with 1783 non-zero entries.


**Phoneme** (N = 256): This dataset contains the covariance matrix of the Phoneme data set accompanied with the Elements of Machine Learning book (Hastie et al., 2001). The original data has 4508 instances of 256 dimensions.  Data available at  https://web.stanford.edu/~hastie/ElemStatLearn/data.html
Hastie, T., Tibshirani, R., Friedman, J., 2001. The Elements of Statistical Learning. Springer New York Inc.


**MiniBooNE** (N = 50): This dataset contains the covariance matrix of the MiniBooNE particle identification data set in the UCI Repository. The original data has 130064 instances of 50 dimensions. Data available at  https://archive.ics.uci.edu/ml/


**Covertype** (N = 54): This dataset contains the covariance matrix of the Covertype data set in the UCI Repository. The original data has 581012 instances of 54 dimensions.


**Mfeat** (N = 649): This dataset contains the covariance matrix of the Multiple Features data set in the UCI Repository. The original data has 2000 instances of 649 dimensions.


**OptDigits** (N = 64): This dataset contains the covariance matrix of the Optical Recognition of Handwritten Digits data set in the UCI Repository. The original data has 5620 instances of 64 dimensions.


**PenDigits** (N = 16): This dataset contains the covariance matrix of the Pen-Based Recognition of Handwritten Digits data set in the UCI Repository. The original data has 10992 instances of 16 dimensions


**Acoustic** (N = 50): This dataset contains acoustic features from a vehicle sound signal, which can be used to classify the type of vehicle. It is a dataset commonly used in machine learning research, and has been made available by the LIBSVM Classification data collection. Data available at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html


**IJCNN** (N = 22): This dataset contains features from the ijcnn data set, which is also commonly used in machine learning research. It consists of binary classification problems with 22 features, and has been made available by the LIBSVM Classification data collection. Data available at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html


**Spam Ham** (N = 448): This dataset is used for email classification practice, with the goal of determining whether an email is spam or ham. It contains 10000 instances with 448 features.


**TIMIT** (N = 390): This dataset is used in speech recognition research, with the goal of identifying spoken words. It contains 151290 instances, each with 390 features that represent Mel Frequency Cepstral Coefficients (MFCCs) calculated over 10 consecutive 30ms windows. Data available at https://catalog.ldc.upenn.edu/LDC93S1


**Votes** (N = 16): This dataset contains voting records from the US Congress, and is often used as a benchmark for supervised learning algorithms. It contains 435 instances with 16 dimensions representing different attributes related to congressional voting.


References:


1.  Ruslan Khalitov, Tong Yu, Lei Cheng, Zhirong Yang, Sparse factorization of square matrices with application to neural attention modeling, Neural Networks, Volume 152, 2022, Pages 160-168



2. Xiao-Yu Hu, Eleftheriou E., Arnold D. M.  "Regular and irregular progressive edge-growth tanner graphs," in IEEE TiT, vol. 51, no. 1, pp. 386-398, Jan. 2005, 


3. Diouf M., Declercq D., Fossorier  M., S. Ouya, B. Vasic, "Improved PEG construction of large girth QC-LDPC codes", 9th International Symposium on Turbo Codes and Iterative Information Processing (ISTC), pp. 146-150,2016.


4.  Usatyuk V., Minenkov A. Progressive edge growth for LDPC code construction C++ and Matlab PEG+ACE implementations, avaliable at
 https://github.com/Lcrypto/classic-PEG-


5. Usatyuk V. , Vorobyev I. "Simulated Annealing Method for Construction of High-Girth QC-LDPC Codes," 2018 41st International Conference on Telecommunications and Signal Processing (TSP), Athens, Greece, 2018, pp. 1-5 Implementation available at: https://github.com/Lcrypto/Simulated-annealing-lifting-QC-LDPC


6. Usatyuk V. S., Egorov S., Svistunov G. Construction of Length and Rate Adaptive MET QC-LDPC Codes by Cyclic Group Decomposition. IEEE East-West Design & Test Symposium (EWDTS), Batumi, Georgia, 2019, pp. 1-5


 
# **Cite this reseach**
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
