# Classical and Quantum Topology Machine Learning: Spherical and Hyperbolic Toric Topologies Code on the Graph based Embedding
Short and brief description of idea, 16 pages articel Topology-Aware Exploration of Energy-Based Models Equilibrium: Toric QC-LDPC Codes and Hyperbolic MET QC-LDPC Codes [https://arxiv.org/abs/2401.14749] . 


The GitHub repositories referenced in this paper, titled “Spherical and Hyperbolic Toric Topology-Based Codes On Graph Embedding for Ising MRF Models: Classical and Quantum Topology Machine Learning”, contain the source code related to the research [https://doi.org/10.48550/arXiv.2307.15778](https://arxiv.org/abs/2307.15778). 
The paper introduces the application of information geometry to describe the ground states of Ising models. This is achieved by utilizing parity-check matrices of cyclic and quasi-cyclic codes on toric and spherical topologies. The approach establishes a connection between machine learning and error-correcting coding, specifically in terms of automorphism and the size of the circulant of the quasi-cyclic code. The proposed approach ingeniously merges Tensor Networks, Bayesian networks, Markov Random Fields, and Factor graphs, harnessing the strengths of each graph model through a brilliant concept centered on the interplay of symmetry and asymmetry originating from Trapping sets. This proposed approach has implications for the development of new embedding methods based on trapping sets.  






This proposed approach has implications for the development of new embedding methods based on trapping sets. Statistical physics and number geometry are utilized to optimize error-correcting codes, leading to these embedding and sparse factorization methods. The paper establishes a direct connection between DNN architecture and error-correcting coding by demonstrating how state-of-the-art DNN Transformer architectures (ChordMixer, Mega, Mega-chunk, CDIL, ...) from the long-range arena can be equivalent to specific types (Cage-graph, Repeat Accumulate) of block and convolutional LDPC codes.

![alt text](https://github.com/Lcrypto/Classical-and-Quantum-Topology-ML-toric-spherical/blob/main/Mega_arch.png)


The Mega-chunk model with two chunks of length 3 is represented in the Figure. A bipartite graph representation shows that such a neural network is equivalent to a parity-check matrix of protograph:
 

$$  H_{MEGA} =  {\left\lbrack \matrix{1 & 1 & 1 & 1\cr 0 & 1 & 1 & 1 \cr 0 & 0 & 1 & 1 \cr 0 & 0 & 0 & 1} \right\rbrack} $$




The Mega and Mega-chunk Attention models use an Generalized Irregular Repeat Accumulate (GeIRA) protograph QC-LDPC codes (Repeat Accumulate (RA)[0] and GeIRA[0], for detail read article https://arxiv.org/abs/2307.15778).


![alt text](https://github.com/Lcrypto/Classical-and-Quantum-Topology-ML-toric-spherical/blob/main/RA_embd.png)



![alt text](https://github.com/Lcrypto/Topology-Signal-Processing/blob/master/RA_codes.png)


The dynamic approach combines: convolutional and block codes by incorporating attention and convolution within a deep neural network (DNN) proposed by Schmidhube at 1991. RA LDPC codes and GeIRA codes under DNN allow to bypass non-linear processing (Fast feature), paper [-1].


![alt text](https://github.com/Lcrypto/Topology-Signal-Processing/blob/master/Fast_slow_weigth.png)







Another state-of-the-art attention architecture from the long-range arena is presented in the article by [1], which is based on the P2P Chord protocol, left.  ChordMixer utilizes Cage graphs as distance graphs to design its attention mechanism, as shown in the research paper by [1]. Using Cage graphs allows ChordMixer to construct the attention mechanism in a way that is equivalent to the parity-check matrix of cage/distance graph LDPC codes (for detail read article https://arxiv.org/abs/2307.15778).



![alt text](https://github.com/Lcrypto/Classical-and-Quantum-Topology-ML-toric-spherical/blob/main/Chord_protocol_cage_distance_graph_parity-check_matrix.png)




Convolutional model ("CDIL"). This is a convolutional code that uses the 2P Chord protocol as its basis and is considered a column weight 3 convolutional code (computation tree of weight 3).


![alt text](https://github.com/Lcrypto/Classical-and-Quantum-Topology-ML-toric-spherical/blob/main/CDIL.png)







QC codes correspond to certain types of chemical elements, with the carbon element being represented by the mixed automorphism Shu-Lin-Fossorier QC-LDPC code. The feasibility of original parity-check matrix of Shu-Lin-Fossorier QC-LDPC code serving as a representation of Nitrogen can be confirmed effortlessly by employing the following logic



$$  
H=\left( \begin{array}{c} {\begin{array}{ccccc} {I_{17} } & {I_{17} } & {I_{17} } & {I_{17} } & {I_{17} } \end{array}} \cr {H_{1} } \cr {H_{2} } \cr {H_{3} } \cr {H_{4} } \end{array}\right) = \left( \begin{array}{c} {\begin{array}{ccccc} {I_{17} } & {I_{17} } & {I_{17} } & {I_{17} } & {I_{17} } \end{array}} \cr {C_{85}^{0} +C_{85}^{24} +C_{85}^{40} +C_{85}^{71} +C_{85}^{84} } \cr {C_{85}^{1} +C_{85}^{49} +C_{85}^{58} +C_{85}^{81} +C_{85}^{84} } \cr {C_{85}^{3} +C_{85}^{14} +C_{85}^{32} +C_{85}^{78} +C_{85}^{84} } \cr {C_{85}^{16} + C_{85}^{33} +C_{85}^{50} +C_{85}^{67} +C_{85}^{84} } \end{array} \right) 
 $$


To determine the values of  spherical coordinate system $\varphi $, $\theta $, we use the criterion for the presence of a cycle of length 6 (call it cycle based gauge) in the quasi-cyclic check matrix, (Fossorier04). Based on the electron cloud associated with an atom of a chemical element, we can refer this equation as the Schrödinger-Heisenberg-Bohr-Fossorier electron cloud Gauge (SHBF Cycle Gauge):

$$ 
\left( \sum_{i=1}^{N-1} \Delta_{ji,ji+1} \left( l_{i} \right) \right) \mod {k}=0 
$$

Now let's collapse the matrix along the radii:


$$ 
H= \left( \begin{array}{c} {\begin{array}{cccccc} {C_{8}^{0} } & {C_{8}^{0} } & {C_{8}^{2} } & {C_{8}^{2} } & {C_{8}^{2} } & {C_{8}^{2} } \end{array}} \cr {C_{48}^{1} +C_{48}^{7} +C_{48}^{13} +C_{48}^{19} +C_{48}^{25} +C_{48}^{31} } \cr {C_{48}^{23} +C_{48}^{17} +C_{48}^{47} +C_{48}^{41} +C_{48}^{35} +C_{48}^{29} } \end{array} \right)  \to \left( \begin{array}{c} { \begin{array}{cccccc} {I_{8} } & {I_{8} } & {I_{8} } & {I_{8}} & {I_{8}} & {I_{8} } \end{array}} \cr {C_{48}^{1} +C_{48}^{7} +C_{48}^{13} +C_{48}^{19} +C_{48}^{25} +C_{48}^{31} } \cr {C_{48}^{23} +C_{48}^{17} +C_{48}^{47} +C_{48}^{41} +C_{48}^{35} +C_{48}^{29} } \end{array} \right) 
$$



The first row, on which the collapse was carried out, contains 6 circulants of size 8 and weight 1. 2 shift 0 circulants correspond to the first energy level, 4 shift 2 circulants correspond to the second energy level (Carbon):


$$
1s^{2} ; 2s^{2} 2p^{2}
$$


Numerous optimization problems, including those in computer vision (Quantum  Computer Vision, \cite{Yu22}), can be converted into a Quadratic Unconstrained Binary Optimization (QUBO) form, \cite{Ble23}. However, QUBO problems are typically NP-complete, which implies that finding solutions through classical means necessitates exploring an exponentially growing solution space as problem size increases. In contrast, quantum computing holds the promise of exponentially faster computation due to the superposition nature of qubits. The exponentially expanding Hilbert space of a quantum system naturally accommodates the solution space of combinatorial optimization problems, offering potential advantages over classical machines in solving such problems. The Quantum Approximate Optimization Algorithm (QAOA) is specifically designed to address QUBO problems by utilizing a quantum circuit to find approximate solutions. 
The Quantum Approximate Optimization Algorithm (QAOA) used in the Sherrington-Kirkpatrick Ising model can be seen as analogous to the back-propagation loss function landscape in training DNNs. This similarity creates a comparable problem with TS pseudo-codeword, resembling the belief propagation method. 
QAOA can solve binary optimization problems known as QUBOs (Quadratic Unconstrained Binary Optimization). QUBO problems fall under the NP-complete class, guaranteeing that any NP-complete problem can be efficiently transformed into a QUBO problem. This mapping of Karp's 21 NP-complete problems to QUBO is extensively discussed in paper \cite{Lu23}. In addition to MaxCut, other relevant optimization problems such as Graph Coloring   \cite{Ta20}, Number Partitioning, and Quadratic Knapsack (\cite{Glo19}) have been successfully formulated as QUBO problems.


In a QUBO problem, the unknown vector $\mathbf{x} = (x_1,\ldots,x_n)$ consists of decision variables taking discrete binary values, i.e., $\mathbf{x} \in \{0,1\}^n$. The problem is defined by a square symmetric matrix $\mathbf{Q} \in \mathbb{R}^{n \times n}$. The objective is to find the optimal vector $\mathbf{x}^*$ that minimizes the cost function:


$$
C(\mathbf{x}) = \mathbf{x}^T \mathbf{Q} \mathbf{x} = \sum_{i, j=1}^{n} Q_{ij}x_i x_j
$$

QUBO problems can also be framed as maximization problems by inverting the sign of the cost function. It is important to note that QUBO problems do not have any constraints on the variables $\mathbf{x}$.

QUBO instances are closely related to Ising models. They can be mapped to each other with a one-to-one correspondence, where the QUBO variables $\mathbf{x} \in \{0,1\}^n$ are replaced by Ising variables $\mathbf{z} \in \{-1,1\}^n$, with $z_i = 2x_i - 1$ for $i=1,\ldots,n$. The Ising Hamiltonian, dependent on $\mathbf{z}$, is equivalent to the QUBO cost function, with a constant term irrelevant for optimization.




Additionally, the layer depth in QAOA correlates to the number of decoding belief propagation iterations in the Wiberg decoding tree.



Overall, this work has the potential to advance multiple fields, from Information Theory, DNN architecture design (sparse and structured prior graph topology), efficient hardware design for Quantum and Classical DPU/TPU (graph, quantize and shift register architect.) to Materials Science and beyond.


# Matrix factorization


Matrix factorization can be considered as a special case of (Ising spin-glass and equivivalent to them code on the graph) embedding, low dimension projection. Codes on the Graph based Sparse Matrix Factorization application represented bellow: 


The Sparse Factorization (SF) can thus be formulated as the following optimization problem, paper [1]:




$$
\mathop{\min }\limits_{W^{(1)} ,\ldots ,W^{(M)} } \left\| X-\prod _{m=1}^{M}W^{(M)}  \right\| _{F}^{2}
$$





where $W^{(M)} $'s are sparse square matrices with non-zero positions specified by the Chord protocol (SF Chord), LDPC codes parity-check  using PEG+ACE and QC-LDPC codes parity-check matrix, MET QC-LDPC codes with circulant more than 1 and product multigraph MET QC-LDPC codes parity-check using SA+EMD, Simulated Annealing with exact cycle extrinsic message degree optimization (EMD). 

![alt text](https://github.com/Lcrypto/Classical-and-Quantum-Topology-ML-toric-spherical/blob/main/Table_1_v2.png)


We modified the Matlab platform from the paper [1], as the base platform for Non-parametric Sparse Factorisation using LDPC codes, MET QC-LDPC codes and Multi-graph Product codes in our work https://github.com/RuslanKhalitov/SparseFactorization for using Code on the Graph: LDPC Codes constructed using Progressive Edge Grown method with ACE optimization [2,3,4]; QC-LDPC Codes, Multi-Edge QC-LDPC Code, Multigraph product code (Chord like) using Simulated Annealing method with EMD and code distance sieving optimization [5, 6]. 





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


**Yeast** (N = 200):This dataset, from the same repository as AuralSonar in Chen et al. (2009), converts the pairwise Smith-Waterman similarities $s_{ij}$ (Lanckriet et al., 2004; Xu et al., 2014) to dissimilarities by $d_{ij}=\sqrt{s_{ii}+s_{jj}-s_{ji}-s_{ij}}.$ 

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


# References:

-1. Schmidhuber J. Learning to control fast-weight memories: An alternative to recurrent nets. Technical Report FKI-147-91, Institut für Informatik, Technische Universität München, 26 March 1991. https://people.idsia.ch/~juergen/FKI-147-91ocr.pdf illustration https://people.idsia.ch/~juergen/fast-weight-programmer-1991-transformer.html


0.Divsalar D., H. Jin, and R. J. McEliece. "Coding theorems for ‘turbo-like’ codes." Proc. 36th Allerton Conf. on Communication, Control and Computing, Allerton, Illinois, Sept. 1998, pp. 201–210. 


0.Divsalar D.,  Jones C.,  Dolinar S. ,  Thorpe J. "Protograph based LDPC codes with minimum distance linearly growing with block size," IEEE Global Telecomm. Conf., 2005., St. Louis, MO, 2005, pp. 5 


0.Liva G. et al.’s "Simple Reconfigurable Low-Density Parity-Check Codes" in IEEE COMMUNICATIONS LETTERS, VOL. 9, NO. 2, FEBRUARY 2005


1.  **Ruslan Khalitov, Tong Yu, Lei Cheng, Zhirong Yang, Sparse factorization of square matrices with application to neural attention modeling, Neural Networks, Volume 152, 2022, Pages 160-168** https://github.com/RuslanKhalitov/SparseFactorization



2. Xiao-Yu Hu, Eleftheriou E., Arnold D. M.  "Regular and irregular progressive edge-growth tanner graphs," in IEEE TiT, vol. 51, no. 1, pp. 386-398, Jan. 2005,  Implementation available at: https://github.com/Lcrypto/classic-PEG-/


3. Diouf M., Declercq D., Fossorier  M., S. Ouya, B. Vasic, "Improved PEG construction of large girth QC-LDPC codes", 9th International Symposium on Turbo Codes and Iterative Information Processing (ISTC), pp. 146-150,2016.


4.  Usatyuk V., Minenkov A. Progressive edge growth for LDPC code construction C++ and Matlab PEG+ACE implementations, avaliable at
 https://github.com/Lcrypto/classic-PEG-


5. Usatyuk V. , Vorobyev I. "Simulated Annealing Method for Construction of High-Girth QC-LDPC Codes," 2018 41st International Conference on Telecommunications and Signal Processing (TSP), Athens, Greece, 2018, pp. 1-5 Implementation available at: https://github.com/Lcrypto/Simulated-annealing-lifting-QC-LDPC


6. Usatyuk V. S., Egorov S., Svistunov G. Construction of Length and Rate Adaptive MET QC-LDPC Codes by Cyclic Group Decomposition. IEEE East-West Design & Test Symposium (EWDTS), Batumi, Georgia, 2019, pp. 1-5  Implementation available at: https://github.com/Lcrypto/Length-und-Rate-adaptive-code


7. Usatyuk V. S., Sapozhnikov D.  Egorov S., Spherical and Hyperbolic Toric Topology-Based Codes On Graph Embedding for Ising MRF Models: Classical and Quantum Topology Machine Learning   https://arxiv.org/abs/2307.15778




# **Cite this reseach**
```
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
```
