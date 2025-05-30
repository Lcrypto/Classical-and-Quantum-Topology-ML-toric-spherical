import networkx as nx
import numpy as np
from scipy import sparse
import scipy.sparse as sp
from scipy.sparse import csc_matrix, spdiags
import random

seed = 0
random.seed(seed)
np.random.seed(seed)
def adjacency_matrix_ER(c, n):
    """
    This function generates an Erdős–Rényi graph with average degree `c` and returns the graph
    as an adjacency matrix representation along with an edge list.

    Usage
    ----------
    `A, edge_list = adjacency_matrix_ER(c, n)`

    Parameters
    ----------
    * `c` : Average degree (float)
    * `n` : Number of nodes (int)

    Returns
    -------
    * `A`         : Sparse adjacency matrix representation (scipy.sparse.csr_matrix)
    * `edge_list` : Graph edge list (numpy.ndarray)
    """
    g = nx.erdos_renyi_graph(n, c/n) # граф Эрдёша-Реньи, взятый из пакета networkx

    A = nx.adjacency_matrix(g)
    I, J = A.nonzero()
    edge_list = np.column_stack((I, J))
    idx = I > J # мы используем неориентированный список рёбер
    edge_list = edge_list[idx,:]
    A = sp.csr_matrix(A)
    return A, edge_list

def adjacency_matrix_DCER(c, theta):
    """
    This function generates a sparse degree-corrected Erdős–Rényi graph. 
    Note that this implementation is not intended for dense graphs.

    Usage
    ----------
    `A, edge_list =  adjacency_matrix_DCER(c, θ)`

    Parameters
    ----------
    * `c`   : Expected average degree (float)
    * `θ`   : Expected normalized degree vector (numpy.ndarray)

    Returns
    -------
    * `A`         : Sparse adjacency matrix representation (scipy.sparse.csr_matrix)
    * `edge_list` : Graph edge list (numpy.ndarray)
    """
    theta_normalized = theta / sum(theta)
    n = len(theta) # количество узлов
    fs = np.random.choice(np.arange(n), size=int(n * c), replace=True, p=theta_normalized)
    ss = np.random.choice(np.arange(n), size=int(n * c), replace=True, p=theta_normalized)
    #ошибочный вариант, потому что описание в Julia не соответствует реализации
    #fs = np.random.choice(np.arange(1, n+1), size=int(n*c), replace=True, p=theta/n)
    #ss = np.random.choice(np.arange(1, n+1), size=int(n*c), replace=True, p=theta/n)

    idx = fs > ss
    fs = fs[idx]
    ss = ss[idx]

    edge_list = np.column_stack((fs, ss))
    edge_list = np.unique(edge_list, axis=0)
    A = sp.csr_matrix((np.ones(len(edge_list)), (edge_list[:, 0], edge_list[:, 1])), shape=(n, n))
    return A + A.T, edge_list

def get_weighted_graph(edge_list, mu=1.0, nu=2.0, p=0.7):
    """
    This function initializes the parameters for the simulation and
    returns a weighted graph. In this case β_N = μ/ν^2
    """
    l = np.ones(len(edge_list[:, 0]))  # label vector
    l[: len(edge_list[:, 0]) // 2] = -1
    # Initialize the J_edge_list array with zeros

    J_edge_list = np.zeros(len(edge_list[:, 0]))

    # With this bit of code we get a weighted graph

    for k in range(len(J_edge_list)):
        a = edge_list[k, 0]
        b = edge_list[k, 1]
        J_edge_list[k] = np.random.normal(mu * l[a] * l[b], nu)

    return J_edge_list

def H_matrix(edge_list, w_edge_list, x, n):
    """
    This function generates the matrix H(x) defined in Equation 12.

    Usage
    ----------
    `H = H_matrix(edge_list, w_edge_list, x, n)`

    Parameters
    ----------
    * `edge_list`   : Undirected edge list containing non-trivial elements of the H matrix (numpy.ndarray)
    * `w_edge_list` : Weight corresponding to each edge in `edge_list` (numpy.ndarray)
    * `x`           : Parameter value x under consideration (float)
    * `n`           : Matrix dimension (int)

    Returns
    -------
    `H` : Sparse representation of the H(x) matrix (scipy.sparse.csr_matrix)
    """
    wd = w_edge_list ** 2 / (x ** 2 - w_edge_list ** 2)
    w = x * w_edge_list / (x ** 2 - w_edge_list ** 2)

    W = sparse.csr_matrix((w, (edge_list[:, 0], edge_list[:, 1])), shape=(n, n))
    W = W + W.T

    Λ = sparse.csr_matrix((wd, (edge_list[:, 0], edge_list[:, 1])), shape=(n, n))
    Λ = Λ + Λ.T
    #Λ = sparse.diags(Λ.sum(axis=1) + 1, dtype=float)
    diag_list = (Λ.sum(axis=1) + 1).ravel().tolist()[0]
    Λ = sparse.diags(diag_list)
    H = Λ - W

    return H

def find_β_F(J_edge_list, c, Φ, ϵ = 2 * 10 ** (-5)):
    """
    This function finds the ferromagnetic to spin glass phase transition temperature β_F. 
    The function accounts for possible degree heterogeneity via the parameter 
    Φ = E[d^2]/E^2[d], where `d` denotes the degree.

    Usage
    ----------
    `β_F = find_β_F(J_edge_list, c, Φ, ϵ)`

    Parameters
    ----------
    * `J_edge_list` : Weight corresponding to each edge in the edge list (numpy.ndarray)
    * `c`           : Average degree (float)
    * `Φ`           : Second moment of the normalized degree distribution (float)
    * `ϵ`           : Estimation precision (float)

    Returns
    -------
    `β_F` : Transition temperature value (float)
    """

    β_small = 0 # β_F > β_small
    β_large = 1/c # init of right interval, which contains β_F

    flag = 0

    while flag == 0:
        f_now = c*Φ*np.mean(np.tanh(β_large*J_edge_list)) - 1
        if f_now > 0:
            flag = 1
        else:
            β_large += 1/c # increment value β_large until c*Φ*mean(tanh.(β_large*J_edge_list)) - 1 > 0

    δ = 1 # solve equation, find roots c*Φ*mean(tanh.(β_large*J_edge_list)) - 1 = 0 using Dichotomy (bisection) method
    while δ > ϵ:
        β_new = (β_small + β_large)/2
        δ = abs(β_large - β_small)
        if c*Φ*np.mean(np.tanh(β_new*J_edge_list)) - 1 > 0:
            β_large = β_new
        else:
            β_small = β_new

    return β_new

def find_β_SG(J_edge_list, c, Φ, ϵ = 2 * 10 ** (-5)):
    """
    This function finds the paramagnetic to spin glass phase transition temperature β_SG. 
    The function accounts for possible degree heterogeneity via the parameter 
    Φ = E[d^2]/E^2[d], where `d` denotes the degree.

    Usage
    ----------
    `β_SG = find_β_SG(J_edge_list, c, Φ, ϵ)`

    Parameters
    ----------
    * `J_edge_list` : Weight corresponding to each edge in the edge list (numpy.ndarray)
    * `c`           : Average degree (float)
    * `Φ`           : Second moment of the normalized degree distribution (float)
    * `ϵ`           : Estimation precision (float)

    Returns
    -------
    `β_SG` : Transition temperature value (float)
    """

    β_small = 0  # β_SG > β_small
    β_large = 1 / c  # initialization of the right edge of the interval containing β_SG
    flag = 0

    while flag == 0:
        f_now = c * Φ * np.mean(np.tanh(β_large * J_edge_list) ** 2) - 1
        if f_now > 0:
            flag = 1
        else:
            β_large += 1 / c  # increase the value of β_large until c*Φ*mean(tanh.(β_large*J_edge_list).^2) - 1 > 0

    # find the solution to c*Φ*mean(tanh.(β_large*J_edge_list).^2) - 1 = 0 with the bisection method

    δ = 1
    while δ > ϵ:
        #global β_new = (β_small + β_large) / 2
        β_new = (β_small + β_large) / 2
        δ = abs(β_large - β_small)
        if c * Φ * np.mean(np.tanh(β_new * J_edge_list) ** 2) - 1 > 0:
            β_large = β_new
        else:
            β_small = β_new

    return β_new


def B_matrix(edge_list, w_edge_list):
    """
    This function generates a weighted adjacency matrix from a list of (directed) graph edges 
    and their corresponding weights. This implementation adapts code from the Erdos.jl package.

    Usage
    ----------
    `B = B_matrix(edge_list, w_edge_list)`

    Parameters
    ----------
    * `edge_list`   : Array containing the graph's edge list (numpy.ndarray of shape [m, 2])
    * `w_edge_list` : Array containing weights for each edge (numpy.ndarray of shape [m])

    Returns
    -------
    * `B` : Sparse representation of the adjacency matrix without back-traces (scipy.sparse.csc_matrix)
    """

    d_edge_list = np.zeros((2*len(edge_list), 2)) # create the directed edge list
    d_edge_list[0:len(edge_list),:] = edge_list
    d_edge_list[len(edge_list):,:] = edge_list[:,::-1]

    d_edge_list = d_edge_list.astype(int)

    w_d_edge_list = np.zeros(2*len(edge_list)) # associate the weights to the directed edge list
    w_d_edge_list[0:len(edge_list)] = w_edge_list
    w_d_edge_list[len(edge_list):] = w_edge_list[::-1]

    edge_id_map = {}
    for i in range(len(d_edge_list)):
        edge = tuple(d_edge_list[i])
        edge_id_map[edge] = i

    nb1 = []
    nb2 = []
    nb_w = []

    n = max(max(edge_list[:, 0]), max(edge_list[:, 1])) + 1
    neighbours = []
    for i in range(n):
        neighbours.append([])
    for edge in edge_list:
        neighbours[edge[0]].append(edge[1])
        neighbours[edge[1]].append(edge[0])

    for (e, u) in edge_id_map.items():
        i, j = e
        for k in neighbours[i]:
            if k == j:
                continue
            v = edge_id_map[(k, i)]
            nb1.append(u)
            nb2.append(v)
            nb_w.append(w_d_edge_list[v])

    B = csc_matrix((nb_w, (nb1, nb2)), shape=(len(d_edge_list), len(d_edge_list))) # create sparse adjacency matrix

    return B


def F_matrix(edge_list, w_edge_list, λ, n):
    """
    This function generates the matrix F(λ) as described in Appendix A of the paper.

    Usage
    ----------
    `F = F_matrix(edge_list, w_edge_list, λ, n)`

    Parameters
    ----------
    * `edge_list`   : Undirected edge list containing non-zero entries of the H matrix (numpy.ndarray of shape [m, 2])
    * `w_edge_list` : Weight corresponding to each edge in `edge_list` (numpy.ndarray of shape [m])
    * `λ`           : Eigenvalue of matrix B under consideration (complex)
    * `n`           : Matrix dimension (int)

    Returns
    -------
    * `F` : Sparse representation of the F(λ) matrix (scipy.sparse.csc_matrix)
    """

    wd = w_edge_list ** 4 / (λ ** 2 - w_edge_list ** 2)
    w = λ * w_edge_list ** 3 / (λ ** 2 - w_edge_list ** 2)

    W = sparse.csr_matrix((w, (edge_list[:, 0], edge_list[:, 1])), shape=(n, n))
    W = W + W.T

    Λ = sparse.csr_matrix((wd, (edge_list[:, 0], edge_list[:, 1])), shape=(n, n))
    Λ = Λ + Λ.T
    Λ = sparse.diags((Λ.sum(axis=1) + 1).flatten().tolist(), [0], dtype=float)

    F = Λ - W

    return F