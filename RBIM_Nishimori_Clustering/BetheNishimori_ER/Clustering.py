import numpy as np
from scipy.sparse import spdiags, csr_matrix
from scipy.sparse.linalg import eigs, eigsh
from sklearn.cluster import KMeans
from Basic_functions import find_β_SG, H_matrix

def clustering_MF(edge_list, J_edge_list, n, N_repeat=8, verbose=1):
    """
    This function performs spectral clustering on a weighted graph using the naive mean field approximation.

    Parameters
    ----------
    * `edge_list`   : Undirected edge list containing non-zero entries of the adjacency matrix A (numpy.ndarray)
    * `J_edge_list` : Weight corresponding to each edge in `edge_list` (numpy.ndarray)
    * `n`           : Matrix dimension (int)

    Optional Parameters
    ------------------
    * `verbose`     : Verbosity level (0 = no output, higher values = more output). Default: 2.
    * `N_repeat`    : Number of repetitions for the k-means algorithm. Default: 8.

    Returns
    -------
    * `X`            : Eigenvector with the largest real part ('LR' sorting)
    * `estimated_ℓ`  : Array containing assigned labels from the set {-1, 1} (numpy.ndarray)
    """
    J = csr_matrix((J_edge_list, (edge_list[:, 0], edge_list[:, 1])), shape=(n, n))
    J = J + J.T

    s, X = eigs(J, k=1, which='LR', return_eigenvectors=True)
    X = X.real
    if verbose >= 1:
        print("\033[92m" + "Running kmeans" + "\033[0m")

        kmeans_models = [KMeans(n_clusters=2, n_init=10).fit(X[:, 0].reshape(-1, 1)) for _ in range(N_repeat)]
        f = [km.inertia_ for km in kmeans_models]
        best = np.argmin(f)
        estimated_l = kmeans_models[best].labels_

    if verbose >= 1:
        print("\033[92m" + "Done!" + "\033[0m\n")

    return X, 2 * estimated_l - 1

def clustering_BH_SG(edge_list, J_edge_list, n, ϵ=2e-5, N_repeat=8, verbose=1, t=1.0):
    """
    This function performs spectral clustering on a weighted graph using the Bethe Hessian matrix 
    at the spin glass transition temperature.

    Usage:
    ----------
    `X, estimated_l = clustering_BH_SG(edge_list, J_edge_list, n)`

    Parameters:
    ----------
    * `edge_list`   : Undirected edge list containing non-zero entries in the adjacency matrix A (numpy.ndarray)
    * `J_edge_list` : Weight corresponding to each edge in `edge_list` (numpy.ndarray)
    * `n`           : Matrix dimension (int)

    Optional Parameters:
    ----------
    * `ε`       : Estimation precision. Default: 2e-5
    * `verbose` : Verbosity level (0 = no output, higher values = more output). Default: 2
    * `N_repeat`: Number of repetitions for the k-means algorithm. Default: 8
    * `t`       : Temperature scaling factor where β = β_SG * t. The x-value is given by 
                sqrt{cΦ * E[tanh²(β * J)]} (float). Default: 1.0

    Returns:
    ----------
    * `X`            : Real-valued eigenvector with the smallest real part ('SA' sorting) (numpy.ndarray)
    * `estimated_l`  : Array containing assigned labels from the set {-1, 1} (numpy.ndarray)
    """
    A = csr_matrix((np.ones(len(edge_list)), (edge_list[:, 0], edge_list[:, 1])), shape=(n, n))
    A = A + A.T
    #d = np.dot(A, np.ones(n))
    d = A * np.ones(n)
    c = np.mean(d)
    Φ = np.mean(d**2) / c**2

    β_SG = find_β_SG(J_edge_list, c, Φ, ϵ)
    if verbose >= 1:
        print(f"\033[93mThe value of β_SG is {round(β_SG, 2):.2f}\n")

    β = t * β_SG
    w_edge_list = np.tanh(β * J_edge_list)
    x = np.sqrt(c * Φ * np.mean(w_edge_list**2))
    H = H_matrix(edge_list, w_edge_list, x, n)
    #sy, Y = eigs(H, k=1, which='SR', return_eigenvectors=True)
    s, X = eigsh(H, k=1, which='SA', return_eigenvectors=True)
    X = X.real
    if verbose >= 1:
        print("Running kmeans")

    kmeans_models = [KMeans(n_clusters=2, n_init=10).fit(X[:, 0].reshape(-1, 1)) for _ in range(N_repeat)]
    f = [km.inertia_ for km in kmeans_models]
    best = np.argmin(f)
    KM = kmeans_models[best]
    estimated_l = KM.labels_

    if verbose >= 1:
        print("Done!\033[0m\n")

    return X[:, 0], 2 * estimated_l - 1

def clustering_signed_Lap(edge_list, J_edge_list, n, eps=2e-5, N_repeat=8, verbose=1):
    """
    Function for spectral clustering on a weighted graph using the signed Laplacian matrix.

    Usage
    ----------
    `X, estimated_ell = clustering_signed_Lap(edge_list, J_edge_list, n, ε=2e-5, N_repeat=8, verbose=1)`

    Parameters
    ----------
    * `edge_list`   : Undirected edge list containing non-zero entries of the adjacency matrix A (numpy.ndarray)
    * `J_edge_list` : Weight corresponding to each edge in `edge_list` (numpy.ndarray)
    * `n`           : Matrix dimension (int)

    Optional Parameters
    ------------------
    * `ε`           : Estimation precision. Default: 2e-5
    * `N_repeat`    : Number of repetitions for the k-means algorithm. Default: 8
    * `verbose`     : Verbosity level (0, 1, 2 with increasing detail). Default: 1

    Returns
    -------
    * `X`             : Informative eigenvector corresponding to the smallest real eigenvalue (numpy.ndarray)
    * `estimated_ell` : Array containing cluster assignment labels (numpy.ndarray)
    """
    L = Lap_matrix(edge_list, J_edge_list, n)  # create the Laplacian matrix
    s, X = eigs(L, k=1, which='SR', return_eigenvectors=True)  # compute the smallest eigenpair
    if verbose >= 1:
        print("\033[95mRunning kmeans")

    fKM = [KMeans(n_clusters=2, n_init=10).fit(X.real[:, 0].reshape(-1, 1)) for _ in range(N_repeat)]  # run k-means on the entries of the eigenvector
    f = [fkm.inertia_ for fkm in fKM]
    best = np.argmin(f)  # pick the best trial
    KM = fKM[best]
    estimated_l = KM.labels_  # find the label assignments as an output of kmeans.

    if verbose >= 1:
        print("Done!\033[90m\n")

    return X.real[:, 0], 2 * estimated_l - 1

def Lap_matrix(edge_list, w, n):
    """
    This function generates the weighted Laplacian matrix corresponding to a given weighted edge list.

    Usage
    ----------
    `L = Lap_matrix(edge_list, J_edge_list, n)`

    Parameters
    ----------
    * `edge_list`   : Undirected edge list corresponding to non-zero entries of the adjacency matrix A (numpy.ndarray)
    * `J_edge_list` : Weight corresponding to each edge in `edge_list` (numpy.ndarray)
    * `n`           : Matrix dimension (int)

    Returns
    -------
    `L` : Sparse weighted Laplacian matrix (scipy.sparse.csr_matrix)
    """
    W = csr_matrix((w, (edge_list[:, 0], edge_list[:, 1])), shape=(n, n))
    W = W + W.T

    absW = csr_matrix((np.abs(w), (edge_list[:, 0], edge_list[:, 1])), shape=(n, n))
    absW = absW + absW.T

    diag_vals = np.array(absW.sum(axis=1)).flatten()
    Lambda = csr_matrix((diag_vals, (np.arange(n), np.arange(n))), shape=(n, n))

    L = Lambda - W

    return L


