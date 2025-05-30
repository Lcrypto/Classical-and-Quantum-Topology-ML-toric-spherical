import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

from Basic_functions import find_β_SG


def clustering_BH_Nishimori(edge_list, J_edge_list, n, ϵ=2e-5, N_repeat=8, is_signed_th=1e-3, beta_threshold=2., verbose=1, n_max=25):
    """
    This function performs spectral clustering on a weighted graph using the Bethe Hessian matrix 
    at the Nishimori temperature.

    Usage
    ----------
    `X, estimated_ℓ = clustering_BH_Nishimori(edge_list, J_edge_list, n)`

    Parameters
    ----------
    * `edge_list`   : Undirected edge list containing non-zero entries of the adjacency matrix A (numpy.ndarray)
    * `J_edge_list` : Weight corresponding to each edge in `edge_list` (numpy.ndarray)
    * `n`           : Matrix dimension (int)

    Optional Parameters
    ------------------
    * `ε`             : Estimation precision (default: 2e-5)
    * `verbose`       : Verbosity level (default: 2)
    * `N_repeat`      : Number of repetitions for the k-means algorithm (default: 8)
    * `is_signed_th`  : Threshold for using signed representation (default: 1e-3)
    * `beta_threshold`: Maximum allowable β_N value (default: 2)
    * `n_max`         : Maximum number of iterations for β_N convergence (default: 25)

    Returns
    -------
    * `X`            : Eigenvector corresponding to the smallest real eigenvalue (numpy.ndarray)
    * `estimated_ℓ`  : Array containing assigned labels from the set {-1, 1} (numpy.ndarray)
    """
    A = sparse.csr_matrix((np.ones(len(edge_list)), (edge_list[:, 0], edge_list[:, 1])), shape=(n, n))
    A = A + A.T
    d = A * np.ones(n)
    c = np.mean(d)
    Phi = np.mean(d**2) / c**2

    β_SG = find_β_SG(J_edge_list, c, Phi, ϵ)
    if verbose >= 1:
        print(f"\033[94mThe value of beta_SG is {round(β_SG, 2):.2f}. Computing beta_N")

    variance_J = np.sqrt(np.var(np.tanh(β_SG * J_edge_list)**2)) / β_SG

    if variance_J < is_signed_th:
        signed = True
        J_edge_list = np.sign(J_edge_list)
        if verbose >= 1:
            print("\nThe signed representation of J is adopted. If you want to use the weighted one, increase the value of `is_signed_th`. The algorithm might have a sensible slow down")
    else:
        signed = False

    if not signed:
        β_N = find_β_N(edge_list, J_edge_list, n, c, β_SG, ϵ=ϵ, verbose=verbose, n_max=n_max)
        L = L_matrix(edge_list, J_edge_list, β_N, n)
        W_Λ = sparse.csr_matrix((np.sinh(β_N * J_edge_list)**2, (edge_list[:, 0], edge_list[:, 1])), shape=(n, n))
        W_Λ = W_Λ + W_Λ.T
        Λ = sparse.diags((1 / np.sqrt(1 + W_Λ.dot(np.ones(n)))).flatten())
        s, X = eigs(L, k=1, which='SR', return_eigenvectors=True)
        s = s.real
        X = X.real
        X = Λ @ X
        #X = np.array([Λ.dot(X)])
    #else:
        # beta_N = find_β_N_signed(edge_list, J_edge_list, n, c, beta_SG, eps=eps, verbose=verbose)
        # r = 1/np.tanh(β_N)
        # H = H_matrix_signed(edge_list, J_edge_list, r, n)
        # s, X = eigs(H, k=1, which='SR', return_eigenvectors=True)

    if verbose >= 1:
        print(f"The value of beta_N is {round(β_N, 2):.2f}")

    if verbose >= 1:
        print("Running kmeans")
    kmeans_models = [KMeans(n_clusters=2, n_init=10).fit(X[:, 0].reshape(-1, 1)) for _ in range(N_repeat)]
    f = [km.inertia_ for km in kmeans_models]
    best = np.argmin(f)
    KM = kmeans_models[best]
    estimated_l = KM.labels_


    if verbose >= 1:
        print("Done!\033[0m")

    return X, 2 * estimated_l - 1

def find_β_N(edge_list, J_edge_list, n, c, β_SG, n_max=25, ϵ=2*10^(-5), β_threshold=2., verbose=1):
    """
    This function estimates the Nishimori temperature for a random graph according to Algorithm 1.

    Usage
    ----------
    `β_N = find_β_N(edge_list, J_edge_list, n, c, β_SG)`

    Parameters
    ----------
    * `edge_list`   : Undirected edge list containing non-zero entries of the adjacency matrix A (numpy.ndarray)
    * `J_edge_list` : Weight corresponding to each edge in `edge_list` (numpy.ndarray)
    * `n`           : Matrix dimension (int)
    * `c`           : Average graph degree (float)
    * `β_SG`        : Spin glass phase transition temperature value (float)

    Optional Parameters
    ----------
    * `n_max`         : Maximum number of iterations for the algorithm (default: 25)
    * `ε`             : Precision for estimated values (default: 2e-5)
    * `β_threshold`   : Maximum computable β_N value = sqrt(c)*β_SG*β_threshold (default: 2)
    * `verbose`       : Verbosity level (0 = no output, higher values = more output) (default: 2)

    Returns
    -------
    `β_N` : Estimated Nishimori temperature (float)
    """
    flag = 0
    counter = 1
    β_old = 2*β_SG

    # Check if β_N > β_SG
    L = L_matrix(edge_list, J_edge_list, β_SG, n)
    s, X = eigs(L, 1, which='SR')
    s = s.real
    X = X.real
    x = X.real

    if s[0] > 0:
        flag = 1
        if verbose >= 1:
            print("\033[91mThe Nishimori temperature cannot be estimated on this matrix\033[0m")
        return β_SG

    β = β_SG

    while flag == 0:

        if counter > 1:

            L = L_matrix(edge_list, J_edge_list, β, n)
            s, X = eigs(L, 1, which='SR')
            s = s.real
            X = X.real
            x = X.real

        flag = exitFromLoop(counter, n_max, verbose, s.real, c, β, J_edge_list, ϵ, β_old, β_threshold, β_SG)

        if flag == 0:

            β_large = find_β_large(β, β_SG, edge_list, J_edge_list, n, x)
            beta_old = β
            β= find_zero(β, β_large, ϵ, c, edge_list, J_edge_list, n, x)

        if verbose >= 2:
            print("\nIteration # ", counter, ": ",
                  "\nThe current estimate of beta_N is ", β,
                  "\nThe smallest eigenvalue is ", s[0]*c*(1+np.mean(np.sinh(β_SG*J_edge_list)**2)), "\n")

        counter += 1

    return β
def L_matrix(edge_list, J_edge_list, β, n):
    """
    This function generates a regularized Laplacian matrix (which differs from Lap_matrix by including temperature),
    corresponding to a given weighted edge list as described in Appendix B.

    Usage
    ----------
    `L = L_matrix(edge_list, J_edge_list, β, n)`

    Parameters
    ----------
    * `edge_list`   : Undirected edge list containing non-zero entries of the H matrix (numpy.ndarray of shape [m, 2])
    * `J_edge_list` : Weights corresponding to each edge in `edge_list` (numpy.ndarray of shape [m])
    * `β`           : Temperature parameter value (float)
    * `n`           : Matrix dimension (int)

    Returns
    -------
    `L` : Sparse regularized Laplacian matrix (scipy.sparse.csc_matrix)
    """

    w = 1/2 * np.sinh(2 * β * J_edge_list)
    w2 = np.power(np.sinh(β * J_edge_list), 2)

    W = sparse.coo_matrix((w, (edge_list[:, 0], edge_list[:, 1])), shape=(n, n))
    W = W + W.T

    Λ = sparse.coo_matrix((w2, (edge_list[:, 0], edge_list[:, 1])), shape=(n, n))
    Λ = Λ + Λ.T

    diag_list = (Λ.sum(axis=1) + 1).ravel().tolist()[0]
    diag_list_05 = np.array(diag_list) ** -0.5
    Λ_05 = sparse.diags(diag_list_05)


    Id = sparse.diags(np.ones(n))

    L = Id - Λ_05 @ W @ Λ_05

    return L.tocsc()
def find_β_large(β_small, β_SG, edge_list, J_edge_list, n, x):
    """
    Finds the value of β where x'*L*x becomes positive.

    Usage
    ----------
    `β_large = find_β_large(β_small, β_SG, edge_list, J_edge_list, n, x)`

    Parameters
    ----------
    * `β_small`     : Left boundary of the interval containing the zero of x'*L*x (float)
    * `β_SG`        : Spin glass phase transition temperature (float)
    * `edge_list`   : Undirected edge list containing non-zero entries of the J matrix (numpy.ndarray)
    * `J_edge_list` : Weight corresponding to each edge in `edge_list` (numpy.ndarray)
    * `n`           : Matrix dimension (int)
    * `x`           : Vector x (numpy.ndarray)

    Returns
    -------
    `β_large` : Right boundary of the interval containing the zero of x'*L*x (float)
    """
    
    flag = 0
    β_large = β_small

    while flag == 0:

        β_large += β_SG
        L = L_matrix(edge_list, J_edge_list, β_large, n)  # compute the matrix L for the new value of β
        f = x.T @ L @ x#np.dot(x, np.dot(L, x.T))
        if f > 0:
            flag = 1

    return β_large
def find_zero(β_small, β_large, ϵ, c, edge_list, J_edge_list, n, x):
    """
    This function estimates the value of β where x'*L*x = 0.

    Usage
    ----------
    `β = find_zero(β_small, β_large, ε, edge_list, J_edge_list, n, x)`

    Parameters
    ----------
    * `β_small`     : Left boundary of the interval containing the zero (float)
    * `β_large`     : Right boundary of the interval containing the zero (float)
    * `ε`           : Estimation precision (float)
    * `edge_list`   : Undirected edge list containing non-zero entries of the J matrix (numpy.ndarray of shape [m, 2])
    * `J_edge_list` : Weight corresponding to each edge in `edge_list` (numpy.ndarray of shape [m])
    * `n`           : Matrix dimension (int)
    * `x`           : Vector x (numpy.ndarray of shape [n])

    Returns
    -------
    `β` : Estimated zero of the function x'*L*x (float)
    """
    
    δ = 1 # инициализация ошибки

    while δ > ϵ:

        β = (β_large + β_small) / 2 # использование метода бисекции для нахождения нуля
        L = L_matrix(edge_list, J_edge_list, β, n) # вычисление матрицы L для нового значения β
        y = x.T @ L @ x#np.dot(x.T, np.dot(L, x)) # используя теорему Куранта-Фишера
        f = y * c * (1 + np.mean(np.sinh(β * J_edge_list)**2))

        if f > 0: # обновление значения β в соответствии с значением f
            β_large = β
        else:
            β_small = β

        δ = min(abs(f), β_large - β_small) # обновление ошибки

    return β_small

def exitFromLoop(counter, n_max, verbose, s, c, beta, J_edge_list, eps, beta_old, beta_threshold, beta_SG):
    """
    This function checks exit conditions for the β_N estimation loop in the weighted case.

    Usage
    ----------
    `flag = exitFromLoop(counter, n_max, verbose, s, c, beta, J_edge_list, eps, beta_old, beta_threshold, beta_SG)`

    Parameters
    ----------
    * `counter`        : Number of completed iterations (int)
    * `n_max`          : Maximum allowed iterations (int)
    * `verbose`        : Message verbosity level (0, 1, or 2) (int)
    * `s`              : Smallest converging eigenvalues of the L matrix (array of shape [1,])
    * `c`              : Average graph degree (float)
    * `beta`           : Current temperature value (float)
    * `J_edge_list`    : Edge weights (array of shape [m])
    * `eps`            : Precision tolerance (float)
    * `beta_old`       : Previous β_N estimate (float)
    * `beta_threshold` : Maximum computable β_N = sqrt(c)*β_SG*beta_threshold (float)
    * `beta_SG`        : Spin glass transition temperature (float)

    Returns
    -------
    `flag` : Boolean (0 or 1). If flag=1, the loop should terminate (bool)
    """

    flag = 0

    # exit from loop, when max iter reached
    if counter > n_max:
        if verbose >= 1:
            print("\n Maximum number of iterations reached. Algorithm stopped. For higher precision, increase the maximum allowed iterations.")
        flag = 1

    # exit from loop, when min eigen value of matrix L for recent  β>0
    if s[0] > 0:
        flag = 1

    # exit from loop, when min eigen value of matrix L close to zero
    if abs(s[0]*(1+c*np.mean(np.sinh(beta_SG*J_edge_list)**2))) < eps:
        flag = 1

    # exit from loop, when change of β estimation < eps/4
    if abs(beta_old - beta) < eps/4:
        if verbose >= 1:
            print("\nChange in β estimates is below eps/4. Algorithm terminated as convergence is unlikely. Increase the eps value to avoid this situation.")
        flag = 1

    # exit from loop, when  β  too big
    if beta > beta_threshold*np.sqrt(c)*beta_SG:
        if verbose >= 1:
            print("\nThe estimated β value is too large: this indicates you are in a scenario where the Nishimori Bethe Hessian becomes numerically unstable, but where the problem is simple enough for mean field approximation. β_N estimation aborted.")
        flag = 1

    return flag
