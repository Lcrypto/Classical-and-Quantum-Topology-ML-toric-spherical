import numpy as np


from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from scipy.stats import zscore
from scipy.optimize import newton
from sklearn.cluster import KMeans
from functools import partial
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from NB_clustering._temperature_optimizations import find_beta_sg_dichotomy
from NB_clustering._similarity import similarity_dict

class NishimoriBetheSpectralClustering:
    """
    Class for spectral clustering using the Bethe-Hessian method and Nishimori temperature.

    Parameters:
    adjaceny_matrix (optional): Adjacency matrix of the graph. Default is None.
    n_clusters (optional): Number of clusters. Default is 2.
    kernel (optional): Kernel for the classifier. Default is None.
    sim (optional): Similarity measure for computing the affinity matrix. Default is None.
    is_z_scoring (optional): Flag for using z-scoring. Default is True.
    eps (optional): Precision for numerical methods. Default is 2e-5.
    random_state (optional): Initial state of the random number generator. Default is 0.
    state (optional): State of the model ('cluster' or 'classifier'). Default is 'cluster'.
    """
    def __init__(self, adjaceny_matrix=None, n_clusters=2,
                 kernel=None, sim=None, is_z_scoring=True,
                 eps=2e-5, random_state=0,
                 state='cluster'
                 ):
        self.adjanecy_matrix = coo_matrix(adjaceny_matrix)
        self.n_clusters = n_clusters
        self.kernel = kernel
        self.sim = sim
        self.is_z_scoring = is_z_scoring
        self.weights = None
        self.classifier = None
        self.random_state = random_state
        self.state=state
        self.embedding = None


    def _regularised_laplacian(self, beta, returnable='eigen'):
        """
        Computes the regularized Laplacian for a given beta value.

        Parameters:
        beta: Value of the beta parameter.
        returnable (optional): Type of return value ('eigen' or 'matrix'). Default is 'eigen'.

        Returns:
        If returnable='eigen', returns the smallest eigenvalue and corresponding eigenvector.
        If returnable='matrix', returns the Laplacian matrix.
        If returnable='all', returns the smallest eigenvalue, corresponding eigenvector, and the Laplacian matrix.
        """
        if True: # hack for choice simplify/accurate algorightm 
            W_data = 1/2 * np.sinh(2 * beta * self.affinity_matrix.data)
            Lam_data = np.sinh(beta * self.affinity_matrix.data) ** 2

        W = coo_matrix((W_data, (self.affinity_matrix.row, self.affinity_matrix.col)),\
                       shape=self.affinity_matrix.shape)
        Lam = coo_matrix((Lam_data, (self.affinity_matrix.row, self.affinity_matrix.col)),\
                       shape=self.affinity_matrix.shape)

        W = W + W.T
        Lam = Lam + Lam.T
        Lam = np.array(Lam.sum(axis=0) + 1)
        Lam = Lam ** -0.5
        Lam = diags(Lam.flatten())
        Id = diags(np.ones(Lam.shape[0]))
        laplacian = Id - Lam @ W @ Lam
        if returnable == 'matrix':
            return laplacian
        s, X = eigsh(laplacian, k=1, which='SA', return_eigenvectors=True)
        if returnable == 'eigen':
            return s[0], X[:, 0]
        return s[0], X[:, 0], laplacian



    def _beta_sg(self):
        """
        Computes the beta value for the spin-glass state using dichotomy method.
        """
        adjaceny = self.adjanecy_matrix + self.adjanecy_matrix.T
        d = adjaceny * np.ones(adjaceny.shape[0])
        c = np.mean(d)
        phi = np.mean(d ** 2) / c ** 2
        return find_beta_sg_dichotomy(self.affinity_matrix.data, c, phi)

    def _find_beta_N(self, num_iter=25, eps=1e-6):
        """
        Finds the beta value for the Nishimori temperature.

        Parameters:
        num_iter (optional): Number of iterations. Default is 25.
        eps (optional): Precision for numerical methods. Default is 1e-6.

        Returns:
        Beta value for the Nishimori temperature.
        """
        beta_sg = self._beta_sg()
        s, X, L = self._regularised_laplacian(beta_sg, returnable='all')
        
        if s > 0:
            print("\033[91mThe Nishimori temperature cannot be estimated on this matrix\033[0m")
            return beta_sg
        
        # Find upper bound for beta where func_lap becomes positive
        beta_positive = beta_sg
        for i in range(num_iter):
            beta_positive += beta_sg
            func_lap = X.T @ self._regularised_laplacian(beta_positive, returnable='matrix') @ X
            if func_lap > 0:
                break
        else:
            print(f'\033[31mPositive eigenvalue not found after {num_iter} iterations. Returning beta_sg.\033[0m')
            return beta_sg

        # Define the function we need to find root for
        def func_beta(beta):
            s_beta, _ = self._regularised_laplacian(beta, returnable='eigen')
            return s_beta

        # Dichotomy method implementation
        beta_low = beta_sg
        beta_high = beta_positive
        
        for _ in range(num_iter):
            beta_mid = (beta_low + beta_high) / 2
            f_mid = func_beta(beta_mid)
            
            if abs(f_mid) < eps:
                return beta_mid
                
            if f_mid < 0:
                beta_low = beta_mid
            else:
                beta_high = beta_mid
        
        # Return the best approximation if not converged
        print(f"\033[33mDichotomy didn't converge to required precision. Returning best approximation.\033[0m")
        return (beta_low + beta_high) / 2


    def bethe_hessian(self):
        pass#Beta_N = find_Beta_N(self)


    def _affinity_matrix(self, train):
        """
        Computes the affinity matrix based on the given similarity measure.

        Parameters:
        train: Training data.
        """
        # Choose the similarity measure
        similarity_measure = self.sim

        # Check if the key exists in the dictionary
        if similarity_measure in similarity_dict:
            # Call the corresponding function
            similarity_function = similarity_dict[similarity_measure]
        else:
            print(f'Invalid similarity measure: {similarity_measure}')
            exit()
        #Labeling for data, rows, and columns of weighted_adjanecy_matrix
        wam_data = self.adjanecy_matrix.data.copy()
        wam_row = self.adjanecy_matrix.row
        wam_col = self.adjanecy_matrix.col
        wam_shape = self.adjanecy_matrix.shape
        counter = 0
        for i, j in zip(wam_row, wam_col):
            wam_data[counter] = similarity_function(train[i], train[j])
            counter += 1
        if self.is_z_scoring:
            wam_data = zscore(wam_data)
        self.affinity_matrix = coo_matrix((wam_data, (wam_row, wam_col)), shape=wam_shape)



    def spectral_embedding(self):
        """
        Computes the spectral embedding of the data.

        Returns:
        Spectral embedding of the data.
        """
        beta_N = self._find_beta_N(25,1e-6)
        s, X, L = self._regularised_laplacian(beta=beta_N, returnable='all')
        W_lam_data = np.sinh(beta_N * self.affinity_matrix.data) ** 2
        W_lam = coo_matrix((W_lam_data, (self.affinity_matrix.row, self.affinity_matrix.col)),\
                       shape=self.affinity_matrix.shape)
        W_lam = W_lam + W_lam.T
        Lam = diags((1 / np.sqrt(1 + W_lam.dot(np.ones(W_lam.shape[0])))).flatten())
        return Lam @ X


    def fit(self, train, train_labels=None):
        """
        Trains the model on the given data.

        Parameters:
        train: Training data.
        train_labels (optional): Labels for the training data. Default is None.

        Returns:
        Cluster or class labels for the training data.
        """
        self._affinity_matrix(train=train)
        embedding = self.spectral_embedding()
        self.embedding = embedding
        self.weights = np.dot(np.linalg.pinv(train), embedding)

        if self.state == 'cluster':
            self.classifier = KMeans(n_clusters=self.n_clusters, n_init='auto', random_state=self.random_state)
            labels = self.classifier.fit_predict(embedding.reshape(-1, 1))

        elif self.state == 'classifier':
            self.classifier = SVC(kernel='poly')
            self.classifier.fit(embedding.reshape(-1, 1), train_labels)
            labels = self.classifier.predict(embedding.reshape(-1, 1))
        return labels


    def predict(self, sample):
        """
        Predicts labels for new data.

        Parameters:
        sample: New data for prediction.

        Returns:
        Cluster or class labels for the new data.
        """
        embedding = sample @ self.weights
        labels = self.classifier.predict(embedding.reshape(-1, 1))
        return labels

