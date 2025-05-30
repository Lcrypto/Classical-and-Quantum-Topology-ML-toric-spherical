import numpy as np

#metric file
def cos_sim(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    similarity = dot_product / (norm_A * norm_B)
    return similarity


def euclidean_sim(A, B):
    distance = np.linalg.norm(A - B)
    similarity = 1 / (1 + distance)
    return similarity


def minkowski_sim(A, B, p=2):
    distance = np.sum(np.abs(A - B) ** p) ** (1 / p)
    similarity = 1 / (1 + distance)
    return similarity


def manhattan_sim(A, B):
    distance = np.sum(np.abs(A - B))
    similarity = 1 / (1 + distance)
    return similarity


def chebyshev_sim(A, B):
    distance = np.max(np.abs(A - B))
    similarity = 1 / (1 + distance)
    return similarity


def mahalanobis_sim(A, B):
    diff = A - B
    cov_matrix = np.cov(np.vstack((A, B)))
    distance = np.sqrt(np.dot(np.dot(diff, np.linalg.inv(cov_matrix)), diff))
    similarity = 1 / (1 + distance)
    return similarity


def pearson_sim(A, B):
    correlation = np.corrcoef(A, B)[0, 1]
    similarity = correlation
    return similarity

similarity_dict =\
    {
        'cos': cos_sim,
        'euclidean': euclidean_sim,
        'minkowski': minkowski_sim,
        'manhattan': manhattan_sim,
        'chebyshev': chebyshev_sim,
        'mahalanobis': mahalanobis_sim,
        'pearson': pearson_sim
    }