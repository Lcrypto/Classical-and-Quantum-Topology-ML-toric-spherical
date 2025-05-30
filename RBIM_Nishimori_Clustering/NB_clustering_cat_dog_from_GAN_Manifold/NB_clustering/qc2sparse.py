import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

def create_circulant_matrix(n, k):
    # Create circulant permutation matrix of n x n with index k
    row = np.arange(n)
    col = (row + k) % n
    data = np.ones(n)
    return csr_matrix((data, (row, col)), shape=(n, n))

def parse_input_file(filename):
    # Parse file for size matrix and numbers  of circulant permutation matrixes
    with open(filename, 'r') as f:
        # read parameters
        n, m, block_size = map(int, f.readline().split())

        # information about cpm
        blocks = []
        for i in range(m):
            row_blocks = f.readline().split()
            for j, block_str in enumerate(row_blocks):
                # print(f"Rows: {i}, Columns: {j}, CPM: {block_str}")
                if block_str == '-1':
                    # Zero circulant block
                    block = csr_matrix((block_size, block_size))
                else:
                    # Block is summ of several circulant ( Multi-edge Graph) 
                    indices = list(map(int, block_str.split('&')))
                    block = sum(create_circulant_matrix(block_size, k) for k in indices)
                blocks.append((i, j, block))

    return n, m, block_size, blocks

def build_sparse_matrix(filename):
    # Construct sparse matrix
    n, m, block_size, blocks = parse_input_file(filename)

    # create initial sparse matrix
    matrix = csr_matrix((m * block_size, n * block_size))

    # Translate to format LIL
    matrix = matrix.tolil()

    # place circulant blocks
    for row, col, block in blocks:
        matrix[row * block_size:(row + 1) * block_size, col * block_size:(col + 1) * block_size] = block

    # Translate format back to CSR format
    matrix = matrix.tocsr()

    return matrix

def display_matrix_portrait(matrix):
    plt.figure(figsize=(10, 10))
    plt.spy(matrix, markersize=5, marker=',', color='green')
    #plt.gca().invert_yaxis()
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.title('Matrix Portrait')
    plt.show()