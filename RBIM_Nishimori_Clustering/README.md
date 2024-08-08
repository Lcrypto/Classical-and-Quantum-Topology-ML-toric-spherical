
```markdown
# Graph Clustering using PEG, QC-LDPC, MET QC-LDPC and ER Models

This repository contains Python code for clustering nodes in graphs generated using the Progressive Edge Growth (PEG) and Erdős–Rényi (ER) models. The code also includes methods for comparing these models using various clustering algorithms and visualizing the results.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- easygui

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```
2. Install the required Python packages:
   ```bash
   pip install numpy matplotlib easygui
   ```

## Usage

1. **Input Graph File**: The script starts by prompting the user to select a text file containing the edge list for the PEG graph using a file dialog.

2. **Graph Construction**: 
   - The edge list for the PEG graph is read and adjusted for Python's zero-based indexing.
   - The edge list for the QC-LDPC, MET QC-LDPC graph is read and adjusted for Python's zero-based indexing.  
   - A weighted graph is generated using the PEG edge list with specified mean (`μ`) and standard deviation (`ν`).
   - An ER graph is also generated with the same parameters.

3. **Clustering Methods**:
   - The script applies multiple clustering algorithms to the PEG, QC-LDPC, MET QC-LDPC and ER graphs:
     - Bethe Hessian (BH) under the Nishimori condition
     - BH for the spin glass model
     - Mean Field (MF) method
     - Signed Laplacian method

4. **Results Visualization**: 
   - Histograms are plotted to compare the clustering results for the PEG and ER graphs.
   - The script calculates and prints the overlap between the true labels and the labels obtained from each clustering method.

5. **Output**:
   - Overlap values for each method are printed, providing insights into the performance of the different clustering algorithms.

## Example Output

```bash
BH Nishimori = 0.85
BH spin glass = 0.78
Mean field = 0.76
Laplacian = 0.72
```

## References

- [PEG Algorithm Matlab, C, Python implementations](https://github.com/Lcrypto/classic-PEG-)
- [Erdős–Rényi Graph](https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model)
- [Simulated Annealing](https://ieeexplore.ieee.org/document/8441303)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

