# OptiBreak Project Rules

## Academic Rules and Constraints
1. Language: Python 3 standard library only for core algorithms and optimization routines.
2. Allowed Standard Imports: `math`, `random`, `time`, `csv`, `os`, and project-internal modules.
3. Third-party Library Restrictions:
   - `numpy` is strictly restricted to `visualization/` and synthetic data generation in `problems/`.
   - `matplotlib` is strictly restricted to `visualization/`.
   - `scipy`, `sklearn`, `torch`, `statsmodels`, `numpy.linalg` are strictly prohibited throughout the entire repository.
4. Linear Solver:
   - All systems of linear equations ($A x = b$) are solved using Gaussian elimination with partial pivoting implemented in `core/linear_systems.py`.
5. Data Structures:
   - Vectors are plain Python lists of floats: `List[float]`.
   - Matrices are plain Python lists of lists: `List[List[float]]`.
   - Vector and matrix operations are performed via helper functions in `core/vectors.py` and `core/matrices.py`.
6. Reproducibility:
   - All stochastic algorithms and data generators must use deterministic random seeds (`random.seed(42)`).
   - All datasets are synthetically generated; no external data files or network calls are permitted.
7. Algorithm Interface and Convergence:
   - Every algorithm returns a standard `Result` object defined in `core/convergence.py`.
   - Standard stopping condition: $\|\nabla f(x_k)\| < \text{tol}$ or $k \ge \text{max\_iters}$.
   - Algorithms must run silently without printing.
8. Testing:
   - All modules must be tested using Python's standard `unittest` framework.
   - Run tests with `python -m unittest discover tests`.
