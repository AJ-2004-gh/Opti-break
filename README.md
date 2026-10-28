# OptiBreak: Optimization Algorithms and Failure Modes

This project implements 23 numerical optimization algorithms from scratch in pure Python and investigates the conditions under which each algorithm fails. For each optimizer, we provide the theoretical update rule, a standard test problem that converges normally, a constructed test case that triggers a specific failure mode, and a detector function that verifies whether the expected failure occurred.

---

## 1. Project Overview & Permitted Libraries

### Project Overview
Standard numerical optimization packages (such as SciPy or PyTorch) typically conceal convergence issues behind early termination criteria or warnings. In this study, we implement each optimizer from first principles to study how and why they break:
1. **Mathematical update rules**: Implemented explicitly without black-box optimization libraries.
2. **Deterministic failure cases**: Constructed test setups that trigger known numerical issues, including step divergence, ill-conditioned zig-zagging, saddle point trapping, step size collapse, stochastic noise floors, limit cycles, and singular constraint matrices.
3. **Detector functions**: Functions that analyze the final state and history of an optimization run to verify whether the expected failure mode occurred.

### Permitted Libraries
- **Core and algorithms**: All optimizers (`algorithms/`), linear algebra routines (`core/`), benchmark functions (`problems/`), and failure detectors (`atlas/`) are written in pure Python using only standard library modules (`math`, `random`, `csv`, `os`, `sys`, `collections`).
- **Plotting and synthetic data**: `numpy` and `matplotlib` are used only in `visualization/` (to render 2D contour plots and decision boundaries) and `problems/` (to generate 2D synthetic datasets).
- **Prohibited libraries**: No use of `scipy`, `sklearn`, `torch`, `statsmodels`, or `numpy.linalg`. Linear equation solving (Gaussian elimination with partial pivoting and LU factorization) is implemented in `core/linear_systems.py`.

---

## 2. All 23 Optimization Algorithms

| ID | Algorithm Name | Source File | Function | Mathematical Update Rule |
|:---|:---|:---|:---|:---|
| **01** | Fixed-Step Gradient Descent | [algorithms/gradient_descent.py](algorithms/gradient_descent.py) | `gradient_descent_fixed` | $x_{k+1} = x_k - \alpha \nabla f(x_k)$ |
| **02** | Exact Line Search GD | [algorithms/gradient_descent.py](algorithms/gradient_descent.py) | `gradient_descent_exact` | $x_{k+1} = x_k - \alpha_k g_k,\quad \alpha_k = \frac{g_k^T g_k}{g_k^T Q g_k}$ |
| **03** | Backtracking Line Search | [algorithms/line_search.py](algorithms/line_search.py) | `backtracking_line_search` | $\alpha \leftarrow \rho \alpha \text{ until } f(x + \alpha d) \le f(x) + c_1 \alpha g^T d$ |
| **04** | Armijo Decrease Condition | [algorithms/line_search.py](algorithms/line_search.py) | `armijo_condition` | $f(x + \alpha d) \le f(x) + c_1 \alpha \nabla f(x)^T d$ |
| **05** | GD with Armijo Backtracking | [algorithms/gradient_descent.py](algorithms/gradient_descent.py) | `gradient_descent_armijo` | $x_{k+1} = x_k - \alpha_k \nabla f(x_k) \text{ via Armijo search}$ |
| **06** | Polyak Heavy Ball Momentum | [algorithms/momentum.py](algorithms/momentum.py) | `polyak_momentum` | $v_{k+1} = \beta v_k + \alpha \nabla f(x_k),\quad x_{k+1} = x_k - v_{k+1}$ |
| **07** | Nesterov Accelerated Gradient | [algorithms/nesterov.py](algorithms/nesterov.py) | `nesterov` | $v_{k+1} = \beta v_k + \alpha \nabla f(x_k - \beta v_k),\quad x_{k+1} = x_k - v_{k+1}$ |
| **08** | Full-Batch GD Linear Regression | [algorithms/regression_optimizers.py](algorithms/regression_optimizers.py) | `gd_linear_regression` | $w_{t+1} = w_t - \alpha \nabla_w \text{MSE}(w, b),\quad b_{t+1} = b_t - \alpha \nabla_b \text{MSE}(w, b)$ |
| **09** | Stochastic Gradient Descent (SGD) | [algorithms/regression_optimizers.py](algorithms/regression_optimizers.py) | `sgd_linear_regression` | $\theta_{t+1} = \theta_t - \alpha \nabla_\theta \mathcal{L}_{i_t}(\theta),\quad i_t \sim \text{Uniform}(1, n)$ |
| **10** | Mini-Batch SGD | [algorithms/regression_optimizers.py](algorithms/regression_optimizers.py) | `minibatch_sgd_linear_regression` | $\theta_{t+1} = \theta_t - \frac{\alpha}{\|B\|} \sum_{i \in B} \nabla_\theta \mathcal{L}_i(\theta)$ |
| **11** | Logistic Regression GD | [algorithms/regression_optimizers.py](algorithms/regression_optimizers.py) | `logistic_regression_gd` | $\theta_{t+1} = \theta_t - \alpha \nabla_\theta \text{BCE}(\theta)$ |
| **12** | AdaGrad Adaptive Gradient | [algorithms/adaptive_optimizers.py](algorithms/adaptive_optimizers.py) | `adagrad` | $G \leftarrow G + g \odot g,\quad x \leftarrow x - \frac{\alpha}{\sqrt{G} + \epsilon} \odot g$ |
| **13** | RMSProp Adaptive Optimizer | [algorithms/adaptive_optimizers.py](algorithms/adaptive_optimizers.py) | `rmsprop` | $v \leftarrow \rho v + (1-\rho)(g \odot g),\quad x \leftarrow x - \frac{\alpha}{\sqrt{v} + \epsilon} \odot g$ |
| **14** | Adam Adaptive Moments | [algorithms/adaptive_optimizers.py](algorithms/adaptive_optimizers.py) | `adam` | $\hat{m}_t = \frac{m_t}{1-\beta_1^t},\ \hat{v}_t = \frac{v_t}{1-\beta_2^t},\ x \leftarrow x - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t$ |
| **15** | Subgradient Method | [algorithms/subgradient.py](algorithms/subgradient.py) | `subgradient_method` | $x_{k+1} = x_k - \alpha_k g_k,\quad g_k \in \partial f(x_k),\ \alpha_k = a/\sqrt{k}$ |
| **16** | Subgradient Lasso Regression | [algorithms/subgradient.py](algorithms/subgradient.py) | `subgradient_lasso` | $w_{k+1} = w_k - \alpha_k \left(\frac{1}{n} X^T(X w_k - y) + \lambda \operatorname{sign}(w_k)\right)$ |
| **17** | Pure Newton's Method | [algorithms/newton.py](algorithms/newton.py) | `newton_method` | $\nabla^2 f(x_k) p_k = -\nabla f(x_k),\quad x_{k+1} = x_k + p_k$ |
| **18** | Damped Newton with Armijo | [algorithms/newton.py](algorithms/newton.py) | `damped_newton_method` | $p_k = -[\nabla^2 f(x_k)]^{-1} \nabla f(x_k),\quad x_{k+1} = x_k + \alpha_k p_k$ |
| **19** | Newton One-Step Exactness | [algorithms/newton.py](algorithms/newton.py) | `verify_newton_one_step` | $x_1 = x_0 - Q^{-1}(Q x_0 - c) = Q^{-1} c \implies \|x_1 - x^*\|_2 = 0$ |
| **20** | BFGS Quasi-Newton | [algorithms/bfgs.py](algorithms/bfgs.py) | `bfgs` | $H_{k+1} = (I - \rho s y^T) H_k (I - \rho y s^T) + \rho s s^T,\quad \rho = 1/(y^T s)$ |
| **21** | Lagrange Multiplier Solver | [algorithms/constrained_optimization.py](algorithms/constrained_optimization.py) | `lagrange_solver` | $\begin{bmatrix} Q & A^T \\ A & 0 \end{bmatrix} \begin{bmatrix} x \\ \lambda \end{bmatrix} = \begin{bmatrix} -c \\ b \end{bmatrix}$ |
| **22** | KKT Optimality Verification | [algorithms/constrained_optimization.py](algorithms/constrained_optimization.py) | `kkt_check` | $\nabla f + A^T \lambda + \nabla g^T \mu = 0,\ Ax = b,\ g(x) \le 0,\ \mu \ge 0,\ \mu_i g_i = 0$ |
| **23** | LICQ Constraint Qualification | [algorithms/constrained_optimization.py](algorithms/constrained_optimization.py) | `licq_check` | $\operatorname{rank}(\{\nabla h_i(x^*)\} \cup \{\nabla g_j(x^*)\}_{j \in \mathcal{A}}) = m$ |

---

## 3. The 23 Failure Atlas Cases

| Case | Algorithm | Failure Title | Theory & Failure Mechanism | Detector Function |
|:---|:---|:---|:---|:---|
| **01** | GD Fixed | Geometric Divergence ($\alpha > 2/L$) | When the step size exceeds $2/\lambda_{\max}(Q)$, Rayleigh quotient updates exceed 1, causing iterates to diverge. | `diverged` |
| **02** | GD Exact | Orthogonal Zig-Zagging ($\kappa \gg 1$) | On an ill-conditioned quadratic, successive gradient descent directions are orthogonal, resulting in slow zig-zagging. | `zigzag_score` |
| **03** | Backtracking LS | Halvings Exhaustion on Non-Descent Ray | When given an ascent direction where $\nabla f^T d > 0$, the Armijo decrease condition cannot hold for any positive step size. | `diverged` |
| **04** | Armijo Condition | Armijo Decrease Violation | An oversized trial step overshoots the quadratic bowl, failing the sufficient decrease criterion. | `diverged` |
| **05** | GD Armijo | Step Collapse in Curved Valley | Steep quadratic walls force repeated step halvings, shrinking $\alpha_k$ to near zero before reaching the optimum. | `step_collapse` |
| **06** | Polyak Momentum | Resonant Energy Overshoot | A high momentum parameter ($\beta \to 1^-$) accumulates excessive velocity, producing large non-monotone oscillations. | `diverged` |
| **07** | Nesterov Acceleration | Saddle Point Trapping | Initialized on an invariant axis of an indefinite saddle, iterates lack transverse momentum and stall at the saddle. | `stalled_at_saddle` |
| **08** | Regression GD | Feature Scale Divergence | Unstandardized features yield an anisotropic Hessian; a step size suitable for small features causes large features to diverge. | `diverged` |
| **09** | Regression SGD | Stochastic Gradient Noise Floor | Single-sample gradient variance does not vanish near the minimum, leaving iterates in a persistent noise floor. | `noise_floor` |
| **10** | Mini-Batch SGD | Small-Batch Variance Floor | Small batch sizes retain noticeable sampling covariance, preventing asymptotic loss convergence. | `noise_floor` |
| **11** | Logistic Regression | Weight Divergence on Separable Data | On linearly separable data without regularization, BCE loss decreases only as weights tend to infinity, causing unbounded growth. | `unbounded_weights` |
| **12** | AdaGrad | Step Size Freezing | Summing squared gradients monotonically increases the denominator, prematurely shrinking effective step sizes. | `step_collapse` |
| **13** | RMSProp | High-Frequency Ridge Limit Cycle | A high learning rate across an anisotropic ravine causes persistent bouncing across valley walls instead of settling. | `tail_oscillation` |
| **14** | Adam | Rosenbrock Valley Limit Cycle | A large step size repeatedly overshoots the narrow Rosenbrock walls, trapping iterates in an oscillatory loop. | `tail_oscillation` |
| **15** | Subgradient Method | Limit Cycle Kink Chattering | Using a constant step size on non-smooth functions like $\|x\|_1$ causes perpetual chattering across the non-differentiable kink. | `tail_oscillation` |
| **16** | Subgradient Lasso | Inability to Achieve Exact Zeros | Subgradient descent lacks coordinate projection; weights oscillate around zero rather than achieving exact sparsity. | `no_exact_zeros` |
| **17** | Pure Newton | Saddle Point / Maximum Attraction | Newton's method solves $\nabla^2 f(x) p = -\nabla f(x)$ without checking positive definiteness, stepping directly toward stationary maxima or saddles. | `stalled_at_saddle` |
| **18** | Damped Newton | Non-Descent Newton Direction | In regions of negative curvature where $\nabla f^T p > 0$, the Newton step is an ascent direction; the line search safely halts. | `non_descent` |
| **19** | Newton One-Step | Truncation Error on Non-Quadratic | Newton's step finds the exact minimizer in one step only for quadratic functions; non-zero higher derivatives leave residual error. | `residual_dist` |
| **20** | BFGS Quasi-Newton | Negative Curvature Update Failure | On non-convex objectives such as $x^4-2x^2+y^2$, $s_k^T y_k \le 0$ can occur in negative curvature regions, forcing update skips. | `skipped_update` |
| **21** | Lagrange Solver | Singular KKT Block Matrix | Linearly dependent equality constraints make the bordered Hessian rank-deficient, preventing matrix inversion. | `singular_kkt` |
| **22** | KKT Check | Constrained Maximizer Passed Stationarity | First-order KKT stationarity holds at local maxima as well as minima; checking the reduced Hessian is necessary to confirm optimality. | `not_minimizer` |
| **23** | LICQ Check | Degeneracy from Collinear Gradients | Linearly dependent active constraint gradients violate LICQ, causing the Gram matrix determinant to drop to zero. | `licq_failed` |

---

## 4. Repository Directory Tree

```
OptiBreak/
├── algorithms/                    # Pure Python optimizer implementations (No NumPy)
│   ├── __init__.py
│   ├── adaptive_optimizers.py     # AdaGrad, RMSProp, Adam, AdamW
│   ├── bfgs.py                    # BFGS Quasi-Newton
│   ├── constrained_optimization.py# Lagrange solver, KKT checker, LICQ checker
│   ├── gradient_descent.py        # Fixed GD, Exact GD, Armijo GD
│   ├── line_search.py             # Backtracking, Exact Quadratic, Armijo
│   ├── momentum.py                # Polyak Heavy Ball Momentum
│   ├── nesterov.py                # Nesterov Accelerated Gradient
│   ├── newton.py                  # Newton, Damped Newton, Exact Verification
│   ├── regression_optimizers.py   # GD, SGD, Mini-batch, Logistic, Lasso ISTA/FISTA
│   └── subgradient.py             # Subgradient, Subgradient Lasso, Projected Subgradient
├── atlas/                         # Failure mode diagnostics & runners (No NumPy)
│   ├── __init__.py
│   ├── cases.py                   # 23 paired failure case specifications
│   ├── detectors.py               # 11 trajectory & landscape failure detectors
│   └── runner.py                  # Case executor, report generator, CSV logger
├── core/                          # Pure Python linear algebra & utility routines
│   ├── __init__.py
│   ├── convergence.py             # Result dataclass & convergence/divergence checks
│   ├── derivatives.py             # Finite difference gradient and Hessian verification
│   ├── linear_systems.py          # LU decomposition, Gaussian elimination, back-substitution
│   ├── matrices.py                # Matrix multiply, transpose, rank, determinant
│   └── vectors.py                 # Vector addition, scaling, dot product, L2 norm
├── experiments/                   # Experiment suites & contour generators
│   ├── __init__.py
│   ├── all_algorithm_contours.py  # Contour plot generator for all 23 algorithms
│   ├── compare_algorithms.py      # GD variants & Momentum vs Nesterov studies
│   ├── constrained_experiments.py # Nonsmooth & constrained optimization pipeline
│   ├── failure_atlas.py           # 23-case Failure Atlas pipeline
│   ├── newton_quadratic.py        # Rosenbrock Newton & Condition Number study
│   └── regression_experiments.py  # Linear & logistic loss landscape pipeline
├── problems/                      # Objective functions & synthetic data generators
│   ├── __init__.py
│   ├── constrained.py             # Equality and inequality constrained problems
│   ├── lasso.py                   # Lasso loss, subgradient, sparse data generator
│   ├── linear_regression.py       # MSE loss, sample gradients, dataset generator
│   ├── logistic_regression.py     # BCE loss, sigmoid, blob data generator
│   ├── nonconvex.py               # Saddle and double-well test functions
│   ├── nonsmooth.py               # L1 norm and non-smooth test functions
│   ├── quadratic.py               # Quadratic forms and condition numbers
│   └── rosenbrock.py              # 2D Rosenbrock function, gradient, Hessian
├── results/                       # Generated outputs and artifacts
│   ├── histories/                 # CSV trajectory and metric histories
│   ├── plots/                     # 23 algorithm contours + 23 Atlas pair plots + 20 experiment figures (66 total)
│   └── reports/                   # failure_atlas.txt summary report
├── tests/                         # Automated test suite (98 tests)
│   ├── __init__.py
│   ├── test_algorithms.py         # GD, line search, momentum, Newton, BFGS
│   ├── test_atlas.py              # Failure cases and detector unit tests
│   ├── test_constraints.py        # Subgradient, Lasso, Lagrange, KKT, LICQ
│   ├── test_convergence.py        # Convergence criteria & Result class
│   ├── test_derivatives.py        # Numerical gradient & Hessian validation
│   ├── test_line_search.py        # Line search & decrease condition tests
│   ├── test_linear_systems.py     # Linear solvers, rank, determinants, inverses
│   ├── test_matrices.py           # Matrix multiplication, transposition
│   ├── test_problems.py           # Quadratic, Rosenbrock, regressions
│   ├── test_regression.py         # SGD, Mini-batch, logistic, adaptive
│   ├── test_smoke.py              # Smoke tests for package imports
│   ├── test_vectors.py            # Vector operations & norm functions
│   └── test_visualization.py      # Plotting tests and non-empty panel guards
├── visualization/                 # Matplotlib + NumPy plotting modules
│   ├── __init__.py
│   ├── contour.py                 # Contours, trajectories, constraints, failure pairs
│   └── model_plots.py             # Regression MSE & Logistic BCE landscapes
├── main.py                        # Interactive menu and command-line runner
├── requirements.txt               # Dependencies for visualization and synthetic data
├── README.md                      # Project documentation
└── RULES.md                       # Architectural and algorithmic rules
```

---

## 5. How to Run

### Installation & Dependencies
The core algorithms and linear algebra routines are **100% pure Python** (using only the standard library). For plotting and synthetic dataset generation, install the visualization dependencies:

```bash
pip install -r requirements.txt
```

### Command-Line Flags
- **Run Everything** (Generates all 23 algorithm contours, 4 experiment suites, and 23 Atlas plots):
  ```bash
  python main.py --all
  ```
- **Run Failure Atlas** (Executes the 23 failure cases and writes `results/reports/failure_atlas.txt`):
  ```bash
  python main.py --atlas
  ```
- **Run Test Suite** (Discovers and executes unit tests):
  ```bash
  python main.py --test
  ```

### Interactive Menu
Run `python main.py` without arguments to open the text menu:
```
============================================================
OptiBreak Interactive Optimization Suite
============================================================
 1. GD family
 2. Momentum/Nesterov
 3. Regression and adaptive
 4. Newton/BFGS
 5. Nonsmooth/Constrained
 6. Failure Atlas
 7. Run everything
 0. Exit
============================================================
Select an option (0-7): 
```

### Running Individual Experiment Scripts
```bash
python experiments/compare_algorithms.py
python experiments/regression_experiments.py
python experiments/newton_quadratic.py
python experiments/constrained_experiments.py
python experiments/failure_atlas.py
```

---

## 6. How to Read an Atlas Plot

Each Failure Atlas case produces a paired comparison image saved to `results/plots/atlas_<case_id>.png`.

```
+-----------------------------------------------------------------------------------+
|                           Comparison: Works vs Breaks                             |
|          Verdict: Case 01: Geometric Divergence (alpha > 2/L) - FLAGGED           |
+------------------------------------------+----------------------------------------+
|              Panel 1: Works              |             Panel 2: Breaks            |
|          (Tuned / Stably Convergent)     |          (Pathological / Divergent)    |
|                                          |                                        |
|  - Stable trajectory                     |  - Oscillating or expanding trajectory |
|  - Monotonic loss reduction              |  - Divergence or limit cycle           |
|  - Reaches expected minimum (star)       |  - Annotated 'diverged' if clipped     |
+------------------------------------------+----------------------------------------+
```

### Visual Elements
- **Banner (Detector Verdict)**: The banner at the top shows the case ID, title, and the detector verdict (`FLAGGED` when the expected failure mode was detected).
- **Left Panel (Works)**: Displays the healthy control run with standard tuning (e.g., step size within theoretical bounds, standardized features, or well-conditioned constraints).
- **Right Panel (Breaks)**: Displays the pathological run illustrating the failure mode under the constructed conditions (e.g., divergence, limit cycles, or degenerate constraints).
- **Report File (Numbers)**: Numerical values, step sizes, loss values, condition numbers, and detector statistics are logged in `results/reports/failure_atlas.txt`.

### Trajectory Symbols
- **Circle (`o`)**: Starting point $x_0$.
- **Star (`*`)**: Final iterate reached.
- **Cross (`x`)**: Analytical minimum $x^*$.
- **Callout Arrow (`diverged`)**: Indicates iterates exceeded the plot boundaries due to divergence.
- **Quiver Arrows**: Active constraint gradients $\nabla g_i(x^*)$ showing whether constraint gradients are linearly independent or collinear.
- **Dashed Curves & Shading**: Inequality constraint boundaries with the infeasible region shaded.
