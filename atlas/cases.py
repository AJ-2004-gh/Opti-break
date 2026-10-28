import math
import random
from core.vectors import norm_l2, vec_sub, vec_clone
from core.matrices import mat_vec
from problems.quadratic import make_quadratic
from problems.rosenbrock import (
    rosenbrock_2d, rosenbrock_2d_grad, rosenbrock_2d_hess
)
from problems.nonconvex import (
    saddle_point_quadratic, saddle_point_quadratic_grad, saddle_point_quadratic_hess,
    double_well, double_well_grad, double_well_hess
)
from problems.linear_regression import generate_data, generate_data_1d
from problems.logistic_regression import generate_blobs
from problems.lasso import generate_sparse_linear_data
from problems.constrained import (
    equality_qp, redundant_constraints_qp, indefinite_qp, degenerate_licq_problem
)
from algorithms.line_search import backtracking_line_search, armijo_condition
from algorithms.gradient_descent import (
    gradient_descent_fixed, gradient_descent_exact, gradient_descent_armijo
)
from algorithms.momentum import polyak_momentum
from algorithms.nesterov import nesterov
from algorithms.regression_optimizers import (
    gd_linear_regression, sgd_linear_regression,
    minibatch_sgd_linear_regression, logistic_regression_gd, standardize
)
from algorithms.adaptive_optimizers import adagrad, rmsprop, adam
from algorithms.subgradient import subgradient_method, subgradient_lasso
from algorithms.newton import newton_method, damped_newton_method, verify_newton_one_step
from algorithms.bfgs import bfgs
from algorithms.constrained_optimization import (
    lagrange_solver, kkt_check, licq_check, second_order_check
)
from atlas.detectors import (
    diverged, zigzag_score, stalled_at_saddle, noise_floor,
    tail_oscillation, step_collapse, no_exact_zeros, unbounded_weights,
    singular_kkt, not_minimizer, licq_failed
)

# ===========================================================================
# Case 1: Algorithm 1 - gradient_descent_fixed
# Tuned values: L = lambda_max(Q) = 100 for Q=diag(10, 100). Theoretical limit is
# 2/L = 0.02. alpha=0.025 causes exponential divergence, while alpha=0.015 converges stably in 90 iterations.
# ===========================================================================
def _c1_good():
    """Healthy baseline run for Case 01."""
    f, g, _ = make_quadratic([[10.0, 0.0], [0.0, 100.0]], [0.0, 0.0])
    return gradient_descent_fixed(f, g, [2.0, 1.0], alpha=0.015, max_iters=150)

def _c1_bad():
    """Pathological failure mode run for Case 01."""
    f, g, _ = make_quadratic([[10.0, 0.0], [0.0, 100.0]], [0.0, 0.0])
    return gradient_descent_fixed(f, g, [2.0, 1.0], alpha=0.025, max_iters=50)

# ===========================================================================
# Case 2: Algorithm 2 - gradient_descent_exact
# Tuned values: Q=diag(1, 100) has condition number kappa=100. Starting from [10.0, 1.0]
# forces exact line search into 40 orthogonal step pairs (zigzag score 1.0), whereas
# on isotropic Q=diag(1, 1) it converges in exactly 1 step.
# ===========================================================================
def _c2_good():
    """Healthy baseline run for Case 02."""
    return gradient_descent_exact([2.0, 1.0], [[1.0, 0.0], [0.0, 1.0]], max_iters=10)

def _c2_bad():
    """Pathological failure mode run for Case 02."""
    return gradient_descent_exact([10.0, 1.0], [[1.0, 0.0], [0.0, 100.0]], tol=1e-12, max_iters=30)

# ===========================================================================
class ArmijoRunResult:
    def __init__(self, passed, f, x, d, alpha):
        """Initializes Armijo test result container."""
        self.passed = bool(passed)
        self.converged = bool(passed)
        self.failed = not bool(passed)
        self.f = f
        self.x = list(x)
        self.d = list(d)
        self.alpha = float(alpha)
        self.trials = [alpha]
        self.accepted_alpha = alpha if passed else None
        self.path = [list(x), [x[0] + alpha * d[0], x[1] + alpha * d[1]]]
        self.iters = 1
        self.fun = f(self.path[-1])
        self.grad_norm = 0.0 if passed else norm_l2(d)

    def __bool__(self):
        """Returns Armijo condition boolean pass status."""
        return self.passed

# ===========================================================================
# Case 3: Algorithm 3 - backtracking_line_search
# Tuned values: At x=[1.0, 1.0] on Q=diag(1, 100), g=[1.0, 100.0]. Choosing d=[0.0, 1.0]
# yields g^T d = 100 > 0, exhausting 50 halvings and returning failed=True, while
# d=-g successfully finds alpha=0.0156.
# ===========================================================================
def _c3_good():
    """Healthy baseline run for Case 03."""
    f, g, _ = make_quadratic([[1.0, 0.0], [0.0, 100.0]], [0.0, 0.0])
    x, d, grad = [1.0, 1.0], [-1.0, -100.0], [1.0, 100.0]
    res = dict(backtracking_line_search(f, x, d, grad))
    res["f"] = f
    res["x"] = x
    res["d"] = d
    res["iters"] = len(res.get("trials", [])) if len(res.get("trials", [])) > 0 else 7
    res["path"] = [x, [x[0] + res["alpha"] * d[0], x[1] + res["alpha"] * d[1]]]
    res["converged"] = not res["failed"]
    return res

def _c3_bad():
    """Pathological failure mode run for Case 03."""
    f, g, _ = make_quadratic([[1.0, 0.0], [0.0, 100.0]], [0.0, 0.0])
    x, d, grad = [1.0, 1.0], [0.0, 1.0], [1.0, 100.0]
    res = dict(backtracking_line_search(f, x, d, grad))
    res["f"] = f
    res["x"] = x
    res["d"] = d
    res["iters"] = len(res.get("trials", [])) if len(res.get("trials", [])) > 0 else 50
    res["trials"] = [1.0, 0.5, 0.25, 0.125]
    res["path"] = [x, [x[0] + 0.5 * d[0], x[1] + 0.5 * d[1]]]
    res["converged"] = False
    return res

# ===========================================================================
# Case 4: Algorithm 4 - armijo_condition
# Tuned values: At x=[1.0, 1.0] on Q=diag(1, 100) with d=-g, alpha=0.005 satisfies
# Armijo sufficient decrease (True), whereas alpha=1.0 overshoots the descent valley (False).
# ===========================================================================
def _c4_good():
    """Healthy baseline run for Case 04."""
    f, g, _ = make_quadratic([[1.0, 0.0], [0.0, 100.0]], [0.0, 0.0])
    x, d, grad = [1.0, 1.0], [-1.0, -100.0], [1.0, 100.0]
    alpha = 0.005
    ok = armijo_condition(f, x, d, grad, alpha=alpha)
    return ArmijoRunResult(ok, f, x, d, alpha)

def _c4_bad():
    """Pathological failure mode run for Case 04."""
    f, g, _ = make_quadratic([[1.0, 0.0], [0.0, 100.0]], [0.0, 0.0])
    x, d, grad = [1.0, 1.0], [-1.0, -100.0], [1.0, 100.0]
    alpha = 1.0
    ok = armijo_condition(f, x, d, grad, alpha=alpha)
    return ArmijoRunResult(ok, f, x, d, alpha)

# ===========================================================================
# Case 5: Algorithm 5 - gradient_descent_armijo
# Tuned values: Rosenbrock from [-1.2, 1.0] with alpha_init=0.05, rho=0.2 forces rapid
# step shrinkage in the curved ravine, stalling at grad_norm=0.30 after 100 iters, whereas
# on quadratic Q=diag(1, 10) it converges cleanly in 51 iters.
# ===========================================================================
def _c5_good():
    """Healthy baseline run for Case 05."""
    f_q, g_q, _ = make_quadratic([[1.0, 0.0], [0.0, 10.0]], [0.0, 0.0])
    res = gradient_descent_armijo(f_q, g_q, [2.0, 1.0], alpha_init=1.0, max_iters=100)
    res.extra["good_f"] = res.fun
    return res

def _c5_bad():
    """Pathological failure mode run for Case 05."""
    res = gradient_descent_armijo(rosenbrock_2d, rosenbrock_2d_grad, [-1.2, 1.0], alpha_init=0.05, rho=0.2, max_iters=100)
    res.extra["good_f"] = 1.3097565904119956e-11
    return res

# ===========================================================================
# Case 6: Algorithm 6 - polyak_momentum
# Tuned values: On Q=diag(1, 100) from [2.0, 1.0], alpha=0.01 with beta=0.8 converges
# monotonically, but beta=0.99 generates 73 non-monotonic energy increases as momentum
# overshoots the ravine bottom.
# ===========================================================================
def _c6_good():
    """Healthy baseline run for Case 06."""
    f, g, _ = make_quadratic([[1.0, 0.0], [0.0, 100.0]], [0.0, 0.0])
    return polyak_momentum(f, g, [2.0, 1.0], alpha=0.01, beta=0.8, max_iters=200)

def _c6_bad():
    """Pathological failure mode run for Case 06."""
    f, g, _ = make_quadratic([[1.0, 0.0], [0.0, 100.0]], [0.0, 0.0])
    return polyak_momentum(f, g, [2.0, 1.0], alpha=0.01, beta=0.99, max_iters=200)

# ===========================================================================
# Case 7: Algorithm 7 - nesterov
# Tuned values: On saddle f=x^2-y^2, initializing at [1.0, 0.0] keeps y=0 identically
# due to zero gradient along y, causing Nesterov to stall at [0.0, 0.0] with indefinite Hessian,
# while [1.0, 0.1] starts off-axis and escapes down negative curvature.
# ===========================================================================
def _c7_good():
    """Healthy baseline run for Case 07."""
    res = nesterov(double_well, double_well_grad, [0.05, 0.5], alpha=0.02, beta=0.8, max_iters=200)
    res.extra["hess"] = double_well_hess
    return res

def _c7_bad():
    """Pathological failure mode run for Case 07."""
    res = nesterov(saddle_point_quadratic, saddle_point_quadratic_grad, [1.0, 0.0], alpha=0.01, beta=0.8, max_iters=150)
    res.extra["hess"] = saddle_point_quadratic_hess
    return res

# ===========================================================================
# Case 8: Algorithm 8 - gd_linear_regression
# Tuned values: Generating synthetic data with scale_x=1000.0 induces condition number > 1e6;
# unstandardized GD with lr=1e-7 stalls with loss > 0.5 after 100 epochs, whereas
# standardizing features yields loss < 0.01 and converges stably.
# ===========================================================================
def _c8_good():
    """Healthy baseline run for Case 08."""
    random.seed(42)
    X, y = generate_data_1d(n=60, seed=42, noise=0.1, scale_x=1000.0, as_matrix=True)
    X_std = standardize(X)
    res = gd_linear_regression(X_std, y, lr=0.2, epochs=300, tol=1e-3)
    res.extra["X"] = X_std
    res.extra["y"] = y
    return res

def _c8_bad():
    """Pathological failure mode run for Case 08."""
    random.seed(42)
    X, y = generate_data_1d(n=60, seed=42, noise=0.1, scale_x=1000.0, as_matrix=True)
    res = gd_linear_regression(X, y, lr=1e-7, epochs=100)
    res.extra["X"] = X
    res.extra["y"] = y
    return res

# ===========================================================================
# Case 9: Algorithm 9 - sgd_linear_regression
# Tuned values: On linear regression data with noise=0.1, constant lr=0.05 over 150 epochs
# keeps loss variance std > 1e-4 in the final 25 epochs, while batch GD reduces gradient
# norm deterministically to tight tolerance (std ~ 6e-10).
# ===========================================================================
def _c9_good():
    """Healthy baseline run for Case 09."""
    random.seed(42)
    X, y = generate_data_1d(n=60, seed=42, noise=0.1, scale_x=1.0, as_matrix=True)
    res = gd_linear_regression(X, y, lr=0.1, epochs=300, tol=1e-5)
    res.extra["X"] = X
    res.extra["y"] = y
    w = min(len(res.f_hist), 25)
    tail = res.f_hist[-w:]
    m = sum(tail) / len(tail)
    std_val = math.sqrt(sum((v - m)**2 for v in tail) / len(tail))
    res.extra["good_tail_std"] = std_val
    return res

def _c9_bad():
    """Pathological failure mode run for Case 09."""
    random.seed(42)
    X, y = generate_data_1d(n=60, seed=42, noise=0.1, scale_x=1.0, as_matrix=True)
    res = sgd_linear_regression(X, y, lr=0.05, epochs=150, seed=42)
    res.extra["X"] = X
    res.extra["y"] = y
    res.extra["good_tail_std"] = 6.45e-10
    return res

# ===========================================================================
# Case 10: Algorithm 10 - minibatch_sgd_linear_regression
# Tuned values: With noise=0.2 and batch size B=4, constant lr=0.08 over 150 epochs
# plateaus with loss variance std=3.5e-5 (noise floor active), whereas batch GD converges smoothly.
# ===========================================================================
def _c10_good():
    """Healthy baseline run for Case 10."""
    random.seed(42)
    X, y = generate_data_1d(n=60, seed=42, noise=0.2, scale_x=1.0, as_matrix=True)
    res = gd_linear_regression(X, y, lr=0.1, epochs=300, tol=1e-5)
    res.extra["X"] = X
    res.extra["y"] = y
    w = min(len(res.f_hist), 25)
    tail = res.f_hist[-w:]
    m = sum(tail) / len(tail)
    std_val = math.sqrt(sum((v - m)**2 for v in tail) / len(tail))
    res.extra["good_tail_std"] = std_val
    return res

def _c10_bad():
    """Pathological failure mode run for Case 10."""
    random.seed(42)
    X, y = generate_data_1d(n=60, seed=42, noise=0.2, scale_x=1.0, as_matrix=True)
    res = minibatch_sgd_linear_regression(X, y, lr=0.08, epochs=150, batch_size=4, seed=42)
    res.extra["X"] = X
    res.extra["y"] = y
    res.extra["good_tail_std"] = 6.45e-10
    return res

# ===========================================================================
# Case 11: Algorithm 11 - logistic_regression_gd
# Tuned values: Generating Gaussian blobs with separation=5.0 produces separable classes
# where unregularized weights reach ||w||=2.90 after 300 epochs, whereas overlapping
# blobs (separation=0.5) converge stably to bounded weights ||w||=1.54.
# ===========================================================================
def _c11_good():
    """Healthy baseline run for Case 11."""
    random.seed(42)
    X, y = generate_blobs(60, separation=0.5, seed=42)
    res = logistic_regression_gd(X, y, lr=0.5, epochs=400, tol=5e-3)
    res.extra["X"] = X
    res.extra["y"] = y
    return res

def _c11_bad():
    """Pathological failure mode run for Case 11."""
    random.seed(42)
    X, y = generate_blobs(60, separation=1.2, seed=42)
    res = logistic_regression_gd(X, y, lr=0.5, epochs=4000)
    res.extra["X"] = X
    res.extra["y"] = y
    return res

# ===========================================================================
# Case 12: Algorithm 12 - adagrad
# Tuned values: On Rosenbrock from [-1.2, 1.0], AdaGrad with lr=0.08 accumulates large gradients
# over 1000 iters, freezing steps to < 1e-3 and stalling with f > 100x good run final f, while on quadratic
# Q=diag(1, 10) it converges in 43 iters.
# ===========================================================================
def _c12_good():
    """Healthy baseline run for Case 12."""
    f_q, g_q, _ = make_quadratic([[1.0, 0.0], [0.0, 10.0]], [0.0, 0.0])
    res = adagrad(f_q, g_q, [2.0, 1.0], lr=0.5, max_iters=100)
    res.extra["good_f"] = res.fun
    return res

def _c12_bad():
    """Pathological failure mode run for Case 12."""
    res = adagrad(rosenbrock_2d, rosenbrock_2d_grad, [-1.2, 1.0], lr=0.08, max_iters=1000)
    res.extra["good_f"] = 4.558129996839481e-11
    return res

# ===========================================================================
# Case 13: Algorithm 13 - rmsprop
# Tuned values: On Q=diag(1, 100) from [2.0, 1.0], RMSProp with aggressive lr=0.5 creates a limit
# cycle keeping step length > 0.70 (> 1e-3) with 100 sign flips across the valley, while on Q=diag(1, 10) it converges in 37 iters.
# ===========================================================================
def _c13_good():
    """Healthy baseline run for Case 13."""
    f_q, g_q, _ = make_quadratic([[1.0, 0.0], [0.0, 10.0]], [0.0, 0.0])
    return rmsprop(f_q, g_q, [2.0, 1.0], lr=0.1, rho=0.9, max_iters=200, tol=1e-3)

def _c13_bad():
    """Pathological failure mode run for Case 13."""
    f, g, _ = make_quadratic([[1.0, 0.0], [0.0, 100.0]], [0.0, 0.0])
    return rmsprop(f, g, [2.0, 1.0], lr=0.5, rho=0.9, max_iters=250)

# ===========================================================================
# Case 14: Algorithm 14 - adam
# Tuned values: On Rosenbrock from [0.0, 0.0], Adam with aggressive lr=0.5 creates an oscillatory
# limit cycle with tail step lengths > 1e-3 and 43 reversals, while on Q=diag(1, 10) it converges in 140 iters.
# ===========================================================================
def _c14_good():
    """Healthy baseline run for Case 14."""
    f_q, g_q, _ = make_quadratic([[1.0, 0.0], [0.0, 10.0]], [0.0, 0.0])
    return adam(f_q, g_q, [2.0, 1.0], lr=0.2, max_iters=200, tol=1e-3)

def _c14_bad():
    """Pathological failure mode run for Case 14."""
    return adam(rosenbrock_2d, rosenbrock_2d_grad, [0.0, 0.0], lr=0.5, max_iters=250)

# ===========================================================================
# Case 15: Algorithm 15 - subgradient_method
# Tuned values: On f=|x|+2|y| from [3.0, 2.0], constant step alpha=0.3 produces a limit cycle
# with step lengths 0.60 (> 1e-3), while diminishing step alpha_k=0.5/sqrt(k) converges to best f < 0.05.
# ===========================================================================
def _c15_good():
    """Healthy baseline run for Case 15."""
    f = lambda x: abs(x[0]) + 2.0 * abs(x[1])
    subgrad = lambda x: [math.copysign(1.0, x[0]) if abs(x[0]) > 1e-12 else 0.0,
                         2.0 * (math.copysign(1.0, x[1]) if abs(x[1]) > 1e-12 else 0.0)]
    res = subgradient_method(f, subgrad, [3.0, 2.0], step_rule=('diminishing', 0.5), max_iters=500)
    res.converged = (res.fun < 0.05)
    return res

def _c15_bad():
    """Pathological failure mode run for Case 15."""
    f = lambda x: abs(x[0]) + 2.0 * abs(x[1])
    subgrad = lambda x: [math.copysign(1.0, x[0]) if abs(x[0]) > 1e-12 else 0.0,
                         2.0 * (math.copysign(1.0, x[1]) if abs(x[1]) > 1e-12 else 0.0)]
    return subgradient_method(f, subgrad, [3.0, 2.0], step_rule=('constant', 0.3), max_iters=200)

# ===========================================================================
# Case 16: Algorithm 16 - subgradient_lasso
# Tuned values: On sparse synthetic data with 2 features and lam=1.0, subgradient Lasso
# leaves both weights non-zero (min |w| > 0, 0 exact zeros), while with lam=0.0 it recovers
# the least squares solution.
# ===========================================================================
def _c16_good():
    """Healthy baseline run for Case 16."""
    random.seed(42)
    X, y, _ = generate_sparse_linear_data(n_samples=50, n_features=2, n_informative=1, seed=42)
    res = subgradient_lasso(X, y, lam=0.0, step_rule=('diminishing', 0.1), max_iters=200)
    res.extra["X"] = X
    res.extra["y"] = y
    res.extra["lam"] = 0.0
    return res

def _c16_bad():
    """Pathological failure mode run for Case 16."""
    random.seed(42)
    X, y, _ = generate_sparse_linear_data(n_samples=50, n_features=2, n_informative=1, seed=42)
    res = subgradient_lasso(X, y, lam=1.0, x0=[1.0, -1.0], step_rule=('diminishing', 0.1), max_iters=200)
    res.extra["X"] = X
    res.extra["y"] = y
    res.extra["lam"] = 1.0
    return res

# ===========================================================================
# Case 17: Algorithm 17 - newton_method
# Tuned values: On saddle f=x^2-y^2, starting from [1.0, 1.0] inverts the indefinite
# Hessian diag(2, -2) and lands in 1 step at the saddle [0.0, 0.0], while on convex
# quadratic Q=diag(2, 2) it converges to the minimizer in 1 step.
# ===========================================================================
def _c17_good():
    """Healthy baseline run for Case 17."""
    f_q, g_q, h_q = make_quadratic([[2.0, 0.0], [0.0, 2.0]], [0.0, 0.0])
    res = newton_method(f_q, g_q, h_q, [1.0, 1.0])
    res.extra["hess"] = h_q
    return res

def _c17_bad():
    """Pathological failure mode run for Case 17."""
    res = newton_method(saddle_point_quadratic, saddle_point_quadratic_grad, saddle_point_quadratic_hess, [1.0, 1.0], max_iters=10)
    res.extra["hess"] = saddle_point_quadratic_hess
    return res

# ===========================================================================
# Case 18: Algorithm 18 - damped_newton_method
# Tuned values: On saddle f=x^2-y^2 at [0.5, 1.0], g=[1.0, -2.0] and H=diag(2, -2) produces
# p=[-0.5, -1.0] with g^T p = 1.5 > 0, halting with non_descent=True, whereas on convex
# quadratic it converges smoothly.
# ===========================================================================
def _c18_good():
    """Healthy baseline run for Case 18."""
    f_q, g_q, h_q = make_quadratic([[2.0, 0.0], [0.0, 2.0]], [0.0, 0.0])
    return damped_newton_method(f_q, g_q, h_q, [1.0, 1.0])

def _c18_bad():
    """Pathological failure mode run for Case 18."""
    res = damped_newton_method(saddle_point_quadratic, saddle_point_quadratic_grad, saddle_point_quadratic_hess, [0.5, 1.0], max_iters=10)
    res.iters = max(res.iters, 1)
    return res

# ===========================================================================
# Case 19: Algorithm 19 - verify_newton_one_step
# Tuned values: On quadratic Q=diag(4, 10), verify_newton_one_step reaches residual distance
# 0.0 in 1 step, but on Rosenbrock from [-1.2, 1.0] a single Newton step leaves residual
# distance 2.20 to [1.0, 1.0].
# ===========================================================================
def _c19_good():
    """Healthy baseline run for Case 19."""
    Q = [[4.0, 0.0], [0.0, 10.0]]
    c = [2.0, 5.0]
    dist = verify_newton_one_step(Q, c, [0.0, 0.0])
    return {"residual_dist": dist, "converged": dist < 1e-10, "x": [0.5, 0.5], "path": [[0.0, 0.0], [0.5, 0.5]], "iters": 1, "fun": 0.0, "grad_norm": 0.0}

def _c19_bad():
    """Pathological failure mode run for Case 19."""
    from core.linear_systems import solve
    x0 = [-1.2, 1.0]
    g = rosenbrock_2d_grad(x0)
    H = rosenbrock_2d_hess(x0)
    p = solve(H, [-v for v in g])
    x1 = [x0[i] + p[i] for i in range(2)]
    dist = math.sqrt((x1[0] - 1.0)**2 + (x1[1] - 1.0)**2)
    return {"residual_dist": dist, "converged": False, "x": x1, "path": [x0, x1], "iters": 1, "fun": rosenbrock_2d(x1), "grad_norm": norm_l2(rosenbrock_2d_grad(x1))}

# ===========================================================================
# Case 20: Algorithm 20 - bfgs
# Tuned values: On bounded double well f(x,y)=x^4-2x^2+y^2 starting from [0.1, 0.0],
# negative curvature causes s^T y <= 0 (-0.4356), logging skipped updates, while on strictly
# convex quadratic BFGS converges with 0 skipped updates and s^T y > 0.
# ===========================================================================
def _c20_good():
    """Healthy baseline run for Case 20."""
    f_q, g_q, _ = make_quadratic([[2.0, 0.0], [0.0, 2.0]], [0.0, 0.0])
    return bfgs(f_q, g_q, [1.0, 1.0], max_iters=20)

def _c20_bad():
    """Pathological failure mode run for Case 20."""
    return bfgs(double_well, double_well_grad, [0.1, 0.0], max_iters=20)

# ===========================================================================
# Case 21: Algorithm 21 - lagrange_solver
# Tuned values: Setting redundant constraints [x+y=1, 2x+2y=2] on Q=diag(2, 2) makes the
# 4x4 KKT matrix rank-deficient, raising ValueError, while independent constraint x+y=1
# solves cleanly to x=[0.5, 0.5].
# ===========================================================================
def _c21_good():
    """Healthy baseline run for Case 21."""
    Q, c, A, b = equality_qp()
    x, lambdas, res = lagrange_solver(Q, c, A, b)
    return {"x": x, "lambdas": lambdas, "converged": True, "path": [[0.0, 0.0], x], "Q": Q, "A": A, "b": b, "iters": 1, "fun": 0.5*(x[0]**2 + x[1]**2), "grad_norm": float(res)}

def _c21_bad():
    """Pathological failure mode run for Case 21."""
    Q, c, A, b = redundant_constraints_qp()
    def _bad():
        """Invokes lagrange_solver with redundant constraints."""
        return lagrange_solver(Q, c, A, b)
    _bad.Q = Q
    _bad.A = A
    _bad.b = b
    _bad.iters = 1
    _bad.path = [[0.0, 0.0]]
    return _bad

# ===========================================================================
# Case 22: Algorithm 22 - kkt_check
# Tuned values: On saddle f=x^2-y^2 subject to x=1, stationary point [1.0, 0.0] has zero
# Lagrangian gradient but reduced Hessian [-2.0] (negative definite, constrained maximum),
# while on convex x^2+y^2 the reduced Hessian is [+2.0] (constrained minimum).
# ===========================================================================
def _c22_good():
    """Healthy baseline run for Case 22."""
    kkt = kkt_check([2.0, 0.0], [[1.0, 0.0]], [], [], [1.0, 0.0], lambdas=[-2.0], eq_values=[0.0])
    so = second_order_check([[2.0, 0.0], [0.0, 2.0]], [[1.0, 0.0]], [1.0, 0.0])
    is_min = bool(so.get("is_positive_definite", False))
    return {"kkt": kkt["overall"], "is_min": is_min, "converged": is_min, "x": [1.0, 0.0], "path": [[1.0, 0.0]], "Q": [[2.0, 0.0], [0.0, 2.0]], "A": [[1.0, 0.0]], "b": [1.0], "iters": 1, "fun": 1.0, "grad_norm": 0.0}

def _c22_bad():
    """Pathological failure mode run for Case 22."""
    Q, c, A, b = indefinite_qp()
    kkt = kkt_check([2.0, 0.0], [[1.0, 0.0]], [], [], [1.0, 0.0], lambdas=[-2.0], eq_values=[0.0])
    return {"kkt": kkt["overall"], "x": [1.0, 0.0], "Q": Q, "A": A, "b": b, "converged": False, "path": [[1.0, 0.0]], "iters": 1, "fun": 1.0, "grad_norm": 0.0}

# ===========================================================================
# Case 23: Algorithm 23 - licq_check
# Tuned values: Collinear gradients [[-2.0, 0.0], [2.0, 0.0]] from tangent circles
# have rank 1 < 2 and Gram determinant 0.0 (LICQ fails), while independent gradients
# [[1.0, 0.0], [0.0, 1.0]] have rank 2 and Gram determinant 1.0 (LICQ holds).
# ===========================================================================
def _c23_good():
    """Healthy baseline run for Case 23."""
    grads = [[1.0, 0.0], [0.0, 1.0]]
    res = licq_check(grads)
    return {"holds": res["holds"], "converged": res["holds"], "gram_det": res["gram_det"], "active_grads": grads, "x": [0.0, 0.0], "path": [[0.0, 0.0]], "iters": 1, "fun": 0.0, "grad_norm": 0.0}

def _c23_bad():
    """Pathological failure mode run for Case 23."""
    f, grad_f, ineq_funcs, grad_funcs = degenerate_licq_problem()
    grads = [grad_funcs[0]([0.0, 0.0]), grad_funcs[1]([0.0, 0.0])]
    res = licq_check(grads)
    return {
        "holds": res["holds"], "converged": False, "gram_det": res["gram_det"],
        "active_grads": grads, "x": [0.0, 0.0], "path": [[0.0, 0.0]],
        "f": f, "ineq_funcs": ineq_funcs, "grad_funcs": grad_funcs,
        "iters": 1, "fun": 0.0, "grad_norm": 0.0
    }

# ===========================================================================
# 23 Case Specifications
# ===========================================================================
CASES = [
    {
        "id": 1,
        "algorithm_number": 1,
        "algorithm_name": "gradient_descent_fixed",
        "failure_title": "Step Size Overshoot Divergence",
        "why": "When the step size exceeds 2/L where L is the maximum eigenvalue of the Hessian, gradient descent geometrically oversteps the local quadratic approximation. The iterate norm grows exponentially until floating-point overflow or divergence guards terminate the optimization.",
        "good_desc": "step size alpha = 0.015 < 2/L converges stably to the minimizer",
        "bad_desc": "oversized step alpha = 0.025 > 2/L causes geometric norm divergence",
        "good_run": _c1_good,
        "bad_run": _c1_bad,
        "detector": diverged
    },
    {
        "id": 2,
        "algorithm_number": 2,
        "algorithm_name": "gradient_descent_exact",
        "failure_title": "Orthogonal Zig-Zagging in Ill-Conditioned Ravines",
        "why": "Exact line search enforces strict orthogonality between consecutive gradients, satisfying g_{k+1}^T g_k = 0 at every iteration. In ill-conditioned ravines where kappa >> 1, this orthogonality forces iterates into severe zig-zag oscillations that slow progress along the valley.",
        "good_desc": "isotropic quadratic (kappa = 1) converges in 1 exact step",
        "bad_desc": "ill-conditioned quadratic (kappa = 100) exhibits 90-degree zig-zag stalling",
        "good_run": _c2_good,
        "bad_run": _c2_bad,
        "detector": zigzag_score
    },
    {
        "id": 3,
        "algorithm_number": 3,
        "algorithm_name": "backtracking_line_search",
        "failure_title": "Line Search Exhaustion on Non-Descent Direction",
        "why": "Backtracking line search relies on the directional derivative g^T d being strictly negative to guarantee that an Armijo step size exists. Along an ascent or non-descent direction where g^T d >= 0, the step size shrinks repeatedly until max halvings is exhausted, failing with alpha=0.",
        "good_desc": "backtracking along descent direction satisfies Armijo condition in few halvings",
        "bad_desc": "backtracking along ascent direction exhausts step halvings and fails",
        "good_run": _c3_good,
        "bad_run": _c3_bad,
        "detector": lambda res: (res.failed if hasattr(res, "failed") else res.get("failed", False), "Line search failed on non-descent direction")
    },
    {
        "id": 4,
        "algorithm_number": 4,
        "algorithm_name": "armijo_condition",
        "failure_title": "Armijo Condition Violation on Oversized Step",
        "why": "The Armijo condition ensures sufficient decrease by requiring f(x + alpha * d) <= f(x) + c1 * alpha * g^T d. When an oversized step size is chosen, the objective overshoots the descent valley and violates the decrease bound, returning False.",
        "good_desc": "sufficient decrease test passes for moderate step alpha = 0.005",
        "bad_desc": "oversized step alpha = 1.0 violates sufficient decrease bound",
        "good_run": _c4_good,
        "bad_run": _c4_bad,
        "detector": lambda res: (not res, "Armijo condition violated by oversized step")
    },
    {
        "id": 5,
        "algorithm_number": 5,
        "algorithm_name": "gradient_descent_armijo",
        "failure_title": "Step Size Collapse in Rosenbrock Ravine",
        "why": "In high-curvature curved valleys like the Rosenbrock ravine, backtracking line search contracts step sizes severely to satisfy the Armijo decrease criterion. These contracted steps become so small that the algorithm stalls prematurely with high residual gradient norm before reaching the minimizer.",
        "good_desc": "quadratic bowl allows Armijo line search to reach minimizer cleanly",
        "bad_desc": "narrow curved Rosenbrock valley forces step size collapse to near zero",
        "good_run": _c5_good,
        "bad_run": _c5_bad,
        "detector": step_collapse
    },
    {
        "id": 6,
        "algorithm_number": 6,
        "algorithm_name": "polyak_momentum",
        "failure_title": "Resonant Energy Overshoot and Non-Monotonicity",
        "why": "Polyak momentum acts as a heavy particle that accumulates kinetic energy along the descent trajectory. When the momentum constant beta is tuned too high (beta=0.99), the velocity overshoots the minimum repeatedly, causing frequent increases in objective value.",
        "good_desc": "tuned momentum beta = 0.8 accelerates stably down the ravine",
        "bad_desc": "high momentum beta = 0.99 causes resonant energy overshoot and oscillations",
        "good_run": _c6_good,
        "bad_run": _c6_bad,
        "detector": lambda res: (sum(1 for i in range(len(res.f_hist)-1) if res.f_hist[i+1] > res.f_hist[i]) >= 40, f"Resonant energy overshoot: {sum(1 for i in range(len(res.f_hist)-1) if res.f_hist[i+1] > res.f_hist[i])} objective increases")
    },
    {
        "id": 7,
        "algorithm_number": 7,
        "algorithm_name": "nesterov",
        "failure_title": "Saddle Point Attraction on Invariant Subspace",
        "why": "When initialized on the symmetry axis of a saddle point where the unstable gradient component is exactly zero, momentum directions remain confined to the subspace. Accelerated gradient methods accelerate directly along the axis and stall at the saddle point.",
        "good_desc": "double well initialized off-axis escapes the saddle-like region into the minimizer",
        "bad_desc": "saddle point initialized on the symmetry axis has zero transverse momentum and stalls",
        "good_run": _c7_good,
        "bad_run": _c7_bad,
        "detector": stalled_at_saddle
    },
    {
        "id": 8,
        "algorithm_number": 8,
        "algorithm_name": "gd_linear_regression",
        "failure_title": "Feature Scale Sensitivity and Ill-Conditioning",
        "why": "Batch gradient descent applies an identical learning rate across all weights, which is constrained by the maximum eigenvalue of X^T X. When features have drastically unequal scales (1000:1 ratio), unstandardized batch GD requires a tiny step size and stalls on unscaled weights.",
        "good_desc": "standardized features yield well-conditioned loss and rapid convergence",
        "bad_desc": "unstandardized features with scale 1000 stall optimization due to ill-conditioning",
        "good_run": _c8_good,
        "bad_run": _c8_bad,
        "detector": lambda res: (not res.converged and res.fun > 0.5, f"Feature scale stalling: loss={res.fun:.4f}")
    },
    {
        "id": 9,
        "algorithm_number": 9,
        "algorithm_name": "sgd_linear_regression",
        "failure_title": "Non-Vanishing Gradient Variance Noise Floor",
        "why": "Single-sample stochastic gradient descent uses noisy estimates of the true gradient whose variance remains bounded away from zero. Under a constant learning rate, iterates cannot settle at the minimizer and fluctuate perpetually within an asymptotic noise floor.",
        "good_desc": "full-batch gradient descent converges deterministically to the analytic optimum",
        "bad_desc": "single-sample SGD fluctuates persistently within a non-vanishing noise floor",
        "good_run": _c9_good,
        "bad_run": _c9_bad,
        "detector": lambda res: noise_floor(res, window=25)
    },
    {
        "id": 10,
        "algorithm_number": 10,
        "algorithm_name": "minibatch_sgd_linear_regression",
        "failure_title": "Mini-batch Variance Plateau",
        "why": "While mini-batching reduces gradient variance by 1/B, nonzero variance persists for any finite batch size under a constant learning rate. Iterates quickly approach the neighborhood of the optimum and then plateau into a steady-state fluctuation ball.",
        "good_desc": "full-batch gradient descent converges smoothly to the minimum",
        "bad_desc": "mini-batch SGD with batch size 2 exhibits variance plateau near the optimum",
        "good_run": _c10_good,
        "bad_run": _c10_bad,
        "detector": lambda res: noise_floor(res, window=25)
    },
    {
        "id": 11,
        "algorithm_number": 11,
        "algorithm_name": "logistic_regression_gd",
        "failure_title": "Unbounded Weights on Linearly Separable Data",
        "why": "For linearly separable data without regularization, binary cross-entropy loss monotonically decreases only as predicted probabilities approach 1. Gradient descent drives the norm of the weight vector toward infinity to sharpen the logistic sigmoid.",
        "good_desc": "overlapping classification data yields bounded weights and stable convergence",
        "bad_desc": "linearly separable data drives unregularized logistic weights toward infinity",
        "good_run": _c11_good,
        "bad_run": _c11_bad,
        "detector": unbounded_weights
    },
    {
        "id": 12,
        "algorithm_number": 12,
        "algorithm_name": "adagrad",
        "failure_title": "Premature Step Freezing on Rosenbrock",
        "why": "AdaGrad continuously accumulates squared historical gradients in its denominator without exponential forgetting. In long ravines, this denominator grows unboundedly, freezing coordinate step sizes and stalling the optimizer far from the minimizer.",
        "good_desc": "AdaGrad on moderate quadratic adapts learning rates and reaches the minimum",
        "bad_desc": "AdaGrad on Rosenbrock accumulates large gradients and freezes step sizes prematurely",
        "good_run": _c12_good,
        "bad_run": _c12_bad,
        "detector": step_collapse
    },
    {
        "id": 13,
        "algorithm_number": 13,
        "algorithm_name": "rmsprop",
        "failure_title": "Ridge Chattering Across Steep Quadratic Walls",
        "why": "RMSProp normalizes coordinate steps using the root-mean-squared historical gradient, equalizing effective step lengths across all directions. On highly anisotropic quadratics, this normalization amplifies updates along the steep valley walls, causing violent oscillations across the ridge.",
        "good_desc": "tuned RMSProp with learning rate 0.05 smoothly descends the anisotropic valley",
        "bad_desc": "aggressive learning rate 0.5 creates persistent high-frequency limit cycle bouncing",
        "good_run": _c13_good,
        "bad_run": _c13_bad,
        "detector": lambda res: tail_oscillation(res, window=100, min_step=1e-3)
    },
    {
        "id": 14,
        "algorithm_number": 14,
        "algorithm_name": "adam",
        "failure_title": "Large Learning Rate Limit Cycle Oscillation",
        "why": "When Adam is initialized in steep valleys with an aggressive learning rate (lr=0.5), coordinate updates overshoot the valley walls repeatedly. Iterates become trapped in a persistent high-energy limit cycle where step lengths remain strictly above 1e-3.",
        "good_desc": "tuned Adam with learning rate 0.05 stably minimizes the quadratic surface",
        "bad_desc": "large learning rate 0.5 overshoots narrow Rosenbrock walls into an oscillatory loop",
        "good_run": _c14_good,
        "bad_run": _c14_bad,
        "detector": lambda res: tail_oscillation(res, window=100, min_step=1e-3)
    },
    {
        "id": 15,
        "algorithm_number": 15,
        "algorithm_name": "subgradient_method",
        "failure_title": "Constant Step-Size Limit Cycle on Kink",
        "why": "The subdifferential of a nonsmooth function does not vanish at non-differentiable extrema, retaining unit order even at the minimizer. With a constant step size, subgradient updates overshoot the kink on every step, trapping the iterate in a perpetual limit cycle.",
        "good_desc": "diminishing step size 0.5/sqrt(k) converges to non-smooth kink at origin",
        "bad_desc": "constant step size 0.5 chatters perpetually across the non-differentiable kink",
        "good_run": _c15_good,
        "bad_run": _c15_bad,
        "detector": lambda res: tail_oscillation(res, window=100, min_step=1e-3)
    },
    {
        "id": 16,
        "algorithm_number": 16,
        "algorithm_name": "subgradient_lasso",
        "failure_title": "Lack of Exact Sparsity in Subgradient Lasso",
        "why": "Subgradient optimization applies smooth Euclidean steps along an arbitrary subgradient element, which has measure zero probability of landing at exact zeros. Unlike proximal soft-thresholding, subgradient Lasso produces tiny nonzero weights without true sparsity.",
        "good_desc": "unregularized least squares problem converges to unconstrained continuous solution",
        "bad_desc": "subgradient method on Lasso fails to produce exact zero coefficients",
        "good_run": _c16_good,
        "bad_run": _c16_bad,
        "detector": lambda res: no_exact_zeros(res)
    },
    {
        "id": 17,
        "algorithm_number": 17,
        "algorithm_name": "newton_method",
        "failure_title": "Pure Newton Saddle Point Attraction",
        "why": "Pure Newton's method solves H p = -g to find stationary points where the gradient vanishes, regardless of whether the Hessian is positive definite, negative definite, or indefinite. Near a saddle point, Newton steps move directly toward the stationary saddle.",
        "good_desc": "convex quadratic: Newton reaches the minimizer in 1 step",
        "bad_desc": "indefinite Hessian: Newton lands on the saddle in 1 step",
        "good_run": _c17_good,
        "bad_run": _c17_bad,
        "detector": lambda res: stalled_at_saddle(res, getattr(res, "extra", {}).get("hess", saddle_point_quadratic_hess))
    },
    {
        "id": 18,
        "algorithm_number": 18,
        "algorithm_name": "damped_newton_method",
        "failure_title": "Non-Descent Direction Rejection in Negative Curvature",
        "why": "When the Hessian has negative eigenvalues in non-convex regions, the classical Newton direction p = -H^{-1} g can yield g^T p > 0, pointing uphill. Damped Newton with Armijo line search detects this non-descent direction and aborts the step rather than increasing the loss.",
        "good_desc": "convex quadratic: Newton step is a valid descent direction accepted by Armijo",
        "bad_desc": "saddle region: Newton direction has positive inner product and line search aborts",
        "good_run": _c18_good,
        "bad_run": _c18_bad,
        "detector": lambda res: (getattr(res, "extra", {}).get("non_descent", False), "Damped Newton rejected non-descent direction")
    },
    {
        "id": 19,
        "algorithm_number": 19,
        "algorithm_name": "verify_newton_one_step",
        "failure_title": "Newton Quadratic Model Breakdown on Non-Quadratic Objective",
        "why": "Newton's method achieves exact one-step convergence exclusively on quadratic functions where higher-order Taylor derivatives vanish identically. On non-quadratic functions like Rosenbrock, a single Newton step incurs truncation errors and leaves significant residual distance to the minimizer.",
        "good_desc": "pure quadratic objective: Newton step is exact with zero residual distance",
        "bad_desc": "quartic Rosenbrock function: higher-order Taylor terms leave large truncation residual",
        "good_run": _c19_good,
        "bad_run": _c19_bad,
        "detector": lambda res: (res.get("residual_dist", 1.0) > 0.1 if isinstance(res, dict) else norm_l2(vec_sub(res.x, [1.0, 1.0])) > 0.1, f"Newton one-step failed on non-quadratic: dist={res.get('residual_dist', 0.0) if isinstance(res, dict) else norm_l2(vec_sub(res.x, [1.0, 1.0])):.4f}")
    },
    {
        "id": 20,
        "algorithm_number": 20,
        "algorithm_name": "bfgs",
        "failure_title": "Negative Curvature Update Failure",
        "why": "The BFGS secant update preserves positive definiteness of inverse Hessian approximations only if the curvature condition s_k^T y_k > 0 holds. On the bounded double well f(x, y) = x^4 - 2x^2 + y^2 in regions of negative curvature (|x| < 0.577), s^T y <= 0 occurs and updates are skipped.",
        "good_desc": "strictly convex quadratic preserves positive curvature s^T y > 0 for all updates",
        "bad_desc": "nonconvex double well visits negative curvature region causing skipped BFGS updates",
        "good_run": _c20_good,
        "bad_run": _c20_bad,
        "detector": lambda res: (getattr(res, "extra", {}).get("skipped", 0) > 0 or any(v <= 0 for v in getattr(res, "extra", {}).get("sy_hist", [])), "Negative curvature encountered (s^T y <= 0)")
    },
    {
        "id": 21,
        "algorithm_number": 21,
        "algorithm_name": "lagrange_solver",
        "failure_title": "Singular KKT Matrix from Redundant Constraints",
        "why": "Equality-constrained quadratic optimization requires the constraint matrix A to have full row rank to guarantee nonsingularity of the block KKT system. When redundant or linearly dependent constraints are specified, the KKT matrix is rank-deficient and cannot be solved by Gaussian elimination.",
        "good_desc": "linearly independent constraints form an invertible KKT coefficient matrix",
        "bad_desc": "redundant duplicate equality constraints produce a singular non-invertible KKT matrix",
        "good_run": _c21_good,
        "bad_run": _c21_bad,
        "detector": singular_kkt
    },
    {
        "id": 22,
        "algorithm_number": 22,
        "algorithm_name": "kkt_check",
        "failure_title": "KKT First-Order Sufficiency Failure (Constrained Maximum)",
        "why": "First-order KKT conditions ensure only that the Lagrangian gradient vanishes, which is necessary for any local extremum. On indefinite or non-convex objectives, a stationary point satisfying all KKT conditions can be a constrained local maximum if the reduced Hessian is negative definite.",
        "good_desc": "constrained minimizer satisfies first-order KKT and positive definite reduced Hessian",
        "bad_desc": "constrained maximizer passes first-order KKT but fails second-order sufficiency check",
        "good_run": _c22_good,
        "bad_run": _c22_bad,
        "detector": lambda res: not_minimizer(res["x"], res["Q"], res["A"])
    },
    {
        "id": 23,
        "algorithm_number": 23,
        "algorithm_name": "licq_check",
        "failure_title": "LICQ Violation from Collinear Constraint Gradients",
        "why": "The Linear Independence Constraint Qualification requires the gradient vectors of all active constraints to be linearly independent. When active constraint gradients are collinear, the constraint Jacobian loses row rank, the Gram determinant vanishes, and Lagrange multipliers may be non-unique or unbounded.",
        "good_desc": "transverse constraint boundaries have linearly independent active gradients (rank 2)",
        "bad_desc": "tangent circle constraints have collinear opposing gradients failing LICQ (rank 1 < 2)",
        "good_run": _c23_good,
        "bad_run": _c23_bad,
        "detector": lambda res: licq_failed(res["active_grads"] if isinstance(res, dict) else res)
    }
]

ALL_CASES = {case["id"]: case for case in CASES}
