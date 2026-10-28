import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
from problems.quadratic import make_quadratic
from problems.rosenbrock import rosenbrock_2d, rosenbrock_2d_grad, rosenbrock_2d_hess
from problems.linear_regression import generate_data
from problems.logistic_regression import generate_blobs
from problems.lasso import generate_sparse_linear_data
from algorithms.line_search import backtracking_line_search
from algorithms.gradient_descent import (
    gradient_descent_fixed, gradient_descent_exact, gradient_descent_armijo
)
from algorithms.momentum import polyak_momentum
from algorithms.nesterov import nesterov
from algorithms.regression_optimizers import (
    gd_linear_regression, sgd_linear_regression,
    minibatch_sgd_linear_regression, logistic_regression_gd
)
from algorithms.adaptive_optimizers import adagrad, rmsprop, adam
from algorithms.subgradient import subgradient_method, subgradient_lasso
from algorithms.newton import newton_method, damped_newton_method, verify_newton_one_step
from algorithms.bfgs import bfgs
from algorithms.constrained_optimization import lagrange_solver, kkt_check, licq_check
from visualization.contour import (
    contour_with_paths, contour_with_line_search, contour_with_constraints,
    plot_armijo_line_search
)
from visualization.model_plots import (
    loss_contour_regression, loss_contour_logistic, loss_contour_lasso
)

def _gen_gd_family_contours():
    """Generates contour plots for Algorithms 1-5."""
    f_q, g_q, _ = make_quadratic([[10.0, 0.0], [0.0, 50.0]], [0.0, 0.0])
    x0 = [2.0, 1.5]
    # Alg 1
    r1 = gradient_descent_fixed(f_q, g_q, x0, alpha=0.015, max_iters=150)
    contour_with_paths(f_q, (-3, 3), (-3, 3), {"Alg 1 Fixed GD": r1.path},
                       "results/plots/alg_01_gradient_descent_fixed.png", "Alg 1: Fixed Step GD", optimum=[0, 0])
    # Alg 2
    r2 = gradient_descent_exact(x0, [[10.0, 0.0], [0.0, 50.0]], max_iters=80)
    contour_with_paths(f_q, (-3, 3), (-3, 3), {"Alg 2 Exact GD": r2.path},
                       "results/plots/alg_02_gradient_descent_exact.png", "Alg 2: Exact Line Search GD", optimum=[0, 0])
    # Alg 3: Backtracking line search with at least 3 rejections
    # Rosenbrock from [-1.2, 1.0] with steepest descent direction d = -grad:
    # alpha_init = 0.02, rho = 0.5 performs 4 rejections before accepting alpha = 0.00125
    x0_rb = [-1.2, 1.0]
    g_rb = rosenbrock_2d_grad(x0_rb)
    d_rb = [-g_rb[0], -g_rb[1]]
    ls = backtracking_line_search(rosenbrock_2d, x0_rb, d_rb, g_rb, alpha_init=0.02, rho=0.5, c1=1e-4)
    contour_with_line_search(rosenbrock_2d, x0_rb, d_rb, ls["trials"], ls["alpha"],
                             "results/plots/alg_03_backtracking_line_search.png", "Alg 3: Backtracking Line Search")
    # Alg 4: 1D Armijo line search decrease condition plot
    nrm_rb = math.hypot(g_rb[0], g_rb[1])
    d_rb_unit = [-g_rb[0] / nrm_rb, -g_rb[1] / nrm_rb]
    ls4 = backtracking_line_search(rosenbrock_2d, x0_rb, d_rb_unit, g_rb, alpha_init=1.0, rho=0.5, c1=1e-4)
    plot_armijo_line_search(rosenbrock_2d, x0_rb, d_rb_unit, g_rb, 1e-4, ls4["trials"], ls4["alpha"],
                            "results/plots/alg_04_armijo_condition.png", "Alg 4: Armijo Condition Verification", alpha_max=1.0)
    # Alg 5
    r5 = gradient_descent_armijo(f_q, g_q, x0, alpha_init=1.0, max_iters=80)
    contour_with_paths(f_q, (-3, 3), (-3, 3), {"Alg 5 Armijo GD": r5.path},
                       "results/plots/alg_05_gradient_descent_armijo.png", "Alg 5: GD with Armijo Backtracking", optimum=[0, 0])

def _gen_momentum_adaptive_contours():
    """Generates contour plots for Momentum, Nesterov, and Adaptive optimizers."""
    f_q, g_q, _ = make_quadratic([[10.0, 0.0], [0.0, 50.0]], [0.0, 0.0])
    x0 = [2.0, 1.5]
    # Alg 6
    r6 = polyak_momentum(f_q, g_q, x0, alpha=0.01, beta=0.8, max_iters=100)
    contour_with_paths(f_q, (-3, 3), (-3, 3), {"Alg 6 Momentum": r6.path},
                       "results/plots/alg_06_polyak_momentum.png", "Alg 6: Polyak Heavy Ball Momentum", optimum=[0, 0])
    # Alg 7
    r7 = nesterov(f_q, g_q, x0, alpha=0.01, beta=0.8, max_iters=100)
    contour_with_paths(f_q, (-3, 3), (-3, 3), {"Alg 7 Nesterov": r7.path},
                       "results/plots/alg_07_nesterov_accelerated.png", "Alg 7: Nesterov Accelerated Gradient", optimum=[0, 0])
    # Alg 12, 13, 14: Tuned on quadratic Q=diag(1, 100) from [2, 1] demonstrating convergence
    # (AdaGrad lr=1.0 reaches grad norm 7.6e-6, RMSProp lr=0.08 reaches 7.1e-6, Adam lr=0.1 reaches 7.2e-6)
    f_adapt, g_adapt, _ = make_quadratic([[1.0, 0.0], [0.0, 100.0]], [0.0, 0.0])
    x0_adapt = [2.0, 1.0]
    # Alg 12
    r12 = adagrad(f_adapt, g_adapt, x0_adapt, lr=1.0, max_iters=200)
    contour_with_paths(f_adapt, (-2.5, 2.5), (-1.5, 1.5), {"Alg 12 AdaGrad": r12.path},
                       "results/plots/alg_12_adagrad.png", "Alg 12: AdaGrad Adaptive Gradient", optimum=[0, 0],
                       converged={"Alg 12 AdaGrad": r12.converged})
    # Alg 13
    r13 = rmsprop(f_adapt, g_adapt, x0_adapt, lr=0.08, max_iters=200)
    contour_with_paths(f_adapt, (-2.5, 2.5), (-1.5, 1.5), {"Alg 13 RMSProp": r13.path},
                       "results/plots/alg_13_rmsprop.png", "Alg 13: RMSProp Adaptive Optimizer", optimum=[0, 0],
                       converged={"Alg 13 RMSProp": r13.converged})
    # Alg 14
    r14 = adam(f_adapt, g_adapt, x0_adapt, lr=0.1, max_iters=300)
    contour_with_paths(f_adapt, (-2.5, 2.5), (-1.5, 1.5), {"Alg 14 Adam": r14.path},
                       "results/plots/alg_14_adam.png", "Alg 14: Adam Adaptive Moments", optimum=[0, 0],
                       converged={"Alg 14 Adam": r14.converged})

def _gen_regression_contours():
    """Generates loss landscape contours for Linear and Logistic Regression."""
    X_raw, y = generate_data(n=50, seed=42, noise=0.1)
    X_1d = [[r[0]] for r in X_raw]
    # Alg 8
    r8 = gd_linear_regression(X_1d, y, lr=0.08, epochs=100)
    loss_contour_regression(X_1d, y, {"Batch GD": r8.path}, "results/plots/alg_08_gd_linear_regression.png", "Alg 8: Batch GD Regression")
    # Alg 9
    r9 = sgd_linear_regression(X_1d, y, lr=0.02, epochs=100, seed=42)
    loss_contour_regression(X_1d, y, {"SGD": r9.path}, "results/plots/alg_09_sgd_linear_regression.png", "Alg 9: Stochastic GD Regression")
    # Alg 10
    r10 = minibatch_sgd_linear_regression(X_1d, y, lr=0.05, epochs=100, batch_size=8, seed=42)
    loss_contour_regression(X_1d, y, {"Mini-batch SGD": r10.path}, "results/plots/alg_10_minibatch_sgd_linear_regression.png", "Alg 10: Mini-batch SGD")
    # Alg 11
    X_log, y_log = generate_blobs(n=50, separation=2.0, seed=42)
    r11 = logistic_regression_gd(X_log, y_log, lr=0.3, epochs=120)
    loss_contour_logistic(X_log, y_log, {"Logistic GD": [p[:2] for p in r11.path]},
                          "results/plots/alg_11_logistic_regression_gd.png", "Alg 11: Logistic Regression GD", b=r11.extra.get("b", 0.0))

def _gen_newton_bfgs_contours():
    """Generates contour plots for Newton and BFGS algorithms."""
    # Alg 17: Pure Newton on Rosenbrock converges in 5 steps; y-range (-4.0, 3.5) accommodates iterate 2
    r17 = newton_method(rosenbrock_2d, rosenbrock_2d_grad, rosenbrock_2d_hess, [-1.2, 1.0], max_iters=20)
    contour_with_paths(rosenbrock_2d, (-2, 2), (-4.0, 3.5), {"Alg 17 Newton": r17.path},
                       "results/plots/alg_17_newton_method.png", "Alg 17: Pure Newton's Method", optimum=[1, 1],
                       converged={"Alg 17 Newton": r17.converged})
    # Alg 18
    r18 = damped_newton_method(rosenbrock_2d, rosenbrock_2d_grad, rosenbrock_2d_hess, [-1.2, 1.0], max_iters=35)
    contour_with_paths(rosenbrock_2d, (-2, 2), (-1, 3), {"Alg 18 Damped Newton": r18.path},
                       "results/plots/alg_18_damped_newton.png", "Alg 18: Damped Newton with Armijo", optimum=[1, 1])
    # Alg 19
    f_q, _, _ = make_quadratic([[4.0, 0.0], [0.0, 10.0]], [0.0, 0.0])
    dist = verify_newton_one_step([[4.0, 0.0], [0.0, 10.0]], [0.0, 0.0], [2.0, 1.0])
    contour_with_paths(f_q, (-3, 3), (-3, 3), {"1-Step Newton": [[2.0, 1.0], [0.0, 0.0]]},
                       "results/plots/alg_19_newton_one_step.png", f"Alg 19: Newton 1-Step Exact (Residual: {dist:.1e})", optimum=[0, 0])
    # Alg 20
    r20 = bfgs(rosenbrock_2d, rosenbrock_2d_grad, [-1.2, 1.0], max_iters=50)
    contour_with_paths(rosenbrock_2d, (-2, 2), (-1, 3), {"Alg 20 BFGS": r20.path},
                       "results/plots/alg_20_bfgs.png", "Alg 20: BFGS Quasi-Newton", optimum=[1, 1])

def _gen_constrained_contours():
    """Generates contour plots for Nonsmooth and Constrained optimization."""
    # Alg 15
    f_sub = lambda pt: abs(pt[0]) + 2.0 * abs(pt[1])
    s_fn = lambda pt: [math.copysign(1.0, pt[0]) if abs(pt[0]) > 1e-12 else 0.0, 2.0 * (math.copysign(1.0, pt[1]) if abs(pt[1]) > 1e-12 else 0.0)]
    r15 = subgradient_method(f_sub, s_fn, [3.0, 2.0], step_rule=('diminishing', 0.5), max_iters=300)
    contour_with_paths(f_sub, (-4, 4), (-4, 4), {"Alg 15 Subgradient": r15.path},
                       "results/plots/alg_15_subgradient_method.png", "Alg 15: Subgradient Method", optimum=[0, 0], log_levels=False)
    # Alg 16
    X_s, y_s, _ = generate_sparse_linear_data(n_samples=40, n_features=2, n_informative=1, seed=42)
    r16 = subgradient_lasso(X_s, y_s, lam=0.2, step_rule=('diminishing', 0.1), max_iters=150)
    loss_contour_lasso(X_s, y_s, 0.2, {"Alg 16 Lasso": r16.path}, "results/plots/alg_16_subgradient_lasso.png", "Alg 16: Subgradient Lasso")
    # Alg 21
    f_sq = lambda pt: pt[0]**2 + pt[1]**2
    sol_x, _, _ = lagrange_solver([[2.0, 0.0], [0.0, 2.0]], [0.0, 0.0], [[1.0, 1.0]], [1.0])
    contour_with_constraints(f_sq, (-1.5, 2.5), (-1.5, 2.5), [lambda pt: pt[0] + pt[1] - 1.0], [], [sol_x],
                             "results/plots/alg_21_lagrange_solver.png", "Alg 21: Lagrange Multiplier Solver")
    # Alg 22
    kkt_pt = [1.0, 0.0]
    contour_with_constraints(f_sq, (-0.5, 2.5), (-1.5, 2.0), [lambda pt: pt[0] - 1.0], [lambda pt: -pt[1]], [kkt_pt],
                             "results/plots/alg_22_kkt_check.png", "Alg 22: KKT Optimality Verification")
    # Alg 23
    f_licq = lambda pt: pt[1]
    ineq_licq = [
        lambda pt: (pt[0] - 1.0)**2 + pt[1]**2 - 1.0,
        lambda pt: (pt[0] + 1.0)**2 + pt[1]**2 - 1.0
    ]
    arrows_licq = [
        ([0.0, 0.0], [-2.0, 0.0], "grad g1 = [-2, 0]"),
        ([0.0, 0.0], [2.0, 0.0], "grad g2 = [2, 0]")
    ]
    contour_with_constraints(f_licq, (-3.0, 3.5), (-2.0, 2.0), [], ineq_licq, {"(0,0)": [0.0, 0.0]},
                             "results/plots/alg_23_licq_check.png", "Alg 23: Degenerate Problem (rank = 1 < 2: LICQ fails)",
                             active_grad_arrows=arrows_licq)

def generate_all_algorithm_contours(verbose=True):
    """Generates dedicated contour plots for all 23 algorithms into results/plots/."""
    os.makedirs("results/plots", exist_ok=True)
    _gen_gd_family_contours()
    _gen_momentum_adaptive_contours()
    _gen_regression_contours()
    _gen_newton_bfgs_contours()
    _gen_constrained_contours()
    if verbose:
        print("Generated individual contour plots for all 23 algorithms in results/plots/.")
