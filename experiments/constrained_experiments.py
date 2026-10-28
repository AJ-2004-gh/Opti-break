import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
import csv
from problems.lasso import generate_sparse_linear_data
from algorithms.subgradient import subgradient_method, subgradient_lasso
from algorithms.constrained_optimization import lagrange_solver, kkt_check, licq_check
from visualization.contour import contour_with_paths, contour_with_constraints
from visualization.model_plots import loss_contour_lasso

def _run_subgrad_lasso_exp():
    """Runs subgradient and subgradient Lasso benchmark experiments."""
    f_sub = lambda pt: abs(pt[0]) + 2.0 * abs(pt[1])
    subg_fn = lambda pt: [math.copysign(1.0, pt[0]) if abs(pt[0]) > 1e-12 else 0.0,
                          2.0 * (math.copysign(1.0, pt[1]) if abs(pt[1]) > 1e-12 else 0.0)]
    r_sub = subgradient_method(f_sub, subg_fn, [3.0, 2.0], step_rule=('diminishing', 0.5), max_iters=400)
    p_sub = "results/plots/constrained_subgradient.png"
    contour_with_paths(f_sub, (-4.0, 4.0), (-4.0, 4.0), {"Diminishing Subgradient": r_sub.path}, p_sub,
                       "Subgradient Method on Nonsmooth Kink |x| + 2|y|", optimum=[0.0, 0.0], log_levels=False)
    X_l, y_l, _ = generate_sparse_linear_data(n_samples=40, n_features=2, n_informative=1, seed=42)
    r_las = subgradient_lasso(X_l, y_l, lam=0.2, step_rule=('diminishing', 0.1), max_iters=200)
    p_las = "results/plots/constrained_lasso.png"
    loss_contour_lasso(X_l, y_l, 0.2, {"Subgradient Lasso": r_las.path}, p_las, "Lasso Loss Landscape & Subgradient Path (lam=0.2)")
    return r_sub, r_las, p_sub, p_las

def _run_lagrange_kkt_licq_exp():
    """Runs Lagrange solver, KKT, and LICQ verification experiments."""
    f_quad = lambda pt: pt[0]**2 + pt[1]**2
    sol_x, sol_lam, _ = lagrange_solver([[2.0, 0.0], [0.0, 2.0]], [0.0, 0.0], [[1.0, 1.0]], [1.0])
    p_lag = "results/plots/constrained_lagrange.png"
    contour_with_constraints(f_quad, (-1.5, 2.5), (-1.5, 2.5), [lambda pt: pt[0] + pt[1] - 1.0], [], [sol_x], p_lag,
                             "Lagrange Multiplier Solution: min x^2 + y^2 s.t. x + y = 1")
    kkt_pt = [1.0, 0.0]
    r_kkt = kkt_check([2.0, 0.0], [[1.0, 0.0]], [0.0], [[0.0, -1.0]], kkt_pt, [-2.0], [0.0])
    p_kkt = "results/plots/constrained_kkt.png"
    contour_with_constraints(f_quad, (-0.5, 2.5), (-1.5, 2.0), [lambda pt: pt[0] - 1.0], [lambda pt: -pt[1]], [kkt_pt], p_kkt,
                             f"KKT Optimality Verification: x*=[1, 0] (All Conditions: {r_kkt['overall']})")
    r_licq = licq_check([[1.0, 0.0], [0.0, 1.0]])
    arrows = [([0.0, 0.0], [1.0, 0.0], "grad g_1 = [1, 0]"), ([0.0, 0.0], [0.0, 1.0], "grad g_2 = [0, 1]")]
    p_licq = "results/plots/constrained_licq.png"
    contour_with_constraints(f_quad, (-1.5, 2.5), (-1.5, 2.5), [], [lambda pt: -pt[0], lambda pt: -pt[1]], [[0.0, 0.0]], p_licq,
                             f"LICQ Verification: Linearly Independent Gradients (Holds: {r_licq['holds']})", active_grad_arrows=arrows)
    return sol_x, r_kkt, r_licq, p_lag, p_kkt, p_licq

def run_constrained_experiments(verbose=True):
    """Executes nonsmooth and constrained optimization experimental suite."""
    os.makedirs("results/plots", exist_ok=True); os.makedirs("results/histories", exist_ok=True)
    r_sub, r_las, p_sub, p_las = _run_subgrad_lasso_exp()
    sol_x, r_kkt, r_licq, p_lag, p_kkt, p_licq = _run_lagrange_kkt_licq_exp()
    recs = [{"method": "Subgradient", "iters": r_sub.iters, "final_val": f"{r_sub.fun:.6e}"},
            {"method": "Subgradient Lasso", "iters": r_las.iters, "final_val": f"{r_las.fun:.6e}"},
            {"method": "Lagrange Solver", "iters": 1, "final_val": f"{sol_x[0]**2 + sol_x[1]**2:.6e}"},
            {"method": "KKT Check", "iters": 1, "final_val": f"{1.0:.6e}"},
            {"method": "LICQ Check", "iters": 1, "final_val": f"{r_licq['gram_det']:.6e}"}]
    csv_p = "results/histories/constrained_experiments.csv"
    with open(csv_p, "w", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=["method", "iters", "final_val"])
        w.writeheader()
        for r in recs: w.writerow(r)
    if verbose:
        print(f"Subgradient: {p_sub}\nLasso: {p_las}\nLagrange: {p_lag}\nKKT: {p_kkt}\nLICQ: {p_licq}\nCSV: {csv_p}")
    return {"subgradient": r_sub, "lasso": r_las, "lagrange_sol": sol_x, "kkt_check": r_kkt, "licq_check": r_licq}

if __name__ == "__main__":
    run_constrained_experiments(verbose=True)
