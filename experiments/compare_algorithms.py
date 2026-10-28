import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import csv
from problems.quadratic import make_quadratic
from algorithms.gradient_descent import (
    gradient_descent_fixed, gradient_descent_exact, gradient_descent_armijo
)
from algorithms.momentum import polyak_momentum
from algorithms.nesterov import nesterov
from visualization.contour import contour_with_paths

def _run_gd_comparison(f, grad, Q, x0):
    """Compares Fixed, Exact, and Armijo gradient descent variants on quadratic."""
    r_fixed = gradient_descent_fixed(f, grad, x0, alpha=0.015, max_iters=200)
    r_exact = gradient_descent_exact(x0, Q, max_iters=100)
    r_armijo = gradient_descent_armijo(f, grad, x0, alpha_init=1.0, max_iters=100)
    gd_paths = {
        "GD Fixed (alpha=0.015)": r_fixed.path,
        "GD Exact Line Search": r_exact.path,
        "GD Armijo Backtracking": r_armijo.path
    }
    plot_p = "results/plots/compare_gd_quadratic.png"
    contour_with_paths(f, (-3.0, 3.0), (-3.0, 3.0), gd_paths, plot_p,
                       "GD Variants on Quadratic: Trajectory Comparison", optimum=[0.0, 0.0])
    return r_fixed, r_exact, r_armijo, plot_p

def _run_momentum_comparison(f, grad, x0):
    """Compares Polyak momentum and Nesterov accelerated gradient on ill-conditioned ravine."""
    r_mom = polyak_momentum(f, grad, x0, alpha=0.01, beta=0.8, max_iters=150)
    r_nes = nesterov(f, grad, x0, alpha=0.01, beta=0.8, max_iters=150)
    acc_paths = {
        "Polyak Momentum (beta=0.8)": r_mom.path,
        "Nesterov Accelerated (beta=0.8)": r_nes.path
    }
    plot_p = "results/plots/compare_momentum_nesterov.png"
    contour_with_paths(f, (-3.0, 3.0), (-3.0, 3.0), acc_paths, plot_p,
                       "Momentum vs Nesterov on Quadratic", optimum=[0.0, 0.0])
    return r_mom, r_nes, plot_p

def run_compare_algorithms(verbose=True):
    """Runs benchmark comparisons for GD family and Momentum vs Nesterov."""
    os.makedirs("results/plots", exist_ok=True); os.makedirs("results/histories", exist_ok=True)
    Q = [[10.0, 0.0], [0.0, 50.0]]
    f, grad, _ = make_quadratic(Q, [0.0, 0.0])
    x0 = [2.0, 1.5]
    r_fix, r_ex, r_arm, p_gd = _run_gd_comparison(f, grad, Q, x0)
    r_mom, r_nes, p_acc = _run_momentum_comparison(f, grad, x0)
    csv_p = "results/histories/compare_algorithms.csv"
    with open(csv_p, "w", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out)
        w.writerow(["algorithm", "iters", "converged", "final_f", "final_grad_norm"])
        for name, r in [("GD Fixed", r_fix), ("GD Exact", r_ex), ("GD Armijo", r_arm),
                        ("Polyak Momentum", r_mom), ("Nesterov Accelerated", r_nes)]:
            w.writerow([name, r.iters, r.converged, f"{r.fun:.6e}", f"{r.grad_norm:.6e}"])
    if verbose:
        print(f"GD variants plot: {p_gd}\nMomentum vs Nesterov: {p_acc}\nCSV: {csv_p}")
    return {"gd_fixed": r_fix, "gd_exact": r_ex, "gd_armijo": r_arm, "momentum": r_mom, "nesterov": r_nes}

run_algorithm_comparison = run_compare_algorithms

if __name__ == "__main__":
    run_compare_algorithms(verbose=True)
