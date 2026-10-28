import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from problems.quadratic import make_quadratic
from problems.rosenbrock import (
    rosenbrock_2d, rosenbrock_2d_grad, rosenbrock_2d_hess
)
from algorithms.newton import newton_method, damped_newton_method
from algorithms.gradient_descent import gradient_descent_armijo
from algorithms.bfgs import bfgs
from visualization.contour import contour_with_paths

def _run_rosenbrock_newton_exp():
    """Compares Newton and BFGS convergence trajectories on Rosenbrock 2D."""
    x0 = [-1.2, 1.0]
    r_newt = newton_method(rosenbrock_2d, rosenbrock_2d_grad, rosenbrock_2d_hess, x0, max_iters=25)
    r_damped = damped_newton_method(rosenbrock_2d, rosenbrock_2d_grad, rosenbrock_2d_hess, x0, max_iters=50)
    r_bfgs = bfgs(rosenbrock_2d, rosenbrock_2d_grad, x0, max_iters=80)
    paths = {"Newton Method": r_newt.path, "Damped Newton": r_damped.path, "BFGS Quasi-Newton": r_bfgs.path}
    plot_p = "results/plots/newton_bfgs_rosenbrock.png"
    contour_with_paths(rosenbrock_2d, (-2.0, 2.0), (-1.0, 3.0), paths, plot_p,
                       "Newton, Damped Newton, and BFGS on Rosenbrock", optimum=[1.0, 1.0])
    return r_newt, r_damped, r_bfgs, plot_p

def _run_condition_number_study():
    """Studies convergence speed of Newton vs GD across condition numbers."""
    kappas = [1, 5, 10, 50, 100, 500, 1000]
    records, n_iters, gd_iters, b_iters = [], [], [], []
    for k in kappas:
        f, g, h = make_quadratic([[float(k), 0.0], [0.0, 1.0]], [float(k), 1.0])
        rn = newton_method(f, g, h, [0.0, 0.0], tol=1e-5)
        rg = gradient_descent_armijo(f, g, [0.0, 0.0], alpha_init=1.0, tol=1e-5, max_iters=1000)
        rb = bfgs(f, g, [0.0, 0.0], tol=1e-5, max_iters=200)
        n_iters.append(rn.iters); gd_iters.append(rg.iters); b_iters.append(rb.iters)
        records.append({"kappa": k, "newton_iters": rn.iters, "newton_converged": rn.converged,
                        "gd_iters": rg.iters, "gd_converged": rg.converged, "bfgs_iters": rb.iters, "bfgs_converged": rb.converged})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(kappas, n_iters, "g-o", lw=2.5, ms=8, label="Newton (O(1) 1-step exact)")
    ax.plot(kappas, b_iters, "b-s", lw=2.0, ms=6, label="BFGS Quasi-Newton")
    ax.plot(kappas, gd_iters, "r-^", lw=2.0, ms=6, label="Gradient Descent (Backtracking)")
    ax.set_xscale("log"); ax.set_title("Optimizer Sensitivity to Condition Number (Quadratic Form)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Condition Number kappa", fontsize=11); ax.set_ylabel("Iterations to Convergence", fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.5); ax.legend(loc="upper left", fontsize=10)
    plot_p = "results/plots/newton_vs_gd_condition_numbers.png"
    fig.savefig(plot_p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return records, plot_p

def run_newton_quadratic_experiment(verbose=True):
    """Executes Newton and BFGS experiments on Rosenbrock and quadratics."""
    os.makedirs("results/plots", exist_ok=True); os.makedirs("results/histories", exist_ok=True)
    r_n, r_d, r_b, p_rosen = _run_rosenbrock_newton_exp()
    records, p_cond = _run_condition_number_study()
    csv_p = "results/histories/newton_quadratic.csv"
    with open(csv_p, "w", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=["kappa", "newton_iters", "newton_converged", "gd_iters", "gd_converged", "bfgs_iters", "bfgs_converged"])
        w.writeheader()
        for r in records: w.writerow(r)
    if verbose:
        print(f"Rosenbrock plot: {p_rosen}\nCondition number plot: {p_cond}\nCSV: {csv_p}")
    return {"newton_rosen": r_n, "damped_rosen": r_d, "bfgs_rosen": r_b, "records": records}

if __name__ == "__main__":
    run_newton_quadratic_experiment(verbose=True)
