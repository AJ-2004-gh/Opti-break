import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import csv
from problems.linear_regression import generate_data
from problems.logistic_regression import generate_blobs
from algorithms.regression_optimizers import (
    gd_linear_regression, sgd_linear_regression,
    minibatch_sgd_linear_regression, logistic_regression_gd
)
from visualization.model_plots import loss_contour_regression, loss_contour_logistic

def _run_linear_reg_exp():
    """Compares Batch GD, SGD, and Mini-batch SGD on linear regression."""
    X_raw, y = generate_data(n=60, seed=42, noise=0.1, scale_x2=1.0)
    X_1d = [[row[0]] for row in X_raw]
    r_bgd = gd_linear_regression(X_1d, y, lr=0.08, epochs=120)
    r_sgd = sgd_linear_regression(X_1d, y, lr=0.02, epochs=120, seed=42)
    r_mb = minibatch_sgd_linear_regression(X_1d, y, lr=0.05, epochs=120, batch_size=8, seed=42)
    paths = {"Batch GD": r_bgd.path, "Mini-batch SGD (B=8)": r_mb.path, "Single-sample SGD": r_sgd.path}
    plot_p = "results/plots/regression_loss_contour.png"
    loss_contour_regression(X_1d, y, paths, plot_p, "Linear Regression Loss Landscape & Fitted Lines", (1.0, 5.0), (-1.0, 3.0))
    return r_bgd, r_sgd, r_mb, plot_p

def _run_logistic_reg_exp():
    """Runs logistic regression GD on 2D synthetic Gaussian blobs."""
    X_log, y_log = generate_blobs(n=60, separation=2.0, seed=42)
    r_log = logistic_regression_gd(X_log, y_log, lr=0.3, epochs=150)
    paths = {"Logistic GD": [p[:2] for p in r_log.path]}
    plot_p = "results/plots/logistic_loss_contour.png"
    loss_contour_logistic(X_log, y_log, paths, plot_p, "Logistic Regression BCE Loss Landscape & Decision Boundary",
                          (-2.0, 4.0), (-3.0, 3.0), b=r_log.extra.get("b", 0.0))
    return r_log, plot_p

def run_regression_experiments(verbose=True):
    """Executes linear and logistic regression experimental suite."""
    os.makedirs("results/plots", exist_ok=True); os.makedirs("results/histories", exist_ok=True)
    r_bgd, r_sgd, r_mb, p_lin = _run_linear_reg_exp()
    r_log, p_log = _run_logistic_reg_exp()
    csv_p = "results/histories/regression_experiments.csv"
    with open(csv_p, "w", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out)
        w.writerow(["model", "optimizer", "iters", "converged", "final_loss"])
        w.writerow(["Linear Regression", "Batch GD", r_bgd.iters, r_bgd.converged, f"{r_bgd.fun:.6e}"])
        w.writerow(["Linear Regression", "Mini-batch SGD", r_mb.iters, r_mb.converged, f"{r_mb.fun:.6e}"])
        w.writerow(["Linear Regression", "Single-sample SGD", r_sgd.iters, r_sgd.converged, f"{r_sgd.fun:.6e}"])
        w.writerow(["Logistic Regression", "GD", r_log.iters, r_log.converged, f"{r_log.fun:.6e}"])
    if verbose:
        print(f"Linear regression plot: {p_lin}\nLogistic regression plot: {p_log}\nCSV: {csv_p}")
    return {"batch_gd": r_bgd, "sgd": r_sgd, "minibatch": r_mb, "logistic_gd": r_log}

if __name__ == "__main__":
    run_regression_experiments(verbose=True)
