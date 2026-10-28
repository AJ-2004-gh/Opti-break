import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def _plot_regression_paths_and_fits(ax_contour, ax_fit, paths, x_sort):
    """Plots regression loss trajectory and fitted line comparison."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (name, path) in enumerate(paths.items()):
        if not path:
            continue
        c = colors[idx % len(colors)]
        ws, bs = [p[0] for p in path], [p[1] for p in path]
        ax_contour.plot(ws, bs, color=c, label=name, linewidth=1.8, alpha=0.85, zorder=5)
        ax_contour.scatter([ws[0]], [bs[0]], color=c, marker="o", s=60, edgecolors="white", linewidths=1.2, zorder=6)
        ax_contour.scatter([ws[-1]], [bs[-1]], color=c, marker="*", s=140, edgecolors="black", linewidths=1.0, zorder=6)
        ax_fit.plot(x_sort, ws[-1] * x_sort + bs[-1], color=c, linewidth=2.0, label=f"Fit: {name}", zorder=4)

def loss_contour_regression(X, y, paths, filepath, title="Linear Regression Loss Landscape",
                            w_range=(-1.0, 5.0), b_range=(-2.0, 4.0), resolution=100):
    """
    Contour of MSE over (w, b) for a one-feature dataset (y = w * x + b),
    with the paths of GD, SGD and mini-batch overlaid (Algorithms 8 to 10),
    plus a side figure/subplot with the fitted line on data points.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    x_data = [float(row[0] if isinstance(row, (list, tuple)) else row) for row in X]
    y_data = [float(v) for v in y]
    n = len(y_data)
    w_vals = np.linspace(w_range[0], w_range[1], resolution)
    b_vals = np.linspace(b_range[0], b_range[1], resolution)
    W, B = np.meshgrid(w_vals, b_vals)
    Z = np.array([[sum((W[i, j] * x_data[k] + B[i, j] - y_data[k])**2 for k in range(n)) / (2.0 * n)
                   for j in range(resolution)] for i in range(resolution)])
    fig, (ax_contour, ax_fit) = plt.subplots(1, 2, figsize=(14, 6))
    min_z, max_z = np.min(Z), np.max(Z)
    levels = np.geomspace(max(min_z, 1e-4), max_z, 30) if (min_z > 0 and (max_z / max(min_z, 1e-12)) > 10) else np.linspace(min_z, max_z, 30)
    cf = ax_contour.contourf(W, B, Z, levels=levels, cmap="viridis", alpha=0.85)
    ax_contour.contour(W, B, Z, levels=levels, colors="white", alpha=0.25, linewidths=0.5)
    fig.colorbar(cf, ax=ax_contour).set_label("MSE Loss", fontsize=10)
    ax_fit.scatter(x_data, y_data, color="black", alpha=0.6, label="Data (y)", s=30, zorder=3)
    x_sort = np.linspace(min(x_data) - 0.5, max(x_data) + 0.5, 100)
    _plot_regression_paths_and_fits(ax_contour, ax_fit, paths, x_sort)
    ax_contour.set_xlim(w_range); ax_contour.set_ylim(b_range); ax_contour.set_title(title, fontsize=12, fontweight="bold")
    ax_contour.set_xlabel("Weight w", fontsize=11); ax_contour.set_ylabel("Bias b", fontsize=11)
    ax_contour.legend(loc="best", framealpha=0.9); ax_contour.grid(True, linestyle=":", alpha=0.4)
    ax_fit.set_title("Fitted Regression Lines", fontsize=12, fontweight="bold")
    ax_fit.set_xlabel("Feature x", fontsize=11); ax_fit.set_ylabel("Target y", fontsize=11)
    ax_fit.legend(loc="best", framealpha=0.9); ax_fit.grid(True, linestyle=":", alpha=0.4)
    fig.savefig(filepath, dpi=150, bbox_inches="tight"); plt.close(fig)

def _calc_bce_grid(x_data, y_data, b, w1_vals, w2_vals, resolution, n):
    """Evaluates binary cross-entropy loss over a 2D weight grid."""
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    Z = np.zeros_like(W1)
    for i in range(resolution):
        for j in range(resolution):
            loss = 0.0
            for k in range(n):
                z = W1[i, j] * x_data[k][0] + W2[i, j] * x_data[k][1] + b
                p = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
                loss += -(y_data[k] * math.log(max(p, 1e-15)) + (1.0 - y_data[k]) * math.log(max(1.0 - p, 1e-15)))
            Z[i, j] = loss / max(1, n)
    return W1, W2, Z

def _plot_logistic_paths_and_boundaries(ax_contour, ax_boundary, paths, x_data, b):
    """Plots BCE parameter trajectory and resulting 2D decision boundary."""
    colors = ["#ff7f0e", "#2ca02c", "#9467bd"]
    for idx, (name, path) in enumerate(paths.items()):
        if not path:
            continue
        c = colors[idx % len(colors)]
        w1_s, w2_s = [p[0] for p in path], [p[1] for p in path]
        curr_b = path[-1][2] if len(path[-1]) > 2 else b
        ax_contour.plot(w1_s, w2_s, color=c, label=name, linewidth=1.8, alpha=0.85, zorder=5)
        ax_contour.scatter([w1_s[0]], [w2_s[0]], color=c, marker="o", s=60, edgecolors="white", linewidths=1.2, zorder=6)
        ax_contour.scatter([w1_s[-1]], [w2_s[-1]], color=c, marker="*", s=140, edgecolors="black", linewidths=1.0, zorder=6)
        if abs(w2_s[-1]) > 1e-4 and x_data:
            x1_min, x1_max = min(r[0] for r in x_data) - 1.0, max(r[0] for r in x_data) + 1.0
            x1_g = np.linspace(x1_min, x1_max, 50)
            ax_boundary.plot(x1_g, -(w1_s[-1] * x1_g + curr_b) / w2_s[-1], color=c, linewidth=2.0, label=f"Boundary: {name}")

def loss_contour_logistic(X, y, paths, filepath, title="Logistic Regression Loss Landscape",
                          w1_range=(-3.0, 3.0), w2_range=(-3.0, 3.0), b=0.0, resolution=100):
    """
    Contour of BCE over two weights with GD path, plus data points and decision boundary (Algorithm 11).
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    n = len(y)
    x_data = [[float(v) for v in row] for row in X]
    y_data = [float(v) for v in y]
    w1_vals, w2_vals = np.linspace(w1_range[0], w1_range[1], resolution), np.linspace(w2_range[0], w2_range[1], resolution)
    W1, W2, Z = _calc_bce_grid(x_data, y_data, b, w1_vals, w2_vals, resolution, n)
    fig, (ax_contour, ax_boundary) = plt.subplots(1, 2, figsize=(14, 6))
    cf = ax_contour.contourf(W1, W2, Z, levels=30, cmap="viridis", alpha=0.85)
    ax_contour.contour(W1, W2, Z, levels=30, colors="white", alpha=0.25, linewidths=0.5)
    fig.colorbar(cf, ax=ax_contour).set_label("BCE Loss", fontsize=10)
    c0_x = [x_data[i][0] for i in range(n) if y_data[i] < 0.5]; c0_y = [x_data[i][1] for i in range(n) if y_data[i] < 0.5]
    c1_x = [x_data[i][0] for i in range(n) if y_data[i] >= 0.5]; c1_y = [x_data[i][1] for i in range(n) if y_data[i] >= 0.5]
    ax_boundary.scatter(c0_x, c0_y, color="#1f77b4", marker="o", s=40, alpha=0.75, label="Class 0", edgecolors="black")
    ax_boundary.scatter(c1_x, c1_y, color="#d62728", marker="s", s=40, alpha=0.75, label="Class 1", edgecolors="black")
    _plot_logistic_paths_and_boundaries(ax_contour, ax_boundary, paths, x_data, b)
    ax_contour.set_xlim(w1_range); ax_contour.set_ylim(w2_range); ax_contour.set_title(title, fontsize=12, fontweight="bold")
    ax_contour.set_xlabel("Weight w_1", fontsize=11); ax_contour.set_ylabel("Weight w_2", fontsize=11)
    ax_contour.legend(loc="best", framealpha=0.9); ax_contour.grid(True, linestyle=":", alpha=0.4)
    ax_boundary.set_title("Dataset & Decision Boundary", fontsize=12, fontweight="bold")
    ax_boundary.set_xlabel("Feature x_1", fontsize=11); ax_boundary.set_ylabel("Feature x_2", fontsize=11)
    ax_boundary.legend(loc="best", framealpha=0.9); ax_boundary.grid(True, linestyle=":", alpha=0.4)
    fig.savefig(filepath, dpi=150, bbox_inches="tight"); plt.close(fig)

def loss_contour_lasso(X, y, lam, paths, filepath, title="Lasso Loss Landscape",
                       w1_range=(-2.0, 3.0), w2_range=(-2.0, 3.0), resolution=100):
    """
    Contour of the lasso loss over (w1, w2) with the subgradient path (Algorithm 16).
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    n = len(y)
    x_data = [[float(v) for v in row] for row in X]
    y_data = [float(v) for v in y]
    w1_vals = np.linspace(w1_range[0], w1_range[1], resolution)
    w2_vals = np.linspace(w2_range[0], w2_range[1], resolution)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    Z = np.array([[sum((W1[i, j] * x_data[k][0] + W2[i, j] * x_data[k][1] - y_data[k])**2 for k in range(n)) / (2.0 * n)
                   + float(lam) * (abs(W1[i, j]) + abs(W2[i, j])) for j in range(resolution)] for i in range(resolution)])
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    cf = ax.contourf(W1, W2, Z, levels=30, cmap="viridis", alpha=0.85)
    ax.contour(W1, W2, Z, levels=30, colors="white", alpha=0.25, linewidths=0.6)
    fig.colorbar(cf, ax=ax).set_label(f"Lasso Objective (lambda={lam})", fontsize=10)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
    for idx, (name, path) in enumerate(paths.items()):
        if not path:
            continue
        c = colors[idx % len(colors)]
        w1_s, w2_s = [p[0] for p in path], [p[1] for p in path]
        ax.plot(w1_s, w2_s, color=c, label=name, linewidth=1.8, alpha=0.85, zorder=5)
        ax.scatter([w1_s[0]], [w2_s[0]], color=c, marker="o", s=60, edgecolors="white", linewidths=1.2, zorder=6)
        ax.scatter([w1_s[-1]], [w2_s[-1]], color=c, marker="*", s=140, edgecolors="black", linewidths=1.0, zorder=6)
    ax.set_xlim(w1_range); ax.set_ylim(w2_range); ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Weight w_1", fontsize=11); ax.set_ylabel("Weight w_2", fontsize=11)
    ax.legend(loc="best", framealpha=0.9); ax.grid(True, linestyle=":", alpha=0.4)
    fig.savefig(filepath, dpi=150, bbox_inches="tight"); plt.close(fig)

def _draw_2d_paths(ax, paths):
    """Draws 2D parameter trajectories on a model loss surface."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, (name, path) in enumerate(paths.items()):
        if not path:
            continue
        c = colors[idx % len(colors)]
        xs = [float(p[0]) for p in path]
        ys = [float(p[1]) for p in path]
        if len(path) > 100:
            ax.plot(xs, ys, color=c, label=name, linewidth=0.8, alpha=0.6, zorder=5)
            ax.scatter(xs, ys, color=c, s=12, alpha=0.45, edgecolors="none", zorder=4)
        else:
            ax.plot(xs, ys, color=c, label=name, linewidth=1.8, alpha=0.85, zorder=5)
        ax.scatter([xs[0]], [ys[0]], color=c, marker="o", s=60, edgecolors="white", linewidths=1.2, zorder=6)
        ax.scatter([xs[-1]], [ys[-1]], color=c, marker="*", s=140, edgecolors="black", linewidths=1.0, zorder=6)

def render_regression_loss_panel(fig, ax, data, default_title="MSE Loss"):
    """Renders a single panel showing regression loss surface and trajectory."""
    X, y = data["X"], data["y"]
    paths = data.get("paths", {})
    x_data = [float(r[0] if isinstance(r, (list, tuple)) else r) for r in X]
    y_data = [float(v) for v in y]
    n = len(y_data)
    w_range = data.get("w_range", None)
    b_range = data.get("b_range", None)
    if w_range is None or b_range is None:
        all_w = [p[0] for path in paths.values() for p in path] or [3.0]
        all_b = [p[1] for path in paths.values() for p in path] or [1.0]
        pad_w = max(0.5, 0.4 * (max(all_w) - min(all_w) + 1e-4))
        pad_b = max(0.5, 0.4 * (max(all_b) - min(all_b) + 1e-4))
        w_range = w_range or (min(all_w) - pad_w, max(all_w) + pad_w)
        b_range = b_range or (min(all_b) - pad_b, max(all_b) + pad_b)
    res = data.get("resolution", 80)
    w_vals = np.linspace(w_range[0], w_range[1], res)
    b_vals = np.linspace(b_range[0], b_range[1], res)
    W, B = np.meshgrid(w_vals, b_vals)
    Z = np.array([[sum((W[i, j] * x_data[k] + B[i, j] - y_data[k])**2 for k in range(n)) / (2.0 * n)
                   for j in range(res)] for i in range(res)])
    min_z, max_z = np.min(Z), np.max(Z)
    levels = np.geomspace(max(min_z, 1e-4), max_z, 25) if (min_z > 0 and (max_z / max(min_z, 1e-12)) > 10) else np.linspace(min_z, max_z, 25)
    cf = ax.contourf(W, B, Z, levels=levels, cmap="viridis", alpha=0.85)
    ax.contour(W, B, Z, levels=levels, colors="white", alpha=0.25, linewidths=0.5)
    fig.colorbar(cf, ax=ax).set_label("MSE Loss", fontsize=9)
    _draw_2d_paths(ax, paths)
    ax.set_xlim(w_range); ax.set_ylim(b_range)
    ax.set_title(data.get("title", default_title), fontsize=11, fontweight="bold")
    ax.set_xlabel("Weight w", fontsize=10); ax.set_ylabel("Bias b", fontsize=10)
    h, l = ax.get_legend_handles_labels()
    if h: ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)

def loss_contour_regression_pair(left_data, right_data, verdict_text, filepath):
    """Generates comparison plot for linear regression cases with optional loss history panel."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    f_hist_l = left_data.get("f_hist") or left_data.get("loss_hist")
    f_hist_r = right_data.get("f_hist") or right_data.get("loss_hist")
    has_third = bool(f_hist_l or f_hist_r)

    if has_third:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6.5))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    fig.suptitle("Comparison: Works vs Breaks", fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.92, f"Verdict: {verdict_text}", ha="center", va="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2cc", edgecolor="#d6b656", lw=1.5))
    render_regression_loss_panel(fig, ax1, left_data, "Works")
    render_regression_loss_panel(fig, ax2, right_data, "Breaks")

    if has_third:
        if f_hist_l:
            ax3.plot(list(range(1, len(f_hist_l) + 1)), f_hist_l,
                     label="Standardized (converging)", color="#1f77b4", lw=2.0)
        if f_hist_r:
            ax3.plot(list(range(1, len(f_hist_r) + 1)), f_hist_r,
                     label="Unstandardized (stalled)", color="#d62728", lw=2.0)
        ax3.set_yscale("log")
        ax3.set_xlabel("Epoch", fontsize=10)
        ax3.set_ylabel("MSE Loss (log scale)", fontsize=10)
        ax3.set_title("Loss vs Epoch (log scale)", fontsize=11, fontweight="bold")
        ax3.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax3.grid(True, which="both", linestyle=":", alpha=0.4)
        plt.subplots_adjust(top=0.81, wspace=0.30)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return (ax1, ax2, ax3)

    plt.subplots_adjust(top=0.85)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return (ax1, ax2)

def render_logistic_loss_panel(fig, ax, data, default_title="BCE Loss"):
    """Renders a single panel showing logistic loss surface and trajectory."""
    X, y = data["X"], data["y"]
    b = float(data.get("b", 0.0))
    paths = data.get("paths", {})
    x_data = [[float(v) for v in row] for row in X]
    y_data = [float(v) for v in y]
    w1_range = data.get("w1_range", (-3.0, 3.0))
    w2_range = data.get("w2_range", (-3.0, 3.0))
    res = data.get("resolution", 80)
    w1_vals = np.linspace(w1_range[0], w1_range[1], res)
    w2_vals = np.linspace(w2_range[0], w2_range[1], res)
    W1, W2, Z = _calc_bce_grid(x_data, y_data, b, w1_vals, w2_vals, res, len(y_data))
    cf = ax.contourf(W1, W2, Z, levels=25, cmap="viridis", alpha=0.85)
    ax.contour(W1, W2, Z, levels=25, colors="white", alpha=0.25, linewidths=0.5)
    fig.colorbar(cf, ax=ax).set_label("BCE Loss", fontsize=9)
    _draw_2d_paths(ax, paths)
    ax.set_xlim(w1_range); ax.set_ylim(w2_range)
    ax.set_title(data.get("title", default_title), fontsize=11, fontweight="bold")
    ax.set_xlabel("Weight w_1", fontsize=10); ax.set_ylabel("Weight w_2", fontsize=10)
    h, l = ax.get_legend_handles_labels()
    if h: ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)

def loss_contour_logistic_pair(left_data, right_data, verdict_text, filepath):
    """Generates 3-panel comparison for Case 11: overlapping, separable, and weight norm vs epoch."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Comparison: Works vs Breaks", fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.92, f"Verdict: {verdict_text}", ha="center", va="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2cc", edgecolor="#d6b656", lw=1.5))
    render_logistic_loss_panel(fig, ax1, left_data, "Works")
    render_logistic_loss_panel(fig, ax2, right_data, "Breaks")
    w_norm_l = left_data.get("w_norm_hist")
    if not w_norm_l and "paths" in left_data:
        p = next(iter(left_data["paths"].values()), [])
        w_norm_l = [math.sqrt(float(pt[0])**2 + float(pt[1])**2) for pt in p]
    w_norm_r = right_data.get("w_norm_hist")
    if not w_norm_r and "paths" in right_data:
        p = next(iter(right_data["paths"].values()), [])
        w_norm_r = [math.sqrt(float(pt[0])**2 + float(pt[1])**2) for pt in p]
    if w_norm_l:
        ax3.plot(list(range(1, len(w_norm_l) + 1)), w_norm_l,
                 label="Overlapping (bounded)", color="#1f77b4", lw=2.0)
    if w_norm_r:
        ax3.plot(list(range(1, len(w_norm_r) + 1)), w_norm_r,
                 label="Separable (growing)", color="#d62728", lw=2.0)
    ax3.set_xscale("log")
    ax3.set_xlabel("Epoch (log scale)", fontsize=10)
    ax3.set_ylabel("Weight Norm ||w||_2", fontsize=10)
    ax3.set_title("Weight Norm vs Epoch", fontsize=11, fontweight="bold")
    if w_norm_l or w_norm_r:
        ax3.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax3.grid(True, which="both", linestyle=":", alpha=0.4)
    plt.subplots_adjust(top=0.85, wspace=0.28)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return (ax1, ax2, ax3)

def render_lasso_loss_panel(fig, ax, data, default_title="Lasso Loss"):
    """Renders a single panel showing Lasso loss surface and trajectory."""
    X, y = data["X"], data["y"]
    lam = float(data.get("lam", 0.0))
    paths = data.get("paths", {})
    x_data = [[float(v) for v in row] for row in X]
    y_data = [float(v) for v in y]
    n = len(y_data)
    w1_range = data.get("w1_range", (-2.0, 2.0))
    w2_range = data.get("w2_range", (-2.5, 1.0))
    res = data.get("resolution", 80)
    w1_vals = np.linspace(w1_range[0], w1_range[1], res)
    w2_vals = np.linspace(w2_range[0], w2_range[1], res)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    Z = np.array([[sum((W1[i, j] * x_data[k][0] + W2[i, j] * x_data[k][1] - y_data[k])**2 for k in range(n)) / (2.0 * n)
                   + lam * (abs(W1[i, j]) + abs(W2[i, j])) for j in range(res)] for i in range(res)])
    cf = ax.contourf(W1, W2, Z, levels=25, cmap="viridis", alpha=0.85)
    ax.contour(W1, W2, Z, levels=25, colors="white", alpha=0.25, linewidths=0.6)
    fig.colorbar(cf, ax=ax).set_label(f"Lasso Loss (lam={lam})", fontsize=9)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    _draw_2d_paths(ax, paths)
    ax.set_xlim(w1_range); ax.set_ylim(w2_range)
    ax.set_title(data.get("title", default_title), fontsize=11, fontweight="bold")
    ax.set_xlabel("Weight w_1", fontsize=10); ax.set_ylabel("Weight w_2", fontsize=10)
    h, l = ax.get_legend_handles_labels()
    if h: ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)

def loss_contour_lasso_pair(left_data, right_data, verdict_text, filepath):
    """Generates side-by-side comparison for Lasso regression cases."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle("Comparison: Works vs Breaks", fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.92, f"Verdict: {verdict_text}", ha="center", va="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2cc", edgecolor="#d6b656", lw=1.5))
    render_lasso_loss_panel(fig, ax1, left_data, "Works")
    render_lasso_loss_panel(fig, ax2, right_data, "Breaks")
    plt.subplots_adjust(top=0.85)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return (ax1, ax2)

def plot_convergence_histories(results_dict, metric="f_hist", title="Convergence History",
                               output_path=None, log_scale=True):
    """Plots convergence curves (objective or gradient norm) across multiple optimizers."""
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    for label, res in results_dict.items():
        hist = getattr(res, metric, None)
        if hist is None or len(hist) == 0:
            continue
        ax.plot(list(range(len(hist))), hist, label=f"{label} ({res.iters} iters)", linewidth=1.8)
    ax.set_title(title, fontsize=13, fontweight="bold"); ax.set_xlabel("Iteration (k)", fontsize=11)
    ax.set_ylabel("Objective Value f(x_k)" if metric == "f_hist" else "Gradient Norm ||g_k||", fontsize=11)
    if log_scale: ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5); ax.legend(loc="best", framealpha=0.9)
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return fig

def plot_regression_fit(X, y, w_dict, title="Regression Fit Comparison", output_path=None):
    """Plots 1D linear regression scatter data and fitted model line."""
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    n, d = len(y), len(X[0])
    if d == 1:
        x_pts = [row[0] for row in X]
        ax.scatter(x_pts, y, color="black", alpha=0.6, label="Data (y)")
        sorted_x = [p[0] for p in sorted(zip(x_pts, range(n)))]
        for label, w in w_dict.items():
            ax.plot(sorted_x, [float(xv * w[0]) for xv in sorted_x], label=f"Model: {label}", linewidth=2.0)
        ax.set_xlabel("Feature x", fontsize=11)
    else:
        indices = list(range(min(n, 50)))
        ax.plot(indices, [y[i] for i in indices], "k-o", label="Actual y", alpha=0.7)
        for label, w in w_dict.items():
            ax.plot(indices, [sum(X[i][j] * w[j] for j in range(d)) for i in indices], "--s", label=f"Pred: {label}", alpha=0.8)
        ax.set_xlabel("Sample Index", fontsize=11)
    ax.set_ylabel("Target / Prediction", fontsize=11); ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5); ax.legend(loc="best")
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return fig

def plot_lasso_path(lambdas, weights_matrix, feature_names=None, title="Lasso Regularization Paths", output_path=None):
    """Plots Lasso parameter evolution trajectory as regularization varies."""
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    n_features = len(weights_matrix[0])
    feature_names = feature_names or [f"w_{j}" for j in range(n_features)]
    for j in range(n_features):
        ax.plot(lambdas, [w[j] for w in weights_matrix], label=feature_names[j], linewidth=1.6)
    ax.set_xscale("log"); ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Regularization Parameter lambda", fontsize=11); ax.set_ylabel("Weight Value w_j", fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.5); ax.legend(loc="upper right", ncol=2, fontsize=8)
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return fig
