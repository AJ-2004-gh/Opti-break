import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def _eval_grid(f, xrange, yrange, resolution=100):
    """Evaluates 2D objective function across a meshgrid."""
    x_vals = np.linspace(xrange[0], xrange[1], resolution)
    y_vals = np.linspace(yrange[0], yrange[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            try:
                Z[i, j] = f([float(X[i, j]), float(Y[i, j])])
            except Exception:
                Z[i, j] = np.nan
    valid = np.isfinite(Z)
    if not np.any(valid):
        return X, Y, np.zeros_like(Z)
    min_z, max_z = np.min(Z[valid]), np.max(Z[valid])
    return X, Y, np.nan_to_num(Z, nan=max_z, posinf=max_z, neginf=min_z)

def _plot_contour_base(fig, ax, X, Y, Z, log_levels=True, levels_n=30, label="Objective f(x, y)"):
    """Plots base filled contours and contour level curves on an axis."""
    min_z, max_z = np.min(Z), np.max(Z)
    if log_levels and min_z > 0 and (max_z / max(min_z, 1e-12)) > 10:
        levels = np.geomspace(max(min_z, 1e-5), max_z, levels_n)
    else:
        levels = np.linspace(min_z, max_z, levels_n)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.85)
    ax.contour(X, Y, Z, levels=levels, colors="white", alpha=0.25, linewidths=0.5)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(label, fontsize=10)
    return cf

def _draw_divergence_annotation(ax, name, last_inside, first_outside, xrange, yrange):
    """Annotates diverging or clipped paths with an arrow callout."""
    target_pt = first_outside if first_outside is not None else last_inside
    dx, dy = target_pt[0] - last_inside[0], target_pt[1] - last_inside[1]
    nrm = math.hypot(dx, dy)
    dx, dy = (dx / nrm, dy / nrm) if nrm > 0 else (1.0, 0.0)
    tip_x = max(xrange[0] + 0.05 * (xrange[1] - xrange[0]), min(xrange[1] - 0.05 * (xrange[1] - xrange[0]), last_inside[0] + 0.15 * dx * (xrange[1] - xrange[0])))
    tip_y = max(yrange[0] + 0.05 * (yrange[1] - yrange[0]), min(yrange[1] - 0.05 * (yrange[1] - yrange[0]), last_inside[1] + 0.15 * dy * (yrange[1] - yrange[0])))
    ax.annotate(f"{name}: diverged", xy=(tip_x, tip_y),
                xytext=(tip_x - 0.25 * (xrange[1] - xrange[0]), tip_y - 0.20 * (yrange[1] - yrange[0])),
                arrowprops=dict(facecolor="#d62728", edgecolor="#d62728", arrowstyle="->", lw=2.0),
                fontsize=9, fontweight="bold", color="#d62728",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff2cc", edgecolor="#d62728", lw=1.2), zorder=10)

def _draw_paths(ax, paths, xrange, yrange, colors=None, converged=None, f=None):
    """Draws optimization trajectories with markers on a contour axis."""
    colors = colors or ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    for idx, (name, val) in enumerate(paths.items()):
        if val is None:
            continue
        is_conv, path = None, val
        if hasattr(val, "path"):
            path = val.path
            is_conv = getattr(val, "converged", None)
        elif isinstance(converged, dict):
            is_conv = converged.get(name)
        elif isinstance(converged, bool):
            is_conv = converged
        if not path:
            continue
        c = colors[idx % len(colors)]
        finite_pts, last_inside, first_outside = [], None, None
        has_finite_f = True
        for pt in path:
            px, py = float(pt[0]), float(pt[1])
            if not (math.isfinite(px) and math.isfinite(py)):
                has_finite_f = False
                break
            if f is not None:
                try:
                    f_val = f([px, py])
                    if not (math.isfinite(f_val) and abs(f_val) < 1e30):
                        has_finite_f = False
                        break
                except Exception:
                    has_finite_f = False
                    break
            finite_pts.append((px, py))
            if (xrange[0] <= px <= xrange[1]) and (yrange[0] <= py <= yrange[1]):
                last_inside = (px, py)
            elif first_outside is None and last_inside is not None:
                first_outside = (px, py)

        diverged = False
        if is_conv is False:
            diverged = True
        elif not has_finite_f:
            diverged = True
        elif is_conv is None and len(finite_pts) > 0:
            final_x, final_y = finite_pts[-1]
            if not (xrange[0] <= final_x <= xrange[1] and yrange[0] <= final_y <= yrange[1]):
                diverged = True

        if finite_pts:
            xs, ys = [p[0] for p in finite_pts], [p[1] for p in finite_pts]
            ax.plot(xs, ys, color=c, label=name, linewidth=2.0, alpha=0.9, zorder=5)
            ax.scatter([xs[0]], [ys[0]], color=c, marker="o", s=70, edgecolors="white", linewidths=1.5, zorder=6)
            ax.scatter([xs[-1]], [ys[-1]], color=c, marker="*", s=140, edgecolors="black", linewidths=1.0, zorder=6)
        if diverged and last_inside is not None:
            _draw_divergence_annotation(ax, name, last_inside, first_outside, xrange, yrange)

def contour_with_paths(f, xrange, yrange, paths, filepath, title, optimum=None, log_levels=True, markers=None, resolution=100, converged=None):
    """Generates 2D contour plot overlaid with named optimizer trajectory paths."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    X, Y, Z = _eval_grid(f, xrange, yrange, resolution)
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    _plot_contour_base(fig, ax, X, Y, Z, log_levels=log_levels, levels_n=30)
    if optimum is not None:
        ax.scatter([optimum[0]], [optimum[1]], color="red", marker="x", s=100, linewidths=2.5, label="Optimum", zorder=7)
    _draw_paths(ax, paths, xrange, yrange, converged=converged, f=f)
    ax.set_xlim(xrange); ax.set_ylim(yrange); ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x", fontsize=11); ax.set_ylabel("y", fontsize=11)
    ax.legend(loc="best", framealpha=0.9); ax.grid(True, linestyle=":", alpha=0.4)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ax

def _draw_constraint_curves(ax, X, Y, eq_constraints, ineq_constraints, resolution):
    """Draws equality curves and shaded inequality constraint boundaries."""
    if eq_constraints:
        for idx, h in enumerate(eq_constraints):
            H_vals = np.array([[h([float(X[i, j]), float(Y[i, j])]) for j in range(resolution)] for i in range(resolution)])
            ax.contour(X, Y, H_vals, levels=[0.0], colors=["#1f77b4"], linestyles="solid", linewidths=2.5)
            if idx == 0:
                ax.plot([], [], color="#1f77b4", linestyle="solid", linewidth=2.5, label="Equality h(x)=0")
    if ineq_constraints:
        for idx, g in enumerate(ineq_constraints):
            G_vals = np.array([[g([float(X[i, j]), float(Y[i, j])]) for j in range(resolution)] for i in range(resolution)])
            ax.contour(X, Y, G_vals, levels=[0.0], colors=["#d62728"], linestyles="dashed", linewidths=2.0)
            ax.contourf(X, Y, G_vals, levels=[0.0, 1e9], colors=["#ff4444"], alpha=0.15)
            if idx == 0:
                ax.plot([], [], color="#d62728", linestyle="dashed", linewidth=2.0, label="Inequality g(x)=0")
                ax.fill_between([], [], [], color="#ff4444", alpha=0.15, label="Infeasible g(x)>0")

def _draw_points_and_arrows(ax, points, active_grad_arrows):
    """Draws marked points and constraint normal gradient arrows."""
    sol_pt = None
    if isinstance(points, dict):
        for name, pt in points.items():
            ax.scatter([pt[0]], [pt[1]], color="gold", marker="*", s=160, edgecolors="black", linewidths=1.2, zorder=8, label=name)
            sol_pt = sol_pt or pt
    elif isinstance(points, (list, tuple)) and points:
        pts = [points] if isinstance(points[0], (int, float)) else points
        for idx, pt in enumerate(pts):
            lbl = "Solution" if idx == 0 else f"Point {idx+1}"
            ax.scatter([pt[0]], [pt[1]], color="gold", marker="*", s=160, edgecolors="black", linewidths=1.2, zorder=8, label=lbl)
            sol_pt = sol_pt or pt
    for arrow in (active_grad_arrows or []):
        pt, grad, lbl = (arrow[0], arrow[1], arrow[2]) if len(arrow) >= 3 else (sol_pt or [0, 0], arrow, "grad g")
        ax.quiver(pt[0], pt[1], grad[0], grad[1], angles='xy', scale_units='xy', scale=1, color="#2c3e50", width=0.012, zorder=9)
        tip_x, tip_y = pt[0] + grad[0], pt[1] + grad[1]
        offset = arrow[3] if len(arrow) > 3 else (0.0, 0.2)
        ax.annotate(lbl, xy=(tip_x, tip_y), xytext=(tip_x + offset[0], tip_y + offset[1]),
                    ha="center", va="bottom", color="#111111", fontweight="bold", fontsize=9.5, zorder=10,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.95))

def contour_with_constraints(f, xrange, yrange, eq_constraints, ineq_constraints, points, filepath, title, active_grad_arrows=None, resolution=100):
    """Generates 2D contour plot overlaid with constraint curves and markers."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    X, Y, Z = _eval_grid(f, xrange, yrange, resolution)
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    _plot_contour_base(fig, ax, X, Y, Z, log_levels=False, levels_n=30, label="f(x, y)")
    _draw_constraint_curves(ax, X, Y, eq_constraints, ineq_constraints, resolution)
    _draw_points_and_arrows(ax, points, active_grad_arrows)
    ax.set_xlim(xrange); ax.set_ylim(yrange); ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x", fontsize=11); ax.set_ylabel("y", fontsize=11)
    ax.legend(loc="best", framealpha=0.9); ax.grid(True, linestyle=":", alpha=0.4)
    fig.savefig(filepath, dpi=150, bbox_inches="tight"); plt.close(fig)

def contour_with_line_search(f, x, d, trial_alphas, accepted_alpha, filepath, title, xrange=None, yrange=None, resolution=100):
    """Plots objective contour overlaid with line search trial step evaluations."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    all_alphas = list(trial_alphas)
    if accepted_alpha is not None and accepted_alpha not in all_alphas:
        all_alphas.append(accepted_alpha)
    pts = [x] + [[x[0] + a * d[0], x[1] + a * d[1]] for a in all_alphas]
    pts_x, pts_y = [p[0] for p in pts], [p[1] for p in pts]
    min_x, max_x = min(pts_x), max(pts_x)
    min_y, max_y = min(pts_y), max(pts_y)
    span_x, span_y = max(max_x - min_x, 0.2), max(max_y - min_y, 0.2)
    pad_x, pad_y = 0.3 * span_x, 0.3 * span_y
    xrange = xrange or (min_x - pad_x, max_x + pad_x)
    yrange = yrange or (min_y - pad_y, max_y + pad_y)
    X, Y, Z = _eval_grid(f, xrange, yrange, resolution)
    min_z, max_z = float(np.min(Z)), float(np.max(Z))
    if min_z > 0 and (max_z / max(min_z, 1e-12)) > 2:
        levels = np.geomspace(min_z, max_z, 30)
    else:
        shift = max(0.0, 1e-4 - min_z)
        levels = np.geomspace(min_z + shift, max_z + shift, 30) - shift
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.85)
    ax.contour(X, Y, Z, levels=levels, colors="white", alpha=0.25, linewidths=0.5)
    fig.colorbar(cf, ax=ax).set_label("f(x, y)", fontsize=10)
    ray_max = max(max(all_alphas or [1.0]) * 1.05, 0.1)
    ray_a = np.linspace(0.0, ray_max, 100)
    ax.plot([x[0] + a * d[0] for a in ray_a], [x[1] + a * d[1] for a in ray_a], "--", color="orange", lw=2.0, label="Search Ray", zorder=4)
    ax.scatter([x[0]], [x[1]], color="#1f77b4", marker="o", s=80, edgecolors="white", lw=1.5, zorder=6, label="Base point x_k")
    _plot_line_search_trials(ax, x, d, trial_alphas, accepted_alpha)
    ax.set_xlim(xrange); ax.set_ylim(yrange); ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x", fontsize=11); ax.set_ylabel("y", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9); ax.grid(True, linestyle=":", alpha=0.4)
    fig.savefig(filepath, dpi=150, bbox_inches="tight"); plt.close(fig)
    return xrange, yrange

def _plot_line_search_trials(ax, x, d, trial_alphas, accepted_alpha):
    """Draws line search trial steps and accepted step marker along search ray."""
    rej_plotted, acc_plotted = False, False
    trial_lines = ["Trial Steps:"]
    for i, a in enumerate(trial_alphas, 1):
        pt = [x[0] + a * d[0], x[1] + a * d[1]]
        is_acc = (accepted_alpha is not None and abs(a - accepted_alpha) < 1e-9)
        if is_acc:
            ax.scatter([pt[0]], [pt[1]], color="gold", marker="*", s=160, edgecolors="black", lw=1.2, zorder=8, label="Accepted point" if not acc_plotted else "")
            acc_plotted = True
            ax.annotate(f"{i}", xy=(pt[0], pt[1]), xytext=(7, 7), textcoords="offset points",
                        fontsize=9.5, fontweight="bold", color="black", zorder=10,
                        bbox=dict(boxstyle="circle,pad=0.2", facecolor="#fff2cc", edgecolor="#d6b656", alpha=0.95))
            trial_lines.append(f" {i}: α = {a:.4g} (accepted)")
        else:
            ax.scatter([pt[0]], [pt[1]], color="#d62728", marker="x", s=80, lw=2.2, zorder=7, label="Rejected trial" if not rej_plotted else "")
            rej_plotted = True
            ax.annotate(f"{i}", xy=(pt[0], pt[1]), xytext=(7, 7), textcoords="offset points",
                        fontsize=9.5, fontweight="bold", color="darkred", zorder=10,
                        bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", edgecolor="#d62728", alpha=0.95))
            trial_lines.append(f" {i}: α = {a:.4g} (rejected)")
    if accepted_alpha is not None and not acc_plotted:
        pt = [x[0] + accepted_alpha * d[0], x[1] + accepted_alpha * d[1]]
        ax.scatter([pt[0]], [pt[1]], color="gold", marker="*", s=160, edgecolors="black", lw=1.2, zorder=8, label="Accepted point")
        ax.annotate("*", xy=(pt[0], pt[1]), xytext=(7, 7), textcoords="offset points",
                    fontsize=9.5, fontweight="bold", color="black", zorder=10,
                    bbox=dict(boxstyle="circle,pad=0.2", facecolor="#fff2cc", edgecolor="#d6b656", alpha=0.95))
    if trial_alphas:
        ax.text(0.03, 0.03, "\n".join(trial_lines), transform=ax.transAxes, fontsize=8.5,
                verticalalignment="bottom", bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92), zorder=10)

def plot_armijo_line_search(f, x0, d, g, c1, trials, accepted_alpha, filepath,
                            title="Alg 4: Armijo Condition Verification", alpha_max=1.0, resolution=200):
    """Plots 1D objective phi(alpha) against the Armijo decrease bound."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    f0 = f(x0)
    gd = sum(g[i] * d[i] for i in range(len(g)))
    alphas = np.linspace(0.0, alpha_max, resolution)
    phi_vals = np.array([f([x0[i] + a * d[i] for i in range(len(x0))]) for a in alphas])
    armijo_vals = np.array([f0 + c1 * a * gd for a in alphas])

    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    ax.plot(alphas, phi_vals, color="#1f77b4", lw=2.2, label=r"Objective $\phi(\alpha) = f(x + \alpha d)$")
    ax.plot(alphas, armijo_vals, color="#d62728", linestyle="--", lw=2.0,
            label=r"Armijo Line: $f(x) + c_1 \alpha \nabla f(x)^T d$")

    armijo_mask = phi_vals <= armijo_vals
    ax.fill_between(alphas, phi_vals, armijo_vals, where=armijo_mask,
                    color="#2ca02c", alpha=0.2, interpolate=True, label="Armijo condition holds (decrease satisfied)")

    for idx, a in enumerate(trials):
        pt = [x0[i] + a * d[i] for i in range(len(x0))]
        f_val = f(pt)
        if a == accepted_alpha:
            ax.scatter([a], [f_val], color="gold", marker="*", s=200, edgecolors="black", lw=1.2,
                       zorder=8, label=f"Accepted Step (α = {a:.4g})")
            ax.annotate(f"Accepted: α={a:.4g}", xy=(a, f_val), xytext=(a + 0.05 * alpha_max, f_val - 5),
                        arrowprops=dict(facecolor="#2ca02c", edgecolor="#2ca02c", arrowstyle="->", lw=1.5),
                        fontsize=9, fontweight="bold", color="#2ca02c", zorder=10)
        else:
            lbl = "Rejected Step" if idx == 0 else ""
            ax.scatter([a], [f_val], color="#d62728", marker="x", s=90, lw=2.2, zorder=7, label=lbl)
            ax.annotate(f"Rejected: α={a:.4g}", xy=(a, f_val), xytext=(a - 0.15 * alpha_max, f_val + 10),
                        arrowprops=dict(facecolor="#d62728", edgecolor="#d62728", arrowstyle="->", lw=1.5),
                        fontsize=9, fontweight="bold", color="#d62728", zorder=10)

    ax.set_xlim(0.0, alpha_max)
    ax.set_xlabel(r"Step size $\alpha$", fontsize=11)
    ax.set_ylabel(r"Function value $f(x + \alpha d)$", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ax

def render_panel(fig, ax, args, default_title):
    """Renders a single contour panel with trajectories and optimum marker."""
    X, Y, Z = _eval_grid(args["f"], args["xrange"], args["yrange"], args.get("resolution", 80))
    _plot_contour_base(fig, ax, X, Y, Z, log_levels=args.get("log_levels", True), levels_n=25)
    opt = args.get("optimum", None)
    if opt is not None:
        ax.scatter([opt[0]], [opt[1]], color="red", marker="x", s=90, lw=2.0, label="Optimum", zorder=7)
    _draw_paths(ax, args.get("paths", {}), args["xrange"], args["yrange"],
                converged=args.get("converged", None), f=args.get("f", None))
    if "start_marker" in args:
        sm = args["start_marker"]
        ax.scatter([sm[0]], [sm[1]], color="#d62728", marker="o", s=80, edgecolors="white", lw=1.5, zorder=8,
                   label=args.get("start_marker_label", "Start x0"))
    for ls in args.get("lines", []):
        if len(ls) == 4:
            ax.plot(ls[0], ls[1], ls[2], label=ls[3], lw=2.0)
    for q in args.get("quivers", []):
        ax.quiver(q[0], q[1], q[2], q[3], angles='xy', scale_units='xy', scale=1, color=q[4], width=0.012, zorder=9)
        ax.annotate(q[5], xy=(q[0] + q[2], q[1] + q[3]), xytext=(q[0] + q[2]*1.1, q[1] + q[3]*1.1), color=q[4], fontweight="bold", fontsize=9, zorder=10)
    for ann in args.get("annotations", []):
        ax.annotate(ann[0], xy=ann[1], xytext=ann[2],
                    arrowprops=dict(facecolor="#d62728", edgecolor="#d62728", arrowstyle="->", lw=1.5),
                    fontsize=9, fontweight="bold", color="#d62728",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff2cc", edgecolor="#d62728", lw=1.2), zorder=10)
    if "inset" in args:
        ins = args["inset"]
        axins = ax.inset_axes(ins.get("bounds", [0.12, 0.55, 0.35, 0.35]))
        tail_pts = ins.get("tail", [])
        if tail_pts:
            xs, ys = [p[0] for p in tail_pts], [p[1] for p in tail_pts]
            axins.plot(xs, ys, ".-", color="#d62728", lw=1.0, ms=3)
            axins.set_title(ins.get("title", "Zoomed Inset"), fontsize=8, fontweight="bold")
            axins.tick_params(labelsize=6)
            ax.indicate_inset_zoom(axins, edgecolor="gray", lw=0.8)
    ax.set_xlim(args["xrange"]); ax.set_ylim(args["yrange"]); ax.set_title(args.get("title", default_title), fontsize=11, fontweight="bold")
    ax.set_xlabel("x", fontsize=10); h, l = ax.get_legend_handles_labels()
    if h: ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)

def failure_pair(left_args, right_args, verdict_text, filepath):
    """Generates side-by-side Works vs Breaks comparison plot with top banner."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle("Comparison: Works vs Breaks", fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.92, f"Verdict: {verdict_text}", ha="center", va="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2cc", edgecolor="#d6b656", lw=1.5))
    render_panel(fig, ax1, left_args, "Works")
    render_panel(fig, ax2, right_args, "Breaks")
    plt.subplots_adjust(top=0.85)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return (ax1, ax2)

def render_constraints_panel(fig, ax, args, default_title="Constrained Problem"):
    """Renders a single panel with constraint boundaries and feasibility regions."""
    X, Y, Z = _eval_grid(args["f"], args["xrange"], args["yrange"], args.get("resolution", 80))
    _plot_contour_base(fig, ax, X, Y, Z, log_levels=False, levels_n=25, label="f(x, y)")
    _draw_constraint_curves(ax, X, Y, args.get("eq_constraints", []), args.get("ineq_constraints", []), args.get("resolution", 80))
    _draw_points_and_arrows(ax, args.get("points", {}), args.get("active_grad_arrows", []))
    ax.set_xlim(args["xrange"]); ax.set_ylim(args["yrange"]); ax.set_title(args.get("title", default_title), fontsize=11, fontweight="bold")
    ax.set_xlabel("x", fontsize=10); ax.set_ylabel("y", fontsize=10)
    h, l = ax.get_legend_handles_labels()
    if h: ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)

def contour_with_constraints_pair(left_args, right_args, verdict_text, filepath):
    """Generates side-by-side comparison for constrained optimization cases."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle("Comparison: Works vs Breaks", fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.92, f"Verdict: {verdict_text}", ha="center", va="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2cc", edgecolor="#d6b656", lw=1.5))
    render_constraints_panel(fig, ax1, left_args, "Works")
    render_constraints_panel(fig, ax2, right_args, "Breaks")
    plt.subplots_adjust(top=0.85)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return (ax1, ax2)

def render_line_search_panel(fig, ax, args, default_title="Line Search"):
    """Renders a single panel displaying line search trial evaluations."""
    f, x, d = args["f"], args["x"], args["d"]
    trial_alphas = args.get("trial_alphas", [])
    accepted_alpha = args.get("accepted_alpha", None)
    all_alphas = list(trial_alphas)
    if accepted_alpha is not None and accepted_alpha not in all_alphas:
        all_alphas.append(accepted_alpha)
    pts = [x] + [[x[0] + a * d[0], x[1] + a * d[1]] for a in all_alphas]
    pts_x, pts_y = [p[0] for p in pts], [p[1] for p in pts]
    min_x, max_x = min(pts_x), max(pts_x)
    min_y, max_y = min(pts_y), max(pts_y)
    span_x, span_y = max(max_x - min_x, 0.2), max(max_y - min_y, 0.2)
    pad_x, pad_y = 0.3 * span_x, 0.3 * span_y
    xrange = args.get("xrange") or (min_x - pad_x, max_x + pad_x)
    yrange = args.get("yrange") or (min_y - pad_y, max_y + pad_y)
    res = args.get("resolution", 80)
    X, Y, Z = _eval_grid(f, xrange, yrange, res)
    min_z, max_z = float(np.min(Z)), float(np.max(Z))
    if min_z > 0 and (max_z / max(min_z, 1e-12)) > 2:
        levels = np.geomspace(min_z, max_z, 25)
    else:
        shift = max(0.0, 1e-4 - min_z)
        levels = np.geomspace(min_z + shift, max_z + shift, 25) - shift
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.85)
    ax.contour(X, Y, Z, levels=levels, colors="white", alpha=0.25, linewidths=0.5)
    fig.colorbar(cf, ax=ax).set_label("f(x, y)", fontsize=9)
    ray_max = max(max(all_alphas or [1.0]) * 1.05, 0.1)
    ray_a = np.linspace(0.0, ray_max, 100)
    ax.plot([x[0] + a * d[0] for a in ray_a], [x[1] + a * d[1] for a in ray_a], "--", color="orange", lw=2.0, label="Search Ray", zorder=4)
    ax.scatter([x[0]], [x[1]], color="#1f77b4", marker="o", s=80, edgecolors="white", lw=1.5, zorder=6, label="Base point x")
    _plot_line_search_trials(ax, x, d, trial_alphas, accepted_alpha)
    ax.set_xlim(xrange); ax.set_ylim(yrange); ax.set_title(args.get("title", default_title), fontsize=11, fontweight="bold")
    ax.set_xlabel("x", fontsize=10); ax.set_ylabel("y", fontsize=10)
    h, l = ax.get_legend_handles_labels()
    if h: ax.legend(loc=args.get("legend_loc", "lower right"), fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)

def contour_with_line_search_pair(left_args, right_args, verdict_text, filepath):
    """Generates side-by-side comparison for line search trial cases."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle("Comparison: Works vs Breaks", fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.92, f"Verdict: {verdict_text}", ha="center", va="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2cc", edgecolor="#d6b656", lw=1.5))
    render_line_search_panel(fig, ax1, left_args, "Works")
    render_line_search_panel(fig, ax2, right_args, "Breaks")
    plt.subplots_adjust(top=0.85)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return (ax1, ax2)

def plot_2d_contour(f, x_range=(-2.0, 2.0), y_range=(-2.0, 2.0), title="Objective Contour Landscape", output_path=None, resolution=100, levels=40):
    """Generates standalone 2D contour plot for mathematical objective."""
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    X, Y, Z = _eval_grid(f, x_range, y_range, resolution)
    cs = ax.contour(X, Y, Z, levels=levels, cmap="viridis", alpha=0.8)
    ax.clabel(cs, inline=True, fontsize=8)
    ax.set_title(title, fontsize=13, fontweight="bold"); ax.set_xlabel("x_0", fontsize=11); ax.set_ylabel("x_1", fontsize=11); ax.grid(True, linestyle="--", alpha=0.5)
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return fig

def plot_optimization_paths(f, paths_dict, x_range=(-2.0, 2.0), y_range=(-2.0, 2.0), title="Optimization Trajectories", output_path=None, resolution=120, levels=35):
    """Overlays optimizer trajectories on a pre-existing matplotlib axis."""
    if output_path is not None:
        contour_with_paths(f, x_range, y_range, paths_dict, output_path, title, log_levels=False, resolution=resolution)
        return None
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    X, Y, Z = _eval_grid(f, x_range, y_range, resolution)
    ax.contour(X, Y, Z, levels=levels, cmap="plasma", alpha=0.6)
    _draw_paths(ax, paths_dict, x_range, y_range)
    ax.set_title(title, fontsize=13, fontweight="bold"); ax.set_xlim(x_range); ax.set_ylim(y_range); ax.legend(loc="best", framealpha=0.9)
    return fig
