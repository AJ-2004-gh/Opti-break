import os
import csv
import math
from atlas.cases import CASES, ALL_CASES
from visualization.contour import (
    failure_pair, contour_with_line_search_pair, contour_with_constraints_pair
)
from visualization.model_plots import (
    loss_contour_regression_pair, loss_contour_logistic_pair, loss_contour_lasso_pair
)
from problems.rosenbrock import rosenbrock_2d
from problems.nonconvex import saddle_point_quadratic

def _extract_metric(res, attr, default=0.0):
    """Safely extracts a scalar metric from a Result object or dictionary."""
    if hasattr(res, attr):
        val = getattr(res, attr)
        return val if val is not None else default
    if isinstance(res, dict):
        return res.get(attr, default)
    return default

def _extract_path(r):
    """Safely extracts iterate trajectory points from a Result or dictionary."""
    if hasattr(r, "path"):
        return r.path
    if isinstance(r, dict) and "path" in r:
        return r["path"]
    if hasattr(r, "x"):
        return [r.x]
    if isinstance(r, dict) and "x" in r:
        return [r["x"]]
    return []

def _render_line_search_case(cid, good_res, bad_res, verdict, filepath):
    """Renders comparison contour for line search failure cases."""
    def _g(obj, attr):
        return obj.get(attr) if isinstance(obj, dict) else getattr(obj, attr, None)
    if cid == 3:
        l_args = {
            "f": _g(good_res, "f"), "x": _g(good_res, "x"), "d": _g(good_res, "d"),
            "trial_alphas": _g(good_res, "trials"), "accepted_alpha": _g(good_res, "alpha"),
            "title": "Descent Direction Succeeded",
            "yrange": (-35.0, 5.0), "xrange": (-0.5, 1.5)
        }
        r_args = {
            "f": _g(bad_res, "f"), "x": _g(bad_res, "x"), "d": _g(bad_res, "d"),
            "trial_alphas": _g(bad_res, "trials"), "accepted_alpha": None,
            "title": "Non-Descent Direction Failed"
        }
    else:
        l_args = {
            "f": _g(good_res, "f"), "x": _g(good_res, "x"), "d": _g(good_res, "d"),
            "trial_alphas": [1.0, 0.5, 0.1, 0.005], "accepted_alpha": 0.005,
            "title": "Armijo Satisfied (alpha=0.005)",
            "yrange": (-35.0, 5.0), "xrange": (-0.5, 1.5)
        }
        r_args = {
            "f": _g(bad_res, "f"), "x": _g(bad_res, "x"), "d": _g(bad_res, "d"),
            "trial_alphas": [1.0, 0.5], "accepted_alpha": None,
            "title": "Armijo Violated (oversized alpha)"
        }
    return contour_with_line_search_pair(l_args, r_args, verdict, filepath)

def _render_regression_case(cid, good_res, bad_res, verdict, filepath):
    """Renders comparison contour for linear regression failure cases."""
    titles = {
        8: ("Standardized Features (Scale 1: Converging)\n[Note: Normalized scale]",
            "Unstandardized Features (Scale 1000: Stalled)\n[Note: Raw feature scale]"),
        9: ("Batch GD: Deterministic Convergence", "SGD: Non-Vanishing Noise Floor"),
        10: ("Batch GD: Deterministic Convergence", "Mini-batch: Variance Plateau")
    }
    gt, bt = titles[cid]
    l_data = {
        "X": good_res.extra["X"], "y": good_res.extra["y"],
        "paths": {"Good Trajectory": good_res.path},
        "title": gt
    }
    r_data = {
        "X": bad_res.extra["X"], "y": bad_res.extra["y"],
        "paths": {"Bad Trajectory": bad_res.path},
        "title": bt
    }
    if cid == 8:
        l_data["f_hist"] = getattr(good_res, "f_hist", [])
        r_data["f_hist"] = getattr(bad_res, "f_hist", [])
    elif cid in (9, 10):
        w_opt, b_opt = good_res.path[-1][0], good_res.path[-1][1]
        bad_tail = bad_res.extra.get("update_path", bad_res.path)[-300:]
        w_tail = [p[0] for p in bad_tail]
        b_tail = [p[1] for p in bad_tail]
        spread_w = max(abs(w - w_opt) for w in w_tail)
        spread_b = max(abs(b - b_opt) for b in b_tail)
        w_box = (w_opt - 3.0 * max(spread_w, 0.02), w_opt + 3.0 * max(spread_w, 0.02))
        b_box = (b_opt - 3.0 * max(spread_b, 0.02), b_opt + 3.0 * max(spread_b, 0.02))
        l_data["w_range"] = w_box; l_data["b_range"] = b_box
        r_data["w_range"] = w_box; r_data["b_range"] = b_box
        r_data["paths"] = {"Noise Ball (last 300 updates)": bad_tail}
        l_data["paths"] = {"Batch GD (Settled Point)": good_res.path}
    return loss_contour_regression_pair(l_data, r_data, verdict, filepath)

def _render_logistic_case(cid, good_res, bad_res, verdict, filepath):
    """Renders comparison contour for logistic regression failure cases."""
    all_w1 = [p[0] for p in bad_res.path + good_res.path]
    all_w2 = [p[1] for p in bad_res.path + good_res.path]
    pad1 = max(1.0, 0.25 * (max(all_w1) - min(all_w1)))
    pad2 = max(1.0, 0.25 * (max(all_w2) - min(all_w2)))
    w1_box = (min(all_w1) - pad1, max(all_w1) + pad1)
    w2_box = (min(all_w2) - pad2, max(all_w2) + pad2)
    l_data = {
        "X": good_res.extra["X"], "y": good_res.extra["y"],
        "paths": {"Overlapping (Bounded)": [p[:2] for p in good_res.path]},
        "title": "Overlapping Blobs: Bounded Weights",
        "w1_range": w1_box, "w2_range": w2_box,
        "w_norm_hist": good_res.extra.get("w_norm_hist", [])
    }
    r_data = {
        "X": bad_res.extra["X"], "y": bad_res.extra["y"],
        "paths": {"Separable (Unbounded)": [p[:2] for p in bad_res.path]},
        "title": "Separable Blobs: Unbounded Weights",
        "w1_range": w1_box, "w2_range": w2_box,
        "w_norm_hist": bad_res.extra.get("w_norm_hist", [])
    }
    return loss_contour_logistic_pair(l_data, r_data, verdict, filepath)

def _render_lasso_case(cid, good_res, bad_res, verdict, filepath):
    """Renders comparison contour for Lasso regression failure cases."""
    l_data = {
        "X": good_res.extra["X"], "y": good_res.extra["y"], "lam": good_res.extra.get("lam", 0.0),
        "paths": {"Least Squares (lam=0)": good_res.path},
        "title": "Least Squares: Continuous Optimum",
        "w1_range": (-2.0, 2.0), "w2_range": (-2.5, 1.0)
    }
    r_data = {
        "X": bad_res.extra["X"], "y": bad_res.extra["y"], "lam": bad_res.extra.get("lam", 1.0),
        "paths": {"Subgradient Lasso (lam=1)": bad_res.path},
        "title": "Subgradient Lasso: No Exact Zeros",
        "w1_range": (-2.0, 2.0), "w2_range": (-2.5, 1.0)
    }
    return loss_contour_lasso_pair(l_data, r_data, verdict, filepath)

def _render_constraints_case(cid, good_res, bad_res, verdict, filepath):
    """Renders comparison contour for constrained optimization failure cases."""
    if cid == 21:
        f = lambda pt: pt[0]**2 + pt[1]**2
        l_args = {
            "f": f, "xrange": (-2, 2), "yrange": (-2, 2),
            "eq_constraints": [lambda pt: pt[0] + pt[1] - 1.0],
            "points": {"Solution [0.5, 0.5]": [0.5, 0.5]},
            "title": "Independent Constraint: Solved"
        }
        r_args = {
            "f": f, "xrange": (-2, 2), "yrange": (-2, 2),
            "eq_constraints": [lambda pt: pt[0] + pt[1] - 1.0, lambda pt: 2.0*pt[0] + 2.0*pt[1] - 2.0],
            "points": {},
            "title": "Redundant Constraints: Singular KKT Matrix"
        }
    elif cid == 22:
        l_args = {
            "f": lambda pt: pt[0]**2 + pt[1]**2, "xrange": (-2, 2), "yrange": (-2, 2),
            "eq_constraints": [lambda pt: pt[0] - 1.0],
            "points": {"Constrained Min": [1.0, 0.0]},
            "title": "Positive Reduced Hessian: Minimizer"
        }
        r_args = {
            "f": lambda pt: pt[0]**2 - pt[1]**2, "xrange": (-2, 2), "yrange": (-2, 2),
            "eq_constraints": [lambda pt: pt[0] - 1.0],
            "points": {"Constrained Max": [1.0, 0.0]},
            "title": "Negative Reduced Hessian: Maximizer"
        }
    else:
        l_args = {
            "f": lambda pt: pt[0]**2 + pt[1]**2, "xrange": (-2.5, 2.5), "yrange": (-2, 2),
            "ineq_constraints": [
                lambda pt: (pt[0] - 1.0)**2 + pt[1]**2 - 1.0,
                lambda pt: pt[0]**2 + (pt[1] - 1.0)**2 - 1.0
            ],
            "points": {"Feasible Vertex (0,0)": [0.0, 0.0]},
            "active_grad_arrows": [([0.0, 0.0], [-2.0, 0.0], "grad g1"), ([0.0, 0.0], [0.0, -2.0], "grad g2", (0.45, 0.0))],
            "title": "LICQ Holds: Linearly Independent Gradients (rank=2)"
        }
        r_args = {
            "f": lambda pt: pt[1], "xrange": (-3.0, 3.5), "yrange": (-2, 2),
            "ineq_constraints": [
                lambda pt: (pt[0] - 1.0)**2 + pt[1]**2 - 1.0,
                lambda pt: (pt[0] + 1.0)**2 + pt[1]**2 - 1.0
            ],
            "points": {"Feasible Point (0,0)": [0.0, 0.0]},
            "active_grad_arrows": [
                ([0.0, 0.0], [-2.0, 0.0], "grad g1 (-2,0)"),
                ([0.0, 0.0], [2.0, 0.0], "grad g2 (2,0)")
            ],
            "title": "rank = 1 < 2: LICQ fails"
        }
    return contour_with_constraints_pair(l_args, r_args, verdict, filepath)

def _render_contour_case(cid, good_res, bad_res, verdict, filepath):
    p_good, p_bad = _extract_path(good_res), _extract_path(bad_res)
    if cid == 1:
        f = lambda pt: 0.5 * (10.0 * pt[0]**2 + 100.0 * pt[1]**2)
        l_args = {"f": f, "xrange": (-3, 3), "yrange": (-3, 3), "paths": {"Good (alpha=0.015)": p_good}, "title": "Stably Convergent"}
        r_args = {"f": f, "xrange": (-3, 3), "yrange": (-3, 3), "paths": {"Bad (alpha=0.025)": p_bad}, "title": "Geometric Divergence"}
    elif cid == 2:
        l_args = {"f": lambda pt: 0.5 * (pt[0]**2 + pt[1]**2), "xrange": (-3, 3), "yrange": (-3, 3), "paths": {"Exact GD (kappa=1)": p_good}, "title": "1-Step Convergence"}
        r_args = {"f": lambda pt: 0.5 * (pt[0]**2 + 100.0 * pt[1]**2), "xrange": (-12, 12), "yrange": (-3, 3), "paths": {"Exact GD (kappa=100)": p_bad}, "title": "Orthogonal Zig-Zagging"}
    elif cid == 5:
        l_args = {"f": lambda pt: 0.5 * (pt[0]**2 + 10.0 * pt[1]**2), "xrange": (-3, 3), "yrange": (-3, 3), "paths": {"Quadratic Descent": p_good}, "title": "Clean Convergence"}
        r_args = {
            "f": rosenbrock_2d, "xrange": (-2, 2), "yrange": (-1, 3),
            "paths": {"Rosenbrock Valley": p_bad}, "title": "Step Size Collapse", "optimum": [1, 1],
            "inset": {"bounds": [0.12, 0.55, 0.35, 0.35], "tail": p_bad[-25:], "title": "Stalled Tail (Tiny Steps)"}
        }
    elif cid == 6:
        f = lambda pt: 0.5 * (pt[0]**2 + 100.0 * pt[1]**2)
        l_args = {"f": f, "xrange": (-3, 3), "yrange": (-3, 3), "paths": {"Tuned Momentum (beta=0.8)": p_good}, "title": "Monotone Descent"}
        r_args = {"f": f, "xrange": (-3, 3), "yrange": (-3, 3), "paths": {"Heavy Ball (beta=0.99)": p_bad}, "title": "Resonant Energy Overshoot"}
    elif cid == 7:
        from problems.nonconvex import double_well
        l_args = {"f": double_well, "xrange": (-2, 2), "yrange": (-1.5, 1.5), "paths": {"Off-Axis Double Well": p_good}, "title": "Double Well: Escape to Min [1, 0]", "optimum": [1.0, 0.0]}
        r_args = {"f": saddle_point_quadratic, "xrange": (-2, 2), "yrange": (-2, 2), "paths": {"Invariant Subspace": p_bad}, "title": "Saddle: Trapped at (0, 0)"}
    elif cid == 12:
        l_args = {"f": lambda pt: 0.5 * (pt[0]**2 + 10.0 * pt[1]**2), "xrange": (-3, 3), "yrange": (-3, 3), "paths": {"Quadratic Progress": p_good}, "title": "Normal Adaptation"}
        r_args = {"f": rosenbrock_2d, "xrange": (-2, 2), "yrange": (-1, 3), "paths": {"Rosenbrock Stalling": p_bad}, "title": "Premature Step Freezing", "optimum": [1, 1]}
    elif cid == 14:
        l_args = {"f": lambda pt: 0.5 * (pt[0]**2 + 10.0 * pt[1]**2), "xrange": (-3, 3), "yrange": (-3, 3), "paths": {"Quadratic Progress": p_good}, "title": "Normal Adaptation"}
        r_args = {"f": rosenbrock_2d, "xrange": (-2, 2), "yrange": (-1, 3), "paths": {"Limit Cycle Oscillation": p_bad}, "title": "Limit Cycle (lr=0.5)", "optimum": [1, 1]}
    elif cid == 13:
        l_args = {"f": lambda pt: 0.5 * (pt[0]**2 + 10.0 * pt[1]**2), "xrange": (-3, 3), "yrange": (-3, 3), "paths": {"Tuned RMSProp": p_good}, "title": "Smooth Descent"}
        r_args = {"f": lambda pt: 0.5 * (pt[0]**2 + 100.0 * pt[1]**2), "xrange": (-3, 3), "yrange": (-3, 3), "paths": {"Aggressive lr=0.5": p_bad}, "title": "Ridge Chattering Limit Cycle"}
    elif cid == 15:
        f = lambda pt: abs(pt[0]) + 2.0 * abs(pt[1])
        l_args = {"f": f, "xrange": (-4, 4), "yrange": (-4, 4), "paths": {"Diminishing Step": p_good}, "title": "Convergence to Kink [0,0]"}
        r_args = {"f": f, "xrange": (-4, 4), "yrange": (-4, 4), "paths": {"Constant Step": p_bad}, "title": "Perpetual Limit Cycle"}
    elif cid == 17:
        l_args = {"f": lambda pt: pt[0]**2 + pt[1]**2, "xrange": (-2, 2), "yrange": (-2, 2), "paths": {"Convex Minimum": p_good}, "title": "Convergence to Minimizer"}
        r_args = {"f": saddle_point_quadratic, "xrange": (-2, 2), "yrange": (-2, 2), "paths": {"Saddle Region": p_bad}, "title": "Saddle Point / Negative Curvature"}
    elif cid == 18:
        l_args = {"f": lambda pt: pt[0]**2 + pt[1]**2, "xrange": (-2, 2), "yrange": (-2, 2), "paths": {"Convex Minimum": p_good}, "title": "Convergence to Minimizer"}
        r_args = {
            "f": saddle_point_quadratic, "xrange": (-2, 2), "yrange": (-2, 2),
            "paths": {"Saddle Region": p_bad}, "title": "Saddle Point / Negative Curvature",
            "start_marker": [0.5, 1.0], "start_marker_label": "Start x0 [0.5, 1.0]",
            "annotations": [
                ("non-descent direction: step rejected", (0.5, 1.0), (-1.3, 1.4))
            ]
        }
    elif cid == 20:
        from problems.nonconvex import double_well
        l_args = {"f": lambda pt: pt[0]**2 + pt[1]**2, "xrange": (-2, 2), "yrange": (-2, 2), "paths": {"Convex Minimum": p_good}, "title": "Convex BFGS Convergence"}
        r_args = {"f": double_well, "xrange": (-2, 2), "yrange": (-2, 2), "paths": {"Double Well (s^T y <= 0)": p_bad}, "title": "Negative Curvature (s^T y <= 0)"}
    else:
        l_args = {"f": lambda pt: 2.0 * pt[0]**2 + 5.0 * pt[1]**2, "xrange": (-2, 2), "yrange": (-2, 2), "paths": {"Quadratic Step": p_good}, "title": "Exact 1-Step Solution"}
        r_args = {"f": rosenbrock_2d, "xrange": (-2, 2), "yrange": (-1, 3), "paths": {"Non-Quadratic Step": p_bad}, "title": "Truncation Residual > 0.5", "optimum": [1, 1]}
    return failure_pair(l_args, r_args, verdict, filepath)

def _render_case_plot(case, good_res, bad_res, detected, filepath):
    """
    Renders side-by-side failure pair contour for each of the 23 cases.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    cid = case["id"]
    verdict = f"Case {cid:02d}: {case['failure_title']} - {'FLAGGED' if detected else 'MISSED'}"
    if cid in (3, 4):
        return _render_line_search_case(cid, good_res, bad_res, verdict, filepath)
    elif cid in (8, 9, 10):
        return _render_regression_case(cid, good_res, bad_res, verdict, filepath)
    elif cid == 11:
        return _render_logistic_case(cid, good_res, bad_res, verdict, filepath)
    elif cid == 16:
        return _render_lasso_case(cid, good_res, bad_res, verdict, filepath)
    elif cid in (21, 22, 23):
        return _render_constraints_case(cid, good_res, bad_res, verdict, filepath)
    else:
        return _render_contour_case(cid, good_res, bad_res, verdict, filepath)

def _write_case_history_csv(case, good_res, bad_res, filepath):
    """
    Saves convergence history of good and bad runs to results/histories/<case>.csv.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    f_good = getattr(good_res, "f_hist", [getattr(good_res, "fun", 0.0)])
    g_good = getattr(good_res, "g_hist", [getattr(good_res, "grad_norm", 0.0)])
    f_bad = getattr(bad_res, "f_hist", [getattr(bad_res, "fun", 0.0)])
    g_bad = getattr(bad_res, "g_hist", [getattr(bad_res, "grad_norm", 0.0)])

    max_len = max(len(f_good), len(f_bad), 1)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "good_f", "good_grad_norm", "bad_f", "bad_grad_norm"])
        for i in range(max_len):
            fg = f_good[i] if i < len(f_good) else (f_good[-1] if f_good else "")
            gg = g_good[i] if i < len(g_good) else (g_good[-1] if g_good else "")
            fb = f_bad[i] if i < len(f_bad) else (f_bad[-1] if f_bad else "")
            gb = g_bad[i] if i < len(g_bad) else (g_bad[-1] if g_bad else "")
            writer.writerow([i, fg, gg, fb, gb])

def _eval_detector(detector, bad_res, good_res):
    """Evaluates case detector function handling optional good_res baseline."""
    try:
        return detector(bad_res, good_res=good_res)
    except TypeError:
        return detector(bad_res)

def run_case(case, plot_dir="results/plots", history_dir="results/histories"):
    """Executes a single case dict, evaluates detector, saves plot and history CSV."""
    if isinstance(case, (int, str)):
        case = ALL_CASES[int(case)]

    good_res, bad_res = case["good_run"](), case["bad_run"]()
    detected, detail = _eval_detector(case["detector"], bad_res, good_res)

    plot_path = os.path.join(plot_dir, f"atlas_{case['id']}.png")
    csv_path = os.path.join(history_dir, f"case_{case['id']}.csv")
    axes = _render_case_plot(case, good_res, bad_res, detected, plot_path)
    _write_case_history_csv(case, good_res, bad_res, csv_path)

    fail_it = _extract_metric(bad_res, "iters", 0) or max(len(_extract_path(bad_res)) - 1, 1)
    good_it = _extract_metric(good_res, "iters", 0) or max(len(_extract_path(good_res)) - 1, 1)

    return {
        "id": case["id"], "algorithm_number": case["algorithm_number"],
        "algorithm_name": case["algorithm_name"], "failure_title": case["failure_title"],
        "good_desc": case.get("good_desc", ""),
        "bad_desc": case.get("bad_desc", ""),
        "why": case["why"], "good_result": good_res, "bad_result": bad_res,
        "detector_outcome": (detected, detail), "detected": detected, "detail": detail,
        "plot_path": plot_path, "plot_paths": [plot_path], "csv_path": csv_path,
        "axes": axes,
        "fail_iters": fail_it, "fail_f": _extract_metric(bad_res, "fun", 0.0),
        "fail_grad_norm": _extract_metric(bad_res, "grad_norm", 0.0),
        "good_iters": good_it, "good_f": _extract_metric(good_res, "fun", 0.0),
        "good_grad_norm": _extract_metric(good_res, "grad_norm", 0.0), "fix_ok": True
    }

def write_report(results, filepath="results/reports/failure_atlas.txt"):
    """Writes textual and markdown Failure Atlas reports with summary table."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    lines = [
        "=" * 80,
        "OPTIBREAK FAILURE ATLAS REPORT",
        "A Systematic Empirical and Theoretical Study of Optimization Breakdown Modes",
        "=" * 80,
        ""
    ]

    for r in results:
        det_str = "YES" if r["detected"] else "NO"
        lines.append("-" * 80)
        lines.append(f"Case {r['id']:02d}: Algorithm {r['algorithm_number']:02d} - {r['algorithm_name']}")
        lines.append(f"Failure Title : {r['failure_title']}")
        lines.append(f"Theory (Why)  : {r['why']}")
        lines.append(f"Good run: {r.get('good_desc', '')}")
        lines.append(f"Bad run: {r.get('bad_desc', '')}")
        lines.append(f"Detected      : {det_str} ({r['detail']})")
        lines.append(f"Bad Run       : iters={r['fail_iters']}, final_f={r['fail_f']:.4e}, grad_norm={r['fail_grad_norm']:.4e}")
        lines.append(f"Good Run      : iters={r['good_iters']}, final_f={r['good_f']:.4e}, grad_norm={r['good_grad_norm']:.4e}")
        lines.append(f"Plot Artifact : {r['plot_path']}")
        lines.append("-" * 80)
        lines.append("")

    # Summary table
    lines.append("=" * 120)
    lines.append("SUMMARY TABLE")
    lines.append("=" * 120)
    lines.append(f"{'Case':<4} | {'Detected':<8} | {'Good run':<50} | {'Bad run':<50}")
    lines.append("-" * 120)
    for r in results:
        det_str = "YES" if r["detected"] else "NO"
        lines.append(f"{r['id']:<4} | {det_str:<8} | {r.get('good_desc', ''):<50} | {r.get('bad_desc', ''):<50}")
    lines.append("-" * 120)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath

def run_all(verbose=True):
    """Executes run_case for all 23 cases and compiles report and CSV summaries."""
    results = []
    if verbose:
        print("Executing all 23 Failure Atlas cases...")
        print("-" * 80)

    for case in CASES:
        res_dict = run_case(case)
        results.append(res_dict)
        if verbose:
            status = "FLAGGED" if res_dict["detected"] else "MISSED"
            print(f"[{res_dict['id']:02d}/23] {res_dict['algorithm_name']:<30} | {status} | {res_dict['detail'][:45]}")

    report_path = write_report(results, "results/reports/failure_atlas.txt")
    if verbose:
        print("-" * 80)
        print(f"Report written to: {report_path}")
        print("All 23 case plots and history CSVs saved successfully.")

    return results

# Backward compatibility alias
run_atlas_case = run_case
run_all_atlas_cases = run_all
generate_atlas_report = lambda results, output_path="results/reports/failure_atlas.txt": write_report(results, output_path)
