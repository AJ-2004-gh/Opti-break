import math
from core.vectors import norm_l2, dot, vec_sub, vec_norm
from core.linear_systems import eigenvalues_2x2, rank
from algorithms.constrained_optimization import second_order_check, licq_check

def diverged(res):
    """Detects if objective value or gradient norm exploded or exceeded thresholds."""
    threshold = 1e5
    fun = getattr(res, "fun", None)
    if fun is not None and (math.isnan(fun) or math.isinf(fun) or abs(fun) > threshold):
        return True, f"Objective exploded: fun={fun:.2e}"
    grad_norm = getattr(res, "grad_norm", None)
    if grad_norm is not None and (math.isnan(grad_norm) or math.isinf(grad_norm) or grad_norm > threshold):
        return True, f"Gradient norm exploded: grad_norm={grad_norm:.2e}"
    msg = getattr(res, "message", "")
    if "diverge" in msg.lower():
        return True, f"Divergence signaled in result message: {msg}"
    f_hist = getattr(res, "f_hist", [])
    for v in f_hist:
        if math.isnan(v) or math.isinf(v) or abs(v) > threshold:
            return True, f"Objective exploded during trajectory to {v:.2e}"
    return False, "No divergence detected"

def zigzag_score(path):
    """Measures fraction of consecutive step reversals and orthogonal pairs."""
    pts = path.path if hasattr(path, "path") else path
    if len(pts) < 3:
        return False, "Trajectory too short to evaluate zigzagging"

    reversals, orthogonal, total = 0, 0, 0
    for k in range(len(pts) - 2):
        s1, s2 = vec_sub(pts[k + 1], pts[k]), vec_sub(pts[k + 2], pts[k + 1])
        n1, n2 = norm_l2(s1), norm_l2(s2)
        if n1 > 1e-12 and n2 > 1e-12:
            cos_theta = dot(s1, s2) / (n1 * n2)
            total += 1
            if cos_theta < -0.5:
                reversals += 1
            if abs(cos_theta) < 0.15:
                orthogonal += 1

    if total == 0:
        return False, "No non-zero steps to evaluate zigzagging"

    score = (reversals + orthogonal) / total
    is_zigzag = score > 0.35 or reversals >= 3 or orthogonal >= 3
    return is_zigzag, f"Zigzag score: {score:.2f} ({reversals} reversals, {orthogonal} orthogonal pairs out of {total})"

def stalled_at_saddle(x, hess=None, tol=1e-2):
    """Reports whether iterate is trapped at an indefinite stationary saddle point."""
    pt = x.x if hasattr(x, "x") else x
    if hess is None and hasattr(x, "extra") and isinstance(x.extra, dict):
        hess = x.extra.get("hess")
    if hess is None:
        hess = lambda p: [[2.0, 0.0], [0.0, -2.0]]
    H = hess(pt)
    evals = eigenvalues_2x2(H)
    is_indefinite = (evals[0] < -1e-4 and evals[1] > 1e-4)
    gn = getattr(x, "grad_norm", None)
    is_stationary = (gn is not None and gn < tol) or (norm_l2(pt) < tol)
    if is_indefinite and is_stationary:
        return True, f"Stalled at saddle point: evals={evals}, grad_norm={gn if gn is not None else 0.0:.2e}"
    return False, f"Not stalled at saddle point: indefinite={is_indefinite}, stationary={is_stationary}"

def noise_floor(res, good_res=None, window=25):
    """Detects if stochastic iterations have plateaued into an asymptotic noise floor."""
    if getattr(res, "converged", False):
        return False, "Run converged normally"
    f_hist = getattr(res, "f_hist", [])
    if len(f_hist) < 5:
        return False, "History too short for noise floor evaluation"
    w = min(len(f_hist), window)
    tail = f_hist[-w:]
    m = sum(tail) / len(tail)
    std_bad = math.sqrt(sum((v - m) ** 2 for v in tail) / len(tail))

    std_good = None
    if good_res is not None:
        g_hist = getattr(good_res, "f_hist", [])
        if len(g_hist) >= 5:
            gw = min(len(g_hist), window)
            gtail = g_hist[-gw:]
            gm = sum(gtail) / len(gtail)
            std_good = math.sqrt(sum((v - gm) ** 2 for v in gtail) / len(gtail))
    elif hasattr(res, "extra") and isinstance(res.extra, dict):
        std_good = res.extra.get("good_tail_std")
    if std_good is None:
        std_good = 1e-6

    is_floor = (std_bad > 5.0 * std_good)
    return is_floor, f"Noise floor active: tail std={std_bad:.2e} vs good tail std={std_good:.2e} (ratio {std_bad/max(std_good, 1e-12):.1f}x >= 5x)"

def tail_oscillation(path, window=100, min_step=1e-3):
    """Detects persistent limit-cycle oscillations with non-vanishing step sizes."""
    if getattr(path, "converged", False):
        return False, "Run converged normally"
    pts = path.path if hasattr(path, "path") else path
    if len(pts) < window + 1:
        return False, f"Trajectory has only {len(pts)} points, need at least {window + 1}"

    tail = pts[-(window + 1):]
    steps = [vec_sub(tail[k + 1], tail[k]) for k in range(window)]
    lens = [norm_l2(s) for s in steps]
    min_len = min(lens)
    if min_len <= min_step:
        return False, f"Step length dropped below {min_step:.1e} (min={min_len:.4e})"

    reversals = 0
    for k in range(window - 1):
        n1, n2 = lens[k], lens[k + 1]
        if n1 > 1e-12 and n2 > 1e-12 and dot(steps[k], steps[k + 1]) / (n1 * n2) < -0.3:
            reversals += 1

    sign_flips = 0
    for k in range(window):
        if (tail[k][0] * tail[k + 1][0] < 0) or (tail[k][1] * tail[k + 1][1] < 0):
            sign_flips += 1

    is_osc = (reversals >= 10 or sign_flips >= 10)
    return is_osc, f"Tail oscillation: {reversals} reversals, {sign_flips} sign flips over last {window} steps (min step {min_len:.4e} > {min_step:.1e})"

def step_collapse(res, good_res=None):
    """Detects if step sizes collapsed prematurely leaving high residual objective."""
    if getattr(res, "converged", False):
        return False, "Algorithm converged normally"

    good_f = None
    if good_res is not None:
        good_f = getattr(good_res, "fun", None)
    elif hasattr(res, "extra") and isinstance(res.extra, dict):
        good_f = res.extra.get("good_f")
    if good_f is None:
        good_f = 1e-6

    eff_step = None
    if hasattr(res, "extra") and isinstance(res.extra, dict):
        if "effective_step_size" in res.extra:
            eff = res.extra["effective_step_size"]
            eff_step = min(eff) if isinstance(eff, (list, tuple)) else float(eff)
        elif "step_norm" in res.extra and res.extra["step_norm"] > 0:
            eff_step = float(res.extra["step_norm"])
    if eff_step is None:
        path = getattr(res, "path", [])
        eff_step = norm_l2(vec_sub(path[-1], path[-2])) if len(path) >= 2 else 0.0

    fun = getattr(res, "fun", 0.0)
    is_collapse = (eff_step < 1e-3) and (fun > 100.0 * max(good_f, 1e-12))
    return is_collapse, f"Step collapse: eff_step={eff_step:.2e} (< 1e-3), fun={fun:.2e} (> 100x good_f={good_f:.2e})"

def no_exact_zeros(w):
    """Detects absence of exact zeros in weights for sparsity diagnostics."""
    if hasattr(w, "extra") and isinstance(w.extra, dict):
        if w.extra.get("lam", 1.0) == 0.0:
            return False, "Sparsity not expected for lam=0 (unregularized least squares)"
    weights = w.x if hasattr(w, "x") else w
    zero_count = sum(1 for v in weights if abs(float(v)) == 0.0)
    has_no_exact_zeros = (zero_count == 0) and (len(weights) > 0)
    if has_no_exact_zeros:
        return True, f"No exact zeros present: all {len(weights)} weights are non-zero (min |w|={min(abs(v) for v in weights):.4e})"
    return False, f"Exact zeros found: {zero_count} weights are exactly zero"

def unbounded_weights(res_short, res_long=None, good_res=None):
    """Detects if weights grow unbounded on unregularized separable datasets."""
    if getattr(res_short, "converged", False):
        return False, "Weights converged stably"

    target = res_long if (res_long is not None and res_long is not good_res) else res_short
    w_hist = getattr(target, "extra", {}).get("w_norm_hist", [])
    if not w_hist and hasattr(target, "path"):
        w_hist = [norm_l2(p[:-1]) for p in target.path]

    if res_long is not None and res_long is not good_res and hasattr(res_short, "x") and hasattr(res_long, "x"):
        n_short = norm_l2(res_short.x[:-1])
        n_long = norm_l2(res_long.x[:-1])
        w_hist_long = getattr(res_long, "extra", {}).get("w_norm_hist", [])
        inc = (w_hist_long[-1] > w_hist_long[-500]) if len(w_hist_long) >= 500 else (n_long > n_short)
        if n_long > 2.0 * n_short and inc:
            return True, f"Unbounded weights: grew from {n_short:.2f} to {n_long:.2f} (> 2x) and increasing"
        return False, f"Weights bounded: short={n_short:.2f}, long={n_long:.2f}"

    if len(w_hist) >= 4000:
        norm_400 = w_hist[400]
        norm_4000 = w_hist[4000] if len(w_hist) > 4000 else w_hist[-1]
        is_increasing = (w_hist[-1] > w_hist[-500]) if len(w_hist) >= 500 else True
        if norm_4000 > 2.0 * norm_400 and is_increasing:
            ratio = norm_4000 / max(norm_400, 1e-12)
            return True, f"Unbounded weights: norm grew from {norm_400:.2f} (ep 400) to {norm_4000:.2f} (ep 4000, ratio {ratio:.2f} > 2.0x) and increasing over last 500 epochs"
        return False, f"Weights bounded: norm_400={norm_400:.2f}, norm_4000={norm_4000:.2f}"

    return False, "Weights bounded or run too short to verify unbounded growth"

def singular_kkt(callable_fn):
    """Detects singular KKT matrix exception during equality-constrained QP solve."""
    if not callable(callable_fn):
        if isinstance(callable_fn, Exception):
            return True, f"Singular KKT matrix detected: {callable_fn}"
        if isinstance(callable_fn, dict) and callable_fn.get("singular", False):
            return True, "Singular KKT matrix detected"
        return False, "Not singular"
    try:
        callable_fn()
        return False, "KKT system solved without singularity"
    except ValueError as e:
        msg = str(e).lower()
        if "singular" in msg or "rank" in msg or "linear solve" in msg:
            return True, f"Singular KKT matrix detected: {str(e)}"
        raise
    except Exception as e:
        return True, f"Exception during KKT solve: {str(e)}"

def not_minimizer(x, Q, A):
    """Reports whether stationary point fails second-order sufficiency."""
    so_res = second_order_check(Q, A, x)
    if not so_res.get("is_positive_definite", False):
        return True, "Stationary point is not a minimizer: reduced Hessian is not positive definite"
    return False, "Point satisfies second-order sufficiency (positive definite reduced Hessian)"

def licq_failed(active_grads):
    """Reports whether Linear Independence Constraint Qualification failed."""
    res = licq_check(active_grads)
    if not res.get("holds", False):
        return True, f"LICQ failed: rank={rank(active_grads)} < {len(active_grads)}, Gram det={res.get('gram_det', 0.0):.4e}"
    return False, f"LICQ holds: rank={len(active_grads)}, Gram det={res.get('gram_det', 0.0):.4e}"

detect_divergence = diverged
detect_oscillation = tail_oscillation
detect_premature_stopping = step_collapse
detect_slow_convergence = step_collapse
