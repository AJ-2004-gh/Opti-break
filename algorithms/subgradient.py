import math
from core.vectors import vec_clone, vec_sub, vec_scale, vec_norm, norm_l2
from core.convergence import Result, check_stopping, is_diverged
from problems.lasso import lasso_loss, lasso_subgradient

def _make_subgrad_res(best_x, best_f, g_norm, iters, conv, msg, path, f_hist, g_hist, best_f_hist):
    """Constructs a Result object for subgradient methods with best-so-far histories."""
    res = Result(x=best_x, fun=best_f, grad_norm=g_norm, iters=iters,
                 converged=conv, message=msg, path=path,
                 f_hist=f_hist, g_hist=g_hist,
                 extra={"best_f_hist": best_f_hist, "best_x": best_x, "best_f": best_f})
    res.best_f_hist = best_f_hist
    res.best_f = best_f
    res.best_x = best_x
    return res

def _step_size(step_rule, k):
    """Computes constant or diminishing step size alpha_k for iteration k."""
    if isinstance(step_rule, (tuple, list)):
        rule, a = str(step_rule[0]).lower(), float(step_rule[1])
    else:
        rule, a = "constant", float(step_rule)
    return a / math.sqrt(k) if rule == "diminishing" else a

def subgradient_method(f, subgrad, x0, step_rule=('diminishing', 0.1), max_iters=1000, tol=1e-12):
    """Subgradient optimization tracking best-so-far iterates."""
    x = vec_clone(x0)
    best_x, best_f = vec_clone(x), f(x)
    path, f_hist, g_hist, best_f_hist = [vec_clone(x)], [best_f], [norm_l2(subgrad(x))], [best_f]
    if is_diverged(best_f, g_hist[0]):
        return _make_subgrad_res(best_x, best_f, g_hist[0], 0, False, "diverged", path, f_hist, g_hist, best_f_hist)

    for k in range(1, max_iters + 1):
        g = subgrad(x)
        gn = norm_l2(g)
        if gn < tol:
            return _make_subgrad_res(best_x, best_f, gn, k - 1, True, "Converged: subgradient norm below tolerance",
                                     path, f_hist, g_hist, best_f_hist)
        alpha_k = _step_size(step_rule, k)
        try:
            x = [x[i] - alpha_k * g[i] for i in range(len(x))]
            cf, cg = f(x), subgrad(x)
            cgn = norm_l2(cg)
        except OverflowError:
            path.append(vec_clone(x)); f_hist.append(float("inf")); best_f_hist.append(best_f); g_hist.append(float("inf"))
            return _make_subgrad_res(best_x, best_f, float("inf"), k, False, "diverged", path, f_hist, g_hist, best_f_hist)

        path.append(vec_clone(x)); f_hist.append(cf); g_hist.append(cgn)
        if cf < best_f:
            best_f, best_x = cf, vec_clone(x)
        best_f_hist.append(best_f)
        if is_diverged(cf, cgn):
            return _make_subgrad_res(best_x, best_f, cgn, k, False, "diverged", path, f_hist, g_hist, best_f_hist)

    return _make_subgrad_res(best_x, best_f, g_hist[-1], max_iters, (g_hist[-1] < tol),
                             "Maximum iterations reached", path, f_hist, g_hist, best_f_hist)

def subgradient_lasso(X, y, lam=0.1, x0=None, step_rule=('diminishing', 0.1), max_iters=1000, tol=1e-12):
    """Subgradient method for unconstrained Lasso regression loss."""
    d = len(X[0])
    if x0 is None:
        x0 = [0.0] * d
    f = lambda w: lasso_loss(w, X, y, lam)
    subgrad = lambda w: lasso_subgradient(w, X, y, lam)
    return subgradient_method(f, subgrad, x0, step_rule=step_rule, max_iters=max_iters, tol=tol)

def subgradient_fixed(f, subgrad, x0, alpha=0.01, tol=1e-5, max_iters=1000):
    """Subgradient method with constant step size."""
    return subgradient_method(f, subgrad, x0, step_rule=("constant", alpha), max_iters=max_iters, tol=tol)

def subgradient_diminishing(f, subgrad, x0, alpha0=0.1, power=0.5, tol=1e-5, max_iters=1000):
    """Subgradient method with diminishing step size."""
    x = vec_clone(x0)
    best_x, best_f = vec_clone(x), f(x)
    path, f_hist, g_hist, best_f_hist = [vec_clone(x)], [best_f], [vec_norm(subgrad(x))], [best_f]
    iters = 0
    while True:
        g = subgrad(x)
        gn = vec_norm(g)
        stop, conv, msg = check_stopping(gn, iters, tol, max_iters)
        if stop:
            r = Result(x=best_x, fun=best_f, grad_norm=gn, iters=iters, converged=conv, message=msg,
                       path=path, f_hist=f_hist, g_hist=g_hist,
                       extra={"alpha0": alpha0, "power": power, "best_f_hist": best_f_hist})
            r.best_f_hist = best_f_hist
            return r
        x = vec_sub(x, vec_scale(alpha0 / ((iters + 1) ** power), g))
        iters += 1
        cf = f(x)
        if cf < best_f:
            best_f, best_x = cf, vec_clone(x)
        best_f_hist.append(best_f); path.append(vec_clone(x)); f_hist.append(cf); g_hist.append(vec_norm(subgrad(x)))

def projected_subgradient(f, subgrad, proj, x0, alpha0=0.1, power=0.5, tol=1e-5, max_iters=1000):
    """Projected subgradient method for convex-constrained non-smooth problems."""
    x = proj(x0)
    best_x, best_f = vec_clone(x), f(x)
    path, f_hist, g_hist, best_f_hist = [vec_clone(x)], [best_f], [vec_norm(subgrad(x))], [best_f]
    iters = 0
    while True:
        g = subgrad(x)
        gn = vec_norm(g)
        stop, conv, msg = check_stopping(gn, iters, tol, max_iters)
        if stop:
            r = Result(x=best_x, fun=best_f, grad_norm=gn, iters=iters, converged=conv, message=msg,
                       path=path, f_hist=f_hist, g_hist=g_hist,
                       extra={"alpha0": alpha0, "power": power, "best_f_hist": best_f_hist})
            r.best_f_hist = best_f_hist
            return r
        x = proj(vec_sub(x, vec_scale(alpha0 / ((iters + 1) ** power), g)))
        iters += 1
        cf = f(x)
        if cf < best_f:
            best_f, best_x = cf, vec_clone(x)
        best_f_hist.append(best_f); path.append(vec_clone(x)); f_hist.append(cf); g_hist.append(vec_norm(subgrad(x)))
