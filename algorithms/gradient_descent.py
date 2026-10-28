import math
from core.vectors import vec_clone, vec_sub, scalar_mul, norm_l2, dot
from core.matrices import mat_vec
from core.convergence import Result, check_stopping
from algorithms.line_search import backtracking_line_search, exact_line_search_quadratic

def gradient_descent_fixed(f, grad, x0, alpha=0.01, tol=1e-5, max_iters=1000):
    """Fixed-step gradient descent with divergence and overflow detection."""
    x = vec_clone(x0)
    path = [vec_clone(x)]
    try:
        f0, g0 = f(x), grad(x)
        g0_norm = norm_l2(g0)
    except OverflowError:
        return Result(x=x, fun=float("inf"), grad_norm=float("inf"), iters=0,
                      converged=False, message="diverged", path=path,
                      f_hist=[float("inf")], g_hist=[float("inf")], extra={"alpha": alpha})

    f_hist, g_hist = [f0], [g0_norm]
    if math.isnan(f0) or math.isinf(f0) or math.isnan(g0_norm) or math.isinf(g0_norm) or abs(f0) > 1e12 or g0_norm > 1e12:
        return Result(x=x, fun=f0, grad_norm=g0_norm, iters=0,
                      converged=False, message="diverged", path=path,
                      f_hist=f_hist, g_hist=g_hist, extra={"alpha": alpha})

    iters = 0
    while True:
        stop, converged, msg = check_stopping(g_hist[-1], iters, tol, max_iters)
        if stop:
            return Result(x=x, fun=f_hist[-1], grad_norm=g_hist[-1], iters=iters,
                          converged=converged, message=msg, path=path,
                          f_hist=f_hist, g_hist=g_hist, extra={"alpha": alpha})
        try:
            x = vec_sub(x, scalar_mul(alpha, g0))
            iters += 1
            curr_f, curr_g = f(x), grad(x)
            curr_gnorm = norm_l2(curr_g)
        except OverflowError:
            path.append(vec_clone(x)); f_hist.append(float("inf")); g_hist.append(float("inf"))
            return Result(x=x, fun=float("inf"), grad_norm=float("inf"), iters=iters,
                          converged=False, message="diverged", path=path,
                          f_hist=f_hist, g_hist=g_hist, extra={"alpha": alpha})

        path.append(vec_clone(x)); f_hist.append(curr_f); g_hist.append(curr_gnorm); g0 = curr_g
        if math.isnan(curr_f) or math.isinf(curr_f) or math.isnan(curr_gnorm) or math.isinf(curr_gnorm) or abs(curr_f) > 1e12 or curr_gnorm > 1e12:
            return Result(x=x, fun=curr_f, grad_norm=curr_gnorm, iters=iters,
                          converged=False, message="diverged", path=path,
                          f_hist=f_hist, g_hist=g_hist, extra={"alpha": alpha})

def gradient_descent_exact(x0, Q, b=None, tol=1e-5, max_iters=1000):
    """Exact line search gradient descent for quadratic objectives."""
    if b is not None:
        f = lambda pt: 0.5 * dot(pt, mat_vec(Q, pt)) - dot(b, pt)
        grad = lambda pt: vec_sub(mat_vec(Q, pt), b)
    else:
        f = lambda pt: 0.5 * dot(pt, mat_vec(Q, pt))
        grad = lambda pt: mat_vec(Q, pt)

    x = vec_clone(x0)
    path = [vec_clone(x)]
    f_hist = [f(x)]
    g_hist = [norm_l2(grad(x))]

    iters = 0
    while True:
        g = grad(x)
        g_norm = norm_l2(g)
        stop, converged, msg = check_stopping(g_norm, iters, tol, max_iters)
        if stop:
            return Result(x=x, fun=f(x), grad_norm=g_norm, iters=iters,
                          converged=converged, message=msg, path=path,
                          f_hist=f_hist, g_hist=g_hist)

        alpha = exact_line_search_quadratic(g, g, Q)
        x = vec_sub(x, scalar_mul(alpha, g))
        iters += 1
        path.append(vec_clone(x))
        f_hist.append(f(x))
        g_hist.append(norm_l2(grad(x)))

def gradient_descent_armijo(f, grad, x0, alpha_init=1.0, rho=0.5, c1=1e-4, tol=1e-5, max_iters=1000, max_halvings=50):
    """Gradient descent with Armijo backtracking line search."""
    x = vec_clone(x0)
    path = [vec_clone(x)]
    f_hist = [f(x)]
    g_hist = [norm_l2(grad(x))]

    iters = 0
    while True:
        g = grad(x)
        g_norm = norm_l2(g)
        stop, converged, msg = check_stopping(g_norm, iters, tol, max_iters)
        if stop:
            return Result(x=x, fun=f(x), grad_norm=g_norm, iters=iters,
                          converged=converged, message=msg, path=path,
                          f_hist=f_hist, g_hist=g_hist, extra={"alpha_init": alpha_init})

        d = scalar_mul(-1.0, g)
        ls_res = backtracking_line_search(f, x, d, g, alpha_init=alpha_init, rho=rho, c1=c1, max_halvings=max_halvings)
        if ls_res["failed"]:
            return Result(x=x, fun=f(x), grad_norm=g_norm, iters=iters,
                          converged=False, message="Line search failed", path=path,
                          f_hist=f_hist, g_hist=g_hist)

        alpha = ls_res["alpha"]
        x = vec_sub(x, scalar_mul(alpha, g))
        iters += 1
        path.append(vec_clone(x))
        f_hist.append(f(x))
        g_hist.append(norm_l2(grad(x)))

gradient_descent_backtracking = gradient_descent_armijo
