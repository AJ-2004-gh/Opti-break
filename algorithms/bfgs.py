from core.vectors import (
    vec_sub, vec_add, scalar_mul, vec_neg, dot, norm_l2
)
from core.matrices import (
    identity, mat_mat, mat_vec, outer, mat_add, mat_sub, mat_scale
)
from core.convergence import Result, check_stopping
from algorithms.line_search import backtracking_line_search

def _bfgs_update_H(H, s, y, sy, n):
    """Applies the BFGS inverse Hessian rank-2 update."""
    rho = 1.0 / sy
    I = identity(n)
    V = mat_sub(I, mat_scale(rho, outer(s, y)))
    VT = mat_sub(I, mat_scale(rho, outer(y, s)))
    return mat_add(mat_mat(mat_mat(V, H), VT), mat_scale(rho, outer(s, s)))

def _bfgs_line_search(f, grad, x, d, g, line_search='armijo'):
    """Performs Armijo backtracking line search along BFGS direction."""
    ls = backtracking_line_search(f, x, d, g, alpha_init=1.0, rho=0.5, c1=1e-4)
    return (0.0, True) if (ls["failed"] or ls["alpha"] <= 0.0) else (ls["alpha"], False)

def bfgs(f, grad, x0, tol=1e-5, max_iters=100, line_search='armijo'):
    """BFGS quasi-Newton optimization with inverse Hessian estimation."""
    x = [float(v) for v in x0]
    n, H, path = len(x), identity(len(x)), [list(x)]
    f_hist, g_hist = [f(x)], [norm_l2(grad(x))]
    extra = {"sy_hist": [], "skipped": 0, "H": H}
    iters = 0
    while True:
        g = grad(x)
        gn = norm_l2(g)
        stop, conv, msg = check_stopping(gn, iters, tol, max_iters)
        if stop:
            extra["H"] = H
            return Result(x=x, fun=f(x), grad_norm=gn, iters=iters,
                          converged=conv, message=msg, path=path,
                          f_hist=f_hist, g_hist=g_hist, extra=extra)
        d = vec_neg(mat_vec(H, g))
        alpha, failed = _bfgs_line_search(f, grad, x, d, g, line_search)
        if failed:
            extra["H"], extra["failure"] = H, "line_search_failed"
            return Result(x=x, fun=f(x), grad_norm=gn, iters=iters,
                          converged=False, message="Line search failed",
                          path=path, f_hist=f_hist, g_hist=g_hist, extra=extra)
        s = scalar_mul(alpha, d)
        x_new = vec_add(x, s)
        y = vec_sub(grad(x_new), g)
        sy = dot(s, y)
        extra["sy_hist"].append(sy)
        if sy <= 1e-10:
            extra["skipped"] += 1
        else:
            H = _bfgs_update_H(H, s, y, sy, n)
        extra["H"] = H
        x, iters = x_new, iters + 1
        path.append(list(x)); f_hist.append(f(x)); g_hist.append(norm_l2(grad(x)))
