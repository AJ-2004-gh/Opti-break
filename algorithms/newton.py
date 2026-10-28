from core.vectors import vec_add, vec_sub, scalar_mul, vec_neg, norm_l2, dot
from core.matrices import mat_diag_add, mat_vec
from core.linear_systems import solve
from core.convergence import Result, check_stopping
from algorithms.line_search import backtracking_line_search

def newton_method(f, grad, hess, x0, tol=1e-5, max_iters=100):
    """Pure Newton optimization solving H p = -g with full unit steps."""
    x = [float(v) for v in x0]
    path, f_hist, g_hist = [list(x)], [f(x)], [norm_l2(grad(x))]
    iters = 0
    while True:
        g = grad(x)
        g_norm = norm_l2(g)
        stop, converged, msg = check_stopping(g_norm, iters, tol, max_iters)
        if stop:
            return Result(x=x, fun=f(x), grad_norm=g_norm, iters=iters,
                          converged=converged, message=msg, path=path,
                          f_hist=f_hist, g_hist=g_hist)
        try:
            p = solve(hess(x), vec_neg(g))
        except ValueError as err:
            return Result(x=x, fun=f(x), grad_norm=g_norm, iters=iters,
                          converged=False, message=f"Linear solve failed: {str(err)}",
                          path=path, f_hist=f_hist, g_hist=g_hist,
                          extra={"failure": "singular_hessian"})
        x = vec_add(x, p)
        iters += 1
        path.append(list(x))
        f_hist.append(f(x))
        g_hist.append(norm_l2(grad(x)))

def damped_newton_method(f, grad, hess, x0, tol=1e-5, max_iters=100, alpha_init=1.0, rho=0.5, c1=1e-4, max_ls_iters=50, reg=0.0):
    """Damped Newton optimization using Armijo backtracking along Newton descent direction."""
    x = [float(v) for v in x0]
    path, f_hist, g_hist = [list(x)], [f(x)], [norm_l2(grad(x))]
    extra = {"non_descent": False, "reg": reg}

    def _ret(cvg, msg):
        """Constructs damped Newton Result with convergence status and diagnostics."""
        return Result(x=x, fun=f_hist[-1], grad_norm=g_hist[-1], iters=iters,
                      converged=cvg, message=msg, path=path, f_hist=f_hist, g_hist=g_hist, extra=extra)

    iters = 0
    while True:
        g, g_norm = grad(x), g_hist[-1]
        stop, converged, msg = check_stopping(g_norm, iters, tol, max_iters)
        if stop:
            return _ret(converged, msg)

        H = mat_diag_add(hess(x), reg) if reg > 0.0 else hess(x)
        try:
            p = solve(H, vec_neg(g))
        except ValueError as err:
            extra["failure"] = "singular_hessian"
            return _ret(False, f"Linear solve failed: {str(err)}")

        if dot(g, p) >= 0.0:
            extra["non_descent"] = True
            return _ret(False, "Non-descent direction: g . p >= 0")

        ls_res = backtracking_line_search(f, x, p, g, alpha_init=alpha_init, rho=rho, c1=c1, max_halvings=max_ls_iters)
        if ls_res["failed"] or ls_res["alpha"] <= 0.0:
            extra["line_search_failed"] = True
            return _ret(False, "Line search failed to satisfy Armijo condition")

        x = vec_add(x, scalar_mul(ls_res["alpha"], p))
        iters += 1
        path.append(list(x)), f_hist.append(f(x)), g_hist.append(norm_l2(grad(x)))

def verify_newton_one_step(Q, c, x0):
    """Verifies one-step Newton exactness on a quadratic by measuring distance to minimizer."""
    p = solve(Q, vec_neg(vec_sub(mat_vec(Q, x0), c)))
    x1 = vec_add(x0, p)
    x_star = solve(Q, c)
    return norm_l2(vec_sub(x1, x_star))

newton_pure = newton_method
newton_damped = damped_newton_method
