from core.vectors import vec_clone, zeros, scalar_mul, vec_add, vec_sub, norm_l2
from core.convergence import Result, check_stopping, is_diverged

def momentum(f, grad, x0, alpha=0.01, beta=0.9, tol=1e-5, max_iters=1000):
    """Polyak heavy-ball momentum optimization."""
    x, v = vec_clone(x0), zeros(len(x0))
    path = [vec_clone(x)]
    extra = {"alpha": alpha, "beta": beta}

    try:
        f0, g0 = f(x), grad(x)
        g0_norm = norm_l2(g0)
    except OverflowError:
        return Result(x=x, fun=float("inf"), grad_norm=float("inf"), iters=0,
                      converged=False, message="diverged", path=path,
                      f_hist=[float("inf")], g_hist=[float("inf")], extra=extra)

    f_hist, g_hist = [f0], [g0_norm]
    if is_diverged(f0, g0_norm):
        return Result(x=x, fun=f0, grad_norm=g0_norm, iters=0,
                      converged=False, message="diverged", path=path,
                      f_hist=f_hist, g_hist=g_hist, extra=extra)

    iters, g = 0, g0
    while True:
        stop, converged, msg = check_stopping(g_hist[-1], iters, tol, max_iters)
        if stop:
            return Result(x=x, fun=f_hist[-1], grad_norm=g_hist[-1], iters=iters,
                          converged=converged, message=msg, path=path,
                          f_hist=f_hist, g_hist=g_hist, extra=extra)

        try:
            v = vec_add(scalar_mul(beta, v), scalar_mul(alpha, g))
            x = vec_sub(x, v)
            iters += 1
            curr_f, curr_g = f(x), grad(x)
            curr_gnorm = norm_l2(curr_g)
        except OverflowError:
            path.append(vec_clone(x))
            f_hist.append(float("inf")), g_hist.append(float("inf"))
            return Result(x=x, fun=float("inf"), grad_norm=float("inf"), iters=iters,
                          converged=False, message="diverged", path=path,
                          f_hist=f_hist, g_hist=g_hist, extra=extra)

        path.append(vec_clone(x))
        f_hist.append(curr_f), g_hist.append(curr_gnorm)
        g = curr_g
        if is_diverged(curr_f, curr_gnorm):
            return Result(x=x, fun=curr_f, grad_norm=curr_gnorm, iters=iters,
                          converged=False, message="diverged", path=path,
                          f_hist=f_hist, g_hist=g_hist, extra=extra)

polyak_momentum = momentum
