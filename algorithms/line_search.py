from core.vectors import dot, scalar_mul, vec_add
from core.matrices import mat_vec

def exact_line_search_quadratic(g, d, Q):
    """Calculates exact step size along direction d for quadratic objective with Hessian Q."""
    num = dot(g, d)
    denom = dot(d, mat_vec(Q, d))
    return (num / denom) if denom > 0.0 else 0.0

def armijo_condition(f, x, d, g, alpha, c1=1e-4):
    """Checks the Armijo sufficient decrease condition f(x + alpha*d) <= f(x) + c1*alpha*(g.d)."""
    fx = f(x)
    step = scalar_mul(alpha, d)
    x_trial = vec_add(x, step)
    try:
        f_trial = f(x_trial)
    except OverflowError:
        return False
    gd = dot(g, d)
    return bool(f_trial <= fx + c1 * alpha * gd)

def backtracking_line_search(f, x, d, g, alpha_init=1.0, rho=0.5, c1=1e-4, max_halvings=50):
    """Performs backtracking Armijo line search returning dict with alpha, trials, and failed."""
    gd = dot(g, d)
    if gd >= 0.0:
        return {"alpha": 0.0, "trials": [], "failed": True}

    alpha = float(alpha_init)
    trials = [alpha]
    halvings = 0

    while True:
        if armijo_condition(f, x, d, g, alpha, c1):
            return {"alpha": alpha, "trials": trials, "failed": False}
        if halvings >= max_halvings:
            return {"alpha": 0.0, "trials": trials, "failed": True}
        alpha *= rho
        trials.append(alpha)
        halvings += 1

def backtracking_armijo(f, grad_f, x, d, alpha0=1.0, c=1e-4, rho=0.5, max_ls_iters=50):
    """Backtracking Armijo line search returning step size scalar."""
    g = grad_f(x)
    res = backtracking_line_search(f, x, d, g, alpha_init=alpha0, rho=rho, c1=c, max_halvings=max_ls_iters)
    return alpha0 if res["failed"] else res["alpha"]
