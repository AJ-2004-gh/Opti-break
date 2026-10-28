from core.vectors import vec_clone
from core.matrices import mat_zeros

def numerical_gradient(f, x, h=1e-6):
    """Computes numerical gradient vector via central finite differences."""
    n = len(x)
    g = []
    x_plus = [float(v) for v in x]
    x_minus = [float(v) for v in x]

    for i in range(n):
        orig = x_plus[i]
        x_plus[i] = orig + h
        x_minus[i] = orig - h
        f_plus = f(x_plus)
        f_minus = f(x_minus)
        g.append((f_plus - f_minus) / (2.0 * h))
        x_plus[i] = orig
        x_minus[i] = orig

    return g

def numerical_hessian(f, x, h=1e-4):
    """Computes numerical Hessian matrix via central finite differences."""
    n = len(x)
    H = mat_zeros(n, n)
    f0 = f(x)
    x_work = [float(v) for v in x]

    for i in range(n):
        for j in range(i, n):
            if i == j:
                orig_i = x_work[i]
                x_work[i] = orig_i + h
                f_plus = f(x_work)
                x_work[i] = orig_i - h
                f_minus = f(x_work)
                x_work[i] = orig_i
                val = (f_plus - 2.0 * f0 + f_minus) / (h * h)
                H[i][i] = val
            else:
                orig_i = x_work[i]
                orig_j = x_work[j]

                x_work[i] = orig_i + h
                x_work[j] = orig_j + h
                f_pp = f(x_work)

                x_work[j] = orig_j - h
                f_pm = f(x_work)

                x_work[i] = orig_i - h
                x_work[j] = orig_j + h
                f_mp = f(x_work)

                x_work[j] = orig_j - h
                f_mm = f(x_work)

                x_work[i] = orig_i
                x_work[j] = orig_j

                val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h)
                H[i][j] = val
                H[j][i] = val

    return H

def check_gradient(f, analytic_grad, x, h=1e-6, tol=1e-4):
    """Verifies analytic gradient against numerical finite-difference approximation."""
    num_g = numerical_gradient(f, x, h=h)
    ana_g = analytic_grad(x)
    max_abs = 0.0
    max_rel = 0.0

    for i in range(len(x)):
        diff = abs(ana_g[i] - num_g[i])
        denom = max(1.0, abs(ana_g[i]), abs(num_g[i]))
        rel = diff / denom
        if diff > max_abs:
            max_abs = diff
        if rel > max_rel:
            max_rel = rel

    is_ok = max_rel <= tol or max_abs <= tol
    return max_abs, max_rel, is_ok

def check_hessian(f, analytic_hess, x, h=1e-4, tol=1e-3):
    """Verifies analytic Hessian against numerical finite-difference approximation."""
    num_H = numerical_hessian(f, x, h=h)
    ana_H = analytic_hess(x)
    n = len(x)
    max_abs = 0.0
    max_rel = 0.0

    for i in range(n):
        for j in range(n):
            diff = abs(ana_H[i][j] - num_H[i][j])
            denom = max(1.0, abs(ana_H[i][j]), abs(num_H[i][j]))
            rel = diff / denom
            if diff > max_abs:
                max_abs = diff
            if rel > max_rel:
                max_rel = rel

    is_ok = max_rel <= tol or max_abs <= tol
    return max_abs, max_rel, is_ok
