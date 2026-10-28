from core.matrices import diag, mat_vec, mat_clone
from core.vectors import dot, vec_clone, vec_sub
from core.linear_systems import solve

# Default ill-conditioned / anisotropic quadratic
Q_DEFAULT = [[1.0, 0.0], [0.0, 100.0]]
MINIMIZER = [0.0, 0.0]
START_POINT = [2.0, 2.0]
PLOT_BOX = [(-3.0, 3.0), (-3.0, 3.0)]

def quadratic_f(x, Q=Q_DEFAULT, c=None):
    """
    f(x) = 0.5 * x^T Q x - c^T x (or 0.5 * x^T Q x when c is None)
    """
    Qx = mat_vec(Q, x)
    val = 0.5 * dot(x, Qx)
    if c is not None:
        val -= dot(c, x)
    return val

def quadratic_grad(x, Q=Q_DEFAULT, c=None):
    """
    grad(x) = Q x - c (or Q x when c is None)
    """
    g = mat_vec(Q, x)
    if c is not None:
        for i in range(len(g)):
            g[i] -= float(c[i])
    return g

def quadratic_hess(x, Q=Q_DEFAULT, c=None):
    """
    hess(x) = Q
    """
    return mat_clone(Q)

def make_quadratic(Q, c=None):
    """Constructs quadratic objective, gradient, and Hessian callables from Q and b."""
    Q_mat = mat_clone(Q)
    c_vec = vec_clone(c) if c is not None else None

    def f(x):
        """Evaluates quadratic objective at x."""
        return quadratic_f(x, Q=Q_mat, c=c_vec)

    def grad(x):
        """Evaluates quadratic gradient at x."""
        return quadratic_grad(x, Q=Q_mat, c=c_vec)

    def hess(x):
        """Evaluates quadratic Hessian at x."""
        return quadratic_hess(x, Q=Q_mat, c=c_vec)

    return f, grad, hess

def make_ill_conditioned_quadratic(dim=2, condition_number=100.0):
    """
    Constructs an ill-conditioned quadratic with condition number kappa.
    """
    d = []
    if dim == 1:
        d = [float(condition_number)]
    else:
        for i in range(dim):
            lambda_i = 1.0 + (condition_number - 1.0) * (i / (dim - 1))
            d.append(lambda_i)
    Q = diag(d)
    return make_quadratic(Q)

def quadratic_optimum(A, b):
    """
    Analytical minimizer of 0.5 x^T A x - b^T x.
    """
    return solve(A, b)
