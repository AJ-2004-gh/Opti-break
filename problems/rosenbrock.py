from core.matrices import mat_zeros

# Constants
MINIMIZER = [1.0, 1.0]
START_POINT = [-1.2, 1.0]
PLOT_BOX = [(-2.0, 2.0), (-1.0, 3.0)]

def rosenbrock_2d(pt):
    """
    f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
    """
    x = float(pt[0])
    y = float(pt[1])
    return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2

def rosenbrock_2d_grad(pt):
    """
    grad(x, y) = [-2(1 - x) - 400 x (y - x^2),  200 (y - x^2)]
    """
    x = float(pt[0])
    y = float(pt[1])
    df_dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x)
    df_dy = 200.0 * (y - x * x)
    return [df_dx, df_dy]

def rosenbrock_2d_hess(pt):
    """Computes exact 2x2 Hessian matrix of 2D Rosenbrock function."""
    x = float(pt[0])
    y = float(pt[1])
    h00 = 2.0 - 400.0 * y + 1200.0 * (x * x)
    h01 = -400.0 * x
    h10 = -400.0 * x
    h11 = 200.0
    return [[h00, h01], [h10, h11]]

# Shorthand aliases
f = rosenbrock_2d
grad = rosenbrock_2d_grad
hess = rosenbrock_2d_hess

def rosenbrock_nd(x):
    """
    N-dimensional generalized Rosenbrock function.
    """
    n = len(x)
    total = 0.0
    for i in range(n - 1):
        xi = float(x[i])
        x_next = float(x[i + 1])
        total += 100.0 * (x_next - xi * xi) ** 2 + (1.0 - xi) ** 2
    return total

def rosenbrock_nd_grad(x):
    """
    Gradient of N-dimensional Rosenbrock function.
    """
    n = len(x)
    g = [0.0] * n
    for i in range(n - 1):
        xi = float(x[i])
        x_next = float(x[i + 1])
        g[i] += -400.0 * xi * (x_next - xi * xi) - 2.0 * (1.0 - xi)
        g[i + 1] += 200.0 * (x_next - xi * xi)
    return g

def rosenbrock_nd_hess(x):
    """
    Hessian of N-dimensional Rosenbrock function.
    """
    n = len(x)
    H = mat_zeros(n, n)
    for i in range(n - 1):
        xi = float(x[i])
        x_next = float(x[i + 1])
        H[i][i] += -400.0 * (x_next - 3.0 * xi * xi) + 2.0
        H[i][i + 1] += -400.0 * xi
        H[i + 1][i] += -400.0 * xi
        H[i + 1][i + 1] += 200.0
    return H
