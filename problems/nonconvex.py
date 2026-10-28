import math

# -------------------------------------------------------------
# 1. Hyperbolic Saddle Point: f(x, y) = x^2 - y^2
# -------------------------------------------------------------
SADDLE_START = [0.5, 0.5]
SADDLE_BOX = [(-2.0, 2.0), (-2.0, 2.0)]
SADDLE_STATIONARY = [0.0, 0.0]

def saddle_f(pt):
    """
    f(x, y) = x^2 - y^2
    """
    x = float(pt[0])
    y = float(pt[1])
    return x * x - y * y

def saddle_grad(pt):
    """
    grad(x, y) = [2*x, -2*y]
    """
    x = float(pt[0])
    y = float(pt[1])
    return [2.0 * x, -2.0 * y]

def saddle_hess(pt):
    """
    hess(x, y) = [[2, 0], [0, -2]]
    """
    return [[2.0, 0.0], [0.0, -2.0]]

# Backward compatibility aliases
saddle_point_quadratic = saddle_f
saddle_point_quadratic_grad = saddle_grad
saddle_point_quadratic_hess = saddle_hess

# -------------------------------------------------------------
# 2. Double Well Potential: f(x, y) = x^4 - 2*x^2 + y^2
# -------------------------------------------------------------
DOUBLE_WELL_MINIMIZERS = [[-1.0, 0.0], [1.0, 0.0]]
DOUBLE_WELL_START = [0.1, 1.0]
DOUBLE_WELL_BOX = [(-2.0, 2.0), (-2.0, 2.0)]

def double_well_f(pt):
    """
    f(x, y) = x^4 - 2*x^2 + y^2
    """
    x = float(pt[0])
    y = float(pt[1])
    return (x ** 4) - 2.0 * (x ** 2) + (y ** 2)

def double_well_grad(pt):
    """
    grad(x, y) = [4*x^3 - 4*x, 2*y]
    """
    x = float(pt[0])
    y = float(pt[1])
    return [4.0 * (x ** 3) - 4.0 * x, 2.0 * y]

def double_well_hess(pt):
    """
    hess(x, y) = [[12*x^2 - 4, 0], [0, 2]]
    """
    x = float(pt[0])
    return [[12.0 * (x ** 2) - 4.0, 0.0], [0.0, 2.0]]

# Compatibility aliases
double_well = double_well_f

# -------------------------------------------------------------
# 3. Multimodal Functions (Rastrigin & Ackley)
# -------------------------------------------------------------
def rastrigin(x, A=10.0):
    """Evaluates 2D Rastrigin multimodal benchmark function."""
    n = len(x)
    total = A * n
    for xi in x:
        fxi = float(xi)
        total += fxi * fxi - A * math.cos(2.0 * math.pi * fxi)
    return total

def rastrigin_grad(x, A=10.0):
    """Computes gradient of 2D Rastrigin function."""
    g = []
    for xi in x:
        fxi = float(xi)
        gi = 2.0 * fxi + 2.0 * math.pi * A * math.sin(2.0 * math.pi * fxi)
        g.append(gi)
    return g
