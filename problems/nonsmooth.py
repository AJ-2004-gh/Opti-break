import math

# Constants
MINIMIZER = [0.0, 0.0]
START_POINT = [2.0, 2.0]
PLOT_BOX = [(-3.0, 3.0), (-3.0, 3.0)]

def nonsmooth_f(pt):
    """
    f(x, y) = abs(x) + 2 * abs(y)
    """
    x = float(pt[0])
    y = float(pt[1])
    return abs(x) + 2.0 * abs(y)

def nonsmooth_subgradient(pt):
    """Computes subgradient of nonsmooth objective |x| + 2|y|."""
    x = float(pt[0])
    y = float(pt[1])

    if x > 0.0:
        gx = 1.0
    elif x < 0.0:
        gx = -1.0
    else:
        gx = 0.0  # At kink x=0, select 0 in [-1, 1]

    if y > 0.0:
        gy = 2.0
    elif y < 0.0:
        gy = -2.0
    else:
        gy = 0.0  # At kink y=0, select 0 in [-2, 2]

    return [gx, gy]

# Shorthand aliases
f = nonsmooth_f
subgradient = nonsmooth_subgradient

def l1_norm(x):
    """Evaluates L1 vector norm ||x||_1."""
    s = 0.0
    for val in x:
        s += abs(float(val))
    return s

def l1_subgradient(x):
    """Computes subgradient of L1 vector norm."""
    g = []
    for val in x:
        fval = float(val)
        if fval > 0.0:
            g.append(1.0)
        elif fval < 0.0:
            g.append(-1.0)
        else:
            g.append(0.0)
    return g

def coupled_nonsmooth_ridge(x):
    """Evaluates non-smooth coupled ridge objective."""
    return abs(x[0] - x[1]) + 2.0 * abs(x[0] + x[1])

def coupled_nonsmooth_subgradient(x):
    """Computes subgradient of non-smooth coupled ridge objective."""
    diff = x[0] - x[1]
    summ = x[0] + x[1]
    s1 = 1.0 if diff > 0.0 else (-1.0 if diff < 0.0 else 0.0)
    s2 = 1.0 if summ > 0.0 else (-1.0 if summ < 0.0 else 0.0)
    return [s1 + 2.0 * s2, -s1 + 2.0 * s2]
