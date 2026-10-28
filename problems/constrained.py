import math
from core.vectors import (
    vec_clone, scalar_mul, dot, norm_l2, vec_add, vec_sub, zeros
)

# -------------------------------------------------------------
# Specific Benchmark Problems Requested
# -------------------------------------------------------------

def equality_qp():
    """Defines equality-constrained quadratic problem: min x^2+y^2 s.t. x+y=1."""
    Q = [[2.0, 0.0], [0.0, 2.0]]
    c = [0.0, 0.0]
    A = [[1.0, 1.0]]
    b = [1.0]
    return Q, c, A, b

def redundant_constraints_qp():
    """Defines degenerate QP with redundant constraints x+y=1 and 2x+2y=2."""
    Q = [[2.0, 0.0], [0.0, 2.0]]
    c = [0.0, 0.0]
    A = [[1.0, 1.0], [2.0, 2.0]]
    b = [1.0, 2.0]
    return Q, c, A, b

def indefinite_qp():
    """Defines indefinite saddle QP: min x^2-y^2 s.t. x=1."""
    Q = [[2.0, 0.0], [0.0, -2.0]]
    c = [0.0, 0.0]
    A = [[1.0, 0.0]]
    b = [1.0]
    return Q, c, A, b

def inequality_problem():
    """Defines inequality-constrained problem: min (x-2)^2+(y-1)^2 s.t. x^2+y^2<=1."""
    def f(pt):
        """Evaluates quadratic distance objective at pt."""
        x, y = float(pt[0]), float(pt[1])
        return (x - 2.0) ** 2 + (y - 1.0) ** 2

    def grad_f(pt):
        """Evaluates objective gradient at pt."""
        x, y = float(pt[0]), float(pt[1])
        return [2.0 * (x - 2.0), 2.0 * (y - 1.0)]

    def g_ineq(pt):
        """Evaluates inequality constraint at pt."""
        x, y = float(pt[0]), float(pt[1])
        return x * x + y * y - 1.0

    def grad_g_ineq(pt):
        """Evaluates inequality gradient at pt."""
        x, y = float(pt[0]), float(pt[1])
        return [2.0 * x, 2.0 * y]

    return f, grad_f, g_ineq, grad_g_ineq

def degenerate_licq_problem():
    """Defines degenerate LICQ problem with two tangent circular boundaries at (0,0)."""
    def f(pt):
        """Evaluates linear coordinate objective at pt."""
        return float(pt[1])

    def grad_f(pt):
        """Evaluates linear coordinate gradient at pt."""
        return [0.0, 1.0]

    def g1(pt):
        """Evaluates first tangent circle constraint at pt."""
        x, y = float(pt[0]), float(pt[1])
        return (x - 1.0) ** 2 + y * y - 1.0

    def grad_g1(pt):
        """Evaluates first tangent circle gradient at pt."""
        x, y = float(pt[0]), float(pt[1])
        return [2.0 * (x - 1.0), 2.0 * y]

    def g2(pt):
        """Evaluates second tangent circle constraint at pt."""
        x, y = float(pt[0]), float(pt[1])
        return (x + 1.0) ** 2 + y * y - 1.0

    def grad_g2(pt):
        """Evaluates second tangent circle gradient at pt."""
        x, y = float(pt[0]), float(pt[1])
        return [2.0 * (x + 1.0), 2.0 * y]

    return f, grad_f, [g1, g2], [grad_g1, grad_g2]

# -------------------------------------------------------------
# Constraint Projections and Penalty Utilities
# -------------------------------------------------------------

def project_box(x, lower, upper):
    """
    Euclidean projection onto box [lower, upper].
    """
    p = []
    for i in range(len(x)):
        xi = float(x[i])
        li = float(lower[i])
        ui = float(upper[i])
        if xi < li:
            p.append(li)
        elif xi > ui:
            p.append(ui)
        else:
            p.append(xi)
    return p

def project_ball(x, radius):
    """
    Euclidean projection onto L2 ball of given radius.
    """
    nrm = norm_l2(x)
    if nrm <= radius or nrm == 0.0:
        return [float(v) for v in x]
    return scalar_mul(float(radius) / nrm, x)

def project_hyperplane(x, a, b):
    """
    Orthogonal projection onto hyperplane a^T x = b.
    """
    a_sq = dot(a, a)
    if a_sq == 0.0:
        return [float(v) for v in x]
    dist = (dot(a, x) - float(b)) / a_sq
    return vec_sub(x, scalar_mul(dist, a))

def quadratic_penalty_objective(f, g_ineq_list, rho):
    """Builds quadratic penalty objective function f(x) + (rho/2)*sum max(0, g_i(x))^2."""
    def penalty_f(x):
        """Evaluates quadratic penalty objective at x."""
        val = f(x)
        pen = 0.0
        for g_func in g_ineq_list:
            viol = g_func(x)
            if viol > 0.0:
                pen += viol * viol
        return val + 0.5 * rho * pen
    return penalty_f

def quadratic_penalty_gradient(grad_f, g_ineq_list, grad_g_ineq_list, rho):
    """Builds gradient of quadratic penalty objective."""
    def penalty_grad(x):
        """Evaluates gradient of quadratic penalty objective at x."""
        g = grad_f(x)
        total_g = vec_clone(g)
        for i in range(len(g_ineq_list)):
            viol = g_ineq_list[i](x)
            if viol > 0.0:
                grad_gi = grad_g_ineq_list[i](x)
                scaled_grad = scalar_mul(rho * viol, grad_gi)
                total_g = vec_add(total_g, scaled_grad)
        return total_g
    return penalty_grad

def log_barrier_objective(f, g_ineq_list, mu):
    """Builds logarithmic barrier objective function f(x) - mu*sum log(-g_i(x))."""
    def barrier_f(x):
        """Evaluates logarithmic barrier objective at x."""
        val = f(x)
        pen = 0.0
        for g_func in g_ineq_list:
            gx = g_func(x)
            if gx >= 0.0:
                return float("inf")
            pen += math.log(-gx)
        return val - mu * pen
    return barrier_f

def log_barrier_gradient(grad_f, g_ineq_list, grad_g_ineq_list, mu):
    """Builds gradient of logarithmic barrier objective."""
    def barrier_grad(x):
        """Evaluates gradient of logarithmic barrier objective at x."""
        g = grad_f(x)
        total_g = vec_clone(g)
        for i in range(len(g_ineq_list)):
            gx = g_ineq_list[i](x)
            if gx >= 0.0:
                return scalar_mul(float("nan"), total_g)
            grad_gi = grad_g_ineq_list[i](x)
            scaled = scalar_mul(-mu / gx, grad_gi)
            total_g = vec_add(total_g, scaled)
        return total_g
    return barrier_grad

def max_constraint_violation(x, g_ineq_list):
    """Computes maximum constraint violation max(0, g_i(x))."""
    max_viol = 0.0
    for g_func in g_ineq_list:
        v = g_func(x)
        if v > max_viol:
            max_viol = v
    return max_viol
