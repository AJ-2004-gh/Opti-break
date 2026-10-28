import math
from core.vectors import norm_l2, dot, zeros, vec_add, scalar_mul, vec_sub
from core.matrices import mat_vec
from core.linear_systems import solve, rank, determinant, is_positive_definite

def lagrange_solver(Q, c, A, b):
    """Solves an equality-constrained quadratic program via the block KKT system."""
    n = len(Q)
    c_vec = [float(v) for v in (c if c is not None else zeros(n))]
    A_mat = [[float(v) for v in A]] if len(A) > 0 and not isinstance(A[0], list) else [[float(v) for v in row] for row in A]
    b_vec = [float(b)] if isinstance(b, (int, float)) else [float(v) for v in b]
    m = len(A_mat)

    total_dim = n + m
    KKT = [[0.0] * total_dim for _ in range(total_dim)]
    for i in range(n):
        for j in range(n):
            KKT[i][j] = float(Q[i][j])
        for k in range(m):
            KKT[i][n + k] = float(A_mat[k][i])
            KKT[n + k][i] = float(A_mat[k][i])

    rhs = [-float(val) for val in c_vec] + [float(val) for val in b_vec]
    try:
        sol = solve(KKT, rhs)
    except ValueError as err:
        raise ValueError(f"Singular KKT matrix: redundant constraints or rank-deficient system ({str(err)})") from err

    x = [float(sol[i]) for i in range(n)]
    lambdas = [float(sol[n + k]) for k in range(m)]
    residual = norm_l2(vec_sub(mat_vec(KKT, sol), rhs))
    return x, lambdas, residual

def second_order_check(Q, A, x=None):
    """Checks positive definiteness of the reduced Hessian on the null space of A."""
    n = len(Q)
    A_mat = [[float(v) for v in A]] if len(A) > 0 and not isinstance(A[0], list) else [[float(v) for v in row] for row in A]
    m = len(A_mat)
    if m == 0 or n != 2:
        is_pd = is_positive_definite(Q)
        basis = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)] if m == 0 else None
        return {"is_positive_definite": is_pd, "reduced_hessian": Q, "basis": basis}

    a1, a2 = float(A_mat[0][0]), float(A_mat[0][1])
    nrm = math.hypot(a1, a2)
    if nrm == 0.0:
        return {"is_positive_definite": is_positive_definite(Q), "reduced_hessian": Q, "basis": None}

    z = [-a2 / nrm, a1 / nrm]
    basis = [[z[0]], [z[1]]]
    red_H_val = dot(z, mat_vec(Q, z))
    return {"is_positive_definite": red_H_val > 1e-10, "reduced_hessian": [[red_H_val]], "basis": basis}

def _to_float_list(vals):
    """Converts a scalar or sequence of numbers to a list of floats."""
    if vals is None:
        return []
    return [float(vals)] if isinstance(vals, (int, float)) else [float(v) for v in vals]

def kkt_check(grad_f, eq_grads, ineq_values, ineq_grads, x, lambdas=None, mus=None, tol=1e-5, eq_values=None):
    """Evaluates first-order KKT optimality conditions at candidate point x."""
    lambdas, mus = _to_float_list(lambdas), _to_float_list(mus)
    gf = grad_f(x) if callable(grad_f) else [float(v) for v in grad_f]
    e_eq_g = [(eg(x) if callable(eg) else [float(v) for v in eg]) for eg in eq_grads]
    e_in_v = [float(iv(x) if callable(iv) else iv) for iv in ineq_values]
    e_in_g = [(ig(x) if callable(ig) else [float(v) for v in ig]) for ig in ineq_grads]

    stat = list(gf)
    for l, eg in zip(lambdas, e_eq_g):
        stat = vec_add(stat, scalar_mul(l, eg))
    for m, ig in zip(mus, e_in_g):
        stat = vec_add(stat, scalar_mul(m, ig))
    stationarity = bool(norm_l2(stat) <= tol)

    p_eq = all(abs(float(ev(x) if callable(ev) else ev)) <= tol for ev in eq_values) if eq_values is not None else True
    primal = all(v <= tol for v in e_in_v) and p_eq
    dual = all(m >= -tol for m in mus)
    slack = all(abs(m * v) <= tol for m, v in zip(mus, e_in_v))
    overall = bool(stationarity and primal and dual and slack)

    return {
        "stationarity": stationarity, "primal_feasibility": primal,
        "dual_feasibility": dual, "complementary_slackness": slack,
        "all_satisfied": overall, "kkt": overall, "kkt_satisfied": overall,
        "overall": overall, "stationarity_norm": norm_l2(stat),
        "stat_norm": norm_l2(stat)
    }

def licq_check(active_grads, tol=1e-9):
    """Checks the Linear Independence Constraint Qualification and Gram determinant."""
    m = len(active_grads)
    if m == 0:
        return {"holds": True, "gram_det": 1.0}

    J = [[float(v) for v in row] for row in active_grads]
    rk = rank(J, tol=tol)
    holds = bool(rk == m)

    G = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            G[i][j] = dot(J[i], J[j])

    gram_det = determinant(G, tol=tol)
    return {"holds": holds, "gram_det": gram_det}
