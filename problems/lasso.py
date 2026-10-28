import numpy as np
from core.vectors import dot, zeros

# Constants
START_POINT = [2.0, 2.0]
PLOT_BOX = [(-3.0, 3.0), (-3.0, 3.0)]

def soft_threshold(z, kappa):
    """
    S_kappa(z) = sign(z) * max(0, |z| - kappa)
    """
    fz = float(z)
    fk = float(kappa)
    if fz > fk:
        return fz - fk
    elif fz < -fk:
        return fz + fk
    else:
        return 0.0

def lasso_2d_loss(w, X, y, lam=0.1):
    """
    loss = 0.5 * mean squared residual + lam * L1 for 2 weights.
    """
    n = len(y)
    total_sq_res = 0.0
    for i in range(n):
        pred = dot(X[i], w)
        res = pred - float(y[i])
        total_sq_res += res * res

    mse = total_sq_res / (2.0 * n)
    l1 = abs(float(w[0])) + abs(float(w[1]))
    return mse + float(lam) * l1

def lasso_2d_subgradient(w, X, y, lam=0.1):
    """Computes subgradient of 2D Lasso objective."""
    n = len(y)
    d = len(w)
    g_smooth = zeros(d)

    for i in range(n):
        res = dot(X[i], w) - float(y[i])
        for j in range(d):
            g_smooth[j] += res * float(X[i][j])

    for j in range(d):
        g_smooth[j] /= float(n)

    g_total = []
    for j in range(d):
        wj = float(w[j])
        if wj > 0.0:
            s_j = 1.0
        elif wj < 0.0:
            s_j = -1.0
        else:
            s_j = 0.0  # Choice: 0 at kink w_j = 0
        g_total.append(g_smooth[j] + float(lam) * s_j)

    return g_total

# Aliases
loss = lasso_2d_loss
subgradient = lasso_2d_subgradient

def generate_lasso_data_2d(n=50, seed=42, noise=0.1):
    """
    Generates synthetic 2D data for 2-weight Lasso.
    """
    rng = np.random.RandomState(seed)
    X_arr = rng.randn(n, 2)
    # True weights: [2.0, 0.0] (second feature is inactive)
    w_true = np.array([2.0, 0.0])
    y_arr = X_arr.dot(w_true) + rng.randn(n) * noise
    return X_arr.tolist(), y_arr.tolist()

# -------------------------------------------------------------
# General Multi-dimensional Functions for Experiments
# -------------------------------------------------------------
def generate_sparse_linear_data(n_samples=100, n_features=20, n_informative=5, noise=0.1, seed=42):
    """Generates synthetic sparse linear regression dataset."""
    rng = np.random.RandomState(seed)
    X_arr = rng.randn(n_samples, n_features)
    w_true_arr = np.zeros(n_features)
    active_idx = rng.choice(n_features, size=n_informative, replace=False)
    w_true_arr[active_idx] = rng.uniform(1.0, 3.0, size=n_informative) * rng.choice([-1.0, 1.0], size=n_informative)
    y_arr = X_arr.dot(w_true_arr) + rng.randn(n_samples) * noise
    return X_arr.tolist(), y_arr.tolist(), w_true_arr.tolist()

def lasso_loss(w, X, y, lmbda):
    """Evaluates full unconstrained Lasso objective MSE + lam*||w||_1."""
    n = len(y)
    total_sq_err = 0.0
    for i in range(n):
        diff = dot(X[i], w) - y[i]
        total_sq_err += diff * diff
    mse = total_sq_err / (2.0 * n)
    l1 = sum(abs(float(v)) for v in w)
    return mse + lmbda * l1

def lasso_smooth_loss(w, X, y):
    """Evaluates smooth MSE loss component of Lasso objective."""
    n = len(y)
    total_sq_err = 0.0
    for i in range(n):
        diff = dot(X[i], w) - y[i]
        total_sq_err += diff * diff
    return total_sq_err / (2.0 * n)

def lasso_smooth_grad(w, X, y):
    """Computes gradient of smooth MSE loss component."""
    n = len(y)
    d = len(w)
    g = zeros(d)
    for i in range(n):
        diff = dot(X[i], w) - y[i]
        for j in range(d):
            g[j] += diff * X[i][j]
    for j in range(d):
        g[j] /= float(n)
    return g

def lasso_subgradient(w, X, y, lmbda):
    """Computes subgradient vector of full Lasso objective."""
    g = lasso_smooth_grad(w, X, y)
    for j in range(len(w)):
        if w[j] > 0.0:
            g[j] += lmbda
        elif w[j] < 0.0:
            g[j] -= lmbda
    return g

def lasso_coordinate_descent_step(w, X, y, lmbda, col_norms_sq=None):
    """Performs single coordinate descent soft-thresholding update."""
    n = len(y)
    d = len(w)
    if col_norms_sq is None:
        col_norms_sq = []
        for j in range(d):
            col_sq = sum(X[i][j] ** 2 for i in range(n))
            col_norms_sq.append(col_sq / float(n))

    preds = [dot(X[i], w) for i in range(n)]

    for j in range(d):
        zj = col_norms_sq[j]
        if zj == 0.0:
            continue
        old_wj = w[j]
        rho_j = 0.0
        for i in range(n):
            residual = (y[i] - preds[i]) + X[i][j] * old_wj
            rho_j += X[i][j] * residual
        rho_j /= float(n)

        new_wj = soft_threshold(rho_j, lmbda) / zj
        w[j] = new_wj

        dw = new_wj - old_wj
        if dw != 0.0:
            for i in range(n):
                preds[i] += X[i][j] * dw

    return w
