import random
import numpy as np
from core.vectors import dot, zeros

def generate_data(n, seed=42, noise=0.1, scale_x2=1.0):
    """Generates synthetic linear dataset y = X w + b + noise."""
    rng = np.random.RandomState(seed)
    x1 = rng.randn(n)
    x2 = rng.randn(n) * scale_x2
    eps = rng.randn(n) * noise
    y_arr = 3.0 * x1 + 2.0 * x2 + 1.0 + eps

    X = [[float(x1[i]), float(x2[i])] for i in range(n)]
    y = [float(v) for v in y_arr]
    return X, y

def mse(w, b, X, y):
    """
    MSE(w, b) = (1 / (2*N)) * sum (w^T x_i + b - y_i)^2
    """
    n = len(y)
    total = 0.0
    for i in range(n):
        pred = dot(X[i], w) + float(b)
        residual = pred - float(y[i])
        total += residual * residual
    return total / (2.0 * n)

def single_sample_grad(w, b, x_i, y_i):
    """Computes gradient of single-sample squared error loss."""
    pred = dot(x_i, w) + float(b)
    residual = pred - float(y_i)
    grad_w = [residual * float(x_i[j]) for j in range(len(w))]
    grad_b = residual
    return grad_w, grad_b

def batch_grad(w, b, X, y):
    """Computes gradient of full-batch MSE loss."""
    n = len(y)
    d = len(w)
    grad_w = zeros(d)
    grad_b = 0.0

    for i in range(n):
        gw, gb = single_sample_grad(w, b, X[i], y[i])
        for j in range(d):
            grad_w[j] += gw[j]
        grad_b += gb

    for j in range(d):
        grad_w[j] /= float(n)
    grad_b /= float(n)

    return grad_w, grad_b

def minibatch_grad(w, b, X, y, batch_indices):
    """Computes gradient of mini-batch MSE loss."""
    k = len(batch_indices)
    d = len(w)
    grad_w = zeros(d)
    grad_b = 0.0

    for idx in batch_indices:
        gw, gb = single_sample_grad(w, b, X[idx], y[idx])
        for j in range(d):
            grad_w[j] += gw[j]
        grad_b += gb

    for j in range(d):
        grad_w[j] /= float(k)
    grad_b /= float(k)

    return grad_w, grad_b

# -------------------------------------------------------------
# 2-Parameter Version (w, b) with 1 Feature for Contour Plotting
# -------------------------------------------------------------
PARAM_2D_START = [0.0, 0.0]
PARAM_2D_BOX = [(-1.0, 5.0), (-2.0, 4.0)]
PARAM_2D_MINIMIZER = [3.0, 1.0]

def generate_data_1d(n=50, seed=42, noise=0.1, scale_x=1.0, as_matrix=False):
    """
    Generates 1D feature data for 2D parameter loss: y = 3 * x + 1 + noise.
    """
    rng = np.random.RandomState(seed)
    x = rng.randn(n) * float(scale_x)
    eps = rng.randn(n) * noise
    y = 3.0 * x + 1.0 + eps
    if as_matrix:
        return [[float(v)] for v in x], [float(v) for v in y]
    return [float(v) for v in x], [float(v) for v in y]

_DEFAULT_X_1D, _DEFAULT_Y_1D = generate_data_1d(n=60, seed=42, noise=0.0)

def loss_2d(param, x_data=None, y_data=None):
    """
    f(w, b) = (1 / (2*N)) * sum (w * x_i + b - y_i)^2
    """
    xs = x_data if x_data is not None else _DEFAULT_X_1D
    ys = y_data if y_data is not None else _DEFAULT_Y_1D
    w = float(param[0])
    b = float(param[1])
    n = len(ys)
    total = 0.0
    for i in range(n):
        xi = float(xs[i][0] if isinstance(xs[i], (list, tuple)) else xs[i])
        res = (w * xi + b) - ys[i]
        total += res * res
    return total / (2.0 * n)

def grad_2d(param, x_data=None, y_data=None):
    """
    grad(w, b) = [df/dw, df/db]
    """
    xs = x_data if x_data is not None else _DEFAULT_X_1D
    ys = y_data if y_data is not None else _DEFAULT_Y_1D
    w = float(param[0])
    b = float(param[1])
    n = len(ys)
    gw = 0.0
    gb = 0.0
    for i in range(n):
        xi = float(xs[i][0] if isinstance(xs[i], (list, tuple)) else xs[i])
        res = (w * xi + b) - ys[i]
        gw += res * xi
        gb += res
    return [gw / float(n), gb / float(n)]

def hess_2d(param=None, x_data=None, y_data=None):
    """
    hess(w, b) = (1/N) * [[sum x_i^2, sum x_i], [sum x_i, N]]
    """
    xs = x_data if x_data is not None else _DEFAULT_X_1D
    n = len(xs)
    clean_xs = [float(v[0] if isinstance(v, (list, tuple)) else v) for v in xs]
    sum_x2 = sum(x * x for x in clean_xs) / float(n)
    sum_x = sum(clean_xs) / float(n)
    return [[sum_x2, sum_x], [sum_x, 1.0]]

# -------------------------------------------------------------
# Backward Compatibility Helpers
# -------------------------------------------------------------
def generate_linear_regression_data(n_samples=100, n_features=5, noise=0.1, seed=42):
    """Generates synthetic linear regression dataset with ground truth."""
    rng = np.random.RandomState(seed)
    X_arr = rng.randn(n_samples, n_features)
    w_true_arr = rng.uniform(-2.0, 2.0, size=n_features)
    y_arr = X_arr.dot(w_true_arr) + rng.randn(n_samples) * noise
    return X_arr.tolist(), y_arr.tolist(), w_true_arr.tolist()

def linear_regression_loss(w, X, y, l2_reg=0.0):
    """Evaluates mean squared error loss for linear regression."""
    n = len(y)
    total = 0.0
    for i in range(n):
        diff = dot(X[i], w) - y[i]
        total += diff * diff
    loss = total / (2.0 * n)
    if l2_reg > 0.0:
        loss += 0.5 * l2_reg * dot(w, w)
    return loss

def linear_regression_grad(w, X, y, l2_reg=0.0):
    """Computes MSE loss gradient for linear regression."""
    n = len(y)
    d = len(w)
    g = zeros(d)
    for i in range(n):
        diff = dot(X[i], w) - y[i]
        for j in range(d):
            g[j] += diff * X[i][j]
    for j in range(d):
        g[j] /= float(n)
        if l2_reg > 0.0:
            g[j] += l2_reg * w[j]
    return g

def linear_regression_hess(w, X, y, l2_reg=0.0):
    """Computes MSE loss Hessian matrix (X^T X / n)."""
    from core.matrices import mat_zeros, mat_diag_add
    n = len(y)
    d = len(w)
    H = mat_zeros(d, d)
    for i in range(n):
        xi = X[i]
        for j in range(d):
            for k in range(j, d):
                H[j][k] += xi[j] * xi[k]
    for j in range(d):
        for k in range(j, d):
            val = H[j][k] / float(n)
            H[j][k] = val
            H[k][j] = val
    if l2_reg > 0.0:
        H = mat_diag_add(H, l2_reg)
    return H

def linear_regression_batch_grad(w, X, y, batch_indices, l2_reg=0.0):
    """Computes MSE gradient on mini-batch of sample indices."""
    k = len(batch_indices)
    d = len(w)
    g = zeros(d)
    for idx in batch_indices:
        diff = dot(X[idx], w) - y[idx]
        for j in range(d):
            g[j] += diff * X[idx][j]
    for j in range(d):
        g[j] /= float(k)
        if l2_reg > 0.0:
            g[j] += l2_reg * w[j]
    return g

def make_linear_regression_problem(X, y, l2_reg=0.0):
    """Constructs objective, gradient, and Hessian callables for regression."""
    return (
        lambda w: linear_regression_loss(w, X, y, l2_reg=l2_reg),
        lambda w: linear_regression_grad(w, X, y, l2_reg=l2_reg),
        lambda w: linear_regression_hess(w, X, y, l2_reg=l2_reg)
    )
