import math
import numpy as np
from core.vectors import dot, zeros
from core.matrices import mat_zeros, mat_diag_add

# Constants
START_POINT = [0.0, 0.0]
PLOT_BOX = [(-4.0, 4.0), (-4.0, 4.0)]

def sigmoid(z):
    """Computes numerically stable scalar sigmoid function 1 / (1 + exp(-z))."""
    fz = float(z)
    if fz >= 0.0:
        return 1.0 / (1.0 + math.exp(-fz))
    else:
        exp_z = math.exp(fz)
        return exp_z / (1.0 + exp_z)

def generate_blobs(n, separation=2.0, seed=42):
    """Generates synthetic 2D binary classification blobs."""
    rng = np.random.RandomState(seed)
    n0 = n // 2
    n1 = n - n0

    center0 = [-separation / 2.0, -separation / 2.0]
    center1 = [separation / 2.0, separation / 2.0]

    # Standard deviation 0.5 per blob ensures clear gap when separation >= 3.0
    pts0 = rng.randn(n0, 2) * 0.5 + center0
    pts1 = rng.randn(n1, 2) * 0.5 + center1

    X = pts0.tolist() + pts1.tolist()
    y = [0.0] * n0 + [1.0] * n1
    return X, y

def bce_loss(w, X, y, l2_reg=0.0):
    """Computes binary cross-entropy loss without bias parameter."""
    n = len(y)
    total_loss = 0.0
    for i in range(n):
        z = dot(X[i], w)
        yi = float(y[i])
        if z >= 0.0:
            log1p = math.log1p(math.exp(-z))
            total_loss += (1.0 - yi) * z + log1p
        else:
            log1p = math.log1p(math.exp(z))
            total_loss += -yi * z + log1p

    loss = total_loss / float(n)
    if l2_reg > 0.0:
        loss += 0.5 * l2_reg * dot(w, w)
    return loss

def bce_grad(w, X, y, l2_reg=0.0):
    """
    Gradient of binary cross-entropy (weights only).
    """
    n = len(y)
    d = len(w)
    g = zeros(d)
    for i in range(n):
        z = dot(X[i], w)
        p = sigmoid(z)
        residual = p - float(y[i])
        for j in range(d):
            g[j] += residual * float(X[i][j])

    for j in range(d):
        g[j] /= float(n)
        if l2_reg > 0.0:
            g[j] += l2_reg * float(w[j])
    return g

def bce_hess(w, X, y, l2_reg=0.0):
    """
    Hessian of binary cross-entropy (weights only).
    """
    n = len(y)
    d = len(w)
    H = mat_zeros(d, d)
    for i in range(n):
        z = dot(X[i], w)
        p = sigmoid(z)
        weight = p * (1.0 - p)
        xi = X[i]
        for j in range(d):
            for k in range(j, d):
                H[j][k] += weight * float(xi[j]) * float(xi[k])

    for j in range(d):
        for k in range(j, d):
            val = H[j][k] / float(n)
            H[j][k] = val
            H[k][j] = val

    if l2_reg > 0.0:
        H = mat_diag_add(H, l2_reg)
    return H

# -------------------------------------------------------------
# Version with Bias: theta = [w0, w1, b]
# -------------------------------------------------------------
def bce_loss_with_bias(theta, X, y, l2_reg=0.0):
    """
    Binary cross-entropy loss with bias term b: theta = [w0, w1, ..., b].
    """
    w = theta[:-1]
    b = float(theta[-1])
    n = len(y)
    total_loss = 0.0
    for i in range(n):
        z = dot(X[i], w) + b
        yi = float(y[i])
        if z >= 0.0:
            log1p = math.log1p(math.exp(-z))
            total_loss += (1.0 - yi) * z + log1p
        else:
            log1p = math.log1p(math.exp(z))
            total_loss += -yi * z + log1p

    loss = total_loss / float(n)
    if l2_reg > 0.0:
        loss += 0.5 * l2_reg * dot(w, w)
    return loss

def bce_grad_with_bias(theta, X, y, l2_reg=0.0):
    """
    Gradient of binary cross-entropy with bias term b: theta = [w0, w1, ..., b].
    """
    w = theta[:-1]
    b = float(theta[-1])
    n = len(y)
    d = len(w)
    g_w = zeros(d)
    g_b = 0.0

    for i in range(n):
        z = dot(X[i], w) + b
        p = sigmoid(z)
        residual = p - float(y[i])
        for j in range(d):
            g_w[j] += residual * float(X[i][j])
        g_b += residual

    grad = []
    for j in range(d):
        val = g_w[j] / float(n)
        if l2_reg > 0.0:
            val += l2_reg * float(w[j])
        grad.append(val)
    grad.append(g_b / float(n))
    return grad

def bce_hess_with_bias(theta, X, y, l2_reg=0.0):
    """
    Hessian of binary cross-entropy with bias term b.
    """
    w = theta[:-1]
    b = float(theta[-1])
    n = len(y)
    d = len(w)
    dim = len(theta)
    H = mat_zeros(dim, dim)

    for i in range(n):
        z = dot(X[i], w) + b
        p = sigmoid(z)
        weight = p * (1.0 - p)
        # augmented feature: [x_i, 1.0]
        x_aug = [float(v) for v in X[i]] + [1.0]
        for j in range(dim):
            for k in range(j, dim):
                H[j][k] += weight * x_aug[j] * x_aug[k]

    for j in range(dim):
        for k in range(j, dim):
            val = H[j][k] / float(n)
            H[j][k] = val
            H[k][j] = val

    if l2_reg > 0.0:
        for j in range(d):
            H[j][j] += l2_reg
    return H

# -------------------------------------------------------------
# Compatibility Helpers
# -------------------------------------------------------------
def generate_logistic_regression_data(n_samples=100, n_features=2, seed=42):
    """Generates synthetic binary classification dataset."""
    X, y = generate_blobs(n_samples, separation=3.0, seed=seed)
    w_true = [1.0] * n_features
    return X, y, w_true

generate_data = generate_blobs
logistic_regression_loss = bce_loss
logistic_regression_grad = bce_grad
logistic_regression_hess = bce_hess

def make_logistic_regression_problem(X, y, l2_reg=0.0):
    """Constructs BCE loss and gradient callables for classification."""
    return (
        lambda w: bce_loss(w, X, y, l2_reg=l2_reg),
        lambda w: bce_grad(w, X, y, l2_reg=l2_reg),
        lambda w: bce_hess(w, X, y, l2_reg=l2_reg)
    )
