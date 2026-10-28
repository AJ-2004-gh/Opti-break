import random
import math
from core.vectors import vec_clone, vec_sub, vec_scale, vec_norm, norm_l2, dot
from core.convergence import Result, check_stopping, is_diverged
from problems.linear_regression import mse, batch_grad, single_sample_grad, minibatch_grad
from problems.logistic_regression import bce_loss_with_bias, bce_grad_with_bias, sigmoid
from problems.lasso import soft_threshold

def standardize(X):
    """Standardizes each feature column of X to zero mean and unit variance."""
    if not X:
        return []
    n, d = len(X), len(X[0])
    means = [sum(X[i][j] for i in range(n)) / float(n) for j in range(d)]
    stds = []
    for j in range(d):
        variance = sum((X[i][j] - means[j]) ** 2 for i in range(n)) / float(n)
        std_val = math.sqrt(variance)
        stds.append(std_val if std_val > 1e-12 else 1.0)
    return [[(X[i][j] - means[j]) / stds[j] for j in range(d)] for i in range(n)]

def _init_reg_run(X, y, extra):
    """Initializes regression weights and checks initial divergence."""
    d = len(X[0])
    w, b = [0.0] * d, 0.0
    theta, path = w + [b], [w + [b]]
    try:
        f0 = mse(w, b, X, y)
        gw, gb = batch_grad(w, b, X, y)
        g0_norm = norm_l2(gw + [gb])
    except OverflowError:
        r = Result(x=theta, fun=float("inf"), grad_norm=float("inf"), iters=0, converged=False,
                   message="diverged", path=path, f_hist=[float("inf")], g_hist=[float("inf")], extra=dict(extra, w=w, b=b))
        r.w, r.b = list(w), float(b)
        return w, b, theta, path, None, None, r
    f_hist, g_hist = [f0], [g0_norm]
    if is_diverged(f0, g0_norm):
        r = Result(x=theta, fun=f0, grad_norm=g0_norm, iters=0, converged=False,
                   message="diverged", path=path, f_hist=f_hist, g_hist=g_hist, extra=dict(extra, w=w, b=b))
        r.w, r.b = list(w), float(b)
        return w, b, theta, path, f_hist, g_hist, r
    return w, b, theta, path, f_hist, g_hist, None

def _finish_reg(theta, w, b, ep, is_conv, msg, extra, path, f_hist, g_hist):
    """Builds a completed Result object for regression methods."""
    res = Result(x=theta, fun=f_hist[-1], grad_norm=g_hist[-1], iters=ep,
                 converged=is_conv, message=msg, path=path, f_hist=f_hist, g_hist=g_hist,
                 extra=dict(extra, w=list(w), b=float(b)))
    res.w, res.b = list(w), float(b)
    return res

def _div_reg(theta, w, b, ep, extra, path, f_hist, g_hist):
    """Builds a divergent Result object for regression methods."""
    res = Result(x=theta, fun=f_hist[-1], grad_norm=g_hist[-1], iters=ep,
                 converged=False, message="diverged", path=path, f_hist=f_hist, g_hist=g_hist,
                 extra=dict(extra, w=list(w), b=float(b)))
    res.w, res.b = list(w), float(b)
    return res

def gd_linear_regression(X, y, lr=0.01, epochs=100, tol=1e-5):
    """Full-batch gradient descent for linear regression."""
    d = len(X[0])
    w, b, theta, path, f_hist, g_hist, init_res = _init_reg_run(X, y, {"lr": lr})
    if init_res:
        return init_res
    gw, gb = batch_grad(w, b, X, y)
    for ep in range(epochs):
        stop, conv, msg = check_stopping(g_hist[-1], ep, tol, epochs)
        if stop:
            return _finish_reg(theta, w, b, ep, conv, msg, {"lr": lr}, path, f_hist, g_hist)
        try:
            w = [w[j] - lr * gw[j] for j in range(d)]
            b = b - lr * gb
            theta = w + [b]
            cf = mse(w, b, X, y)
            gw, gb = batch_grad(w, b, X, y)
            cgn = norm_l2(gw + [gb])
        except OverflowError:
            path.append(vec_clone(theta)); f_hist.append(float("inf")); g_hist.append(float("inf"))
            return _div_reg(theta, w, b, ep + 1, {"lr": lr}, path, f_hist, g_hist)
        path.append(vec_clone(theta)); f_hist.append(cf); g_hist.append(cgn)
        if is_diverged(cf, cgn):
            return _div_reg(theta, w, b, ep + 1, {"lr": lr}, path, f_hist, g_hist)
    is_conv = g_hist[-1] < tol
    msg = "Converged: gradient norm below tolerance" if is_conv else "Maximum iterations reached"
    return _finish_reg(theta, w, b, epochs, is_conv, msg, {"lr": lr}, path, f_hist, g_hist)

def sgd_linear_regression(X, y, lr=0.01, epochs=100, seed=42, tol=1e-5):
    """Stochastic gradient descent for linear regression."""
    rng = random.Random(seed)
    n, d = len(y), len(X[0])
    w, b, theta, path, f_hist, g_hist, init_res = _init_reg_run(X, y, {"lr": lr})
    if init_res:
        return init_res
    indices = list(range(n))
    update_path = []
    for ep in range(epochs):
        stop, conv, msg = check_stopping(g_hist[-1], ep, tol, epochs)
        if stop:
            return _finish_reg(theta, w, b, ep, conv, msg, {"lr": lr, "update_path": update_path}, path, f_hist, g_hist)
        rng.shuffle(indices)
        try:
            for idx in indices:
                sg_w, sg_b = single_sample_grad(w, b, X[idx], y[idx])
                w = [w[j] - lr * sg_w[j] for j in range(d)]
                b = b - lr * sg_b
                update_path.append(w + [b])
            theta = w + [b]
            cf = mse(w, b, X, y)
            gw, gb = batch_grad(w, b, X, y)
            cgn = norm_l2(gw + [gb])
        except OverflowError:
            path.append(vec_clone(theta)); f_hist.append(float("inf")); g_hist.append(float("inf"))
            return _div_reg(theta, w, b, ep + 1, {"lr": lr, "update_path": update_path}, path, f_hist, g_hist)
        path.append(vec_clone(theta)); f_hist.append(cf); g_hist.append(cgn)
        if is_diverged(cf, cgn):
            return _div_reg(theta, w, b, ep + 1, {"lr": lr, "update_path": update_path}, path, f_hist, g_hist)
    is_conv = g_hist[-1] < tol
    msg = "Converged: gradient norm below tolerance" if is_conv else "Maximum iterations reached"
    return _finish_reg(theta, w, b, epochs, is_conv, msg, {"lr": lr, "update_path": update_path}, path, f_hist, g_hist)

def minibatch_sgd_linear_regression(X, y, lr=0.01, epochs=100, batch_size=16, seed=42, tol=1e-5):
    """Mini-batch SGD for linear regression."""
    rng = random.Random(seed)
    n, d = len(y), len(X[0])
    w, b, theta, path, f_hist, g_hist, init_res = _init_reg_run(X, y, {"lr": lr, "batch_size": batch_size})
    if init_res:
        return init_res
    indices = list(range(n))
    update_path = []
    for ep in range(epochs):
        stop, conv, msg = check_stopping(g_hist[-1], ep, tol, epochs)
        if stop:
            return _finish_reg(theta, w, b, ep, conv, msg, {"lr": lr, "batch_size": batch_size, "update_path": update_path}, path, f_hist, g_hist)
        rng.shuffle(indices)
        try:
            for start_idx in range(0, n, batch_size):
                b_idx = indices[start_idx : start_idx + batch_size]
                mb_w, mb_b = minibatch_grad(w, b, X, y, b_idx)
                w = [w[j] - lr * mb_w[j] for j in range(d)]
                b = b - lr * mb_b
                update_path.append(w + [b])
            theta = w + [b]
            cf = mse(w, b, X, y)
            gw, gb = batch_grad(w, b, X, y)
            cgn = norm_l2(gw + [gb])
        except OverflowError:
            path.append(vec_clone(theta)); f_hist.append(float("inf")); g_hist.append(float("inf"))
            return _div_reg(theta, w, b, ep + 1, {"lr": lr, "batch_size": batch_size, "update_path": update_path}, path, f_hist, g_hist)
        path.append(vec_clone(theta)); f_hist.append(cf); g_hist.append(cgn)
        if is_diverged(cf, cgn):
            return _div_reg(theta, w, b, ep + 1, {"lr": lr, "batch_size": batch_size, "update_path": update_path}, path, f_hist, g_hist)
    is_conv = g_hist[-1] < tol
    msg = "Converged: gradient norm below tolerance" if is_conv else "Maximum iterations reached"
    return _finish_reg(theta, w, b, epochs, is_conv, msg, {"lr": lr, "batch_size": batch_size, "update_path": update_path}, path, f_hist, g_hist)

def _compute_logistic_metrics(theta, X, y, d, n, lr, path, f_hist, g_hist, tol, w_norm_hist=None):
    """Computes accuracy, decision boundary, and terminal Result for logistic regression."""
    w, b = theta[:-1], float(theta[-1])
    correct = sum(1 for i in range(n) if (1.0 if sigmoid(dot(X[i], w) + b) >= 0.5 else 0.0) == float(y[i]))
    accuracy = correct / float(n)
    db = {"w": list(w), "b": b, "slope": (-w[0] / w[1]) if (d == 2 and abs(w[1]) > 1e-12) else 0.0,
          "intercept": (-b / w[1]) if (d == 2 and abs(w[1]) > 1e-12) else 0.0}
    is_conv = g_hist[-1] < tol
    msg = "Converged: gradient norm below tolerance" if is_conv else "Maximum iterations reached"
    extra = {"w": list(w), "b": b, "lr": lr, "accuracy": accuracy, "decision_boundary": db}
    if w_norm_hist is not None:
        extra["w_norm_hist"] = list(w_norm_hist)
    res = Result(x=theta, fun=f_hist[-1], grad_norm=g_hist[-1], iters=len(path) - 1,
                 converged=is_conv, message=msg, path=path, f_hist=f_hist, g_hist=g_hist, extra=extra)
    res.accuracy, res.decision_boundary, res.w, res.b = accuracy, db, list(w), b
    return res

def logistic_regression_gd(X, y, lr=0.1, epochs=100, tol=1e-5):
    """Batch gradient descent for binary logistic regression."""
    n, d = len(y), len(X[0])
    theta = [0.0] * (d + 1)
    path = [vec_clone(theta)]
    w_norm_hist = [norm_l2(theta[:-1])]
    try:
        f0, g0 = bce_loss_with_bias(theta, X, y), bce_grad_with_bias(theta, X, y)
        g0_norm = norm_l2(g0)
    except OverflowError:
        return Result(x=theta, fun=float("inf"), grad_norm=float("inf"), iters=0, converged=False,
                      message="diverged", path=path, f_hist=[float("inf")], g_hist=[float("inf")], extra={"theta": theta, "lr": lr, "w_norm_hist": w_norm_hist})
    f_hist, g_hist, curr_g = [f0], [g0_norm], g0
    if is_diverged(f0, g0_norm):
        return Result(x=theta, fun=f0, grad_norm=g0_norm, iters=0, converged=False,
                      message="diverged", path=path, f_hist=f_hist, g_hist=g_hist, extra={"theta": theta, "lr": lr, "w_norm_hist": w_norm_hist})
    for ep in range(epochs):
        stop, _, _ = check_stopping(g_hist[-1], ep, tol, epochs)
        if stop:
            break
        try:
            theta = [theta[j] - lr * curr_g[j] for j in range(d + 1)]
            w_norm_hist.append(norm_l2(theta[:-1]))
            cf = bce_loss_with_bias(theta, X, y)
            curr_g = bce_grad_with_bias(theta, X, y)
            cgn = norm_l2(curr_g)
        except OverflowError:
            path.append(vec_clone(theta)); f_hist.append(float("inf")); g_hist.append(float("inf"))
            return Result(x=theta, fun=float("inf"), grad_norm=float("inf"), iters=ep + 1, converged=False,
                          message="diverged", path=path, f_hist=f_hist, g_hist=g_hist, extra={"theta": theta, "lr": lr, "w_norm_hist": w_norm_hist})
        path.append(vec_clone(theta)); f_hist.append(cf); g_hist.append(cgn)
        if is_diverged(cf, cgn):
            return Result(x=theta, fun=cf, grad_norm=cgn, iters=ep + 1, converged=False,
                          message="diverged", path=path, f_hist=f_hist, g_hist=g_hist, extra={"theta": theta, "lr": lr, "w_norm_hist": w_norm_hist})
    return _compute_logistic_metrics(theta, X, y, d, n, lr, path, f_hist, g_hist, tol, w_norm_hist)

def sgd(f, batch_grad_fn, full_grad_fn, x0, n_samples, alpha=0.01, tol=1e-5, max_iters=1000, seed=42):
    """Generic stochastic gradient descent using single sample evaluations."""
    rng = random.Random(seed)
    x = vec_clone(x0)
    path, f_hist, g_hist = [vec_clone(x)], [f(x)], [vec_norm(full_grad_fn(x))]
    iters = 0
    while True:
        gn = vec_norm(full_grad_fn(x))
        stop, conv, msg = check_stopping(gn, iters, tol, max_iters)
        if stop:
            return Result(x=x, fun=f(x), grad_norm=gn, iters=iters, converged=conv, message=msg,
                          path=path, f_hist=f_hist, g_hist=g_hist, extra={"alpha": alpha})
        sample_idx = [rng.randint(0, n_samples - 1)]
        x = vec_sub(x, vec_scale(alpha, batch_grad_fn(x, sample_idx)))
        iters += 1
        path.append(vec_clone(x)); f_hist.append(f(x)); g_hist.append(vec_norm(full_grad_fn(x)))

def minibatch_gd(f, batch_grad_fn, full_grad_fn, x0, n_samples, batch_size=32, alpha=0.01, tol=1e-5, max_iters=1000, seed=42):
    """Generic mini-batch gradient descent using subset sample evaluations."""
    rng = random.Random(seed)
    x = vec_clone(x0)
    path, f_hist, g_hist = [vec_clone(x)], [f(x)], [vec_norm(full_grad_fn(x))]
    iters, all_indices = 0, list(range(n_samples))
    while True:
        gn = vec_norm(full_grad_fn(x))
        stop, conv, msg = check_stopping(gn, iters, tol, max_iters)
        if stop:
            return Result(x=x, fun=f(x), grad_norm=gn, iters=iters, converged=conv, message=msg,
                          path=path, f_hist=f_hist, g_hist=g_hist, extra={"alpha": alpha, "batch_size": batch_size})
        b_idx = rng.sample(all_indices, min(batch_size, n_samples))
        x = vec_sub(x, vec_scale(alpha, batch_grad_fn(x, b_idx)))
        iters += 1
        path.append(vec_clone(x)); f_hist.append(f(x)); g_hist.append(vec_norm(full_grad_fn(x)))

def coordinate_descent(f, x0, grad=None, alpha=0.1, tol=1e-5, max_iters=1000):
    """Cyclical coordinate descent optimization using finite-difference gradients."""
    x = vec_clone(x0)
    n, path, f_hist = len(x), [vec_clone(x)], [f(x)]
    g_hist = [vec_norm(grad(x)) if grad is not None else 1.0]
    iters = 0
    while True:
        gn = vec_norm(grad(x)) if grad is not None else abs(f_hist[-1] - (f_hist[-2] if len(f_hist) > 1 else f_hist[-1]))
        stop, conv, msg = check_stopping(gn, iters, tol, max_iters)
        if stop:
            return Result(x=x, fun=f(x), grad_norm=gn, iters=iters, converged=conv, message=msg, path=path, f_hist=f_hist, g_hist=g_hist)
        h = 1e-5
        for i in range(n):
            xp, xm = vec_clone(x), vec_clone(x)
            xp[i] += h; xm[i] -= h
            x[i] -= alpha * ((f(xp) - f(xm)) / (2.0 * h))
        iters += 1
        path.append(vec_clone(x)); f_hist.append(f(x))
        g_hist.append(vec_norm(grad(x)) if grad is not None else gn)

def ista(f, grad_smooth, x0, lmbda=0.1, L=1.0, tol=1e-5, max_iters=1000):
    """Iterative soft-thresholding algorithm for L1-regularized objectives."""
    x = vec_clone(x0)
    n, path, f_hist, g_hist = len(x), [vec_clone(x)], [f(x)], [vec_norm(grad_smooth(x))]
    step_size, threshold = 1.0 / L, lmbda * (1.0 / L)
    iters = 0
    while True:
        gs = grad_smooth(x)
        x_trial = [soft_threshold(x[i] - step_size * gs[i], threshold) for i in range(n)]
        mapping_norm = vec_norm(vec_sub(x, x_trial)) / step_size
        stop, conv, msg = check_stopping(mapping_norm, iters, tol, max_iters)
        if stop:
            return Result(x=x, fun=f(x), grad_norm=mapping_norm, iters=iters, converged=conv, message=msg,
                          path=path, f_hist=f_hist, g_hist=g_hist, extra={"L": L, "lmbda": lmbda})
        x = x_trial
        iters += 1
        path.append(vec_clone(x)); f_hist.append(f(x)); g_hist.append(mapping_norm)

def fista(f, grad_smooth, x0, lmbda=0.1, L=1.0, tol=1e-5, max_iters=1000):
    """Fast iterative soft-thresholding algorithm with Nesterov momentum."""
    x, y, t = vec_clone(x0), vec_clone(x0), 1.0
    n, path, f_hist, g_hist = len(x), [vec_clone(x)], [f(x)], [vec_norm(grad_smooth(x))]
    step_size, threshold = 1.0 / L, lmbda * (1.0 / L)
    iters = 0
    while True:
        gy = grad_smooth(y)
        x_next = [soft_threshold(y[i] - step_size * gy[i], threshold) for i in range(n)]
        mapping_norm = vec_norm(vec_sub(x_next, x)) / step_size
        stop, conv, msg = check_stopping(mapping_norm, iters, tol, max_iters)
        if stop:
            return Result(x=x, fun=f(x), grad_norm=mapping_norm, iters=iters, converged=conv, message=msg,
                          path=path, f_hist=f_hist, g_hist=g_hist, extra={"L": L, "lmbda": lmbda})
        t_next = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        factor = (t - 1.0) / t_next
        y = [x_next[i] + factor * (x_next[i] - x[i]) for i in range(n)]
        x, t = x_next, t_next
        iters += 1
        path.append(vec_clone(x)); f_hist.append(f(x)); g_hist.append(mapping_norm)
