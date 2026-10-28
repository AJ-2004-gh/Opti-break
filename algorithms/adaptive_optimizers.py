import math
from core.vectors import vec_clone, norm_l2
from core.convergence import Result, check_stopping, is_diverged

def _init_opt(f, grad, x0, extra):
    """Initializes trajectories and checks initial divergence for adaptive optimizers."""
    x = vec_clone(x0)
    path = [vec_clone(x)]
    try:
        f0, g0 = f(x), grad(x)
        g0_n = norm_l2(g0)
    except OverflowError:
        r = Result(x=x, fun=float("inf"), grad_norm=float("inf"), iters=0,
                   converged=False, message="diverged", path=path,
                   f_hist=[float("inf")], g_hist=[float("inf")], extra=dict(extra))
        return x, path, None, None, r
    f_hist, g_hist = [f0], [g0_n]
    if is_diverged(f0, g0_n):
        r = Result(x=x, fun=f0, grad_norm=g0_n, iters=0,
                   converged=False, message="diverged", path=path,
                   f_hist=f_hist, g_hist=g_hist, extra=dict(extra))
        return x, path, f_hist, g_hist, r
    return x, path, f_hist, g_hist, None

def _div_step(x, path, f_hist, g_hist, iters, extra, attrs):
    """Builds a divergent Result object when overflow or divergence occurs."""
    r = Result(x=x, fun=f_hist[-1], grad_norm=g_hist[-1], iters=iters,
               converged=False, message="diverged", path=path,
               f_hist=f_hist, g_hist=g_hist, extra=dict(extra))
    for k, v in attrs.items():
        setattr(r, k, v)
    return r

def _finish_adaptive(f, x, g_norm, iters, conv, msg, path, f_hist, g_hist, extra, attrs):
    """Builds a completed Result object attaching optimizer diagnostic attributes."""
    r = Result(x=x, fun=f(x), grad_norm=g_norm, iters=iters,
               converged=conv, message=msg, path=path,
               f_hist=f_hist, g_hist=g_hist, extra=dict(extra))
    for k, v in attrs.items():
        setattr(r, k, v)
    return r

def adagrad(f, grad, x0, lr=0.01, tol=1e-5, max_iters=1000, eps=1e-8, alpha=None):
    """AdaGrad adaptive optimizer accumulating squared historical coordinate gradients."""
    lr = alpha if alpha is not None else lr
    x, path, f_hist, g_hist, init_res = _init_opt(f, grad, x0, {"lr": lr, "eps": eps})
    if init_res:
        init_res.G = [0.0] * len(x)
        return init_res
    n, G, iters, eff_hist, step_norm = len(x), [0.0] * len(x), 0, [], 0.0
    while True:
        g = grad(x)
        gn = norm_l2(g)
        stop, conv, msg = check_stopping(gn, iters, tol, max_iters)
        if stop:
            eff = [lr / (math.sqrt(G[i]) + eps) for i in range(n)] if iters > 0 else [lr / eps] * n
            ex = {"lr": lr, "eps": eps, "G": list(G), "effective_step_size": eff,
                  "effective_step_sizes": eff_hist, "effective_step_hist": eff_hist, "step_norm": step_norm}
            return _finish_adaptive(f, x, gn, iters, conv, msg, path, f_hist, g_hist, ex,
                                    {"G": list(G), "effective_step_size": eff, "effective_step": step_norm})
        x_prev = vec_clone(x)
        try:
            cur_eff = []
            for i in range(n):
                G[i] += g[i] * g[i]
                eta_i = lr / (math.sqrt(G[i]) + eps)
                cur_eff.append(eta_i)
                x[i] -= eta_i * g[i]
            eff_hist.append(cur_eff)
            step_norm = norm_l2([x[i] - x_prev[i] for i in range(n)])
            iters += 1
            cf, cg = f(x), grad(x)
            cgn = norm_l2(cg)
        except OverflowError:
            path.append(vec_clone(x)); f_hist.append(float("inf")); g_hist.append(float("inf"))
            return _div_step(x, path, f_hist, g_hist, iters, {"lr": lr, "G": list(G)}, {"G": list(G)})
        path.append(vec_clone(x)); f_hist.append(cf); g_hist.append(cgn)
        if is_diverged(cf, cgn):
            return _div_step(x, path, f_hist, g_hist, iters, {"lr": lr, "G": list(G)}, {"G": list(G)})

def rmsprop(f, grad, x0, lr=0.001, rho=0.9, tol=1e-5, max_iters=1000, eps=1e-8, beta=None, alpha=None):
    """RMSProp adaptive optimizer using an exponential moving average of squared gradients."""
    lr = alpha if alpha is not None else lr
    rho = beta if beta is not None else rho
    x, path, f_hist, g_hist, init_res = _init_opt(f, grad, x0, {"lr": lr, "rho": rho, "eps": eps})
    if init_res:
        init_res.v = [0.0] * len(x)
        return init_res
    n, v, iters, eff_hist, step_norm = len(x), [0.0] * len(x), 0, [], 0.0
    while True:
        g = grad(x)
        gn = norm_l2(g)
        stop, conv, msg = check_stopping(gn, iters, tol, max_iters)
        if stop:
            eff = [lr / (math.sqrt(v[i]) + eps) for i in range(n)] if iters > 0 else [lr / eps] * n
            ex = {"lr": lr, "rho": rho, "eps": eps, "v": list(v), "effective_step_size": eff,
                  "effective_step_sizes": eff_hist, "effective_step_hist": eff_hist, "step_norm": step_norm}
            return _finish_adaptive(f, x, gn, iters, conv, msg, path, f_hist, g_hist, ex,
                                    {"v": list(v), "effective_step_size": eff, "effective_step": step_norm})
        x_prev = vec_clone(x)
        try:
            cur_eff = []
            for i in range(n):
                v[i] = rho * v[i] + (1.0 - rho) * (g[i] * g[i])
                eta_i = lr / (math.sqrt(v[i]) + eps)
                cur_eff.append(eta_i)
                x[i] -= eta_i * g[i]
            eff_hist.append(cur_eff)
            step_norm = norm_l2([x[i] - x_prev[i] for i in range(n)])
            iters += 1
            cf, cg = f(x), grad(x)
            cgn = norm_l2(cg)
        except OverflowError:
            path.append(vec_clone(x)); f_hist.append(float("inf")); g_hist.append(float("inf"))
            return _div_step(x, path, f_hist, g_hist, iters, {"lr": lr, "v": list(v)}, {"v": list(v)})
        path.append(vec_clone(x)); f_hist.append(cf); g_hist.append(cgn)
        if is_diverged(cf, cgn):
            return _div_step(x, path, f_hist, g_hist, iters, {"lr": lr, "v": list(v)}, {"v": list(v)})

def adam(f, grad, x0, lr=0.001, b1=0.9, b2=0.999, tol=1e-5, max_iters=1000, eps=1e-8, beta1=None, beta2=None, alpha=None):
    """Adam optimizer with bias-corrected first and second gradient moments."""
    lr = alpha if alpha is not None else lr
    b1 = beta1 if beta1 is not None else b1
    b2 = beta2 if beta2 is not None else b2
    x, path, f_hist, g_hist, init_res = _init_opt(f, grad, x0, {"lr": lr, "b1": b1, "b2": b2, "eps": eps})
    if init_res:
        init_res.m = [0.0] * len(x); init_res.v = [0.0] * len(x)
        return init_res
    n, m, v, iters, eff_hist, step_norm = len(x), [0.0] * len(x), [0.0] * len(x), 0, [], 0.0
    while True:
        g = grad(x)
        gn = norm_l2(g)
        stop, conv, msg = check_stopping(gn, iters, tol, max_iters)
        if stop:
            t = max(1, iters); bc2 = 1.0 - (b2 ** t)
            eff = [lr / (math.sqrt(v[i] / bc2) + eps) for i in range(n)] if iters > 0 else [lr / eps] * n
            ex = {"lr": lr, "b1": b1, "b2": b2, "eps": eps, "m": list(m), "v": list(v), "effective_step_size": eff,
                  "effective_step_sizes": eff_hist, "effective_step_hist": eff_hist, "step_norm": step_norm}
            return _finish_adaptive(f, x, gn, iters, conv, msg, path, f_hist, g_hist, ex,
                                    {"m": list(m), "v": list(v), "effective_step_size": eff, "effective_step": step_norm})
        t, bc1, bc2 = iters + 1, 1.0 - (b1 ** (iters + 1)), 1.0 - (b2 ** (iters + 1))
        x_prev = vec_clone(x)
        try:
            cur_eff = []
            for i in range(n):
                m[i] = b1 * m[i] + (1.0 - b1) * g[i]
                v[i] = b2 * v[i] + (1.0 - b2) * (g[i] * g[i])
                eta_i = lr / (math.sqrt(v[i] / bc2) + eps)
                cur_eff.append(eta_i)
                x[i] -= eta_i * (m[i] / bc1)
            eff_hist.append(cur_eff)
            step_norm = norm_l2([x[i] - x_prev[i] for i in range(n)])
            iters += 1
            cf, cg = f(x), grad(x)
            cgn = norm_l2(cg)
        except OverflowError:
            path.append(vec_clone(x)); f_hist.append(float("inf")); g_hist.append(float("inf"))
            return _div_step(x, path, f_hist, g_hist, iters, {"lr": lr, "m": list(m), "v": list(v)}, {"m": list(m), "v": list(v)})
        path.append(vec_clone(x)); f_hist.append(cf); g_hist.append(cgn)
        if is_diverged(cf, cgn):
            return _div_step(x, path, f_hist, g_hist, iters, {"lr": lr, "m": list(m), "v": list(v)}, {"m": list(m), "v": list(v)})

def adamw(f, grad, x0, lr=0.001, b1=0.9, b2=0.999, weight_decay=0.01, tol=1e-5, max_iters=1000, eps=1e-8, beta1=None, beta2=None, alpha=None):
    """AdamW optimizer with decoupled weight decay and moment-based adaptive steps."""
    lr = alpha if alpha is not None else lr
    b1 = beta1 if beta1 is not None else b1
    b2 = beta2 if beta2 is not None else b2
    x, path, f_hist, g_hist, init_res = _init_opt(f, grad, x0, {"lr": lr, "b1": b1, "b2": b2, "weight_decay": weight_decay, "eps": eps})
    if init_res:
        init_res.m = [0.0] * len(x); init_res.v = [0.0] * len(x)
        return init_res
    n, m, v, iters, eff_hist, step_norm = len(x), [0.0] * len(x), [0.0] * len(x), 0, [], 0.0
    while True:
        g = grad(x)
        gn = norm_l2(g)
        stop, conv, msg = check_stopping(gn, iters, tol, max_iters)
        if stop:
            t = max(1, iters); bc2 = 1.0 - (b2 ** t)
            eff = [lr / (math.sqrt(v[i] / bc2) + eps) for i in range(n)] if iters > 0 else [lr / eps] * n
            ex = {"lr": lr, "b1": b1, "b2": b2, "weight_decay": weight_decay, "eps": eps, "m": list(m), "v": list(v),
                  "effective_step_size": eff, "effective_step_sizes": eff_hist, "effective_step_hist": eff_hist, "step_norm": step_norm}
            return _finish_adaptive(f, x, gn, iters, conv, msg, path, f_hist, g_hist, ex,
                                    {"m": list(m), "v": list(v), "effective_step_size": eff, "effective_step": step_norm})
        t, bc1, bc2 = iters + 1, 1.0 - (b1 ** (iters + 1)), 1.0 - (b2 ** (iters + 1))
        x_prev = vec_clone(x)
        try:
            cur_eff = []
            for i in range(n):
                x[i] -= lr * weight_decay * x[i]
                m[i] = b1 * m[i] + (1.0 - b1) * g[i]
                v[i] = b2 * v[i] + (1.0 - b2) * (g[i] * g[i])
                eta_i = lr / (math.sqrt(v[i] / bc2) + eps)
                cur_eff.append(eta_i)
                x[i] -= eta_i * (m[i] / bc1)
            eff_hist.append(cur_eff)
            step_norm = norm_l2([x[i] - x_prev[i] for i in range(n)])
            iters += 1
            cf, cg = f(x), grad(x)
            cgn = norm_l2(cg)
        except OverflowError:
            path.append(vec_clone(x)); f_hist.append(float("inf")); g_hist.append(float("inf"))
            return _div_step(x, path, f_hist, g_hist, iters, {"lr": lr, "m": list(m), "v": list(v)}, {"m": list(m), "v": list(v)})
        path.append(vec_clone(x)); f_hist.append(cf); g_hist.append(cgn)
        if is_diverged(cf, cgn):
            return _div_step(x, path, f_hist, g_hist, iters, {"lr": lr, "m": list(m), "v": list(v)}, {"m": list(m), "v": list(v)})
