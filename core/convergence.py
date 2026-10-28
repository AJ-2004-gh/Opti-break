class Result:
    """
    Standard optimization result container.
    """
    def __init__(self, x, fun, grad_norm, iters, converged, message, path=None, f_hist=None, g_hist=None, extra=None):
        self.x = list(x) if x is not None else []
        self.fun = float(fun)
        self.grad_norm = float(grad_norm)
        self.iters = int(iters)
        self.converged = bool(converged)
        self.message = str(message)
        self.path = list(path) if path is not None else []
        self.f_hist = list(f_hist) if f_hist is not None else []
        self.g_hist = list(g_hist) if g_hist is not None else []
        self.extra = dict(extra) if extra is not None else {}

    def __repr__(self):
        return (f"Result(converged={self.converged}, iters={self.iters}, "
                f"fun={self.fun:.6e}, grad_norm={self.grad_norm:.6e}, message='{self.message}')")

def should_stop(g_norm, k, tol, max_iters):
    """Checks if gradient norm falls below tolerance or max iterations reached."""
    return bool(float(g_norm) < float(tol) or int(k) >= int(max_iters))

def check_stopping(grad_norm, iters, tol, max_iters):
    """Checks gradient tolerance, maximum iterations, and non-finite breakdown."""
    if grad_norm < tol:
        return True, True, "Converged: gradient norm below tolerance"
    if iters >= max_iters:
        return True, False, "Maximum iterations reached"
    return False, False, ""

import math

def is_diverged(f_val, g_val, threshold=1e12):
    """Checks if function value or gradient norm indicates numerical divergence."""
    return bool(
        math.isnan(f_val) or math.isinf(f_val) or
        math.isnan(g_val) or math.isinf(g_val) or
        abs(f_val) > threshold or g_val > threshold
    )
