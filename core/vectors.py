import math

def validate_vector(v):
    """
    Validates that v is a non-empty list of numeric values.
    """
    if not isinstance(v, list) or len(v) == 0:
        raise ValueError("Vector must be a non-empty list")
    for val in v:
        if not isinstance(val, (int, float)):
            raise ValueError("Vector elements must be numbers")

def zeros(n):
    """
    Returns zero vector of dimension n.
    """
    if n <= 0:
        raise ValueError("Dimension must be positive")
    out = []
    for _ in range(n):
        out.append(0.0)
    return out

def vec_add(u, v):
    """Computes vector addition w = u + v."""
    if len(u) != len(v):
        raise ValueError(f"Vector length mismatch: {len(u)} != {len(v)}")
    w = []
    for i in range(len(u)):
        w.append(float(u[i]) + float(v[i]))
    return w

def vec_sub(u, v):
    """Computes vector subtraction w = u - v."""
    if len(u) != len(v):
        raise ValueError(f"Vector length mismatch: {len(u)} != {len(v)}")
    w = []
    for i in range(len(u)):
        w.append(float(u[i]) - float(v[i]))
    return w

def scalar_mul(c, v):
    """Computes scalar-vector multiplication w = c * v."""
    c_val = float(c)
    w = []
    for val in v:
        w.append(c_val * float(val))
    return w

def dot(u, v):
    """Computes Euclidean inner product dot(u, v) = u^T v."""
    if len(u) != len(v):
        raise ValueError(f"Vector length mismatch: {len(u)} != {len(v)}")
    s = 0.0
    for i in range(len(u)):
        s += float(u[i]) * float(v[i])
    return s

def norm_l1(v):
    """Computes L1 Manhattan vector norm."""
    s = 0.0
    for val in v:
        s += abs(float(val))
    return s

def norm_l2(v):
    """Computes L2 Euclidean vector norm."""
    s = 0.0
    for val in v:
        fval = float(val)
        s += fval * fval
    return math.sqrt(s)

def norm_inf(v):
    """Computes infinity Chebyshev vector norm."""
    if len(v) == 0:
        raise ValueError("Cannot compute infinity norm of empty vector")
    m = 0.0
    for val in v:
        aval = abs(float(val))
        if aval > m:
            m = aval
    return m

def hadamard(u, v):
    """Computes elementwise vector Hadamard product."""
    if len(u) != len(v):
        raise ValueError(f"Vector length mismatch: {len(u)} != {len(v)}")
    w = []
    for i in range(len(u)):
        w.append(float(u[i]) * float(v[i]))
    return w

def vec_div_elementwise(u, v, eps=1e-12):
    """Computes elementwise vector division with epsilon safety."""
    if len(u) != len(v):
        raise ValueError(f"Vector length mismatch: {len(u)} != {len(v)}")
    w = []
    for i in range(len(u)):
        denom = float(v[i])
        if abs(denom) < eps:
            denom = eps if denom >= 0.0 else -eps
        w.append(float(u[i]) / denom)
    return w

# Aliases and helper routines for compatibility across repository
vec_scale = scalar_mul
vec_dot = dot
vec_zero = zeros
vec_hadamard = hadamard
vec_div = vec_div_elementwise

def vec_norm(v, p=2):
    """
    Computes vector p-norm (defaults to L2).
    """
    if p == 1:
        return norm_l1(v)
    elif p == float("inf") or p == "inf":
        return norm_inf(v)
    return norm_l2(v)

def vec_clone(v):
    """
    Returns copy of vector.
    """
    return [float(x) for x in v]

def vec_ones(n):
    """
    Returns vector of ones of length n.
    """
    return [1.0] * n

def vec_neg(v):
    """
    w = -v
    """
    return [-float(x) for x in v]

def vec_dist(u, v, p=2):
    """
    d = ||u - v||
    """
    diff = vec_sub(u, v)
    if p == 1:
        return norm_l1(diff)
    elif p == float("inf") or p == "inf":
        return norm_inf(diff)
    return norm_l2(diff)

def vec_clip(v, min_val, max_val):
    """
    Clips vector elements into [min_val, max_val].
    """
    w = []
    for val in v:
        fval = float(val)
        if fval < min_val:
            w.append(float(min_val))
        elif fval > max_val:
            w.append(float(max_val))
        else:
            w.append(fval)
    return w

def vec_lincomb(c1, u, c2, v):
    """
    w = c1 * u + c2 * v
    """
    return vec_add(scalar_mul(c1, u), scalar_mul(c2, v))

def vec_project_box(v, lower, upper):
    """
    Projects vector onto box [lower, upper].
    """
    w = []
    for i in range(len(v)):
        fval = float(v[i])
        l = float(lower[i])
        u = float(upper[i])
        if fval < l:
            w.append(l)
        elif fval > u:
            w.append(u)
        else:
            w.append(fval)
    return w

def vec_project_ball(v, radius):
    """
    Projects vector onto L2 ball of given radius.
    """
    nrm = norm_l2(v)
    if nrm <= radius or nrm == 0.0:
        return vec_clone(v)
    scale = float(radius) / nrm
    return scalar_mul(scale, v)
