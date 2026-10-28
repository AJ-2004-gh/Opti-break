import math
from core.vectors import dot, zeros

def identity(n):
    """
    Returns an n x n identity matrix I.
    """
    if n <= 0:
        raise ValueError("Matrix dimension must be positive")
    A = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                row.append(0.0)
        A.append(row)
    return A

def transpose(A):
    """Computes matrix transpose B = A^T."""
    rows = len(A)
    if rows == 0:
        return []
    cols = len(A[0])
    T = []
    for j in range(cols):
        row = []
        for i in range(rows):
            row.append(float(A[i][j]))
        T.append(row)
    return T

def mat_vec(A, v):
    """Computes matrix-vector product y = A x."""
    rows = len(A)
    if rows == 0:
        return []
    cols = len(A[0])
    if len(v) != cols:
        raise ValueError(f"Matrix columns ({cols}) do not match vector length ({len(v)})")
    y = []
    for i in range(rows):
        y.append(dot(A[i], v))
    return y

def mat_mat(A, B):
    """Computes matrix-matrix product C = A B."""
    rows_A = len(A)
    if rows_A == 0:
        return []
    cols_A = len(A[0])
    rows_B = len(B)
    if rows_B == 0:
        return []
    cols_B = len(B[0])

    if cols_A != rows_B:
        raise ValueError(f"Matrix dimensions mismatch: ({rows_A}x{cols_A}) and ({rows_B}x{cols_B})")

    C = []
    for i in range(rows_A):
        row = [0.0] * cols_B
        C.append(row)

    for i in range(rows_A):
        for k in range(cols_A):
            aik = float(A[i][k])
            if aik != 0.0:
                for j in range(cols_B):
                    C[i][j] += aik * float(B[k][j])
    return C

def outer(u, v):
    """Computes outer product matrix A = u v^T."""
    m = len(u)
    n = len(v)
    M = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append(float(u[i]) * float(v[j]))
        M.append(row)
    return M

def mat_add(A, B):
    """Computes elementwise matrix addition C = A + B."""
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0
    if len(B) != rows or (rows > 0 and len(B[0]) != cols):
        raise ValueError("Matrix dimensions must match for addition")
    C = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(float(A[i][j]) + float(B[i][j]))
        C.append(row)
    return C

def mat_scale(c, A):
    """Computes scalar-matrix multiplication B = c * A."""
    c_val = float(c)
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0
    C = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(c_val * float(A[i][j]))
        C.append(row)
    return C

def diag(d):
    """
    Constructs a square diagonal matrix from list d.
    """
    n = len(d)
    A = []
    for i in range(n):
        row = [0.0] * n
        row[i] = float(d[i])
        A.append(row)
    return A

def is_symmetric(A, tol=1e-9):
    """
    Returns True if square matrix A satisfies A == A^T within tol.
    """
    rows = len(A)
    if rows == 0:
        return True
    cols = len(A[0])
    if rows != cols:
        return False
    for i in range(rows):
        for j in range(i + 1, cols):
            if abs(float(A[i][j]) - float(A[j][i])) > tol:
                return False
    return True

# Aliases and helper functions
mat_identity = identity
mat_transpose = transpose
mat_vec_mul = mat_vec
mat_mat_mul = mat_mat
outer_product = outer
mat_diag = diag

def mat_zeros(rows, cols):
    """
    Returns a rows x cols matrix filled with zeros.
    """
    A = []
    for _ in range(rows):
        row = [0.0] * cols
        A.append(row)
    return A

def mat_clone(A):
    """
    Returns a deep copy of matrix A.
    """
    C = []
    for row in A:
        C.append([float(x) for x in row])
    return C

def mat_shape(A):
    """
    Returns (rows, cols) dimensions of matrix A.
    """
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0
    return (rows, cols)

def mat_sub(A, B):
    """
    C = A - B
    """
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0
    C = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(float(A[i][j]) - float(B[i][j]))
        C.append(row)
    return C

def mat_trace(A):
    """
    tr(A) = sum(A_ii)
    """
    n = len(A)
    s = 0.0
    for i in range(n):
        s += float(A[i][i])
    return s

def mat_diag_add(A, val):
    """
    C = A + val * I
    """
    n = len(A)
    C = mat_clone(A)
    fval = float(val)
    for i in range(n):
        C[i][i] += fval
    return C

def quadratic_form(A, v):
    """
    q = v^T * A * v
    """
    Av = mat_vec(A, v)
    return dot(v, Av)

def mat_frobenius_norm(A):
    """
    ||A||_F = sqrt(sum(A_ij^2))
    """
    s = 0.0
    for row in A:
        for val in row:
            fval = float(val)
            s += fval * fval
    return math.sqrt(s)
