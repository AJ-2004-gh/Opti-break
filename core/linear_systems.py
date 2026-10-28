import math
from core.vectors import zeros
from core.matrices import identity, is_symmetric

def solve(A, b, tol=1e-12):
    """Solves linear system A x = b via Gaussian elimination with partial pivoting."""
    n = len(A)
    if n == 0 or len(b) != n:
        raise ValueError("Matrix and vector dimensions must match and be non-empty")

    # Clone matrix and rhs vector
    M = [[float(val) for val in row] for row in A]
    rhs = [float(val) for val in b]

    # Forward elimination with partial pivoting
    for k in range(n):
        # Find pivot index
        pivot_row = k
        max_val = abs(M[k][k])
        for i in range(k + 1, n):
            val = abs(M[i][k])
            if val > max_val:
                max_val = val
                pivot_row = i

        # Check for singularity
        if max_val < tol:
            raise ValueError("singular matrix")

        # Swap rows
        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]
            rhs[k], rhs[pivot_row] = rhs[pivot_row], rhs[k]

        pivot = M[k][k]
        for i in range(k + 1, n):
            factor = M[i][k] / pivot
            M[i][k] = 0.0
            for j in range(k + 1, n):
                M[i][j] -= factor * M[k][j]
            rhs[i] -= factor * rhs[k]

    # Back substitution
    x = zeros(n)
    for i in range(n - 1, -1, -1):
        accum = rhs[i]
        for j in range(i + 1, n):
            accum -= M[i][j] * x[j]
        x[i] = accum / M[i][i]

    return x

def inverse(A, tol=1e-12):
    """
    Computes A^{-1} by solving A x_j = e_j for each column of the identity matrix.
    """
    n = len(A)
    I = identity(n)
    inv_cols = []
    for j in range(n):
        e_j = [I[i][j] for i in range(n)]
        x_j = solve(A, e_j, tol=tol)
        inv_cols.append(x_j)

    # Reconstruct matrix by rows from columns
    inv_A = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(inv_cols[j][i])
        inv_A.append(row)

    return inv_A

def determinant(A, tol=1e-12):
    """
    Computes det(A) using Gaussian elimination with partial pivoting.
    """
    n = len(A)
    if n == 0:
        return 1.0
    if n == 1:
        return float(A[0][0])

    M = [[float(val) for val in row] for row in A]
    sign = 1.0
    det = 1.0

    for k in range(n):
        pivot_row = k
        max_val = abs(M[k][k])
        for i in range(k + 1, n):
            val = abs(M[i][k])
            if val > max_val:
                max_val = val
                pivot_row = i

        if max_val < tol:
            return 0.0

        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]
            sign = -sign

        pivot = M[k][k]
        det *= pivot

        for i in range(k + 1, n):
            factor = M[i][k] / pivot
            for j in range(k + 1, n):
                M[i][j] -= factor * M[k][j]

    return sign * det

def rank(A, tol=1e-9):
    """
    Computes rank of matrix A by reduction to row echelon form.
    """
    rows = len(A)
    if rows == 0:
        return 0
    cols = len(A[0])
    M = [[float(val) for val in row] for row in A]

    rank_count = 0
    lead = 0

    for r in range(rows):
        if lead >= cols:
            break
        i = r
        while abs(M[i][lead]) < tol:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if lead == cols:
                    break
        if lead == cols:
            break

        # Swap rows r and i
        M[r], M[i] = M[i], M[r]

        pivot = M[r][lead]
        for i in range(r + 1, rows):
            factor = M[i][lead] / pivot
            for j in range(lead, cols):
                M[i][j] -= factor * M[r][j]

        lead += 1
        rank_count += 1

    return rank_count

def is_positive_definite(A, tol=1e-12):
    """
    Checks if symmetric matrix A is positive definite via Cholesky decomposition.
    """
    n = len(A)
    if not is_symmetric(A, tol=1e-7):
        return False

    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            accum = float(A[i][j])
            for k in range(j):
                accum -= L[i][k] * L[j][k]
            if i == j:
                if accum <= tol:
                    return False
                L[i][j] = math.sqrt(accum)
            else:
                L[i][j] = accum / L[j][j]
    return True

def eigenvalues_2x2(A):
    """Computes real eigenvalues of a 2x2 matrix analytically in ascending order."""
    if len(A) != 2 or len(A[0]) != 2 or len(A[1]) != 2:
        raise ValueError("Matrix must be 2x2")

    a = float(A[0][0])
    b = float(A[0][1])
    c = float(A[1][0])
    d = float(A[1][1])

    trace = a + d
    det = a * d - b * c
    discriminant = trace * trace - 4.0 * det

    if discriminant < 0.0:
        discriminant = 0.0

    sqrt_disc = math.sqrt(discriminant)
    lam1 = (trace - sqrt_disc) / 2.0
    lam2 = (trace + sqrt_disc) / 2.0

    return sorted([lam1, lam2])

# Aliases
solve_linear_system = solve
matrix_inverse = inverse
