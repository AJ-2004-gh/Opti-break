import unittest
from core.linear_systems import (
    solve, inverse, determinant, rank, is_positive_definite, eigenvalues_2x2
)
from core.matrices import mat_mat

class TestLinearSystems(unittest.TestCase):
    def test_solve_3x3_known_answer(self):
        A = [
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 4.0],
            [5.0, 6.0, 0.0]
        ]
        # For x = [1, 2, 3], A*x = [14, 14, 17]
        b = [14.0, 14.0, 17.0]
        x = solve(A, b)
        self.assertAlmostEqual(x[0], 1.0, places=9)
        self.assertAlmostEqual(x[1], 2.0, places=9)
        self.assertAlmostEqual(x[2], 3.0, places=9)

    def test_solve_needs_pivoting(self):
        # A[0][0] is 0, so Gaussian elimination without pivoting would divide by zero
        A = [
            [0.0, 2.0, 1.0],
            [1.0, -2.0, -3.0],
            [-1.0, 1.0, 2.0]
        ]
        b = [-8.0, 0.0, 3.0]
        x = solve(A, b)
        # Verify A * x == b
        self.assertAlmostEqual(2.0 * x[1] + 1.0 * x[2], -8.0, places=9)
        self.assertAlmostEqual(1.0 * x[0] - 2.0 * x[1] - 3.0 * x[2], 0.0, places=9)
        self.assertAlmostEqual(-1.0 * x[0] + 1.0 * x[1] + 2.0 * x[2], 3.0, places=9)

    def test_singular_matrix_raises_value_error(self):
        A = [[1.0, 2.0], [2.0, 4.0]]
        b = [1.0, 2.0]
        with self.assertRaises(ValueError) as ctx:
            solve(A, b)
        self.assertIn("singular", str(ctx.exception).lower())

    def test_inverse_identity_within_1e9(self):
        A = [
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 1.0],
            [2.0, 0.0, 1.0]
        ]
        A_inv = inverse(A)
        prod = mat_mat(A, A_inv)
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(prod[i][j], expected, delta=1e-9)

    def test_determinant_known_3x3(self):
        A = [
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 4.0],
            [5.0, 6.0, 0.0]
        ]
        # det = 1*(0 - 24) - 2*(0 - 20) + 3*(0 - 5) = -24 + 40 - 15 = 1.0
        self.assertAlmostEqual(determinant(A), 1.0, places=9)

    def test_rank_matrix(self):
        self.assertEqual(rank([[1.0, 2.0], [2.0, 4.0]]), 1)
        self.assertEqual(rank([[1.0, 0.0], [0.0, 1.0]]), 2)

    def test_is_positive_definite(self):
        self.assertTrue(is_positive_definite([[2.0, 0.0], [0.0, 3.0]]))
        self.assertFalse(is_positive_definite([[1.0, 0.0], [0.0, -1.0]]))

    def test_eigenvalues_2x2_sorted(self):
        eigs = eigenvalues_2x2([[2.0, 0.0], [0.0, -2.0]])
        self.assertEqual(eigs, [-2.0, 2.0])

if __name__ == "__main__":
    unittest.main()
